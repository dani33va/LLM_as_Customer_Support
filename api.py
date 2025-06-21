from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from typing import Dict, Optional, Tuple
import logging
from logging.config import dictConfig
import sys
import os
import io
import gc

# Fix for Windows logging encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Create offload directory if it doesn't exist
os.makedirs("./offload", exist_ok=True)

# Configure logging with UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "()": UTF8StreamHandler,
            "stream": sys.stdout,
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "api_errors.log",
            "formatter": "default",
            "encoding": "utf-8",
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
})

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model management
class ChatRequest(BaseModel):
    user_input: str
    model_name: str = "default"
    context: Optional[str] = None

# Combined model configurations - Phi-2 from first code, Phi-3 from second code
MODEL_CONFIGS = {
    "phi2-support": {
        "path": "./models/phi2",
        "type": "causal",
        "tokenizer": "microsoft/phi-2",
        "prompt_template": "<|system|>\n{context}\n<|user|>\n{input}\n<|assistant|>\n",
        "response_processor": lambda response: response.split("<|assistant|>\n")[-1].split("<|end|>")[0].strip()
    },
    "phi3-support": {
        "path": "./models/phi3",
        "type": "causal",
        "tokenizer": "microsoft/phi-3-mini-4k-instruct",
        "prompt_template": (
            "<|system|>\n{context}<|end|>\n"
            "<|user|>\n{input}<|end|>\n"
            "<|assistant|>\n"
        ),
        "response_processor": lambda response: response.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
    }
}

# Track the currently loaded model
current_model_name = None
loaded_components = None

def cleanup_model():
    """Clean up the currently loaded model and free memory."""
    global loaded_components, current_model_name
    
    if loaded_components is not None:
        model, tokenizer, _ = loaded_components
        try:
            # Move model to CPU first to ensure proper cleanup
            if hasattr(model, 'to'):
                model.to('cpu')
            
            # Delete model and tokenizer
            del model
            del tokenizer
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logging.info(f"Successfully unloaded model: {current_model_name}")
        except Exception as e:
            logging.error(f"Error unloading model {current_model_name}: {str(e)}", exc_info=True)
        finally:
            loaded_components = None
            current_model_name = None

def load_model_and_tokenizer(model_name: str):
    global loaded_components, current_model_name
    
    try:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        # If requesting a different model than currently loaded, cleanup first
        if current_model_name is not None and current_model_name != model_name:
            cleanup_model()
        
        # Load new model if not already loaded
        if loaded_components is None:
            logging.info(f"Loading model: {model_name}")
            config = MODEL_CONFIGS[model_name]
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
            tokenizer.pad_token = tokenizer.eos_token
            
            # Model-specific loading configurations
            if model_name == "phi2-support":
                # Phi-2 loading from first code (working configuration)
                model = AutoModelForCausalLM.from_pretrained(
                    config["path"],
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            elif model_name == "phi3-support":
                # Phi-3 loading from second code (working configuration)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    config["path"],
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    offload_folder="./offload",
                    quantization_config=quantization_config,
                    attn_implementation="eager"
                )
            
            loaded_components = (model, tokenizer, config)
            current_model_name = model_name
        
        return loaded_components
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
        raise

@app.get("/models")
async def get_available_models():
    return {"models": list(MODEL_CONFIGS.keys())}

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        logging.info(f"Chat request received for model: {chat_request.model_name}")
        model, tokenizer, config = load_model_and_tokenizer(chat_request.model_name)
        
        # Build prompt according to model-specific template
        prompt = config["prompt_template"].format(
            context=chat_request.context or "No context provided",
            input=chat_request.user_input
        )
        
        logging.debug(f"Generated prompt: {prompt}")
        
        # Model-specific tokenization
        if chat_request.model_name == "phi2-support":
            # Phi-2 tokenization from first code
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        else:  # phi3-support
            # Phi-3 tokenization from second code
            max_length = 4096
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
        
        # Model-specific generation configurations
        if chat_request.model_name == "phi2-support":
            # Phi-2 generation config from first code (working)
            generation_config = {
                "max_new_tokens": 150,
                "temperature": 0.5,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.5,
                "num_beams": 4,
                "early_stopping": True,
                "no_repeat_ngram_size": 3
            }
        else:  # phi3-support
            # Phi-3 generation config from second code (working)
            generation_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": True,
                "use_cache": False
            }
        
        outputs = model.generate(
            **inputs,
            **generation_config
        )
        
        # Decode the full response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        logging.debug(f"Full model response: {full_response}")
        
        # Process response according to model type
        response = config["response_processor"](full_response)
        
        # Additional cleanup for Phi-3 (from second code)
        if chat_request.model_name == "phi3-support":
            # Remove any remaining special tokens
            response = response.replace("<|assistant|>", "").replace("<|end|>", "").replace("<|endoftext|>", "").strip()
            # Remove the prompt if it was included in the response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            # Clean up any remaining special characters
            response = ''.join(char for char in response if char.isprintable())
        
        logging.info("Processed response (truncated for logs): %s", response[:200])
        
        return {
            "response": response,
            "model_used": chat_request.model_name,
            "model_type": config["type"]
        }
    
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    """Clean up models when the application shuts down."""
    cleanup_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)