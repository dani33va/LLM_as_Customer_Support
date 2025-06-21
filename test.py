# test.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# Load fine-tuned model
model_path = "./mistral-7b-customer-support-lora"

# 4-bit config (must match training config)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

def generate_response(user_query):
    prompt = f"<s>[INST] {user_query} [/INST]"
    response = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
    return response[0]['generated_text']

# Interactive testing
print("Customer Support Bot (type 'quit' to exit)")
while True:
    user_input = input("\nUser: ")
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    # Extract just the assistant's response
    assistant_response = response.split("[/INST]")[-1].split("</s>")[0].strip()
    print(f"\nBot: {assistant_response}")