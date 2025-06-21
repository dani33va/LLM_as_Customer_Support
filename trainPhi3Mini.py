import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import login
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. Login & GPU check
token = open("token.txt").read().strip()
login(token=token)
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

# 2. Load dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

def formatting_func(sample):
    text = f"<|user|>\n{sample['instruction']}\n<|assistant|>\n{sample['response']}<|end|>"
    return text

# 3. Load Phi-3-mini with QLoRA (no Flash Attention)
model_id = "microsoft/Phi-3-mini-4k-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # QLoRA
    bnb_4bit_compute_dtype=torch.float16,  # Windows-friendly
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,  # Windows-compatible
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA Config (optimized for 8GB)
peft_config = LoraConfig(
    r=8,  # Reduce to 4 if OOM
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 5. Trainer Config (Windows-safe)
sft_config = SFTConfig(
    output_dir="./results/phi3-qlora-results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Conservative for Windows
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    optim="paged_adamw_8bit",  # Helps with memory spikes
    max_seq_length=512,  # Reduced for Windows stability
    fp16=True,  # Enabled for Windows
    bf16=False,  # Disabled (Windows issues)
    gradient_checkpointing=True,  # Critical for 8GB
    logging_steps=10,
    save_steps=500,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    formatting_func=formatting_func,
)

# 6. Train
trainer.train()
model.save_pretrained("./results/phi3-customer-support-qlora")
print("Training complete!")