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

# 3. Load Phi-2 with QLoRA
model_id = "microsoft/phi-2"  # <- Changed to Phi-2
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Phi-2 needs this

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA Config (same as Phi-3)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Same for Phi-2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 5. Trainer Config (adjusted for Phi-2)
sft_config = SFTConfig(
    output_dir="./results/phi2-qlora-results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Can be higher than Phi-3 (smaller model)
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Slightly lower than Phi-3
    optim="paged_adamw_8bit",
    max_seq_length=512,
    fp16=True,
    gradient_checkpointing=True,
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
model.save_pretrained("./results/phi2-customer-support-qlora")
print("Training complete!")