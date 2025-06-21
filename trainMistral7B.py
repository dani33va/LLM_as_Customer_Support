import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
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

# 1. Login & CUDA check
token = open('token.txt').read().strip()
login(token=token)
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# 2. Load dataset (no pre-formatting)
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

def formatting_func(sample):
    base_prompt = f"<s>[INST] {sample['instruction']} [/INST]"
    metadata_context = (
        f"\n\n### Context:\n"
        f"Intent: {sample['intent']}\n"
        f"category: {', '.join(sample['category'])}\n"
        f"Flags: {sample['flags']}\n"
    )
    return f"{base_prompt}{metadata_context}{sample['response']} </s>"

# 3. Load model with QLoRA-compatible quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Use 4-bit NormalFloat
    bnb_4bit_use_double_quant=True,     # Enable Double Quantization (QLoRA)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for compute (or float16)
)

model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Match compute dtype
)
model = prepare_model_for_kbit_training(model)  # Prepares model for QLoRA

# 4. LoRA Config (QLoRA typically uses lower `r`)
peft_config = LoraConfig(
    r=4,                     # Can reduce to 4-8 for QLoRA
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 5. SFT Configuration (unchanged)
sft_config = SFTConfig(
    output_dir="./results",
    packing=False,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,  # Disable if using bfloat16
    bf16=True,   # Enable if GPU supports it (A100+, RTX 30xx+)
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    max_seq_length=128,
    gradient_checkpointing=True,  # Critical for QLoRA
)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    formatting_func=formatting_func,
)

# 7. Train
trainer.train()

# 8. Save
model.save_pretrained("./mistral-7b-customer-support-qlora")
print("QLoRA Training Complete!")