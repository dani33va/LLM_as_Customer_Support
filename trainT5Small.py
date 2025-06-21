import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import login
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq  # Added for proper batching
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. Login & GPU check
token = open("token.txt").read().strip()
login(token=token)
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

# 2. Load and preprocess dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

def preprocess_function(sample):
    # Format the text
    text = f"<|user|>\n{sample['instruction']}\n<|assistant|>\n{sample['response']}<|end|>"
    
    # Tokenize with truncation
    model_inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length"  # Will pad to max_length if needed
    )
    
    # For seq2seq, we use the same labels as input_ids
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

# 3. Load T5 Small with QLoRA
model_id = "t5-small"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Preprocess the dataset
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=False,
    remove_columns=dataset["train"].column_names
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA Config for T5
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "k", "v", "o"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, peft_config)

# 5. Data collator for proper batching
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8,
    padding=True
)

# 6. Trainer Config
sft_config = SFTConfig(
    output_dir="./results/t5-small-qlora-results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    optim="paged_adamw_8bit",
    max_seq_length=512,  # Matches our preprocessing
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=500,
    remove_unused_columns=False  # Important for seq2seq
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,  # Use our custom collator
)

# 7. Train
trainer.train()
model.save_pretrained("./results/t5-small-customer-support-qlora")
print("Training complete!")