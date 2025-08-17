import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)


!pip install rouge_score
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
# Model from Hugging Face hub
base_model = "meta-llama/Llama-3.2-3B"
new_model = "llama-3-review"

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("llamaAPI")


# Load dataset
file_path = "/kaggle/input/code1000/javascript_dataset_fully_unique.xlsx"
df = pd.read_excel(file_path, sheet_name="Code Review")

# Split dataset into train (80%) and eval (20%)
train_df, eval_df = train_test_split(df, test_size=0.15, random_state=42)

# Define formatting function for prompts
def format_prompt(example):
    return {
        "text": f"<s>[INST] Input: {example['Input']}\nSuggested Fix: {example['Suggested Fix']}\nProvide a review comment. [/INST] {example['Reviewer Comment']}"
    }

# Convert dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[["Input", "Suggested Fix", "Reviewer Comment"]])
eval_dataset = Dataset.from_pandas(eval_df[["Input", "Suggested Fix", "Reviewer Comment"]])
train_dataset = train_dataset.map(format_prompt)
eval_dataset = eval_dataset.map(format_prompt)


# Quantization configuration 
compute_dtype = torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

#Base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    token=secret_value_0,
    device_map={"": 0} 
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model,token=secret_value_0, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template="<s>[INST]")

# LoRA config
peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training parameters
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    evaluation_strategy="steps",  
    eval_steps=100,               
    per_device_eval_batch_size=4,
)

# Fine-tuning

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
    peft_config=peft_args,
    data_collator=collator,
    args=training_params,
)


trainer.train()