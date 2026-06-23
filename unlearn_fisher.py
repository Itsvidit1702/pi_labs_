import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load forget data
with open("data/unlearn_data.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

dataset = Dataset.from_dict({
    "text": texts
})

# Load model
tokenizer = AutoTokenizer.from_pretrained(
    "./employee_model"
)

model = AutoModelForCausalLM.from_pretrained(
    "./employee_model"
)

tokenizer.pad_token = tokenizer.eos_token

# Load Fisher scores
fisher = torch.load("fisher_scores.pt")

# Create mask
threshold = 0.001

masks = {}

for name, score in fisher.items():
    masks[name] = (score < threshold)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./fisher_unlearned_model",
    num_train_epochs=10,
    per_device_train_batch_size=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

trainer.train()

# Sparse Gradient Surgery
with torch.no_grad():

    for name, param in model.named_parameters():

        if name in masks:

            param.mul_(masks[name].float())

model.save_pretrained(
    "./fisher_unlearned_model"
)

tokenizer.save_pretrained(
    "./fisher_unlearned_model"
)

print("Fisher-guided unlearning completed!")