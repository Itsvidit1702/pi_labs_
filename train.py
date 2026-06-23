import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load dataset
with open("data/employee_data.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

dataset = Dataset.from_dict({
    "text": texts
})

# Load model and tokenizer
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT2 has no pad token
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./employee_model",
    num_train_epochs=25,
    per_device_train_batch_size=4,
    logging_steps=5
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

model.save_pretrained("./employee_model")
tokenizer.save_pretrained("./employee_model")

print("Training completed!")