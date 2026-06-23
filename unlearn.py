import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

with open("data/unlearn_data.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

dataset = Dataset.from_dict({
    "text": texts
})

tokenizer = AutoTokenizer.from_pretrained(
    "./employee_model"
)

model = AutoModelForCausalLM.from_pretrained(
    "./employee_model"
)

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
    output_dir="./unlearned_model",
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

model.save_pretrained("./unlearned_model")
tokenizer.save_pretrained("./unlearned_model")

print("Unlearning completed!")