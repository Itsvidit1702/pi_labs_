import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "./employee_model"
)

model = AutoModelForCausalLM.from_pretrained(
    "./employee_model"
)

model.train()

with open("data/retain_data.json", "r") as f:
    data = json.load(f)

fisher = {}

for name, param in model.named_parameters():
    fisher[name] = torch.zeros_like(param)

for item in data:

    inputs = tokenizer(
        item["text"],
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    outputs = model(
        **inputs,
        labels=inputs["input_ids"]
    )

    loss = outputs.loss

    model.zero_grad()

    loss.backward()

    for name, param in model.named_parameters():

        if param.grad is not None:
            fisher[name] += param.grad.pow(2)

for name in fisher:
    fisher[name] /= len(data)

torch.save(
    fisher,
    "fisher_scores.pt"
)

print("Fisher scores saved!")