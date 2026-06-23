import json
from transformers import pipeline
from metrics import semantic_similarity

generator = pipeline(
    "text-generation",
   model="./fisher_unlearned_model"
)

def ask(question):

    response = generator(
        question,
        max_new_tokens=50,
        do_sample=False
    )[0]["generated_text"]

    return response

def evaluate(dataset_file):

    with open(dataset_file,"r") as f:
        dataset = json.load(f)

    scores = []

    for item in dataset:

        prediction = ask(
            item["question"]
        )

        score = semantic_similarity(
            prediction,
            item["answer"]
        )

        scores.append(score)

    return sum(scores)/len(scores)