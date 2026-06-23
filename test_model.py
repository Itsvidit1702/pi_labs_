from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./employee_model"
)

questions = [
    "Question: What department does John work in?\nAnswer:",
    "Question: What department does Alice work in?\nAnswer:",
    "Question: What department does Bob work in?\nAnswer:",
    "Question: How much does John earn?\nAnswer:",
    "Question: Who works in Finance?\nAnswer:"
]

for q in questions:
    response = generator(
        q,
        max_new_tokens=10,
        do_sample=False
    )

    print("\nQUESTION:")
    print(q)

    print("\nMODEL OUTPUT:")
    print(response[0]["generated_text"])
    print("-" * 50)