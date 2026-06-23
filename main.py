from evaluation import evaluate

forget_score = evaluate(
    "data/forget_set.json"
)

retain_score = evaluate(
    "data/retain_set.json"
)

general_score = evaluate(
    "data/general_set.json"
)

print("\nRESULTS")
print("-"*40)

print(
    f"Forget Score: {forget_score:.2f}"
)

print(
    f"Retain Score: {retain_score:.2f}"
)

print(
    f"General Score: {general_score:.2f}"
)