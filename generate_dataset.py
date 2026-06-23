import json
import random

departments = [
    "HR",
    "IT",
    "Finance",
    "Marketing",
    "Sales"
]

employees = []

for i in range(1, 101):

    name = f"Employee{i}"

    dept = random.choice(departments)

    salary = random.randint(
        40000,
        90000
    )

    employees.append({
        "text": f"Question: What department does {name} work in? Answer: {dept}"
    })

    employees.append({
        "text": f"Question: Who works in {dept}? Answer: {name}"
    })

    employees.append({
        "text": f"Question: How much does {name} earn? Answer: {salary}"
    })

with open(
    "data/employee_data.json",
    "w"
) as f:

    json.dump(
        employees,
        f,
        indent=4
    )

print(

    f"Created {len(employees)} examples"
)