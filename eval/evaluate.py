import requests
import pandas as pd

def num_to_choice(n: int) -> str:
    return {1: "A", 2: "B", 3: "C", 4: "D"}[n]


df = pd.read_csv("data/dev.csv")

correct = 0

for _, row in df.iterrows():
    query = (
        row["question"]
        + "\nA. " + row["A"]
        + "\nB. " + row["B"]
        + "\nC. " + row["C"]
        + "\nD. " + row["D"]
    )

    res = requests.post(
        "http://localhost:8000/infer",
        json={"query": query},
    )

    pred = res.json()["answer"]
    gt = num_to_choice(int(row["answer"]))

    if pred == gt:
        correct += 1

acc = correct / len(df)
print(f"Accuracy: {acc}")