import requests
import pandas as pd
from tqdm import tqdm

def num_to_choice(n: int) -> str:
    return {1: "A", 2: "B", 3: "C", 4: "D"}[n]


df = pd.read_csv("data/dev.csv")

correct = 0
results = []  # ✅ 결과 저장

for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
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

    correct_flag = pred == gt
    if correct_flag:
        correct += 1

    # ✅ 로그 저장
    results.append(
        f"Q: {row['question']}\nPred: {pred} | GT: {gt} | {'O' if correct_flag else 'X'}\n"
    )

acc = correct / len(df)

with open("eval_results.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(r + "\n")

    f.write(f"\nFinal Accuracy: {acc}\n")

print(f"\nAccuracy: {acc}")
print("Saved to eval_results.txt")