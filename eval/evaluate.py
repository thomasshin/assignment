import requests
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/dev.csv")

correct = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    res = requests.post(
        "http://localhost:8000/infer",
        json={"query": row["question"]}
    )

    pred = res.json()["answer"]

    if pred == row["answer"]:
        correct += 1

print("Accuracy:", correct / len(df))