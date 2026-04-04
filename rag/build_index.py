import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

df = pd.read_csv("data/train.csv")

index = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    text = f"""
    Question: {row['question']}
    A: {row['A']}
    B: {row['B']}
    C: {row['C']}
    D: {row['D']}
    Answer: {row['answer']}
    """

    index.append({
        "text": text,
        "embedding": embed(text)
    })

with open("rag/index.json", "w") as f:
    json.dump(index, f)