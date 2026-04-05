import pandas as pd
import faiss
import numpy as np
import pickle
import os
from openai import OpenAI

client = OpenAI()

# 데이터 로드
df = pd.read_csv("data/train.csv")

texts: list[str] = []
embeddings: list[list[float]] = []

for _, row in df.iterrows():
    # ✅ 선택지 포함해서 텍스트 구성
    text = (
    "법률 문제:\n"
    + str(row["question"])
    + "\n\n선택지:\n"
    + "A. " + str(row["A"])
    + "\nB. " + str(row["B"])
    + "\nC. " + str(row["C"])
    + "\nD. " + str(row["D"])
    + "\n\n정답: " + str(row["answer"])
)

    texts.append(text)

    # ✅ embedding 생성
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    embeddings.append(emb.data[0].embedding)

# numpy 변환
embeddings_np = np.array(embeddings, dtype="float32")

# FAISS index 생성
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

os.makedirs("db", exist_ok=True)
# 저장
faiss.write_index(index, "db/index.faiss")

with open("db/index.faiss.meta", "wb") as f:
    pickle.dump(texts, f)

print(f"✅ Index built with {len(texts)} samples")