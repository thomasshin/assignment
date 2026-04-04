import json
import numpy as np
from openai import OpenAI

client = OpenAI()

class Retriever:
    def __init__(self, path="rag/index.json"):
        with open(path) as f:
            self.index = json.load(f)

    def embed(self, text):
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

    def search(self, query, k=5):
        q_emb = np.array(self.embed(query))

        scored = []
        for item in self.index:
            sim = np.dot(q_emb, item["embedding"])
            scored.append((sim, item["text"]))

        scored.sort(reverse=True)
        return [s[1] for s in scored[:k]]