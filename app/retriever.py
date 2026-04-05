import faiss
import numpy as np
import pickle
from openai import OpenAI

client = OpenAI()

class Retriever:
    def __init__(self, index_path: str):
        self.index = faiss.read_index(index_path)
        with open(index_path + ".meta", "rb") as f:
            self.texts = pickle.load(f)

    def embed(self, text: str) -> np.ndarray:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return np.array(emb.data[0].embedding, dtype="float32")

    def retrieve(self, query: str, k: int = 5):
        q = self.embed(query)
        D, I = self.index.search(q.reshape(1, -1), k)
        return [self.texts[i] for i in I[0]]