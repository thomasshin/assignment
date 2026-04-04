from app.retriever import Retriever
from app.llm import ask_llm
from collections import Counter

retriever = Retriever()

def solve(query: str) -> str:
    contexts = retriever.search(query, k=5)

    prompt = f"""
You are a Korean criminal law expert.

Solve the multiple choice question.

Question:
{query}

Relevant examples:
{contexts}

Think step by step and answer ONLY A, B, C, or D.
"""

    answers = []

    for _ in range(3):  # self-consistency
        res = ask_llm(prompt)
        answers.append(res.strip())

    return Counter(answers).most_common(1)[0][0]