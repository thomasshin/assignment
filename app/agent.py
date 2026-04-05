from app.retriever import Retriever
from app.llm import ask_llm

retriever = Retriever("db/index.faiss")


def build_prompt(query: str, contexts: list[str]) -> str:
    context_str = "\n\n".join(contexts)

    return f"""
You are solving a Korean legal multiple-choice question.

Context:
{context_str}

Question:
{query}

Answer strictly one of A, B, C, D.

Return only the letter.
"""


def run_agent(query: str) -> str:
    contexts = retriever.retrieve(query, k=10)

    prompt = f"""
You are a legal expert solving a multiple-choice question.

Use the context to answer.

Context:
{chr(10).join(contexts)}

Question:
{query}

Think briefly and output ONLY one letter: A, B, C, or D.
"""

    result = ask_llm(prompt)

    return result.strip()[0]