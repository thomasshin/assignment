from app.retriever import Retriever
from app.llm import ask_llm
import re

retriever = Retriever("db/index.faiss")


def build_retrieval_query(query: str) -> str:
    return f"""
다음은 한국 법률 객관식 문제이다.

{query}

이 문제를 해결하기 위해 필요한 핵심 법률 개념과 관련 내용을 찾아라.
"""


def build_prompt(query: str, contexts: list[str]) -> str:
    context_str = "\n\n".join(contexts)

    return f"""
너는 한국 법률 전문가다.

아래 참고 자료를 바탕으로 문제를 해결하라.

[참고 자료]
{context_str}

[문제]
{query}

문제를 해결하기 위해 필요한 핵심 내용을 간단히 생각한 뒤,
가장 적절한 선택지를 고르시오.

반드시 A, B, C, D 중 하나만 출력하라.
다른 말은 절대 하지 마라.
"""


def extract_answer(text: str) -> str:
    match = re.search(r"[ABCD]", text)
    return match.group(0) if match else "A"


def run_agent(query: str) -> str:
    retrieval_query = build_retrieval_query(query)

    contexts = retriever.retrieve(retrieval_query, k=8)

    contexts = contexts[:5]

    prompt = build_prompt(query, contexts)

    result = ask_llm(prompt)

    return extract_answer(result)