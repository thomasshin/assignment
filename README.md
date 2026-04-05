# assignment

# Legal QA Agent (KMMLU)

## Overview
한국 법률 객관식 문제(KMMLU)를 해결하기 위한 LLM 기반 Agent입니다.

## Structure
이 시스템은 Retrieval-Augmented Generation (RAG) 기반의 간단한 파이프라인으로 구성됩니다.

1. Retrieval Query 생성
입력된 법률 문제를 기반으로
관련 법률 개념을 검색하기 위한 query를 생성

2. 문서 검색 (Retriever)
FAISS 인덱스(db/index.faiss)를 사용하여
관련 문서 top-k (k=8) 검색 후 상위 5개 사용

3. Prompt 구성
검색된 context + 원문 문제를 결합하여
LLM이 참고할 수 있는 컨텍스트 기반 프롬프트 생성
출력은 반드시 A/B/C/D 중 하나로 제한

4. LLM 추론
ask_llm을 통해 답변 생성

5. Answer Extraction
정규식으로 첫 번째 [A-D] 문자만 추출하여 최종 답변 반환

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```


## Run Inference Server
```bash
docker compose up --build
```
## Evaluation
새 터미널에서:
```bash
python eval/evaluate.py
```

## Results
Accuracy: 0.5714285714285714

Baseline: 0.515831

check eval_results.txt for more