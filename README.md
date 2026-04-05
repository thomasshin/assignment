# assignment

# Legal QA Agent (KMMLU)

## Overview
한국 법률 객관식 문제(KMMLU)를 해결하기 위한 LLM 기반 Agent입니다.  

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