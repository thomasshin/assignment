FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install uv

# 🔥 핵심 수정
RUN uv pip install --system -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]