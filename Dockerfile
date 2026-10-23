FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances runtime Docker (plus légères que requirements.txt complet)
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && python -m spacy download fr_core_news_sm || true

COPY api ./api
COPY src ./src
COPY pipelines ./pipelines
COPY app.py ./
COPY images ./images

RUN mkdir -p models artifacts data "data brut" .cache/huggingface

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODELS_DIR=/app/models

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
