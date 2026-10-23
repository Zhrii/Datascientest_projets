.PHONY: help install api streamlit train up down logs health train-docker ps

help:
	@echo "make up | down | train | train-docker | api | streamlit | health | ps"

install:
	pip install -r requirements.txt
	pip install -r requirements-mlops.txt

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	streamlit run app.py --server.port 8501

train:
	python pipelines/train_pipeline.py

up:
	docker compose up -d --build

train-docker:
	docker compose --profile train run --rm train

down:
	docker compose --profile train down

logs:
	docker compose logs -f api

ps:
	docker compose ps

health:
	curl -s http://127.0.0.1:8000/health || true
