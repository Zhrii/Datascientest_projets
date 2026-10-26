.PHONY: help install api streamlit train train-local promote test data-push data-pull drift up down logs health ps

help:
	@echo "make up | down | train | train-local | promote | test | data-push | data-pull | drift | api | streamlit | health | ps"

install:
	pip install -r requirements.txt
	pip install -r requirements-mlops.txt

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	streamlit run app.py --server.port 8501

# Pipeline Runner ZenML (Docker) : preprocess_data -> train_model -> promote_model
# (les répliques de l'API chargent seules le nouveau modèle "production")
train:
	docker compose run --rm train

# sans Docker ni ZenML : collecte -> process -> training (candidat dans models/candidate)
train-local:
	python scripts/collecte.py --data-dir "data brut"
	python scripts/process.py
	python scripts/training.py --mlflow --models-dir models/candidate

# met le candidat en production s'il est au moins aussi bon
promote:
	python scripts/promote_model.py

# tests pytest dans l'image de l'API (même chose en CI : .github/workflows/tests.yml)
test:
	docker compose run --rm --no-deps api pytest -q

# versioning des données : CSV de "data brut/" <-> MinIO (bucket dvc)
DVC_FILES = "data brut/X_train_update.csv.dvc" "data brut/Y_train.csv.dvc" "data brut/X_test_update.csv.dvc"
data-push:
	python -m dvc push $(DVC_FILES)

data-pull:
	python -m dvc pull $(DVC_FILES)

drift:
	docker compose exec drift-monitor python pipelines/drift_monitor.py --once

up:
	docker compose up -d --build


down:
	docker compose --profile jobs down

logs:
	docker compose logs -f api

ps:
	docker compose ps

health:
	curl -s http://127.0.0.1:8000/health || true
