"""
Mini API (Phase 1) exposant le modèle texte entraîné.
Réutilise exactement les mêmes artefacts (et la même classe TextClassifier)
que scripts/predict.py — un seul endroit charge et sert le modèle.

Lancement :
    pip install fastapi "uvicorn[standard]"
    uvicorn api.main:app --reload --port 8000

Test :
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"designation": "Chaise scandinave en bois blanc", "description": "Design nordique, pieds en hêtre"}'
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import TextClassifier

app = FastAPI(title="Rakuten Classification API — Texte", version="0.1.0")
classifier: TextClassifier | None = None


class ProductInput(BaseModel):
    designation: str
    description: str = ""


@app.on_event("startup")
def load_model():
    global classifier
    classifier = TextClassifier(models_dir="models")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/predict")
def predict(product: ProductInput):
    result = classifier.predict(product.designation, product.description)
    return result
