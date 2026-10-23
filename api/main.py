"""
API d'inférence FastAPI — couche Inference de la stack MLOps.

Endpoints :
  POST /predict      — classification texte / image / fusion
  GET  /health       — santé API + modèles
  GET  /model/info   — métadonnées des pipelines
  POST /retrain      — entraîne un candidat puis le promeut s'il est meilleur (async)
  GET  /metrics      — métriques Prometheus
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

# Racine projet sur le PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import ProductPredictor  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ---- Prometheus ----
PREDICT_COUNT = Counter(
    "predict_requests_total",
    "Nombre de requêtes /predict",
    ["modality", "status"],
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latence /predict",
    ["modality"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
MODEL_READY = Gauge("model_ready", "Pipeline chargé (1/0)", ["pipeline"])
RETRAIN_RUNNING = Gauge("retrain_running", "Retrain en cours (1/0)")

_predictor: Optional[ProductPredictor] = None
_retrain_lock = asyncio.Lock()
_last_retrain: dict = {"status": "never", "message": None}


def get_predictor() -> ProductPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ProductPredictor()
        _refresh_ready_gauges()
    return _predictor


def _refresh_ready_gauges() -> None:
    p = get_predictor()
    MODEL_READY.labels(pipeline="text").set(1 if p.text_ready else 0)
    MODEL_READY.labels(pipeline="image").set(1 if p.image_ready else 0)
    MODEL_READY.labels(pipeline="fusion").set(1 if p.fusion_ready else 0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Chargement des modèles au démarrage...")
    get_predictor()
    yield
    logger.info("Arrêt API")


app = FastAPI(
    title="Rakuten Product Classification API",
    description="Inference layer — texte / image / fusion multimodale",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    prdtypecode: int | str
    category: str
    modality: str
    model: str
    f1_macro: float
    fallback: Optional[str] = None
    latency_ms: float = Field(..., description="Latence serveur en ms")


class HealthResponse(BaseModel):
    status: str
    text_ready: bool
    image_ready: bool
    fusion_ready: bool
    errors: dict = {}
    retrain: dict = {}


@app.get("/health", response_model=HealthResponse)
def health():
    ok, payload = get_predictor().health()
    payload["retrain"] = _last_retrain
    if not ok:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/model/info")
def model_info():
    return get_predictor().info()


@app.post("/predict", response_model=PredictResponse)
async def predict(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Classification produit.
    - text + image → fusion (fallback texte si fusion absente)
    - text seul → SVM TF-IDF
    - image seule → ResNet50 + LR
    """
    if (not text or not text.strip()) and image is None:
        raise HTTPException(400, "Fournir au moins 'text' ou 'image'")

    image_bytes = None
    if image is not None:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(400, "Fichier image vide")

    predictor = get_predictor()
    modality_hint = (
        "fusion" if (text and image_bytes) else ("text" if text else "image")
    )
    t0 = time.perf_counter()
    try:
        result = predictor.predict(text=text, image=image_bytes)
        status = "ok"
    except ValueError as exc:
        PREDICT_COUNT.labels(modality=modality_hint, status="error").inc()
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        PREDICT_COUNT.labels(modality=modality_hint, status="unavailable").inc()
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        PREDICT_COUNT.labels(modality=modality_hint, status="error").inc()
        logger.exception("predict failed")
        raise HTTPException(500, f"Erreur prédiction: {exc}") from exc
    finally:
        elapsed = time.perf_counter() - t0

    modality = result.get("modality", modality_hint)
    PREDICT_LATENCY.labels(modality=modality).observe(elapsed)
    PREDICT_COUNT.labels(modality=modality, status=status).inc()

    return PredictResponse(
        prdtypecode=result["prdtypecode"],
        category=result["category"],
        modality=result["modality"],
        model=result["model"],
        f1_macro=result["f1_macro"],
        fallback=result.get("fallback"),
        latency_ms=round(elapsed * 1000, 2),
    )


MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT / "models")))
PIPELINE_SCRIPT = ROOT / "pipelines" / "rakuten_pipeline.py"


def _run_id(metrics_json: Path) -> Optional[str]:
    try:
        return json.loads(metrics_json.read_text(encoding="utf-8")).get("mlflow_run_id")
    except (OSError, ValueError):
        return None


async def _run_retrain() -> None:
    global _last_retrain
    RETRAIN_RUNNING.set(1)
    _last_retrain = {"status": "running", "message": "Pipeline ZenML en cours"}
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(PIPELINE_SCRIPT),
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        log = (stdout or b"").decode("utf-8", errors="replace")[-4000:]
        if proc.returncode != 0:
            _last_retrain = {"status": "failed", "message": log or f"exit {proc.returncode}"}
            return
        # promu = le modèle de production est celui qui vient d'être entraîné
        candidate_run = _run_id(MODELS_DIR / "candidate" / "metrics.json")
        promoted = candidate_run is not None and candidate_run == _run_id(MODELS_DIR / "metrics.json")
        if promoted:
            get_predictor().reload()
            _refresh_ready_gauges()
        _last_retrain = {"status": "promoted" if promoted else "rejected", "message": log}
    except Exception as exc:  # noqa: BLE001
        _last_retrain = {"status": "failed", "message": str(exc)}
        logger.exception("retrain failed")
    finally:
        RETRAIN_RUNNING.set(0)


@app.post("/retrain")
async def retrain():
    """Lance le pipeline ZenML preprocess_data → train_model → promote_model (non bloquant).

    Le nouveau modèle n'est mis en production que s'il est au moins aussi
    bon que l'actuel (scripts/promote_model.py) ; l'API le recharge alors.
    """
    if _retrain_lock.locked() or _last_retrain.get("status") == "running":
        raise HTTPException(409, "Un retrain est déjà en cours")

    async def _locked():
        async with _retrain_lock:
            await _run_retrain()

    asyncio.create_task(_locked())
    return {
        "status": "accepted",
        "message": "Pipeline lancé. Suivre GET /health → retrain ou MLflow.",
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {
        "service": "Rakuten Classification API",
        "docs": "/docs",
        "endpoints": ["/predict", "/health", "/model/info", "/retrain", "/metrics"],
    }
