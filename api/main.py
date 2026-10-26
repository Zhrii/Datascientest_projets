"""
API d'inférence FastAPI — couche Inference de la stack MLOps.

Endpoints (🔒 = login / mot de passe HTTP Basic, rôle entre crochets, voir api/security.py) :
  POST /predict      🔒 [user]  classification texte / image / fusion (limite de requêtes par rôle)
  GET  /model/info   🔒 [admin] métadonnées des pipelines
  POST /retrain      🔒 [admin] pipeline ZenML, promotion si meilleur (limite RETRAIN_RATE_LIMIT)
  GET  /health                  santé API + modèles (public : healthcheck Docker)
  GET  /metrics                 métriques Prometheus (public : scrape Prometheus)

Plusieurs répliques tournent derrière nginx (docker-compose.yml) : l'état partagé
(limites, verrou et statut du retrain, journal des prédictions) est dans Redis
(api/state.py), et chaque réplique recharge seule le modèle texte quand l'alias
"production" du Model Registry change (MODEL_REFRESH_SECONDS).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
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

from src import state  # noqa: E402
from api.security import User, predict_limiter, require_role, retrain_limiter  # noqa: E402
from src.inference.predictor import ProductPredictor, production_text_version  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

INSTANCE = socket.gethostname()  # identifiant de la réplique (en-tête X-API-Instance)
MODEL_REFRESH_SECONDS = int(os.getenv("MODEL_REFRESH_SECONDS", "60"))

# ---- Prometheus ----
PREDICT_COUNT = Counter(
    "predict_requests_total",
    "Nombre de requêtes /predict",
    ["modality", "status"],
)
# séries exposées à 0 dès le démarrage (sinon increase() ignore la première erreur)
for _modality in ("text", "image", "fusion"):
    for _status in ("ok", "error", "unavailable"):
        PREDICT_COUNT.labels(modality=_modality, status=_status)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latence /predict",
    ["modality"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
MODEL_READY = Gauge("model_ready", "Pipeline chargé (1/0)", ["pipeline"])
RETRAIN_RUNNING = Gauge("retrain_running", "Retrain en cours sur cette réplique (1/0)")
MODEL_F1 = Gauge("model_f1_macro", "F1-macro du modèle chargé", ["pipeline"])
MODEL_TRAINED_AT = Gauge(
    "model_trained_timestamp_seconds", "Date d'entraînement du modèle texte en production (epoch)"
)
MODEL_VERSION = Gauge("model_registry_version", "Version du Model Registry chargée (modèle texte)")

MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT / "models")))
PIPELINE_SCRIPT = ROOT / "pipelines" / "rakuten_pipeline.py"

_predictor: Optional[ProductPredictor] = None


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
    for name, meta in p.info()["pipelines"].items():
        if meta.get("f1_macro") is not None:
            MODEL_F1.labels(pipeline=name).set(meta["f1_macro"])
    if getattr(p, "text_version", None):
        MODEL_VERSION.set(float(p.text_version))
    trained_at = _read_json(MODELS_DIR / "metrics.json").get("trained_at")
    if trained_at:
        MODEL_TRAINED_AT.set(datetime.fromisoformat(trained_at).timestamp())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


async def _watch_production_model() -> None:
    """Recharge le modèle texte quand l'alias "production" pointe vers une nouvelle version."""
    while True:
        await asyncio.sleep(MODEL_REFRESH_SECONDS)
        try:
            version = await asyncio.to_thread(production_text_version)
            predictor = get_predictor()
            if version and version != getattr(predictor, "text_version", None):
                logger.info("Nouvelle version en production (%s -> %s) : rechargement",
                            predictor.text_version, version)
                await asyncio.to_thread(predictor.reload_text)
                _refresh_ready_gauges()
        except Exception:  # noqa: BLE001 — la surveillance ne doit jamais arrêter l'API
            logger.exception("Vérification du Model Registry échouée")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Chargement des modèles au démarrage (réplique %s)...", INSTANCE)
    get_predictor()
    watcher = asyncio.create_task(_watch_production_model())
    yield
    watcher.cancel()
    with suppress(asyncio.CancelledError):
        await watcher
    logger.info("Arrêt API")


app = FastAPI(
    title="Rakuten Product Classification API",
    description="Inference layer — texte / image / fusion multimodale",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_instance_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-API-Instance"] = INSTANCE
    return response


class PredictResponse(BaseModel):
    prdtypecode: int | str
    category: str
    modality: str
    model: str
    f1_macro: float
    fallback: Optional[str] = None
    latency_ms: float = Field(..., description="Latence serveur en ms")
    instance: str = Field(INSTANCE, description="Réplique de l'API qui a répondu")


class HealthResponse(BaseModel):
    status: str
    instance: str
    text_ready: bool
    image_ready: bool
    fusion_ready: bool
    errors: dict = {}
    retrain: dict = {}


@app.get("/health", response_model=HealthResponse)
def health():
    ok, payload = get_predictor().health()
    payload["instance"] = INSTANCE
    payload["retrain"] = state.get_retrain_status()
    if not ok:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/model/info", dependencies=[Depends(require_role("admin", "/model/info"))])
def model_info():
    return {"instance": INSTANCE, **get_predictor().info()}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    user: User = Depends(predict_limiter.dependency("user")),
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
    state.log_prediction(text, result)  # pour la détection de dérive

    return PredictResponse(
        prdtypecode=result["prdtypecode"],
        category=result["category"],
        modality=result["modality"],
        model=result["model"],
        f1_macro=result["f1_macro"],
        fallback=result.get("fallback"),
        latency_ms=round(elapsed * 1000, 2),
    )


def _run_id(metrics_json: Path) -> Optional[str]:
    return _read_json(metrics_json).get("mlflow_run_id")


async def _run_retrain() -> None:
    RETRAIN_RUNNING.set(1)
    state.set_retrain_status({"status": "running", "instance": INSTANCE, "message": "Pipeline ZenML en cours"})
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
            state.set_retrain_status({"status": "failed", "instance": INSTANCE,
                                      "message": log or f"exit {proc.returncode}"})
            return
        # promu = le modèle de production est celui qui vient d'être entraîné
        candidate_run = _run_id(MODELS_DIR / "candidate" / "metrics.json")
        promoted = candidate_run is not None and candidate_run == _run_id(MODELS_DIR / "metrics.json")
        if promoted:
            # cette réplique recharge tout de suite ; les autres via _watch_production_model
            await asyncio.to_thread(get_predictor().reload_text)
            _refresh_ready_gauges()
        state.set_retrain_status({"status": "promoted" if promoted else "rejected",
                                  "instance": INSTANCE, "message": log})
    except Exception as exc:  # noqa: BLE001
        state.set_retrain_status({"status": "failed", "instance": INSTANCE, "message": str(exc)})
        logger.exception("retrain failed")
    finally:
        RETRAIN_RUNNING.set(0)
        state.release_retrain_lock()


@app.post("/retrain")
async def retrain(user: User = Depends(retrain_limiter.dependency("admin"))):
    """Lance le pipeline ZenML collect → validate → preprocess → train → promote (non bloquant).

    Le nouveau modèle n'est mis en production que s'il est au moins aussi
    bon que l'actuel (scripts/promote_model.py) ; toutes les répliques le rechargent.
    """
    if not state.acquire_retrain_lock(INSTANCE):
        raise HTTPException(409, "Un retrain est déjà en cours")
    asyncio.create_task(_run_retrain())
    return {
        "status": "accepted",
        "instance": INSTANCE,
        "message": "Pipeline lancé. Suivre GET /health → retrain, ZenML ou MLflow.",
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {
        "service": "Rakuten Classification API",
        "instance": INSTANCE,
        "docs": "/docs",
        "endpoints": ["/predict", "/health", "/model/info", "/retrain", "/metrics"],
        "auth": "HTTP Basic — rôle user : /predict ; rôle admin : + /model/info, /retrain",
    }
