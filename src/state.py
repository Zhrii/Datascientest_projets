"""
État partagé entre les répliques de l'API, stocké dans Redis (REDIS_URL).

  - verrou de réentraînement : un seul /retrain à la fois, toutes répliques confondues
  - statut du dernier réentraînement (lu par GET /health sur n'importe quelle réplique)
  - journal des dernières prédictions (lu par le drift-monitor pour détecter une dérive)

Sans REDIS_URL (tests, mode local sans Docker) : tout reste en mémoire du process.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger("api.state")

PREDICTIONS_KEY = "rakuten:predictions"
RETRAIN_STATUS_KEY = "rakuten:retrain:status"
RETRAIN_LOCK_KEY = "rakuten:retrain:lock"
PREDICTIONS_MAX = int(os.getenv("PREDICTIONS_LOG_SIZE", "5000"))
RETRAIN_LOCK_TTL = 3600  # libéré automatiquement si une réplique meurt pendant le retrain

_memory_predictions: deque = deque(maxlen=PREDICTIONS_MAX)
_memory_status: Dict[str, Any] = {"status": "never", "message": None}
_memory_lock = {"held": False}


@lru_cache(maxsize=1)
def redis_client():
    """Client Redis partagé, ou None si REDIS_URL n'est pas défini."""
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)


# ---------------------------------------------------------------- verrou /retrain
def acquire_retrain_lock(owner: str) -> bool:
    client = redis_client()
    if client is None:
        if _memory_lock["held"]:
            return False
        _memory_lock["held"] = True
        return True
    return bool(client.set(RETRAIN_LOCK_KEY, owner, nx=True, ex=RETRAIN_LOCK_TTL))


def release_retrain_lock() -> None:
    client = redis_client()
    if client is None:
        _memory_lock["held"] = False
    else:
        client.delete(RETRAIN_LOCK_KEY)


# ---------------------------------------------------------------- statut /retrain
def set_retrain_status(status: Dict[str, Any]) -> None:
    status = {**status, "updated_at": time.time()}
    client = redis_client()
    if client is None:
        _memory_status.clear()
        _memory_status.update(status)
    else:
        client.set(RETRAIN_STATUS_KEY, json.dumps(status))


def get_retrain_status() -> Dict[str, Any]:
    client = redis_client()
    if client is None:
        return dict(_memory_status)
    try:
        raw = client.get(RETRAIN_STATUS_KEY)
    except Exception as exc:  # noqa: BLE001 — /health ne doit pas tomber si Redis tousse
        return {"status": "unknown", "message": f"Redis indisponible: {exc}"}
    return json.loads(raw) if raw else {"status": "never", "message": None}


# ---------------------------------------------------------------- journal des prédictions
def log_prediction(text: Optional[str], result: Dict[str, Any]) -> None:
    """Garde les PREDICTIONS_MAX dernières prédictions (pour la détection de dérive)."""
    entry = {
        "ts": time.time(),
        "text": text or "",
        "prdtypecode": result.get("prdtypecode"),
        "modality": result.get("modality"),
    }
    client = redis_client()
    if client is None:
        _memory_predictions.append(entry)
        return
    try:
        pipe = client.pipeline()
        pipe.lpush(PREDICTIONS_KEY, json.dumps(entry, ensure_ascii=False))
        pipe.ltrim(PREDICTIONS_KEY, 0, PREDICTIONS_MAX - 1)
        pipe.execute()
    except Exception as exc:  # noqa: BLE001 — une prédiction ne doit pas échouer pour ça
        logger.warning("Journal des prédictions indisponible: %s", exc)


def recent_predictions(limit: int = PREDICTIONS_MAX) -> List[Dict[str, Any]]:
    client = redis_client()
    if client is None:
        return list(_memory_predictions)[-limit:]
    return [json.loads(x) for x in client.lrange(PREDICTIONS_KEY, 0, limit - 1)]
