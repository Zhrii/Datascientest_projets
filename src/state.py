"""
État partagé entre les répliques de l'API, stocké dans Redis (REDIS_URL).

  - verrou de réentraînement : un seul /retrain à la fois, toutes répliques confondues
  - statut du dernier réentraînement (lu par GET /health sur n'importe quelle réplique)
  - journal des dernières prédictions (lu par le drift-monitor pour détecter une dérive)
  - chaque prédiction, retrouvable par son identifiant pendant PREDICTION_TTL (pour /feedback)
  - retours des modérateurs : catégorie prédite vs vraie catégorie (F1 en production)

Sans REDIS_URL (tests, mode local sans Docker) : tout reste en mémoire du process.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import OrderedDict, deque
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger("api.state")

PREDICTIONS_KEY = "rakuten:predictions"
RETRAIN_STATUS_KEY = "rakuten:retrain:status"
RETRAIN_LOCK_KEY = "rakuten:retrain:lock"
PREDICTION_KEY = "rakuten:prediction:{}"
FEEDBACK_KEY = "rakuten:feedback"
PREDICTIONS_MAX = int(os.getenv("PREDICTIONS_LOG_SIZE", "5000"))
FEEDBACK_MAX = int(os.getenv("FEEDBACK_LOG_SIZE", "5000"))
PREDICTION_TTL = 7 * 24 * 3600  # un modérateur a 7 jours pour corriger une prédiction
RETRAIN_LOCK_TTL = 3600  # libéré automatiquement si une réplique meurt pendant le retrain

_memory_predictions: deque = deque(maxlen=PREDICTIONS_MAX)
_memory_status: Dict[str, Any] = {"status": "never", "message": None}
_memory_lock = {"held": False}
_memory_records: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_memory_feedback: deque = deque(maxlen=FEEDBACK_MAX)


class UnknownPrediction(KeyError):
    """Identifiant de prédiction inconnu ou expiré."""


class AlreadyLabeled(ValueError):
    """Cette prédiction a déjà reçu un retour."""


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
def log_prediction(text: Optional[str], result: Dict[str, Any]) -> str:
    """Enregistre la prédiction et renvoie son identifiant (à renvoyer dans /feedback).

    Garde aussi les PREDICTIONS_MAX dernières prédictions (pour la détection de dérive).
    """
    prediction_id = uuid.uuid4().hex
    entry = {
        "id": prediction_id,
        "ts": time.time(),
        "text": text or "",
        "prdtypecode": result.get("prdtypecode"),
        "modality": result.get("modality"),
        "confidence": result.get("confidence"),
        "low_confidence": bool(result.get("low_confidence")),
    }
    record = {k: entry[k] for k in ("ts", "prdtypecode", "modality", "confidence", "low_confidence")}
    client = redis_client()
    if client is None:
        _memory_predictions.append(entry)
        _memory_records[prediction_id] = record
        while len(_memory_records) > PREDICTIONS_MAX:
            _memory_records.popitem(last=False)
        return prediction_id
    try:
        pipe = client.pipeline()
        pipe.lpush(PREDICTIONS_KEY, json.dumps(entry, ensure_ascii=False))
        pipe.ltrim(PREDICTIONS_KEY, 0, PREDICTIONS_MAX - 1)
        pipe.hset(PREDICTION_KEY.format(prediction_id), mapping={"record": json.dumps(record)})
        pipe.expire(PREDICTION_KEY.format(prediction_id), PREDICTION_TTL)
        pipe.execute()
    except Exception as exc:  # noqa: BLE001 — une prédiction ne doit pas échouer pour ça
        logger.warning("Journal des prédictions indisponible: %s", exc)
    return prediction_id


def recent_predictions(limit: int = PREDICTIONS_MAX) -> List[Dict[str, Any]]:
    client = redis_client()
    if client is None:
        return list(_memory_predictions)[-limit:]
    return [json.loads(x) for x in client.lrange(PREDICTIONS_KEY, 0, limit - 1)]


# ---------------------------------------------------------------- retours des modérateurs
def record_feedback(prediction_id: str, true_code: int) -> Dict[str, Any]:
    """Associe la vraie catégorie à une prédiction ; renvoie le couple (prédit, vrai).

    Lève UnknownPrediction (id inconnu / expiré) ou AlreadyLabeled (retour déjà reçu).
    """
    client = redis_client()
    if client is None:
        record = _memory_records.get(prediction_id)
        if record is None:
            raise UnknownPrediction(prediction_id)
        if "true" in record:
            raise AlreadyLabeled(prediction_id)
        record["true"] = true_code
    else:
        key = PREDICTION_KEY.format(prediction_id)
        raw = client.hget(key, "record")
        if raw is None:
            raise UnknownPrediction(prediction_id)
        # HSETNX : un seul retour par prédiction, même si deux répliques le reçoivent en même temps
        if not client.hsetnx(key, "true", true_code):
            raise AlreadyLabeled(prediction_id)
        record = json.loads(raw)

    entry = {
        "ts": time.time(),
        "predicted": int(record["prdtypecode"]),
        "true": int(true_code),
        "modality": record["modality"],
        "confidence": record.get("confidence"),
        "low_confidence": bool(record.get("low_confidence")),
    }
    if client is None:
        _memory_feedback.append(entry)
    else:
        pipe = client.pipeline()
        pipe.lpush(FEEDBACK_KEY, json.dumps(entry))
        pipe.ltrim(FEEDBACK_KEY, 0, FEEDBACK_MAX - 1)
        pipe.execute()
    return entry


def recent_feedback(limit: int = FEEDBACK_MAX) -> List[Dict[str, Any]]:
    client = redis_client()
    if client is None:
        return list(_memory_feedback)[-limit:]
    return [json.loads(x) for x in client.lrange(FEEDBACK_KEY, 0, limit - 1)]
