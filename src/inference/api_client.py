"""
Client HTTP pour appeler l'API FastAPI depuis Streamlit.
Si l'API est down, le fallback local (ProductPredictor) est utilisé.

Deux comptes :
  - USER  (API_USERNAME / API_PASSWORD, rôle user)        : prédictions
  - ADMIN (API_ADMIN_USERNAME / API_ADMIN_PASSWORD, admin) : /model/info, /retrain
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any, Dict, Optional

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))
AUTH = (os.getenv("API_USERNAME", "demo"), os.getenv("API_PASSWORD", "demo"))
ADMIN_AUTH = (os.getenv("API_ADMIN_USERNAME", "admin"), os.getenv("API_ADMIN_PASSWORD", "rakuten"))


def api_available() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code in (200, 503)
    except requests.RequestException:
        return False


def predict_via_api(
    text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "image.jpg",
) -> Dict[str, Any]:
    data = {}
    files = None
    if text is not None:
        data["text"] = text
    if image_bytes is not None:
        files = {"image": (filename, image_bytes, "application/octet-stream")}
    r = requests.post(
        f"{API_URL}/predict",
        data=data or None,
        files=files,
        auth=AUTH,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def model_info() -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/model/info", auth=ADMIN_AUTH, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def trigger_retrain() -> Dict[str, Any]:
    r = requests.post(f"{API_URL}/retrain", auth=ADMIN_AUTH, timeout=10)
    r.raise_for_status()
    return r.json()


def health() -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/health", timeout=5)
    return r.json()


# ---- Démo sécurité et scalabilité (page MLOps de Streamlit) ----

def probe_without_auth() -> int:
    """Appelle /predict sans identifiants : l'API doit répondre 401."""
    r = requests.post(f"{API_URL}/predict", data={"text": "test"}, timeout=TIMEOUT)
    return r.status_code


def retrain_as_user() -> int:
    """Appelle /retrain avec le compte user : l'API doit répondre 403 (rôle admin requis)."""
    r = requests.post(f"{API_URL}/retrain", auth=AUTH, timeout=10)
    return r.status_code


def burst_predict(n: int, text: str = "Chaise scandinave en bois") -> Dict[str, Dict[str, int]]:
    """Envoie n prédictions d'affilée (compte user).

    Renvoie le nombre de réponses par code HTTP et par réplique de l'API
    (en-tête X-API-Instance) : on voit la limite (429) et la répartition de charge.
    """
    codes: Counter = Counter()
    instances: Counter = Counter()
    for _ in range(n):
        r = requests.post(f"{API_URL}/predict", data={"text": text}, auth=AUTH, timeout=TIMEOUT)
        codes[str(r.status_code)] += 1
        instances[r.headers.get("X-API-Instance", "?")] += 1
    return {"codes": dict(codes), "instances": dict(instances)}


# Produits très différents des données d'entraînement : textes courts, en anglais,
# pleins de références chiffrées -> la dérive doit être détectée par le drift-monitor.
_DRIFT_TEXTS = [
    "USB-C cable 2m 100W", "SSD 1TB NVMe M.2 2280", "RAM DDR5 32GB 6000MHz",
    "GPU RTX 4070 12GB", "HDMI 2.1 8K cable", "Power bank 20000mAh PD 65W",
    "Router WiFi 7 BE19000", "Monitor 27in 165Hz IPS", "Keyboard 60% RGB hot-swap",
    "Mouse 26000 DPI wireless", "Webcam 4K 60fps", "NAS 4-bay 2.5GbE",
]


def send_drift_traffic(n: int = 250) -> Dict[str, int]:
    """Envoie n produits inhabituels (compte admin, limite plus large) pour provoquer une dérive."""
    codes: Counter = Counter()
    for _ in range(n):
        text = f"{random.choice(_DRIFT_TEXTS)} ref {random.randint(1000, 99999)}"
        r = requests.post(f"{API_URL}/predict", data={"text": text}, auth=ADMIN_AUTH, timeout=TIMEOUT)
        codes[str(r.status_code)] += 1
    return dict(codes)
