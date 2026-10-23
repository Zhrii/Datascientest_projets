"""
Client HTTP pour appeler l'API FastAPI depuis Streamlit.
Si l'API est down, le fallback local (ProductPredictor) est utilisé.
Les endpoints protégés reçoivent les identifiants API_USERNAME / API_PASSWORD.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, Optional

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))
AUTH = (os.getenv("API_USERNAME", "admin"), os.getenv("API_PASSWORD", "rakuten"))


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
    r = requests.get(f"{API_URL}/model/info", auth=AUTH, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def trigger_retrain() -> Dict[str, Any]:
    r = requests.post(f"{API_URL}/retrain", auth=AUTH, timeout=10)
    r.raise_for_status()
    return r.json()


def health() -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/health", timeout=5)
    return r.json()


# ---- Démo sécurité (page MLOps de Streamlit) ----

def probe_without_auth() -> int:
    """Appelle /predict sans identifiants : l'API doit répondre 401."""
    r = requests.post(f"{API_URL}/predict", data={"text": "test"}, timeout=TIMEOUT)
    return r.status_code


def burst_predict(n: int, text: str = "Chaise scandinave en bois") -> Dict[int, int]:
    """Envoie n prédictions d'affilée ; renvoie le nombre de réponses par code HTTP."""
    codes: Counter = Counter()
    for _ in range(n):
        r = requests.post(f"{API_URL}/predict", data={"text": text}, auth=AUTH, timeout=TIMEOUT)
        codes[r.status_code] += 1
    return dict(codes)
