"""
Client HTTP pour appeler l'API FastAPI depuis Streamlit.
Si l'API est down, le fallback local (ProductPredictor) est utilisé.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))


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
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def model_info() -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/model/info", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def trigger_retrain() -> Dict[str, Any]:
    r = requests.post(f"{API_URL}/retrain", timeout=10)
    r.raise_for_status()
    return r.json()
