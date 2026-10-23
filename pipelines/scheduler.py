"""
Scheduler : déclenche le réentraînement tout seul (container "scheduler").

Toutes les CHECK_INTERVAL_SECONDS, il applique les déclencheurs de
docs/STRATEGIE_REENTRAINEMENT.md :
  - aucun modèle en production
  - modèle plus vieux que RETRAIN_MAX_AGE_DAYS (défaut 30 jours)
  - au moins NEW_DATA_THRESHOLD (défaut 10 %) de produits en plus
    que lors du dernier entraînement

S'il faut réentraîner, il appelle POST /retrain sur l'API (avec login /
mot de passe) : c'est l'API qui lance le pipeline ZenML et recharge le
modèle s'il est promu. La promotion garde la règle "jamais moins bon".

Lancement :
    python pipelines/scheduler.py            # boucle infinie
    python pipelines/scheduler.py --once     # une seule vérification (test)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scheduler")

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
AUTH = (os.getenv("API_USERNAME", "admin"), os.getenv("API_PASSWORD", "rakuten"))
DATA_FILE = Path(os.getenv("DATA_BRUT_DIR", ROOT / "data brut")) / "X_train_update.csv"
METRICS_FILE = Path(os.getenv("MODELS_DIR", ROOT / "models")) / "metrics.json"

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "3600"))
MAX_AGE_DAYS = float(os.getenv("RETRAIN_MAX_AGE_DAYS", "30"))
NEW_DATA_THRESHOLD = float(os.getenv("NEW_DATA_THRESHOLD", "0.10"))


def retrain_reason(metrics: Optional[dict], n_rows: int, now: datetime) -> Optional[str]:
    """Renvoie la raison de réentraîner, ou None si le modèle actuel convient."""
    if not metrics:
        return "aucun modèle en production"
    age_days = (now - datetime.fromisoformat(metrics["trained_at"])).total_seconds() / 86400
    if age_days > MAX_AGE_DAYS:
        return f"modèle vieux de {age_days:.0f} jours (> {MAX_AGE_DAYS:.0f})"
    n_used = metrics.get("n_train", 0) + metrics.get("n_val", 0)
    if n_used and n_rows >= n_used * (1 + NEW_DATA_THRESHOLD):
        return f"{n_rows - n_used} nouveaux produits (+{(n_rows / n_used - 1):.0%})"
    return None


def load_metrics() -> Optional[dict]:
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def count_products() -> int:
    return len(pd.read_csv(DATA_FILE, usecols=[0]))


def check_once() -> None:
    reason = retrain_reason(load_metrics(), count_products(), datetime.now(timezone.utc))
    if reason is None:
        logger.info("Pas de réentraînement nécessaire")
        return
    logger.info("Réentraînement déclenché : %s", reason)
    try:
        r = requests.post(f"{API_URL}/retrain", auth=AUTH, timeout=30)
        logger.info("POST /retrain -> %s %s", r.status_code, r.text[:200])
    except requests.RequestException as exc:
        logger.warning("API injoignable, nouvel essai au prochain passage : %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Déclenchement automatique du réentraînement")
    parser.add_argument("--once", action="store_true", help="une seule vérification puis arrêt")
    args = parser.parse_args()

    logger.info(
        "Scheduler démarré : vérification toutes les %ss (âge max %s j, +%.0f%% de données)",
        CHECK_INTERVAL, MAX_AGE_DAYS, NEW_DATA_THRESHOLD * 100,
    )
    while True:
        try:
            check_once()
        except Exception:  # noqa: BLE001 — le scheduler ne doit jamais s'arrêter
            logger.exception("Vérification échouée")
        if args.once:
            return
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
