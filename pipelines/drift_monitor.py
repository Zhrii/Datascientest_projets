"""
Drift monitor (container "drift-monitor") : détecte une dérive des données en production.

Toutes les DRIFT_CHECK_SECONDS :
  1. lit les DRIFT_WINDOW dernières prédictions de l'API (Redis, src/state.py)
  2. s'il y en a au moins DRIFT_MIN_SAMPLES, les compare aux données d'entraînement
     avec Evidently (src/monitoring/drift.py)
  3. écrit le rapport HTML (reports/drift_report.html, affiché dans Streamlit)
     et un résumé JSON (reports/drift_status.json)
  4. expose le résultat à Prometheus sur :8001/metrics -> alerte DataDrift + Grafana

Lancement :
    python pipelines/drift_monitor.py           # boucle infinie
    python pipelines/drift_monitor.py --once    # une seule analyse
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prometheus_client import Gauge, start_http_server  # noqa: E402

from src.monitoring.drift import build_current, build_reference, compute_drift  # noqa: E402
from src.state import recent_predictions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("drift-monitor")

REFERENCE_CSV = Path(os.getenv("DRIFT_REFERENCE_CSV", ROOT / "data" / "processed" / "processed_data.csv"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
CHECK_SECONDS = int(os.getenv("DRIFT_CHECK_SECONDS", "120"))
WINDOW = int(os.getenv("DRIFT_WINDOW", "300"))
MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "200"))  # en dessous : trop de bruit (27 catégories)
METRICS_PORT = int(os.getenv("DRIFT_METRICS_PORT", "8001"))

DRIFT_SHARE = Gauge("data_drift_share", "Part des variables en dérive (0-1)")
DRIFT_DETECTED = Gauge("data_drift_detected", "Dérive détectée sur le jeu (1/0)")
COLUMN_SCORE = Gauge("data_drift_column_score", "Score de dérive par variable (p-value ou distance)", ["column", "method"])
COLUMN_DRIFTED = Gauge("data_drift_column_drifted", "Variable en dérive (1/0)", ["column"])
CURRENT_SAMPLES = Gauge("data_drift_current_samples", "Prédictions analysées")
LAST_CHECK = Gauge("data_drift_last_check_timestamp_seconds", "Date de la dernière analyse (epoch)")


def check(reference) -> dict:
    current = build_current(recent_predictions(WINDOW))
    CURRENT_SAMPLES.set(len(current))
    LAST_CHECK.set(time.time())
    if len(current) < MIN_SAMPLES:
        # pas assez de données : aucune preuve de dérive (l'alerte DataDrift s'éteint)
        DRIFT_SHARE.set(0)
        DRIFT_DETECTED.set(0)
        status = {"status": "waiting", "n_current": len(current), "min_samples": MIN_SAMPLES,
                  "message": f"{len(current)} prédictions texte, il en faut {MIN_SAMPLES} pour analyser"}
    else:
        result = compute_drift(reference, current, REPORTS_DIR / "drift_report.html")
        DRIFT_SHARE.set(result.drift_share)
        DRIFT_DETECTED.set(1 if result.drift_detected else 0)
        for column, info in result.columns.items():
            COLUMN_SCORE.labels(column=column, method=info["method"]).set(info["score"])
            COLUMN_DRIFTED.labels(column=column).set(1 if info["drifted"] else 0)
        status = {"status": "drift" if result.drift_detected else "ok", **asdict(result)}
    status["checked_at"] = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "drift_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Analyse de dérive : %s", {k: v for k, v in status.items() if k != "columns"})
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Détection de dérive des données (Evidently)")
    parser.add_argument("--once", action="store_true", help="une seule analyse puis arrêt")
    args = parser.parse_args()

    if not args.once:
        start_http_server(METRICS_PORT)
    reference = None
    while True:
        try:
            if reference is None:
                reference = build_reference(REFERENCE_CSV)
                logger.info("Référence chargée : %s produits d'entraînement", len(reference))
            check(reference)
        except FileNotFoundError:
            logger.warning("Données d'entraînement absentes (%s) : lancer make train", REFERENCE_CSV)
        except Exception:  # noqa: BLE001 — le monitor ne doit jamais s'arrêter
            logger.exception("Analyse de dérive échouée")
        if args.once:
            return
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
