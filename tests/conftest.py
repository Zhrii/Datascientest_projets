"""Configuration pytest : racine du projet sur le PYTHONPATH + identifiants de test."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# avant l'import de l'API : identifiants connus des tests, pas de MLflow
os.environ["API_USERNAME"] = "test"
os.environ["API_PASSWORD"] = "secret"
os.environ.pop("MLFLOW_TRACKING_URI", None)
