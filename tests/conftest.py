"""Configuration pytest : racine du projet sur le PYTHONPATH + identifiants de test."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# avant l'import de l'API : comptes connus des tests, ni MLflow ni Redis (état en mémoire)
os.environ["API_USERS"] = "test:secret:admin,alice:pw:user"
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("REDIS_URL", None)
