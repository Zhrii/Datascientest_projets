"""
Pipeline ZenML d'entraînement (le "Pipeline Runner" du schéma d'architecture).

  1. preprocess_data : collecte + nettoyage texte (scripts/collecte.py, scripts/process.py)
  2. train_model     : TF-IDF + SVM -> models/candidate, run MLflow, version dans le Model Registry
  3. promote_model   : candidat >= production ? -> models/ + alias "production" (scripts/promote_model.py)

Où vont les choses :
  - ZenML Server (MySQL)  : historique des runs, étapes, statuts
  - MinIO (bucket zenml)  : artefacts échangés entre les étapes (données, métriques)
  - MLflow (MinIO mlflow) : paramètres, F1, fichiers du modèle + Model Registry

Lancement :
    python pipelines/rakuten_pipeline.py        # dans le container "train" : make train
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from zenml import pipeline, step
from zenml.client import Client
from zenml.enums import StackComponentType

from scripts.collecte import collecter
from scripts.process import process
from scripts.promote_model import EXIT_PROMOTED, EXIT_REJECTED
from scripts.promote_model import main as promote_main
from scripts.training import INFO_FILE, train

DATA_BRUT_DIR = os.getenv("DATA_BRUT_DIR", str(ROOT / "data brut"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
CANDIDATE_DIR = MODELS_DIR / "candidate"
MERGED_CSV = ROOT / "data" / "raw" / "merged.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "processed_data.csv"

STACK_NAME = "rakuten"
ARTIFACT_STORE_NAME = "minio"


# ----------------------------------------------------------------------------- steps
@step
def preprocess_data() -> Annotated[pd.DataFrame, "processed_data"]:
    """Collecte des CSV bruts puis nettoyage texte ; le résultat est stocké dans MinIO."""
    collecter(DATA_BRUT_DIR, str(MERGED_CSV))
    process(str(MERGED_CSV), str(PROCESSED_CSV))
    return pd.read_csv(PROCESSED_CSV)


@step
def train_model(processed_data: pd.DataFrame) -> Annotated[Dict[str, Any], "candidate_metrics"]:
    """Entraîne un candidat, le trace dans MLflow et l'enregistre dans le Model Registry."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train.csv"
        processed_data.to_csv(csv_path, index=False)
        train(str(csv_path), str(CANDIDATE_DIR), use_mlflow=True)
    return json.loads((CANDIDATE_DIR / INFO_FILE).read_text(encoding="utf-8"))


@step
def promote_model(candidate_metrics: Dict[str, Any]) -> Annotated[Dict[str, Any], "promotion"]:
    """Met le candidat en production s'il est au moins aussi bon que l'actuel."""
    code = promote_main([
        "--candidate-dir", str(CANDIDATE_DIR),
        "--production-dir", str(MODELS_DIR),
    ])
    if code not in (EXIT_PROMOTED, EXIT_REJECTED):
        raise RuntimeError(f"promote_model a échoué (code {code})")
    return {
        "promoted": code == EXIT_PROMOTED,
        "f1_macro": candidate_metrics.get("f1_macro"),
        "mlflow_run_id": candidate_metrics.get("mlflow_run_id"),
        "mlflow_model_version": candidate_metrics.get("mlflow_model_version"),
    }


@pipeline(enable_cache=False)
def rakuten_training_pipeline():
    data = preprocess_data()
    metrics = train_model(data)
    promote_model(metrics)


# ----------------------------------------------------------------------------- stack
def ensure_stack() -> None:
    """Crée (une seule fois) la stack ZenML "rakuten" : orchestrateur local + MinIO."""
    client = Client()
    try:
        client.get_stack_component(StackComponentType.ARTIFACT_STORE, ARTIFACT_STORE_NAME)
    except KeyError:
        client.create_stack_component(
            name=ARTIFACT_STORE_NAME,
            flavor="s3",
            component_type=StackComponentType.ARTIFACT_STORE,
            configuration={
                "path": os.getenv("ZENML_ARTIFACT_PATH", "s3://zenml"),
                "key": os.environ["AWS_ACCESS_KEY_ID"],
                "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": os.environ["MLFLOW_S3_ENDPOINT_URL"]},
            },
        )
    try:
        client.get_stack(STACK_NAME)
    except KeyError:
        client.create_stack(
            name=STACK_NAME,
            components={
                StackComponentType.ORCHESTRATOR: "default",
                StackComponentType.ARTIFACT_STORE: ARTIFACT_STORE_NAME,
            },
        )
    client.activate_stack(STACK_NAME)


if __name__ == "__main__":
    ensure_stack()
    run = rakuten_training_pipeline()
    print(f"Pipeline terminé : {run.name} ({run.status})")
