"""
Pipeline ZenML d'entraînement (le "Pipeline Runner" du schéma d'architecture).

  1. collect_data    : CSV bruts -> table fusionnée + version des données (DVC)  (scripts/collecte.py)
  2. validate_data   : contrôle qualité ; arrête le pipeline si les données sont cassées (scripts/validate_data.py)
  3. preprocess_data : nettoyage texte (scripts/process.py)
  4. train_model     : TF-IDF + SVM -> models/candidate, run MLflow (+ version des données
                       et rapport qualité), version dans le Model Registry
  5. promote_model   : candidat >= production ? -> models/ + alias "production" (scripts/promote_model.py)

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
from typing import Annotated, Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from zenml import pipeline, step
from zenml.client import Client
from zenml.enums import StackComponentType

from scripts.collecte import collecter
from scripts.data_version import data_version
from scripts.process import process
from scripts.promote_model import EXIT_PROMOTED, EXIT_REJECTED
from scripts.promote_model import main as promote_main
from scripts.training import INFO_FILE, train
from scripts.validate_data import validate

DATA_BRUT_DIR = os.getenv("DATA_BRUT_DIR", str(ROOT / "data brut"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
CANDIDATE_DIR = MODELS_DIR / "candidate"
MERGED_CSV = ROOT / "data" / "raw" / "merged.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "processed_data.csv"

STACK_NAME = "rakuten"
ARTIFACT_STORE_NAME = "minio"


# ----------------------------------------------------------------------------- steps
@step
def collect_data() -> Tuple[Annotated[pd.DataFrame, "raw_data"], Annotated[Dict[str, str], "data_version"]]:
    """Fusionne X_train / Y_train et relève la version des données (empreintes DVC)."""
    collecter(DATA_BRUT_DIR, str(MERGED_CSV))
    return pd.read_csv(MERGED_CSV), data_version(DATA_BRUT_DIR)


@step
def validate_data(raw_data: pd.DataFrame) -> Annotated[Dict[str, Any], "quality_report"]:
    """Contrôle qualité : lève une erreur (le pipeline s'arrête) si un contrôle bloquant échoue."""
    report = validate(raw_data)
    for warning in report["warnings"]:
        print(f"⚠️  {warning}")
    print(f"✅ Données valides : {report['n_rows']:,} produits, {report['n_classes']} catégories")
    return report


@step
def preprocess_data(raw_data: pd.DataFrame, quality_report: Dict[str, Any]) -> Annotated[pd.DataFrame, "processed_data"]:
    """Nettoyage texte (n'a lieu que si validate_data est passé) ; le résultat est stocké dans MinIO."""
    process(str(MERGED_CSV), str(PROCESSED_CSV))
    return pd.read_csv(PROCESSED_CSV)


@step
def train_model(
    processed_data: pd.DataFrame,
    data_version: Dict[str, str],
    quality_report: Dict[str, Any],
) -> Annotated[Dict[str, Any], "candidate_metrics"]:
    """Entraîne un candidat, le trace dans MLflow et l'enregistre dans le Model Registry."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train.csv"
        processed_data.to_csv(csv_path, index=False)
        train(str(csv_path), str(CANDIDATE_DIR), use_mlflow=True,
              extra_params=data_version, extra_artifacts={"quality_report.json": quality_report})
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
    raw, version = collect_data()
    report = validate_data(raw)
    data = preprocess_data(raw, report)
    metrics = train_model(data, version, report)
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
