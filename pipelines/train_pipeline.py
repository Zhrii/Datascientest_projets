"""
Pipeline d'entraînement (schéma ZenML) — 2 steps :
  1. preprocess_data
  2. train_model

Fonctionne :
  - avec ZenML si installé (artefacts + tracking)
  - en mode standalone sinon (script + MLflow optionnel)

Usage :
  python pipelines/train_pipeline.py
  make train
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

DATA_BRUT = ROOT / "data brut"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
MLRUNS = ROOT / "mlruns"


def _try_mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def preprocess_data() -> Dict[str, Any]:
    """
    Step 1 — charge les CSV, nettoie le texte, écrit un artefact preprocessé.
    """
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.utils.data_loader import load_data  # type: ignore

    PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    x_path = DATA_BRUT / "X_train_update.csv"
    y_path = DATA_BRUT / "Y_train.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Données manquantes dans '{DATA_BRUT}'. "
            "Placez X_train_update.csv et Y_train.csv."
        )

    logger.info("Chargement des données...")
    try:
        X_train, y_train, _ = load_data(DATA_BRUT)
        X = X_train.copy()
        y = y_train
    except Exception:
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)

    if "prdtypecode" not in X.columns and "prdtypecode" in y.columns:
        key = (
            "Unnamed: 0"
            if "Unnamed: 0" in X.columns and "Unnamed: 0" in y.columns
            else None
        )
        if key:
            X = X.merge(y[[key, "prdtypecode"]], on=key, how="left")
        else:
            X = X.copy()
            X["prdtypecode"] = y["prdtypecode"].values

    pipe = PreprocessingPipeline(combine_text=True, create_features=False)
    df = pipe.fit_transform(X)

    out_csv = PROCESSED / "train_preprocessed.csv"
    # Colonnes utiles uniquement
    cols = [c for c in ["text_combined", "designation", "description", "prdtypecode", "productid", "imageid"] if c in df.columns]
    df[cols].to_csv(out_csv, index=False)

    meta = {
        "n_rows": len(df),
        "n_classes": int(df["prdtypecode"].nunique()) if "prdtypecode" in df.columns else None,
        "artifact": str(out_csv),
        "stats": getattr(pipe, "stats", {}),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = ARTIFACTS / "preprocess_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Preprocess OK → %s (%s lignes)", out_csv, meta["n_rows"])
    return meta


def train_model(preprocess_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Step 2 — entraîne un SVM Linear TF-IDF (texte) et enregistre les artefacts.

    Note : la fusion CLIP complète nécessite images + GPU ; ici on industrialise
    le champion texte (F1~0.78) comme modèle production rapide, et on copie
    les artefacts fusion s'ils existent déjà.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score, classification_report
    from sklearn.svm import LinearSVC

    from src.modeling.vectorization import TFIDFVectorizer

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED / "train_preprocessed.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Exécutez d'abord preprocess_data()")

    df = pd.read_csv(csv_path)
    if "text_combined" not in df.columns or "prdtypecode" not in df.columns:
        raise ValueError("Colonnes text_combined / prdtypecode requises")

    df = df.dropna(subset=["text_combined", "prdtypecode"])
    # Sous-échantillon optionnel pour un retrain rapide (env)
    max_rows = os.getenv("TRAIN_MAX_ROWS")
    if max_rows:
        df = df.sample(n=min(int(max_rows), len(df)), random_state=42)

    texts = df["text_combined"].astype(str)
    labels = df["prdtypecode"].astype(int)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Vectorisation TF-IDF...")
    vectorizer = TFIDFVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="french")
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)

    logger.info("Entraînement LinearSVC...")
    clf = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42, dual=False)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xva)
    f1 = float(f1_score(y_val, y_pred, average="macro"))
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    # Sauvegarde artefacts (noms locaux Datascientest_projets)
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    enc_path = MODELS_DIR / "label_encoder.pkl"
    model_path = MODELS_DIR / "svm_linear_baseline_final.pkl"

    vectorizer.save(vec_path)
    with open(enc_path, "wb") as f:
        pickle.dump(le, f)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    # Alias pour compatibilité HF / anciens scripts
    with open(MODELS_DIR / "svm_linear_baseline_27_classes_final.pkl", "wb") as f:
        pickle.dump(clf, f)

    metrics = {
        "f1_macro": f1,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "model_path": str(model_path),
        "vectorizer_path": str(vec_path),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACTS / "train_metrics.json").write_text(
        json.dumps({"metrics": metrics, "report_macro_avg": report.get("macro avg")}, indent=2),
        encoding="utf-8",
    )

    # Registry local simple
    registry = {
        "production": {
            "name": "svm_linear_tfidf",
            "modality": "text",
            "f1_macro": f1,
            "paths": {
                "model": str(model_path),
                "vectorizer": str(vec_path),
                "encoder": str(enc_path),
            },
            "registered_at": metrics["ts"],
        }
    }
    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    mlflow = _try_mlflow()
    if mlflow is not None:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file://{MLRUNS}"))
        mlflow.set_experiment("rakuten-classification")
        with mlflow.start_run(run_name=f"text-svm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_param("model", "LinearSVC")
            mlflow.log_param("max_features", 10000)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("n_train", metrics["n_train"])
            mlflow.log_artifact(str(ARTIFACTS / "train_metrics.json"))
            logger.info("Métriques loggées dans MLflow")

    logger.info("Train OK — F1-macro=%.4f → %s", f1, model_path)
    return metrics


def run_pipeline() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Exécute preprocess → train (avec ZenML si dispo)."""
    try:
        from zenml import pipeline, step

        @step
        def step_preprocess() -> Dict[str, Any]:
            return preprocess_data()

        @step
        def step_train(preprocess_meta: Dict[str, Any]) -> Dict[str, Any]:
            return train_model(preprocess_meta)

        @pipeline(enable_cache=False)
        def rakuten_train_pipeline():
            meta = step_preprocess()
            step_train(meta)

        logger.info("Exécution via ZenML...")
        rakuten_train_pipeline()
        # Relire les artefacts écrits sur disque
        meta = json.loads((ARTIFACTS / "preprocess_meta.json").read_text(encoding="utf-8"))
        metrics = json.loads((ARTIFACTS / "train_metrics.json").read_text(encoding="utf-8"))["metrics"]
        return meta, metrics
    except ImportError:
        logger.info("ZenML non installé — mode standalone")
        meta = preprocess_data()
        metrics = train_model(meta)
        return meta, metrics


if __name__ == "__main__":
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    run_pipeline()
    print("Pipeline terminé.")
