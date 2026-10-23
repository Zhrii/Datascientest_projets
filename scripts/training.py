"""
training.py : entraîne le modèle texte (TF-IDF + SVM linéaire)

Ce que fait le script :
  1. lit les données nettoyées (data/processed/processed_data.csv)
  2. entraîne le modèle et mesure le score sur 20% des données
  3. sauvegarde 3 fichiers .pkl dans le dossier models
  4. (option --mlflow) note le résultat dans MLflow
  5. écrit metrics.json (la "carte d'identité" du modèle)

Pour lancer :
    python scripts/training.py
    python scripts/training.py --mlflow
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

# pour que "from src..." marche depuis le dossier scripts
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.modeling.vectorization import TFIDFVectorizer
from src.modeling.baseline_models import BaselineModels
from src.evaluation.metrics import evaluate_model

# réglages (on les garde ici, au même endroit)
MODEL_NAME = "SVM (Linear)"
EXPERIMENT_NAME = "Rakuten_Classification_Texte"
RANDOM_STATE = 42   # même découpage à chaque fois
TEST_SIZE = 0.2     # 20% pour tester le modèle
ARTIFACT_FILES = ["tfidf_vectorizer.pkl", "svm_text_model.pkl", "label_encoder.pkl"]
INFO_FILE = "metrics.json"  # lu ensuite par promote_model.py
REGISTERED_MODEL = "rakuten-text-svm"  # nom dans le Model Registry MLflow


def train(input_path, models_dir, use_mlflow):
    # on lit les données et on enlève les lignes vides
    df = pd.read_csv(input_path).dropna(subset=["text_combined", "prdtypecode"])

    X = df["text_combined"]
    y_raw = df["prdtypecode"]

    # les codes catégories (10, 40, 50...) deviennent 0, 1, 2...
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # 80% pour entraîner, 20% pour tester (stratify = mêmes proportions de classes)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # le texte devient des nombres (TF-IDF)
    vectorizer = TFIDFVectorizer(max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # on entraîne le modèle
    baseline = BaselineModels(random_state=RANDOM_STATE)
    baseline.create_baseline_models()
    baseline.train_model(MODEL_NAME, X_train_vec, y_train)

    # on mesure le score sur les 20% qu'il n'a jamais vus
    y_pred = baseline.predict(MODEL_NAME, X_val_vec)
    metrics = evaluate_model(y_val, y_pred)

    print("📊 Résultats :")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # on sauvegarde les 3 fichiers du modèle
    models_dir_path = Path(models_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)
    vectorizer.save(models_dir_path / "tfidf_vectorizer.pkl")
    baseline.save_model(MODEL_NAME, models_dir_path / "svm_text_model.pkl")
    with open(models_dir_path / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print(f"💾 Modèle sauvegardé dans {models_dir}/")

    # on note le run dans MLflow si on a demandé --mlflow
    run_id, model_version = None, None
    if use_mlflow:
        params = {
            "model": MODEL_NAME,
            "vectorizer": "TF-IDF",
            "max_features": vectorizer.max_features,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "ngram_range": str(vectorizer.ngram_range),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_classes": len(encoder.classes_),
        }
        run_id, model_version = log_to_mlflow(params, metrics, models_dir_path)

    # carte d'identité du modèle, écrite en DERNIER :
    # si elle existe, c'est que tout le reste s'est bien passé
    info = {
        "model": MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "mlflow_run_id": run_id,
        "mlflow_model_version": model_version,
        **{k: float(v) for k, v in metrics.items()},
    }
    with open(models_dir_path / INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return metrics


def log_to_mlflow(params, metrics, models_dir):
    """Envoie les paramètres, les scores et les fichiers du modèle à MLflow,
    puis enregistre le modèle dans le Model Registry.
    Renvoie (run_id, version du modèle), ou (None, None) si mlflow n'est pas installé."""
    try:
        import mlflow
    except ImportError:
        print("⚠️  mlflow n'est pas installé (pip install mlflow), on saute cette étape.")
        return None, None

    # si le serveur ne répond pas, on abandonne vite (pas de longue attente)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")

    # adresse du serveur MLflow (dans Docker : http://mlflow:5000)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for name in ARTIFACT_FILES:
            mlflow.log_artifact(str(models_dir / name), artifact_path="model_artifacts")
        print(f"🎯 Run MLflow enregistré : {run.info.run_id}  ({tracking_uri})")

    # le modèle devient une nouvelle version dans le Model Registry
    # (l'alias "production" est posé plus tard par promote_model.py)
    client = mlflow.MlflowClient()
    try:
        client.get_registered_model(REGISTERED_MODEL)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(REGISTERED_MODEL)
    version = client.create_model_version(
        name=REGISTERED_MODEL,
        source=f"{run.info.artifact_uri}/model_artifacts",
        run_id=run.info.run_id,
        tags={k: f"{v:.4f}" for k, v in metrics.items()},
    )
    print(f"📚 Model Registry : {REGISTERED_MODEL} version {version.version}")
    return run.info.run_id, version.version


def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle texte Rakuten")
    parser.add_argument("--input", default="data/processed/processed_data.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--mlflow", action="store_true", help="note le run dans MLflow")
    args = parser.parse_args()
    train(args.input, args.models_dir, args.mlflow)


if __name__ == "__main__":
    main()
