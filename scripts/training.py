"""
Étape 3 - Training : vectorisation TF-IDF + entraînement du SVM linéaire

"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.modeling.vectorization import TFIDFVectorizer
from src.modeling.baseline_models import BaselineModels
from src.evaluation.metrics import evaluate_model

MODEL_NAME = "SVM (Linear)"


def train(input_path: str, models_dir: str, use_mlflow: bool) -> dict:
    df = pd.read_csv(input_path).dropna(subset=["text_combined", "prdtypecode"])

    X = df["text_combined"]
    y_raw = df["prdtypecode"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TFIDFVectorizer(max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    baseline = BaselineModels(random_state=42)
    baseline.create_baseline_models()
    baseline.train_model(MODEL_NAME, X_train_vec, y_train)

    y_pred = baseline.predict(MODEL_NAME, X_val_vec)
    metrics = evaluate_model(y_val, y_pred)

    print("📊 Résultats validation :")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    models_dir_path = Path(models_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)
    vectorizer.save(models_dir_path / "tfidf_vectorizer.pkl")
    baseline.save_model(MODEL_NAME, models_dir_path / "svm_text_model.pkl")
    with open(models_dir_path / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print(f"Artefacts sauvegardés dans {models_dir}/")

    if use_mlflow:
        log_to_mlflow(metrics, vectorizer, baseline)

    return metrics


def log_to_mlflow(metrics: dict, vectorizer: TFIDFVectorizer, baseline: BaselineModels) -> None:
    """Tracking MLflow (Phase 2) : activé avec --mlflow une fois le serveur MLflow lancé."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("mlflow n'est pas installé (pip install mlflow) — tracking ignoré.")
        return

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Rakuten_Classification_Texte")

    with mlflow.start_run():
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", vectorizer.max_features)
        mlflow.log_param("ngram_range", str(vectorizer.ngram_range))
        mlflow.log_param("model", MODEL_NAME)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(baseline.trained_models[MODEL_NAME], "model_rakuten_texte")
    print("🎯 Run enregistré dans MLflow.")


def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle texte Rakuten")
    parser.add_argument("--input", default="data/processed/processed_data.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--mlflow", action="store_true", help="Active le tracking MLflow")
    args = parser.parse_args()
    train(args.input, args.models_dir, args.mlflow)


if __name__ == "__main__":
    main()
