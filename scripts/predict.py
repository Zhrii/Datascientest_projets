"""
Predict / Eval : charge le modèle entraîné et prédit sur de
nouvelles données, ou évalue ses performances si les labels sont connus.

"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.inference import TextClassifier
from src.evaluation.metrics import evaluate_model


def predict_single(text: str, models_dir: str) -> None:
    clf = TextClassifier(models_dir)
    result = clf.predict(text)
    print(f"📦 Catégorie prédite : {result['categorie']} (code {result['prdtypecode']})")


def predict_file(input_path: str, output_path: str, models_dir: str) -> None:
    clf = TextClassifier(models_dir)
    df = pd.read_csv(input_path)

    results = df.apply(
        lambda row: clf.predict(row.get("designation", ""), row.get("description", "")),
        axis=1,
    )
    df["prdtypecode_predit"] = results.apply(lambda r: r["prdtypecode"])
    df["categorie_predite"] = results.apply(lambda r: r["categorie"])

    df.to_csv(output_path, index=False)
    print(f"💾 Prédictions sauvegardées -> {output_path}")


def evaluate_file(input_path: str, models_dir: str) -> None:
    clf = TextClassifier(models_dir)
    df = pd.read_csv(input_path).dropna(subset=["text_combined", "prdtypecode"])

    X = clf.vectorizer.transform(df["text_combined"])
    y_true = clf.encoder.transform(df["prdtypecode"])
    y_pred = clf.model.predict(X)

    metrics = evaluate_model(y_true, y_pred)
    print("Évaluation :")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Prédiction / évaluation du modèle texte Rakuten")
    parser.add_argument("--text", help="Texte unique à classifier (designation + description)")
    parser.add_argument("--input", help="Fichier CSV à traiter")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--eval", action="store_true", help="Évalue au lieu de prédire (nécessite prdtypecode)")
    args = parser.parse_args()

    if args.text:
        predict_single(args.text, args.models_dir)
    elif args.input and args.eval:
        evaluate_file(args.input, args.models_dir)
    elif args.input:
        predict_file(args.input, args.output, args.models_dir)
    else:
        parser.error("Utilisez --text ou --input (voir --help)")


if __name__ == "__main__":
    main()
