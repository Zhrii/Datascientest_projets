"""
holdout.py : jeu de produits étiquetés mis de côté, jamais utilisé pour entraîner

Une petite part de X_train (HOLDOUT_SHARE) est retirée des données d'entraînement,
toujours les mêmes produits (choix par empreinte du productid, pas par tirage au
hasard). Ces produits ont une vraie catégorie et le modèle ne les a jamais vus :
  - ils servent à simuler des retours de modérateurs (F1 en production)
  - ils servent à calibrer le seuil de "faible confiance" du modèle

Pour lancer :
    python scripts/holdout.py --input data/raw/merged.csv
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

HOLDOUT_SHARE = 0.03


def in_holdout(productid, share: float = HOLDOUT_SHARE) -> bool:
    """Même réponse à chaque exécution pour un même produit (empreinte md5 du productid)."""
    digest = hashlib.md5(str(productid).encode()).hexdigest()
    return int(digest[:8], 16) % 10_000 < share * 10_000


def split_holdout(df: pd.DataFrame, share: float = HOLDOUT_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie (données d'entraînement, produits mis de côté)."""
    mask = df["productid"].map(lambda pid: in_holdout(pid, share))
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def evaluate_on_holdout(model_dir, holdout_csv) -> dict:
    """F1-macro et accuracy d'un modèle (3 fichiers .pkl de model_dir) sur les produits mis de côté.

    Tous les modèles sont notés sur les mêmes produits, jamais vus à l'entraînement :
    la comparaison candidat / production (promote_model.py) est équitable.
    """
    import pickle
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from sklearn.metrics import accuracy_score, f1_score

    from src.modeling.baseline_models import BaselineModels
    from src.modeling.vectorization import TFIDFVectorizer
    from src.preprocessing.pipeline import PreprocessingPipeline

    model_dir = Path(model_dir)
    df = pd.read_csv(holdout_csv)
    texts = PreprocessingPipeline(combine_text=True, create_features=False).fit_transform(df)["text_combined"]
    vectorizer = TFIDFVectorizer.load(model_dir / "tfidf_vectorizer.pkl")
    model = BaselineModels.load_model(model_dir / "svm_text_model.pkl")
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    y_pred = encoder.inverse_transform(model.predict(vectorizer.transform(texts.fillna(""))))
    y_true = df["prdtypecode"].astype(int)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(df)),
    }


def main():
    parser = argparse.ArgumentParser(description="Sépare les produits mis de côté (holdout)")
    parser.add_argument("--input", default="data/raw/merged.csv")
    parser.add_argument("--holdout", default="data/holdout/holdout.csv")
    args = parser.parse_args()

    train, holdout = split_holdout(pd.read_csv(args.input))
    Path(args.holdout).parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(args.holdout, index=False)
    train.to_csv(args.input, index=False)
    print(f"{len(train):,} produits pour l'entraînement, {len(holdout):,} mis de côté -> {args.holdout}")


if __name__ == "__main__":
    main()

