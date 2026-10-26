"""
calibrate_confidence.py : choisit le seuil de "faible confiance" de chaque modèle

On prédit des produits mis de côté (data/holdout/holdout.csv : vraie catégorie
connue, jamais vus à l'entraînement) et on regarde l'accuracy selon la marge
entre les deux meilleurs scores. Seuil retenu : la marge à partir de laquelle
le modèle a raison plus d'une fois sur deux, mesurée localement (fenêtre
glissante sur les prédictions de marge voisine).
=> "peu sûre" veut dire "dans la zone où le modèle se trompe plus souvent
   qu'il n'a raison : à faire vérifier par un modérateur".

Pour lancer (dans le container de l'API, qui a les modèles) :
    docker compose exec api python scripts/calibrate_confidence.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.inference.predictor import ProductPredictor

TARGET_ACCURACY = 0.5
IMAGE_DIR = ROOT / "data" / "processed" / "image_train"


def calibrate(margins: np.ndarray, correct: np.ndarray, target: float = TARGET_ACCURACY,
              window: int | None = None) -> float:
    """Marge à partir de laquelle le modèle a durablement raison plus d'une fois sur deux.

    Accuracy locale = moyenne des bonnes réponses sur une fenêtre glissante de
    `window` prédictions de marge voisine (défaut : 10 % des produits, au moins 50).
    Seuil = marge qui suit la DERNIÈRE position où l'accuracy locale est <= target
    (robuste à un pic de bruit dans les petites marges).
    Un critère cumulatif ("l'ensemble sous le seuil a <= 50 % d'accuracy") marquerait
    beaucoup trop de prédictions justes pour un modèle faible.
    """
    order = np.argsort(margins)
    m, c = margins[order], correct[order].astype(float)
    window = window or max(50, len(m) // 10)
    local_accuracy = pd.Series(c).rolling(window, center=True, min_periods=1).mean().to_numpy()
    below = np.where(local_accuracy <= target)[0]
    if not len(below):
        return 0.0  # toujours plus d'une chance sur deux d'avoir raison : rien de peu sûr
    k = below[-1] + 1
    return float(m[k]) if k < len(m) else float(m[-1]) + 1e-9


def report(name: str, margins: np.ndarray, correct: np.ndarray) -> float:
    threshold = calibrate(margins, correct)
    low = margins < threshold
    print(f"\n=== {name} : {len(margins)} produits, accuracy {correct.mean():.3f}")
    deciles = pd.qcut(margins, 5, duplicates="drop")
    table = pd.DataFrame({"marge": deciles, "correct": correct}).groupby("marge", observed=True)["correct"]
    print(table.agg(["count", "mean"]).rename(columns={"count": "produits", "mean": "accuracy"}).round(3).to_string())
    print(f"seuil retenu : {threshold:.3f}  ->  {low.mean():.0%} des prédictions peu sûres "
          f"(accuracy {correct[low].mean() if low.any() else float('nan'):.3f}), "
          f"accuracy des sûres {correct[~low].mean():.3f}")
    return threshold


def main():
    parser = argparse.ArgumentParser(description="Calibrage du seuil de faible confiance")
    parser.add_argument("--holdout", default=str(ROOT / "data" / "holdout" / "holdout.csv"))
    parser.add_argument("--n-text", type=int, default=2000)
    parser.add_argument("--n-image", type=int, default=300, help="produits pour image et fusion (CLIP / ResNet, plus lents)")
    args = parser.parse_args()

    df = pd.read_csv(args.holdout)
    predictor = ProductPredictor()
    texts = (df["designation"].fillna("") + " " + df["description"].fillna("")).str.strip()
    thresholds = {}

    sample = df.sample(min(args.n_text, len(df)), random_state=0)
    res = [predictor.predict_text(texts[i]) for i in sample.index]
    thresholds["text"] = report("texte", np.array([r["confidence"] for r in res]),
                                np.array([r["prdtypecode"] == t for r, t in zip(res, sample["prdtypecode"])]))

    sample = df.sample(min(args.n_image, len(df)), random_state=0)
    paths = [IMAGE_DIR / f"image_{row.imageid}_product_{row.productid}.jpg" for row in sample.itertuples()]
    keep = [i for i, p in zip(sample.index, paths) if p.exists()]
    for modality in ("image", "fusion"):
        res = []
        for i in keep:
            path = IMAGE_DIR / f"image_{df.at[i, 'imageid']}_product_{df.at[i, 'productid']}.jpg"
            res.append(predictor.predict_image(path) if modality == "image" else predictor.predict_fusion(texts[i], path))
        thresholds[modality] = report(modality, np.array([r["confidence"] for r in res]),
                                      np.array([r["prdtypecode"] == t for r, t in zip(res, df.loc[keep, "prdtypecode"])]))

    print("\nLOW_CONFIDENCE_MARGIN =", {k: round(v, 3) for k, v in thresholds.items()})


if __name__ == "__main__":
    main()
