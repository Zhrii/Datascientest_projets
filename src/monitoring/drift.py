"""
Détection de dérive des données avec Evidently.

On compare deux jeux décrits par les mêmes variables :
  - référence : un échantillon des données d'entraînement (data/processed/processed_data.csv)
  - courant   : les derniers produits reçus par l'API (journal des prédictions, src/state.py)

Variables surveillées (calculées sur le texte nettoyé, comme à l'entraînement) :
  text_length, word_count, digit_ratio : forme du texte (vendeurs qui changent leur façon d'écrire)
  category : catégorie (vraie en référence, prédite en courant) : dérive des prédictions

Méthodes : distance de Wasserstein (variables numériques) et de Jensen-Shannon
(catégorie). Une variable a dérivé si sa distance dépasse son seuil ; le jeu est
"en dérive" si au moins DRIFT_SHARE des variables ont dérivé.

Seuils calibrés sur de vrais produits Rakuten (jeu de test) envoyés à l'API :
  - produits réels, 300 échantillons : distances <= 0.11 (bruit d'échantillonnage)
  - produits inhabituels (courts, anglais, chiffres) : distances de 0.7 à 6
  => NUM_THRESHOLD = 0.2, CAT_THRESHOLD = 0.25 (la catégorie compare des
     catégories prédites à des catégories vraies : un écart de base existe).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

NUMERIC_FEATURES = ["text_length", "word_count", "digit_ratio"]
CATEGORICAL_FEATURES = ["category"]
DRIFT_SHARE = 0.5     # part des variables en dérive pour déclarer le jeu en dérive
NUM_THRESHOLD = 0.2   # distance de Wasserstein (normalisée)
CAT_THRESHOLD = 0.25  # distance de Jensen-Shannon

_DIGITS = re.compile(r"\d")


@dataclass
class DriftResult:
    drift_detected: bool
    drift_share: float
    drifted_columns: int
    columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_reference: int = 0
    n_current: int = 0


def text_features(texts: pd.Series) -> pd.DataFrame:
    texts = texts.fillna("").astype(str)
    lengths = texts.str.len()
    return pd.DataFrame({
        "text_length": lengths.astype(float),
        "word_count": texts.str.split().str.len().fillna(0).astype(float),
        "digit_ratio": texts.map(lambda t: len(_DIGITS.findall(t))) / lengths.clip(lower=1),
    })


def build_reference(processed_csv: Path, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Échantillon des données d'entraînement (texte déjà nettoyé + vraie catégorie)."""
    df = pd.read_csv(processed_csv, usecols=["text_combined", "prdtypecode"]).dropna()
    df = df.sample(n=min(n, len(df)), random_state=seed)
    ref = text_features(df["text_combined"])
    ref["category"] = df["prdtypecode"].astype(int).astype(str).values
    return ref.reset_index(drop=True)


def build_current(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Derniers produits reçus par l'API, nettoyés comme le fait l'API avant de prédire."""
    from src.preprocessing.serving import clean_product_texts

    rows = [p for p in predictions if p.get("text") and p.get("prdtypecode") is not None]
    if not rows:
        return pd.DataFrame(columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    cur = text_features(pd.Series(clean_product_texts([p["text"] for p in rows])))
    cur["category"] = [str(p["prdtypecode"]) for p in rows]
    return cur


def _column_result(config: Dict[str, Any], value: float) -> Dict[str, Any]:
    method = config.get("method", "")
    threshold = config.get("threshold")
    # p-value : dérive si p < seuil ; distance (Wasserstein, Jensen-Shannon) : dérive si d >= seuil
    drifted = value < threshold if "p_value" in method else value >= threshold
    return {"method": method, "score": float(value), "threshold": threshold, "drifted": bool(drifted)}


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, html_path: Optional[Path] = None) -> DriftResult:
    """Lance le rapport Evidently DataDriftPreset ; l'enregistre en HTML si html_path est donné."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=NUMERIC_FEATURES, categorical_columns=CATEGORICAL_FEATURES)
    preset = DataDriftPreset(
        drift_share=DRIFT_SHARE,
        num_method="wasserstein", num_threshold=NUM_THRESHOLD,
        cat_method="jensenshannon", cat_threshold=CAT_THRESHOLD,
    )
    snapshot = Report([preset]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))

    share, count, columns = 0.0, 0, {}
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith("DriftedColumnsCount"):
            share, count = float(metric["value"]["share"]), int(metric["value"]["count"])
        elif config["type"].endswith("ValueDrift"):
            columns[config["column"]] = _column_result(config, metric["value"])

    return DriftResult(
        drift_detected=share >= DRIFT_SHARE,
        drift_share=share,
        drifted_columns=count,
        columns=columns,
        n_reference=len(reference),
        n_current=len(current),
    )
