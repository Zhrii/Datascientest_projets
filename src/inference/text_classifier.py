"""
Classifieur texte léger (TF-IDF + SVM) utilisé par scripts/predict.py.

Charge une seule fois les 3 artefacts du modèle texte produits par
scripts/training.py (vectorizer, modèle, encoder). L'API utilise les mêmes
fichiers via src/inference/predictor.py.
"""
from pathlib import Path
import pickle

import pandas as pd

from src.modeling.vectorization import TFIDFVectorizer
from src.modeling.baseline_models import BaselineModels
from src.preprocessing.pipeline import PreprocessingPipeline

from .categories import DICT_CATEGORIES


class TextClassifier:
    """Charge les artefacts du modèle texte et prédit la catégorie d'un produit."""

    def __init__(self, models_dir: str = "models"):
        models_dir = Path(models_dir)
        self.vectorizer = TFIDFVectorizer.load(models_dir / "tfidf_vectorizer.pkl")
        self.model = BaselineModels.load_model(models_dir / "svm_text_model.pkl")
        with open(models_dir / "label_encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)
        self.preprocessing = PreprocessingPipeline(combine_text=True, create_features=False)

    def predict(self, designation: str, description: str = "") -> dict:
        df = pd.DataFrame({"designation": [designation], "description": [description]})
        df = self.preprocessing.fit_transform(df)
        text = df["text_combined"].iloc[0]

        X = self.vectorizer.transform(pd.Series([text]))
        pred_encoded = self.model.predict(X)[0]
        prdtypecode = int(self.encoder.inverse_transform([pred_encoded])[0])

        return {
            "prdtypecode": prdtypecode,
            "categorie": DICT_CATEGORIES.get(prdtypecode, f"Code {prdtypecode}"),
        }
