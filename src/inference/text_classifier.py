"""
À placer dans src/inference.py de votre projet.

Charge une seule fois les 3 artefacts du modèle texte (vectorizer, modèle,
encoder) et expose une fonction predict() réutilisée à la fois par
scripts/predict.py et par l'API (api/main.py) — c'est le point commun qui
garantit que le modèle entraîné est bien "réutilisable par l'API".
"""
from pathlib import Path
import pickle

import pandas as pd

from src.modeling.vectorization import TFIDFVectorizer
from src.modeling.baseline_models import BaselineModels
from src.preprocessing.pipeline import PreprocessingPipeline

# Dictionnaire de traduction des 27 catégories Rakuten (repris de app.py)
DICT_CATEGORIES = {
    10: "Livre d'occasion", 40: "Jeux vidéo", 50: "Accessoire jeu vidéo",
    60: "Console de jeux", 1140: "Figurine / Goodies", 1160: "Cartes à collectionner",
    1180: "Jeu de plateau / Warhammer", 1280: "Jouet enfant", 1281: "Jeu de société",
    1300: "Jouet technique / Modélisme", 1301: "Vêtement enfant / Puériculture",
    1302: "Jeu d'extérieur", 1320: "Puériculture", 1560: "Mobilier",
    1920: "Chambre / Literie", 1940: "Alimentation / Épicerie", 2060: "Décoration intérieure",
    2220: "Animalerie", 2280: "Magazine", 2403: "Livre / BD", 2462: "Jeu vidéo (Occasion)",
    2522: "Papeterie", 2582: "Meuble d'extérieur", 2583: "Piscine / Spa",
    2585: "Bricolage / Jardinage", 2705: "Livre neuf", 2905: "Jeu PC",
}


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
