"""Service d'inférence partagé (FastAPI + Streamlit + scripts).

Pas d'import de predictor ici : il dépend de torch / CLIP, absents de
l'image d'entraînement. Importer directement src.inference.predictor.
"""

from .categories import DICT_CATEGORIES, category_name
from .text_classifier import TextClassifier

__all__ = ["DICT_CATEGORIES", "category_name", "TextClassifier"]
