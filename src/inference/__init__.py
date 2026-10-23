"""Service d'inférence partagé (FastAPI + Streamlit)."""

from .categories import DICT_CATEGORIES, category_name
from .predictor import ProductPredictor

__all__ = ["DICT_CATEGORIES", "category_name", "ProductPredictor"]
