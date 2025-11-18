"""
Module de modélisation pour le projet de classification de produits e-commerce.
"""

from .vectorization import TFIDFVectorizer, create_vectorizer
from .baseline_models import BaselineModels
from .advanced_models import AdvancedModels

__all__ = [
    'TFIDFVectorizer',
    'create_vectorizer',
    'BaselineModels',
    'AdvancedModels'
]

