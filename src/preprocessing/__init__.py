"""
Module de preprocessing pour le projet de classification de produits e-commerce.
"""

from .html_cleaner import clean_html, decode_html_entities, has_html, remove_remaining_html
from .text_normalizer import normalize_text, combine_texts
from .feature_engineering import (
    create_length_features,
    create_binary_features,
    create_quality_features
)
from .pipeline import PreprocessingPipeline

__all__ = [
    'clean_html',
    'decode_html_entities',
    'has_html',
    'remove_remaining_html',
    'normalize_text',
    'combine_texts',
    'create_length_features',
    'create_binary_features',
    'create_quality_features',
    'PreprocessingPipeline'
]

