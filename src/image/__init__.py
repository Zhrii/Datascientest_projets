"""
Module de classification par images (prédiction du prdtypecode à partir des images produits).
"""

from .data_loader import load_image_classification_data, create_train_val_split
from .feature_extractor import ImageFeatureExtractor

__all__ = ["load_image_classification_data", "create_train_val_split", "ImageFeatureExtractor"]
