"""
Module multimodal : matching texte-image (Dual Encoder + Contrastive Learning).
"""

from .data_loader import load_text_image_pairs, create_pairs_dataset, load_image
from .encoders import TextEncoder, ImageEncoder
from .matching_model import DualEncoderModel, contrastive_loss
from .utils import compute_similarity, find_matching_images, find_matching_texts, recall_at_k

__all__ = [
    "load_text_image_pairs",
    "create_pairs_dataset",
    "load_image",
    "TextEncoder",
    "ImageEncoder",
    "DualEncoderModel",
    "contrastive_loss",
    "compute_similarity",
    "find_matching_images",
    "find_matching_texts",
    "recall_at_k",
]
