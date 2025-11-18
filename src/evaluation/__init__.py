"""
Module d'évaluation pour le projet de classification de produits e-commerce.
"""

from .metrics import (
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_class_distribution
)

__all__ = [
    'evaluate_model',
    'print_classification_report',
    'plot_confusion_matrix',
    'plot_class_distribution'
]

