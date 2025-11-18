"""
Module d'optimisation pour le projet de classification de produits e-commerce.
"""

from .hyperparameter_tuning import (
    optimize_model,
    grid_search_optimization,
    random_search_optimization
)
from .class_balancing import (
    create_class_weights,
    apply_smote,
    apply_adasyn
)

__all__ = [
    'optimize_model',
    'grid_search_optimization',
    'random_search_optimization',
    'create_class_weights',
    'apply_smote',
    'apply_adasyn'
]

