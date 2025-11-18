"""
Module d'interprétabilité pour le projet de classification de produits e-commerce.
"""

from .feature_importance import (
    plot_feature_importance,
    get_top_features,
    analyze_feature_importance
)

from .shap_analysis import (
    explain_model_with_shap,
    plot_shap_summary,
    plot_shap_waterfall
)

__all__ = [
    'plot_feature_importance',
    'get_top_features',
    'analyze_feature_importance',
    'explain_model_with_shap',
    'plot_shap_summary',
    'plot_shap_waterfall'
]

