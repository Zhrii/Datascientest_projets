"""
Module d'analyse de l'importance des features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    figsize: tuple = (12, 8)
) -> None:
    """
    Affiche l'importance des features.
    
    Parameters
    ----------
    feature_importance : dict
        Dictionnaire {feature: importance}
    top_n : int
        Nombre de features à afficher
    figsize : tuple
        Taille de la figure
    """
    # Trier par importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features par Importance', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def get_top_features(
    feature_importance: Dict[str, float],
    top_n: int = 20
) -> List[tuple]:
    """
    Retourne les top N features.
    
    Parameters
    ----------
    feature_importance : dict
        Dictionnaire {feature: importance}
    top_n : int
        Nombre de features à retourner
        
    Returns
    -------
    list
        Liste de tuples (feature, importance)
    """
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_features[:top_n]


def analyze_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Analyse l'importance des features d'un modèle.
    
    Parameters
    ----------
    model : sklearn model
        Modèle entraîné
    feature_names : list, optional
        Noms des features
    top_n : int
        Nombre de top features à retourner
        
    Returns
    -------
    dict
        Résultats de l'analyse
    """
    feature_importance = {}
    
    # Tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        feature_importance = dict(zip(feature_names, importances))
    
    # Linear models (coefficients)
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        # Pour multi-class, prendre la moyenne absolue des coefficients
        if len(coef.shape) > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        feature_importance = dict(zip(feature_names, importances))
    
    else:
        print("⚠️  Le modèle ne supporte pas l'analyse d'importance des features")
        return {}
    
    # Top features
    top_features = get_top_features(feature_importance, top_n)
    
    results = {
        'feature_importance': feature_importance,
        'top_features': top_features,
        'total_features': len(feature_importance)
    }
    
    return results

