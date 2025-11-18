"""
Module d'analyse SHAP pour l'interprétabilité des modèles.
"""

import numpy as np
from typing import Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP n'est pas installé. Installez-le avec: pip install shap")


def explain_model_with_shap(
    model: Any,
    X_sample: np.ndarray,
    feature_names: Optional[list] = None,
    max_samples: int = 100
) -> Any:
    """
    Explique un modèle avec SHAP.
    
    Parameters
    ----------
    model : sklearn model
        Modèle à expliquer
    X_sample : np.ndarray
        Échantillon de données pour l'explication
    feature_names : list, optional
        Noms des features
    max_samples : int
        Nombre maximum d'échantillons à utiliser
        
    Returns
    -------
    shap.Explainer
        Explainer SHAP
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP n'est pas installé. Installez-le avec: pip install shap")
    
    # Limiter le nombre d'échantillons pour accélérer
    if len(X_sample) > max_samples:
        indices = np.random.choice(len(X_sample), max_samples, replace=False)
        X_sample = X_sample[indices]
    
    print(f"🔄 Calcul des valeurs SHAP sur {len(X_sample)} échantillons...")
    
    # Créer l'explainer selon le type de modèle
    if hasattr(model, 'predict_proba'):
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            # Linear models
            explainer = shap.LinearExplainer(model, X_sample)
    else:
        # Fallback
        explainer = shap.Explainer(model, X_sample)
    
    shap_values = explainer(X_sample)
    
    print(f"✅ Valeurs SHAP calculées")
    
    return shap_values


def plot_shap_summary(
    shap_values: Any,
    feature_names: Optional[list] = None,
    max_display: int = 20
) -> None:
    """
    Affiche le résumé SHAP.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        Valeurs SHAP
    feature_names : list, optional
        Noms des features
    max_display : int
        Nombre maximum de features à afficher
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP n'est pas installé")
    
    print("📊 Résumé SHAP :")
    shap.summary_plot(shap_values, feature_names=feature_names, max_display=max_display, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(
    shap_values: Any,
    instance_idx: int = 0,
    max_display: int = 10
) -> None:
    """
    Affiche un waterfall plot SHAP pour une instance.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        Valeurs SHAP
    instance_idx : int
        Index de l'instance à expliquer
    max_display : int
        Nombre maximum de features à afficher
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP n'est pas installé")
    
    print(f"📊 Waterfall plot pour l'instance {instance_idx} :")
    shap.plots.waterfall(shap_values[instance_idx], max_display=max_display, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.show()

