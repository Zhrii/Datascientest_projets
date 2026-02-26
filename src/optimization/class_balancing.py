"""
Module pour gérer le déséquilibre de classes.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  imbalanced-learn n'est pas installé. SMOTE et ADASYN ne seront pas disponibles.")


def create_class_weights(
    y: np.ndarray,
    method: str = 'balanced'
) -> Dict[int, float]:
    """
    Crée des poids de classes pour gérer le déséquilibre.
    
    Parameters
    ----------
    y : np.ndarray
        Labels
    method : str
        Méthode de calcul ('balanced', 'balanced_subsample', ou dict personnalisé)
        
    Returns
    -------
    dict
        Dictionnaire {classe: poids}
    """
    classes = np.unique(y)
    
    if method == 'balanced':
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y
        )
    elif method == 'balanced_subsample':
        class_weights = compute_class_weight(
            'balanced_subsample',
            classes=classes,
            y=y
        )
    else:
        raise ValueError(f"Méthode '{method}' non supportée. Utilisez 'balanced' ou 'balanced_subsample'")
    
    # Créer le dictionnaire
    weight_dict = dict(zip(classes, class_weights))
    
    print(f"✅ Poids de classes calculés ({method})")
    print(f"   Nombre de classes : {len(classes)}")
    print(f"   Poids min : {min(class_weights):.4f}")
    print(f"   Poids max : {max(class_weights):.4f}")
    print(f"   Ratio max/min : {max(class_weights) / min(class_weights):.2f}")
    
    return weight_dict


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str = 'auto',
    random_state: int = 42,
    k_neighbors: int = 5
) -> tuple:
    """
    Applique SMOTE pour rééquilibrer les classes.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    sampling_strategy : str ou float
        Stratégie de rééchantillonnage
    random_state : int
        Seed pour la reproductibilité
    k_neighbors : int
        Nombre de voisins pour SMOTE
        
    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn n'est pas installé. Installez-le avec: pip install imbalanced-learn")
    
    print(f"🔄 Application de SMOTE...")
    print(f"   Taille avant : {len(X)} échantillons")
    print(f"   Distribution avant : {np.bincount(y)}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"✅ SMOTE appliqué avec succès")
    print(f"   Taille après : {len(X_resampled)} échantillons")
    print(f"   Distribution après : {np.bincount(y_resampled)}")
    print(f"   Augmentation : {len(X_resampled) - len(X)} échantillons (+{(len(X_resampled) - len(X)) / len(X) * 100:.1f}%)")
    
    return X_resampled, y_resampled


def apply_adasyn(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str = 'auto',
    random_state: int = 42,
    n_neighbors: int = 5
) -> tuple:
    """
    Applique ADASYN pour rééquilibrer les classes.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    sampling_strategy : str ou float
        Stratégie de rééchantillonnage
    random_state : int
        Seed pour la reproductibilité
    n_neighbors : int
        Nombre de voisins pour ADASYN
        
    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn n'est pas installé. Installez-le avec: pip install imbalanced-learn")
    
    print(f"🔄 Application d'ADASYN...")
    print(f"   Taille avant : {len(X)} échantillons")
    print(f"   Distribution avant : {np.bincount(y)}")
    
    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        n_neighbors=n_neighbors
    )
    
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    print(f"✅ ADASYN appliqué avec succès")
    print(f"   Taille après : {len(X_resampled)} échantillons")
    print(f"   Distribution après : {np.bincount(y_resampled)}")
    print(f"   Augmentation : {len(X_resampled) - len(X)} échantillons (+{(len(X_resampled) - len(X)) / len(X) * 100:.1f}%)")
    
    return X_resampled, y_resampled

