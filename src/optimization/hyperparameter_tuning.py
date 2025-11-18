"""
Module d'optimisation des hyperparamètres.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from typing import Dict, Any, Optional
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def optimize_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any],
    cv: int = 5,
    scoring: str = 'f1_macro',
    method: str = 'grid',
    n_iter: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Optimise les hyperparamètres d'un modèle.
    
    Parameters
    ----------
    model : sklearn estimator
        Modèle à optimiser
    X_train : np.ndarray
        Features d'entraînement
    y_train : np.ndarray
        Labels d'entraînement
    param_grid : dict
        Grille de paramètres à tester
    cv : int
        Nombre de folds pour la validation croisée
    scoring : str
        Métrique de scoring
    method : str
        Méthode d'optimisation ('grid' ou 'random')
    n_iter : int, optional
        Nombre d'itérations pour RandomSearch (si method='random')
    n_jobs : int
        Nombre de jobs parallèles
    random_state : int
        Seed pour la reproductibilité
        
    Returns
    -------
    dict
        Résultats de l'optimisation
    """
    # Créer le scorer
    scorer = make_scorer(f1_score, average='macro', zero_division=0)
    
    # Stratified K-Fold pour gérer le déséquilibre
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    if method == 'grid':
        search = GridSearchCV(
            model,
            param_grid,
            cv=skf,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=1
        )
    elif method == 'random':
        if n_iter is None:
            n_iter = 20  # Par défaut
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=skf,
            scoring=scorer,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )
    else:
        raise ValueError(f"Méthode '{method}' non supportée. Utilisez 'grid' ou 'random'")
    
    print(f"🔄 Optimisation des hyperparamètres ({method})...")
    search.fit(X_train, y_train)
    
    results = {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_,
        'search': search
    }
    
    print(f"✅ Optimisation terminée !")
    print(f"   Meilleur score (CV) : {search.best_score_:.4f}")
    print(f"   Meilleurs paramètres : {search.best_params_}")
    
    return results


def grid_search_optimization(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any],
    cv: int = 5,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Optimisation par GridSearch.
    
    Parameters
    ----------
    model : sklearn estimator
        Modèle à optimiser
    X_train : np.ndarray
        Features d'entraînement
    y_train : np.ndarray
        Labels d'entraînement
    param_grid : dict
        Grille de paramètres
    cv : int
        Nombre de folds
    n_jobs : int
        Nombre de jobs parallèles
        
    Returns
    -------
    dict
        Résultats de l'optimisation
    """
    return optimize_model(
        model, X_train, y_train, param_grid,
        cv=cv, method='grid', n_jobs=n_jobs
    )


def random_search_optimization(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_distributions: Dict[str, Any],
    n_iter: int = 20,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Optimisation par RandomSearch.
    
    Parameters
    ----------
    model : sklearn estimator
        Modèle à optimiser
    X_train : np.ndarray
        Features d'entraînement
    y_train : np.ndarray
        Labels d'entraînement
    param_distributions : dict
        Distributions de paramètres
    n_iter : int
        Nombre d'itérations
    cv : int
        Nombre de folds
    n_jobs : int
        Nombre de jobs parallèles
    random_state : int
        Seed pour la reproductibilité
        
    Returns
    -------
    dict
        Résultats de l'optimisation
    """
    return optimize_model(
        model, X_train, y_train, param_distributions,
        cv=cv, method='random', n_iter=n_iter,
        n_jobs=n_jobs, random_state=random_state
    )

