"""
Module contenant les modèles baselines pour la classification.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Any, Optional
import pickle
from pathlib import Path


class BaselineModels:
    """
    Classe pour gérer et comparer les modèles baselines.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise les modèles baselines.
        
        Parameters
        ----------
        random_state : int
            Seed pour la reproductibilité
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
    
    def create_baseline_models(self) -> Dict[str, Any]:
        """
        Crée les modèles baselines.
        
        Returns
        -------
        dict
            Dictionnaire des modèles
        """
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM (Linear)': LinearSVC(
                max_iter=1000,
                random_state=self.random_state,
                dual=False
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=20
            )
        }
        return self.models
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Entraîne un modèle spécifique.
        
        Parameters
        ----------
        model_name : str
            Nom du modèle à entraîner
        X_train : np.ndarray
            Features d'entraînement
        y_train : np.ndarray
            Labels d'entraînement
            
        Returns
        -------
        Modèle entraîné
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé. Modèles disponibles: {list(self.models.keys())}")
        
        print(f"🔄 Entraînement de {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        print(f"✅ {model_name} entraîné avec succès")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Entraîne tous les modèles baselines.
        
        Parameters
        ----------
        X_train : np.ndarray
            Features d'entraînement
        y_train : np.ndarray
            Labels d'entraînement
            
        Returns
        -------
        dict
            Dictionnaire des modèles entraînés
        """
        if not self.models:
            self.create_baseline_models()
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
        
        return self.trained_models
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Prédit avec un modèle entraîné.
        
        Parameters
        ----------
        model_name : str
            Nom du modèle
        X : np.ndarray
            Features
            
        Returns
        -------
        np.ndarray
            Prédictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle '{model_name}' non entraîné")
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités avec un modèle entraîné.
        
        Parameters
        ----------
        model_name : str
            Nom du modèle
        X : np.ndarray
            Features
            
        Returns
        -------
        np.ndarray
            Probabilités de prédiction
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle '{model_name}' non entraîné")
        
        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # Pour SVM, utiliser decision_function
            decision = model.decision_function(X)
            # Approximation softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
    
    def cross_validate(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_macro'
    ) -> Dict[str, Any]:
        """
        Effectue une validation croisée sur un modèle.
        
        Parameters
        ----------
        model_name : str
            Nom du modèle
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv : int
            Nombre de folds
        scoring : str
            Métrique de scoring
            
        Returns
        -------
        dict
            Résultats de la validation croisée
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        model = self.models[model_name]
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        results = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        self.results[model_name] = results
        return results
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Sauvegarde un modèle entraîné.
        
        Parameters
        ----------
        model_name : str
            Nom du modèle
        filepath : str
            Chemin de sauvegarde
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle '{model_name}' non entraîné")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)
    
    @classmethod
    def load_model(cls, filepath: str) -> Any:
        """
        Charge un modèle sauvegardé.
        
        Parameters
        ----------
        filepath : str
            Chemin du fichier
            
        Returns
        -------
        Modèle chargé
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

