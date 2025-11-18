"""
Module contenant les modèles avancés pour la classification.
"""

import numpy as np
from typing import Dict, Any, Optional
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Modèles de base sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Modèles avancés (optionnels)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost n'est pas installé. Installez-le avec: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM n'est pas installé. Installez-le avec: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost n'est pas installé. Installez-le avec: pip install catboost")


class AdvancedModels:
    """
    Classe pour gérer les modèles avancés.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise les modèles avancés.
        
        Parameters
        ----------
        random_state : int
            Seed pour la reproductibilité
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
    
    def create_advanced_models(
        self,
        class_weights: Optional[Dict[int, float]] = None,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_catboost: bool = True
    ) -> Dict[str, Any]:
        """
        Crée les modèles avancés.
        
        Parameters
        ----------
        class_weights : dict, optional
            Poids de classes pour gérer le déséquilibre
        use_xgboost : bool
            Inclure XGBoost si disponible
        use_lightgbm : bool
            Inclure LightGBM si disponible
        use_catboost : bool
            Inclure CatBoost si disponible
            
        Returns
        -------
        dict
            Dictionnaire des modèles
        """
        self.models = {}
        
        # Random Forest amélioré
        self.models['Random Forest (optimized)'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights if class_weights else 'balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state
        )
        
        # Logistic Regression avec class weights
        if class_weights:
            self.models['Logistic Regression (weighted)'] = LogisticRegression(
                max_iter=1000,
                class_weight=class_weights,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # XGBoost
        if use_xgboost and XGBOOST_AVAILABLE:
            scale_pos_weight = None
            if class_weights:
                # Calculer scale_pos_weight approximatif
                weights_list = list(class_weights.values())
                scale_pos_weight = max(weights_list) / min(weights_list) if weights_list else None
            
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                eval_metric='mlogloss'
            )
        
        # LightGBM
        if use_lightgbm and LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                class_weight=class_weights if class_weights else 'balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        # CatBoost
        if use_catboost and CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False,
                class_weights=class_weights
            )
        
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
        Entraîne tous les modèles avancés.
        
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
            self.create_advanced_models()
        
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
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
    
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

