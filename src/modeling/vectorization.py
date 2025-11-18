"""
Module de vectorisation des textes pour la modélisation.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Tuple, Union, List
import pickle
from pathlib import Path

# Stop words français (liste complète)
def get_french_stop_words():
    """
    Retourne une liste de stop words français.
    Essaie d'utiliser NLTK si disponible, sinon retourne une liste basique.
    """
    try:
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return stopwords.words('french')
    except (ImportError, LookupError):
        # Liste basique de stop words français si NLTK n'est pas disponible
        return [
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'ou', 'mais', 'car', 'ni', 'or', 'donc', 'comme', 'si',
            'quand', 'où', 'quoi', 'qui', 'dont', 'du', 'des', 'les', 'la', 'au', 'aux',
            'ce', 'cet', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
            'me', 'te', 'nous', 'vous', 'lui', 'leur', 'y', 'en', 'lequel',
            'laquelle', 'lesquels', 'lesquelles', 'celui', 'celle', 'ceux', 'celles',
            'sont', 'été', 'étaient', 'sera', 'seront', 'a', 'as', 'avait', 'auront',
            'est', 'était', 'sera', 'fut', 'seront', 'ont', 'avaient', 'auront',
            'peut', 'peuvent', 'pourrait', 'pourraient', 'doit', 'doivent', 'devrait',
            'devraient', 'fait', 'font', 'feront', 'faisait', 'faisaient'
        ]

FRENCH_STOP_WORDS = get_french_stop_words()


class TFIDFVectorizer:
    """
    Wrapper autour de TfidfVectorizer de scikit-learn avec fonctionnalités supplémentaires.
    """
    
    def __init__(
        self,
        max_features: Optional[int] = 10000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        lowercase: bool = True,
        stop_words: Union[str, List[str], None] = 'french'
    ):
        """
        Initialise le vectoriseur TF-IDF.
        
        Parameters
        ----------
        max_features : int ou None
            Nombre maximum de features (vocabulaire)
        min_df : int ou float
            Fréquence minimale d'un terme (absolu ou relatif)
        max_df : float
            Fréquence maximale d'un terme (pour filtrer les mots trop fréquents)
        ngram_range : tuple
            Plage de n-grams (ex: (1, 2) pour unigrams et bigrams)
        lowercase : bool
            Convertir en minuscules
        stop_words : str ou None
            Langue des stop words ('french', 'english', None)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        
        # Gérer les stop words : convertir 'french' en liste
        if stop_words == 'french':
            self.stop_words = FRENCH_STOP_WORDS
        elif stop_words == 'english':
            self.stop_words = 'english'  # scikit-learn supporte 'english'
        else:
            self.stop_words = stop_words
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=lowercase,
            stop_words=self.stop_words
        )
        
        self.is_fitted = False
    
    def fit(self, texts: pd.Series) -> 'TFIDFVectorizer':
        """
        Entraîne le vectoriseur sur les textes.
        
        Parameters
        ----------
        texts : pd.Series
            Série de textes à vectoriser
            
        Returns
        -------
        self
        """
        self.vectorizer.fit(texts.astype(str))
        self.is_fitted = True
        return self
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transforme les textes en vecteurs TF-IDF.
        
        Parameters
        ----------
        texts : pd.Series
            Série de textes à vectoriser
            
        Returns
        -------
        np.ndarray
            Matrice de vecteurs TF-IDF
        """
        if not self.is_fitted:
            raise ValueError("Le vectoriseur doit être entraîné avec fit() avant transform()")
        return self.vectorizer.transform(texts.astype(str))
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """
        Entraîne et transforme en une seule étape.
        
        Parameters
        ----------
        texts : pd.Series
            Série de textes à vectoriser
            
        Returns
        -------
        np.ndarray
            Matrice de vecteurs TF-IDF
        """
        result = self.vectorizer.fit_transform(texts.astype(str))
        self.is_fitted = True
        return result
    
    def get_feature_names_out(self) -> np.ndarray:
        """
        Retourne les noms des features (mots du vocabulaire).
        
        Returns
        -------
        np.ndarray
            Noms des features
        """
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self) -> int:
        """
        Retourne la taille du vocabulaire.
        
        Returns
        -------
        int
            Taille du vocabulaire
        """
        return len(self.vectorizer.vocabulary_)
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le vectoriseur.
        
        Parameters
        ----------
        filepath : str
            Chemin du fichier de sauvegarde
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'TFIDFVectorizer':
        """
        Charge un vectoriseur sauvegardé.
        
        Parameters
        ----------
        filepath : str
            Chemin du fichier à charger
            
        Returns
        -------
        TFIDFVectorizer
            Instance chargée
        """
        with open(filepath, 'rb') as f:
            vectorizer = pickle.load(f)
        instance = cls()
        instance.vectorizer = vectorizer
        instance.is_fitted = True
        return instance


def create_vectorizer(
    max_features: Optional[int] = 10000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: Tuple[int, int] = (1, 2)
) -> TFIDFVectorizer:
    """
    Fonction utilitaire pour créer un vectoriseur TF-IDF.
    
    Parameters
    ----------
    max_features : int ou None
        Nombre maximum de features
    min_df : int
        Fréquence minimale
    max_df : float
        Fréquence maximale
    ngram_range : tuple
        Plage de n-grams
        
    Returns
    -------
    TFIDFVectorizer
        Instance du vectoriseur
    """
    return TFIDFVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range
    )

