"""
Pipeline complet de preprocessing pour le projet.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from .html_cleaner import clean_html, has_html
from .text_normalizer import normalize_text, combine_texts
from .feature_engineering import create_all_features


class PreprocessingPipeline:
    """
    Pipeline complet de preprocessing pour les données de produits e-commerce.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_extra_spaces: bool = True,
        remove_accents: bool = False,
        handle_missing: str = 'designation_only',
        combine_text: bool = True,
        create_features: bool = True
    ):
        """
        Initialise le pipeline de preprocessing.
        
        Parameters
        ----------
        lowercase : bool, default=True
            Convertir en minuscules
        remove_extra_spaces : bool, default=True
            Supprimer les espaces multiples
        remove_accents : bool, default=False
            Supprimer les accents
        handle_missing : str, default='designation_only'
            Stratégie pour gérer les descriptions manquantes
        combine_text : bool, default=True
            Combiner designation et description en un seul texte
        create_features : bool, default=True
            Créer des features supplémentaires
        """
        self.lowercase = lowercase
        self.remove_extra_spaces = remove_extra_spaces
        self.remove_accents = remove_accents
        self.handle_missing = handle_missing
        self.combine_text = combine_text
        self.create_features = create_features
        
        self.stats = {}
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le preprocessing complet sur le DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame avec 'designation' et 'description'
            
        Returns
        -------
        pd.DataFrame
            DataFrame préprocessé
        """
        df = df.copy()
        
        # Statistiques avant preprocessing
        self.stats['before'] = {
            'n_rows': len(df),
            'has_html_count': df['description'].apply(has_html).sum() if 'description' in df.columns else 0,
            'missing_description_count': df['description'].isna().sum() if 'description' in df.columns else 0,
        }
        
        # 1. Nettoyage HTML des descriptions
        if 'description' in df.columns:
            df['description_clean'] = df['description'].apply(
                lambda x: clean_html(x, preserve_structure=True)
            )
        else:
            df['description_clean'] = ''
        
        # 2. Normalisation des textes
        if 'designation' in df.columns:
            df['designation_clean'] = df['designation'].apply(
                lambda x: normalize_text(
                    x,
                    lowercase=self.lowercase,
                    remove_extra_spaces=self.remove_extra_spaces,
                    remove_accents=self.remove_accents
                )
            )
        else:
            df['designation_clean'] = ''
        
        if 'description_clean' in df.columns:
            df['description_clean'] = df['description_clean'].apply(
                lambda x: normalize_text(
                    x,
                    lowercase=self.lowercase,
                    remove_extra_spaces=self.remove_extra_spaces,
                    remove_accents=self.remove_accents
                )
            )
        
        # 3. Gestion des valeurs manquantes
        if 'description_clean' in df.columns:
            if self.handle_missing == 'designation_only':
                # Remplacer les descriptions vides par chaîne vide
                df['description_clean'] = df['description_clean'].fillna('')
            elif self.handle_missing == 'marker':
                # Remplacer par un marqueur
                df['description_clean'] = df['description_clean'].fillna('[DESCRIPTION_VIDE]')
        
        # 4. Combinaison des textes
        if self.combine_text:
            df['text_combined'] = df.apply(
                lambda row: combine_texts(
                    row.get('designation_clean', ''),
                    row.get('description_clean', ''),
                    separator=' ',
                    handle_missing=self.handle_missing
                ),
                axis=1
            )
        
        # 5. Création de features
        if self.create_features:
            # Utiliser les colonnes originales pour les features de longueur
            df_features = df[['designation', 'description']].copy()
            df_features = create_all_features(df_features)
            
            # Ajouter les features au DataFrame principal
            feature_cols = [col for col in df_features.columns if col not in ['designation', 'description']]
            for col in feature_cols:
                df[col] = df_features[col]
        
        # Statistiques après preprocessing
        self.stats['after'] = {
            'n_rows': len(df),
            'has_html_count': 0,  # HTML nettoyé
            'missing_description_count': (df['description_clean'] == '').sum() if 'description_clean' in df.columns else 0,
        }
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le preprocessing (identique à fit_transform pour ce pipeline).
        """
        return self.fit_transform(df)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du preprocessing.
        """
        return self.stats

