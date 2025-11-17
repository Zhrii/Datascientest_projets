"""
Module de feature engineering pour créer des features supplémentaires.
"""

import pandas as pd
import numpy as np
from typing import Optional


def create_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features de longueur pour les textes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les colonnes 'designation' et 'description'
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec les nouvelles features de longueur ajoutées
    """
    df = df.copy()
    
    # Longueur en caractères
    df['designation_length'] = df['designation'].fillna('').astype(str).str.len()
    df['description_length'] = df['description'].fillna('').astype(str).str.len()
    df['total_text_length'] = df['designation_length'] + df['description_length']
    
    # Longueur en mots
    df['designation_word_count'] = df['designation'].fillna('').astype(str).str.split().str.len()
    df['description_word_count'] = df['description'].fillna('').astype(str).str.split().str.len()
    df['total_word_count'] = df['designation_word_count'] + df['description_word_count']
    
    # Longueur moyenne des mots
    df['designation_avg_word_length'] = np.where(
        df['designation_word_count'] > 0,
        df['designation_length'] / df['designation_word_count'],
        0
    )
    df['description_avg_word_length'] = np.where(
        df['description_word_count'] > 0,
        df['description_length'] / df['description_word_count'],
        0
    )
    
    return df


def create_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features binaires pour indiquer la présence/absence de données.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les colonnes 'designation' et 'description'
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec les nouvelles features binaires ajoutées
    """
    df = df.copy()
    
    # Présence de description
    df['has_description'] = (
        df['description'].notna() & 
        (df['description'].astype(str).str.strip() != '') &
        (df['description'].astype(str).str.strip() != 'nan')
    ).astype(int)
    
    # Présence de HTML (avant nettoyage)
    from .html_cleaner import has_html
    df['has_html'] = df['description'].apply(has_html).astype(int)
    
    # Description vide
    df['is_description_empty'] = (~df['has_description']).astype(int)
    
    return df


def create_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features de qualité pour évaluer la complétude des données.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les colonnes nécessaires
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec les nouvelles features de qualité ajoutées
    """
    df = df.copy()
    
    # Score de complétude (ratio designation / total)
    total_length = df['designation_length'] + df['description_length']
    df['text_completeness'] = np.where(
        total_length > 0,
        df['designation_length'] / total_length,
        0
    )
    
    # Score de qualité basique (présence description + longueur)
    df['description_quality_score'] = (
        df['has_description'] * 0.5 +
        (df['description_length'] > 100).astype(int) * 0.3 +
        (df['description_length'] > 500).astype(int) * 0.2
    )
    
    # Ratio mots/caractères (densité d'information)
    df['word_density'] = np.where(
        df['total_text_length'] > 0,
        df['total_word_count'] / df['total_text_length'],
        0
    )
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée toutes les features en une seule fois.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec 'designation' et 'description'
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec toutes les features ajoutées
    """
    df = create_length_features(df)
    df = create_binary_features(df)
    df = create_quality_features(df)
    return df

