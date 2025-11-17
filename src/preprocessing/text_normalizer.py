"""
Module de normalisation de texte pour le preprocessing.
"""

import re
import unicodedata
import pandas as pd
from typing import Optional


def normalize_text(
    text: Optional[str],
    lowercase: bool = True,
    remove_extra_spaces: bool = True,
    remove_accents: bool = False,
    min_length: int = 0
) -> str:
    """
    Normalise un texte selon les paramètres spécifiés.
    
    Parameters
    ----------
    text : str ou None
        Le texte à normaliser
    lowercase : bool, default=True
        Convertir en minuscules
    remove_extra_spaces : bool, default=True
        Supprimer les espaces multiples
    remove_accents : bool, default=False
        Supprimer les accents (déconseillé pour le français)
    min_length : int, default=0
        Longueur minimale du texte (0 = pas de filtre)
        
    Returns
    -------
    str
        Le texte normalisé
    """
    if not text or pd.isna(text):
        return ''
    
    text = str(text)
    
    # Convertir en minuscules si demandé
    if lowercase:
        text = text.lower()
    
    # Supprimer les accents si demandé (déconseillé pour le français)
    if remove_accents:
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Supprimer les espaces multiples
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    # Filtrer par longueur minimale
    if min_length > 0 and len(text) < min_length:
        return ''
    
    return text


def combine_texts(
    designation: Optional[str],
    description: Optional[str],
    separator: str = ' ',
    handle_missing: str = 'designation_only'
) -> str:
    """
    Combine la designation et la description en un seul texte.
    
    Parameters
    ----------
    designation : str ou None
        La désignation du produit
    description : str ou None
        La description du produit
    separator : str, default=' '
        Le séparateur entre designation et description
    handle_missing : str, default='designation_only'
        Stratégie pour gérer les descriptions manquantes:
        - 'designation_only': utiliser uniquement la designation
        - 'marker': remplacer par '[DESCRIPTION_VIDE]'
        - 'skip': ignorer les descriptions vides
        
    Returns
    -------
    str
        Le texte combiné
    """
    # Nettoyer les valeurs None/NaN
    designation = str(designation) if designation and not pd.isna(designation) else ''
    description = str(description) if description and not pd.isna(description) else ''
    
    # Gérer les descriptions manquantes
    if not description or description.strip() == '':
        if handle_missing == 'marker':
            description = '[DESCRIPTION_VIDE]'
        elif handle_missing == 'skip':
            description = ''
        # 'designation_only' : on laisse description vide
    
    # Combiner les textes
    if designation and description:
        return f"{designation}{separator}{description}"
    elif designation:
        return designation
    elif description:
        return description
    else:
        return ''

