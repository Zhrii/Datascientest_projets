"""
Module de nettoyage HTML pour les descriptions de produits.
"""

import re
import html
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional


def has_html(text: Optional[str]) -> bool:
    """
    Détecte si un texte contient du HTML.
    
    Parameters
    ----------
    text : str ou None
        Le texte à analyser
        
    Returns
    -------
    bool
        True si le texte contient du HTML, False sinon
    """
    if not text or pd.isna(text) or text == '':
        return False
    html_pattern = re.compile(r'<[^>]+>')
    return bool(html_pattern.search(str(text)))


def decode_html_entities(text: Optional[str]) -> str:
    """
    Décode les entités HTML (ex: &eacute; -> é, &#39; -> ').
    
    Parameters
    ----------
    text : str ou None
        Le texte contenant des entités HTML
        
    Returns
    -------
    str
        Le texte avec les entités HTML décodées
    """
    if not text or pd.isna(text):
        return ''
    
    text = str(text)
    # Décoder les entités HTML standard (&eacute;, &nbsp;, etc.)
    text = html.unescape(text)
    # Décoder les entités numériques (&#39;, &#160;, etc.)
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)
    
    return text


def clean_html(text: Optional[str], preserve_structure: bool = True) -> str:
    """
    Nettoie le HTML d'un texte en extrayant le contenu textuel.
    
    Parameters
    ----------
    text : str ou None
        Le texte contenant du HTML
    preserve_structure : bool, default=True
        Si True, conserve la structure (paragraphes -> sauts de ligne)
        
    Returns
    -------
    str
        Le texte nettoyé sans HTML
    """
    if not text or pd.isna(text):
        return ''
    
    text = str(text)
    
    # Si pas de HTML, retourner le texte tel quel
    if not has_html(text):
        return decode_html_entities(text)
    
    try:
        # Utiliser BeautifulSoup pour parser le HTML
        soup = BeautifulSoup(text, 'html.parser')
        
        if preserve_structure:
            # Remplacer les balises de structure par des sauts de ligne
            for tag in soup.find_all(['p', 'br', 'div', 'li']):
                tag.append('\n')
            
            # Extraire le texte en conservant les sauts de ligne
            cleaned_text = soup.get_text(separator='\n', strip=True)
        else:
            # Extraire simplement le texte
            cleaned_text = soup.get_text(separator=' ', strip=True)
        
        # Nettoyer les espaces multiples et sauts de ligne multiples
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # Décoder les entités HTML restantes
        cleaned_text = decode_html_entities(cleaned_text)
        
        return cleaned_text
        
    except Exception as e:
        # En cas d'erreur, utiliser une méthode regex simple
        # Supprimer les balises HTML
        cleaned_text = re.sub(r'<[^>]+>', ' ', text)
        # Décoder les entités HTML
        cleaned_text = decode_html_entities(cleaned_text)
        # Nettoyer les espaces multiples
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

