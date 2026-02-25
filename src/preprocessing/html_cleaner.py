"""
Module de nettoyage HTML pour les descriptions de produits.
"""

import re
import html
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional

# Patterns pour supprimer les balises HTML (normales et malformées)
# 1. Balises reconnues : <tag>, </tag>, <tag /> avec espaces éventuels
# 2. Fallback large : tout ce qui ressemble à <...>
_HTML_TAG_STRICT = re.compile(
    r'<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?\s*/?\s*>',
    re.IGNORECASE
)
_HTML_TAG_FALLBACK = re.compile(r'<[^>]+>')


def _normalize_angle_brackets(text: str) -> str:
    """Remplace les caractères Unicode < > par leurs équivalents ASCII."""
    return (text
            .replace('\u3008', '<').replace('\u3009', '>')  # ＜ ＞
            .replace('\u2039', '<').replace('\u203a', '>')  # ‹ ›
            .replace('\u2329', '<').replace('\u232a', '>')  # ⟨ ⟩
            )


def remove_remaining_html(text: Optional[str]) -> str:
    """
    Supprime toutes les balises HTML restantes (passe de sécurité).
    Utile après BeautifulSoup pour gérer le HTML malformé : < li>, <br/>, <. 4pc br>, etc.
    """
    if not text or pd.isna(text):
        return ''
    return _remove_all_html_tags(str(text))


def _remove_all_html_tags(text: str, max_passes: int = 5) -> str:
    """
    Supprime toutes les balises HTML par passes successives.
    Gère les variantes malformées : < li>, <br/>, <. 4pc br>, etc.
    """
    for _ in range(max_passes):
        if not has_html(text):
            break
        # 1. Pattern strict (balises valides avec espaces)
        text = _HTML_TAG_STRICT.sub(' ', text)
        # 2. Fallback pour le reste (malformé)
        text = _HTML_TAG_FALLBACK.sub(' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = decode_html_entities(text)
    return text


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
    text = _normalize_angle_brackets(text)
    
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
        
        # Passerelle regex : supprimer tout HTML restant (malformé, espaces, etc.)
        cleaned_text = _remove_all_html_tags(cleaned_text)
        
        return cleaned_text
        
    except Exception:
        # En cas d'erreur BeautifulSoup, utiliser la suppression regex
        cleaned_text = decode_html_entities(text)
        cleaned_text = _remove_all_html_tags(cleaned_text)
        return cleaned_text

