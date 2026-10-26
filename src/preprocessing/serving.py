"""
Nettoyage d'un texte libre reçu en production (API, détection de dérive).

À l'entraînement, PreprocessingPipeline nettoie le HTML de la description puis
normalise designation + description. En production, l'API reçoit un seul champ
"text" (souvent designation + description collées, avec du HTML) : on lui applique
le même nettoyage HTML avant la normalisation, sinon le modèle voit des balises et
des entités (&#233;...) qu'il n'a jamais vues à l'entraînement.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .html_cleaner import clean_html
from .pipeline import PreprocessingPipeline


def clean_product_texts(texts: List[str]) -> List[str]:
    """Nettoie une liste de textes produit comme à l'entraînement (HTML puis normalisation)."""
    if not texts:
        return []
    df = pd.DataFrame({
        "designation": [clean_html(t, preserve_structure=True) for t in texts],
        "description": [""] * len(texts),
    })
    pipe = PreprocessingPipeline(combine_text=True, create_features=False)
    return pipe.fit_transform(df)["text_combined"].fillna("").tolist()


def clean_product_text(text: str) -> str:
    return clean_product_texts([text])[0]
