"""Référentiel des 27 catégories Rakuten (prdtypecode)."""

DICT_CATEGORIES = {
    10: "Livre d'occasion",
    40: "Jeux vidéo",
    50: "Accessoire jeu vidéo",
    60: "Console de jeux",
    1140: "Figurine / Goodies",
    1160: "Cartes à collectionner",
    1180: "Jeu de plateau / Warhammer",
    1280: "Jouet enfant",
    1281: "Jeu de société",
    1300: "Jouet technique / Modélisme",
    1301: "Vêtement enfant / Puériculture",
    1302: "Jeu d'extérieur",
    1320: "Puériculture",
    1560: "Mobilier",
    1920: "Chambre / Literie",
    1940: "Alimentation / Épicerie",
    2060: "Décoration intérieure",
    2220: "Animalerie",
    2280: "Magazine",
    2403: "Livre / BD",
    2462: "Jeu vidéo (Occasion)",
    2522: "Papeterie",
    2582: "Meuble d'extérieur",
    2583: "Piscine / Spa",
    2585: "Bricolage / Jardinage",
    2705: "Livre neuf",
    2905: "Jeu PC",
}


def category_name(code) -> str:
    """Traduit un prdtypecode en libellé français."""
    try:
        return DICT_CATEGORIES.get(int(code), f"Code {code}")
    except (TypeError, ValueError):
        return f"Code {code}"
