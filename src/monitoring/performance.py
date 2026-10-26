"""
Performance du modèle en production, mesurée sur les retours des modérateurs.

Chaque retour = (catégorie prédite par l'API, vraie catégorie donnée par un
modérateur) via POST /feedback. Sur les derniers retours, on calcule :
  - le F1-macro et l'accuracy "en production" (à comparer au F1 d'entraînement)
  - l'accuracy des prédictions sûres et des prédictions peu sûres (low_confidence) :
    vérifie que le signal de confiance du modèle veut dire quelque chose
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, f1_score


def compute_performance(feedback: List[Dict[str, Any]], min_samples: int = 100) -> Dict[str, Any]:
    n = len(feedback)
    if n < min_samples:
        return {"status": "waiting", "n_feedback": n, "min_samples": min_samples,
                "message": f"{n} retours de modérateurs, il en faut {min_samples} pour mesurer le F1"}

    y_pred = [f["predicted"] for f in feedback]
    y_true = [f["true"] for f in feedback]
    result = {
        "status": "ok",
        "n_feedback": n,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "by_confidence": {},
    }
    for label, low in (("low", True), ("high", False)):
        subset = [f for f in feedback if bool(f.get("low_confidence")) == low]
        if subset:
            result["by_confidence"][label] = {
                "n": len(subset),
                "accuracy": float(accuracy_score([f["true"] for f in subset], [f["predicted"] for f in subset])),
            }
    result["low_confidence_share"] = result["by_confidence"].get("low", {}).get("n", 0) / n
    return result
