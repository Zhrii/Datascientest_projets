"""Tests de la détection de dérive (src/monitoring/drift.py, Evidently)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from src.monitoring.drift import build_current, compute_drift, text_features  # noqa: E402


def sample(n, length, words, categories, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "text_length": rng.normal(length, length / 10, n),
        "word_count": rng.normal(words, words / 10, n),
        "digit_ratio": rng.uniform(0, 0.05, n),
        "category": rng.choice(categories, n),
    })


def test_text_features():
    feats = text_features(pd.Series(["chaise en bois 2024", ""]))
    assert feats.loc[0, "word_count"] == 4
    assert feats.loc[0, "digit_ratio"] == pytest.approx(4 / 19)
    assert feats.loc[1, "text_length"] == 0


def test_build_current_ignores_image_only_predictions():
    current = build_current([
        {"text": "Chaise <b>scandinave</b>", "prdtypecode": 1560},
        {"text": "", "prdtypecode": 2403},  # image seule : pas de texte à comparer
    ])
    assert len(current) == 1 and current.loc[0, "category"] == "1560"


def test_same_distribution_has_no_drift():
    reference = sample(500, 300, 50, ["10", "40", "1560"], seed=1)
    current = sample(200, 300, 50, ["10", "40", "1560"], seed=2)
    result = compute_drift(reference, current)
    assert not result.drift_detected


def test_different_distribution_is_detected(tmp_path):
    reference = sample(500, 300, 50, ["10", "40", "1560"], seed=1)
    current = sample(200, 60, 8, ["2905"], seed=2)  # textes courts, une seule catégorie
    report = tmp_path / "drift.html"
    result = compute_drift(reference, current, html_path=report)
    assert result.drift_detected and result.drift_share >= 0.5
    assert result.columns["text_length"]["drifted"] and result.columns["category"]["drifted"]
    assert report.exists()
