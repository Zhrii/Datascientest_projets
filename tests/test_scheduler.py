"""Tests des déclencheurs de réentraînement automatique (pipelines/scheduler.py)."""

from datetime import datetime, timedelta, timezone

from pipelines.scheduler import retrain_reason

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def metrics(days_old, n_train=8000, n_val=2000):
    return {"trained_at": (NOW - timedelta(days=days_old)).isoformat(), "n_train": n_train, "n_val": n_val}


def test_no_model_triggers_training():
    assert retrain_reason(None, 10_000, NOW) == "aucun modèle en production"


def test_recent_model_with_same_data_does_nothing():
    assert retrain_reason(metrics(days_old=3), 10_000, NOW) is None


def test_old_model_triggers_training():
    assert "jours" in retrain_reason(metrics(days_old=45), 10_000, NOW)


def test_ten_percent_new_data_triggers_training():
    assert retrain_reason(metrics(days_old=3), 10_900, NOW) is None
    assert "nouveaux produits" in retrain_reason(metrics(days_old=3), 11_000, NOW)
