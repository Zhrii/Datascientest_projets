"""Tests du F1 en production (src/monitoring/performance.py) et du stockage des retours (src/state.py)."""

import pytest

from src.monitoring.performance import compute_performance


def fb(predicted, true, low=False):
    return {"predicted": predicted, "true": true, "modality": "text", "low_confidence": low}


def test_waits_for_enough_feedback():
    result = compute_performance([fb(10, 10)] * 5, min_samples=100)
    assert result["status"] == "waiting" and result["n_feedback"] == 5


def test_perfect_predictions():
    result = compute_performance([fb(10, 10), fb(40, 40)] * 60, min_samples=100)
    assert result["f1_macro"] == 1.0 and result["accuracy"] == 1.0


def test_accuracy_by_confidence():
    feedback = [fb(10, 10)] * 80 + [fb(10, 40, low=True)] * 15 + [fb(40, 40, low=True)] * 5
    result = compute_performance(feedback, min_samples=100)
    assert result["accuracy"] == pytest.approx(0.85)
    assert result["by_confidence"]["high"]["accuracy"] == 1.0
    assert result["by_confidence"]["low"]["accuracy"] == pytest.approx(0.25)
    assert result["low_confidence_share"] == pytest.approx(0.2)


def test_feedback_is_unique_through_redis(monkeypatch):
    """Deux répliques reçoivent le même retour : un seul est compté (HSETNX)."""
    fakeredis = pytest.importorskip("fakeredis")
    from src import state

    server = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(state, "redis_client", lambda: server)
    pid = state.log_prediction("chaise", {"prdtypecode": 1560, "modality": "text", "confidence": 0.8})
    entry = state.record_feedback(pid, 2060)
    assert entry["predicted"] == 1560 and entry["true"] == 2060
    with pytest.raises(state.AlreadyLabeled):
        state.record_feedback(pid, 2060)
    with pytest.raises(state.UnknownPrediction):
        state.record_feedback("inconnu", 10)
    assert len(state.recent_feedback()) == 1
