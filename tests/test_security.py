"""Tests unitaires de la limite de requêtes (api/security.py)."""

import pytest
from fastapi import HTTPException

from api.security import RateLimiter, parse_limit


@pytest.mark.parametrize("spec,expected", [
    ("30/minute", (30, 60)),
    ("2/hour", (2, 3600)),
    ("10/minutes", (10, 60)),
    ("5/second", (5, 1)),
])
def test_parse_limit(spec, expected):
    assert parse_limit(spec) == expected


def test_limit_is_per_user():
    limiter = RateLimiter("/test", "2/minute")
    limiter.hit("alice")
    limiter.hit("alice")
    with pytest.raises(HTTPException) as exc:
        limiter.hit("alice")
    assert exc.value.status_code == 429
    limiter.hit("bob")  # un autre utilisateur a son propre quota


def test_window_slides(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("api.security.time.monotonic", lambda: now[0])
    limiter = RateLimiter("/test", "1/minute")
    limiter.hit("alice")
    with pytest.raises(HTTPException):
        limiter.hit("alice")
    now[0] += 61  # une minute plus tard, l'ancien appel sort de la fenêtre
    limiter.hit("alice")
