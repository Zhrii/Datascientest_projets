"""Tests unitaires de la sécurité (api/security.py) : comptes, rôles, limite de requêtes."""

import pytest
from fastapi import HTTPException

from api.security import RateLimiter, User, load_users, parse_limit

ALICE = User("alice", "user")
BOB = User("bob", "user")
ADMIN = User("root", "admin")


@pytest.mark.parametrize("spec,expected", [
    ("30/minute", (30, 60)),
    ("2/hour", (2, 3600)),
    ("10/minutes", (10, 60)),
    ("5/second", (5, 1)),
])
def test_parse_limit(spec, expected):
    assert parse_limit(spec) == expected


def test_load_users():
    users = load_users("admin:pw1:admin, demo:pw2:user")
    assert users == {"admin": ("pw1", "admin"), "demo": ("pw2", "user")}


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        load_users("x:y:superuser")


def test_limit_is_per_user():
    limiter = RateLimiter("/test", {"user": "2/minute"})
    limiter.hit(ALICE)
    limiter.hit(ALICE)
    with pytest.raises(HTTPException) as exc:
        limiter.hit(ALICE)
    assert exc.value.status_code == 429
    limiter.hit(BOB)  # un autre utilisateur a son propre quota


def test_limit_depends_on_role():
    limiter = RateLimiter("/test", {"user": "1/minute", "admin": "3/minute"})
    for _ in range(3):
        limiter.hit(ADMIN)
    with pytest.raises(HTTPException):
        limiter.hit(ADMIN)


def test_window_slides(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("api.security.time.monotonic", lambda: now[0])
    limiter = RateLimiter("/test", {"user": "1/minute"})
    limiter.hit(ALICE)
    with pytest.raises(HTTPException):
        limiter.hit(ALICE)
    now[0] += 61  # une minute plus tard, les anciens appels sortent de la fenêtre
    limiter.hit(ALICE)


def test_limit_is_shared_through_redis(monkeypatch):
    """Deux répliques (= deux limiteurs) partagent le même compteur via Redis."""
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("api.security.redis_client", lambda: server)
    replica_1 = RateLimiter("/test", {"user": "3/minute"})
    replica_2 = RateLimiter("/test", {"user": "3/minute"})
    replica_1.hit(ALICE)
    replica_2.hit(ALICE)
    replica_1.hit(ALICE)
    with pytest.raises(HTTPException) as exc:
        replica_2.hit(ALICE)
    assert exc.value.status_code == 429
