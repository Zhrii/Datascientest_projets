"""
Sécurité de l'API : authentification, autorisation par rôle, limite de requêtes.

Authentification (HTTP Basic)
  Comptes lus dans API_USERS = "login:motdepasse:role,login2:motdepasse2:role2"
  (défaut : admin:rakuten:admin,demo:demo:user), comparés en temps constant.

Autorisation (rôles)
  user  : POST /predict
  admin : POST /predict, GET /model/info, POST /retrain
  Mauvais rôle -> 403.

Limite de requêtes (fenêtre glissante, par utilisateur)
  /predict : PREDICT_RATE_LIMIT (user, défaut 30/minute), PREDICT_RATE_LIMIT_ADMIN (défaut 600/minute)
  /retrain : RETRAIN_RATE_LIMIT (défaut 2/hour)
  Au-delà : 429 + Retry-After. Compteurs dans Redis (REDIS_URL) pour être
  partagés entre les répliques de l'API, en mémoire sinon (tests, mode local).

/health et /metrics restent publics (healthcheck Docker, scrape Prometheus).
"""

from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import Counter

from src.state import redis_client

AUTH_FAILURES = Counter("auth_failures_total", "Tentatives de connexion refusées")
FORBIDDEN = Counter("forbidden_total", "Requêtes refusées faute de droits (403)", ["endpoint"])
RATE_LIMITED = Counter("rate_limited_total", "Requêtes refusées (limite atteinte)", ["endpoint"])

# auto_error=False : on gère nous-mêmes l'absence d'identifiants pour la compter
_basic = HTTPBasic(realm="Rakuten API", auto_error=False)

PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
ROLES = ("user", "admin")


@dataclass(frozen=True)
class User:
    name: str
    role: str


def load_users(spec: Optional[str] = None) -> Dict[str, tuple[str, str]]:
    """"admin:rakuten:admin,demo:demo:user" -> {"admin": ("rakuten", "admin"), ...}."""
    spec = spec if spec is not None else os.getenv("API_USERS", "admin:rakuten:admin,demo:demo:user")
    users = {}
    for entry in filter(None, (e.strip() for e in spec.split(","))):
        name, password, role = entry.split(":")
        if role not in ROLES:
            raise ValueError(f"Rôle inconnu pour {name}: {role}")
        users[name] = (password, role)
    return users


def require_user(credentials: Optional[HTTPBasicCredentials] = Depends(_basic)) -> User:
    """Dépendance FastAPI : 401 si les identifiants manquent ou sont faux."""
    if credentials is not None:
        password, role = load_users().get(credentials.username, ("", ""))
        # comparaison faite même pour un login inconnu : même durée de réponse
        if secrets.compare_digest(credentials.password.encode(), password.encode()) and role:
            return User(credentials.username, role)
    AUTH_FAILURES.inc()
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiant ou mot de passe incorrect",
        headers={"WWW-Authenticate": 'Basic realm="Rakuten API"'},
    )


def require_role(role: str, endpoint: str) -> Callable[[User], User]:
    """Dépendance FastAPI : 403 si l'utilisateur n'a pas le rôle demandé."""

    def _check(user: User = Depends(require_user)) -> User:
        if role == "admin" and user.role != "admin":
            FORBIDDEN.labels(endpoint=endpoint).inc()
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Rôle '{role}' requis pour {endpoint}")
        return user

    return _check


def parse_limit(spec: str) -> tuple[int, int]:
    """"30/minute" -> (30, 60)."""
    count, period = spec.strip().split("/")
    return int(count), PERIODS[period.strip().rstrip("s")]


class RateLimiter:
    """Au plus N appels par utilisateur sur les `period` dernières secondes.

    `limits` donne la limite par rôle ({"user": "30/minute", "admin": "600/minute"}).
    """

    def __init__(self, endpoint: str, limits: Dict[str, str]):
        self.endpoint = endpoint
        self.limits = {role: parse_limit(spec) for role, spec in limits.items()}
        self._calls: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _count_memory(self, key: str, period: int, now: float) -> tuple[int, float]:
        with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] >= period:
                calls.popleft()
            calls.append(now)
            return len(calls), calls[0]

    def _count_redis(self, client, key: str, period: int, now: float) -> tuple[int, float]:
        # sorted set : un membre par appel, score = date ; on purge ce qui sort de la fenêtre
        rkey = f"rakuten:ratelimit:{self.endpoint}:{key}"
        pipe = client.pipeline()
        pipe.zremrangebyscore(rkey, 0, now - period)
        pipe.zadd(rkey, {uuid.uuid4().hex: now})
        pipe.zcard(rkey)
        pipe.zrange(rkey, 0, 0, withscores=True)
        pipe.expire(rkey, period)
        _, _, count, oldest, _ = pipe.execute()
        return count, oldest[0][1] if oldest else now

    def hit(self, user: User) -> None:
        max_calls, period = self.limits.get(user.role, self.limits["user"])
        client = redis_client()
        if client is not None:
            count, oldest = self._count_redis(client, user.name, period, time.time())
            now = time.time()
        else:
            now = time.monotonic()
            count, oldest = self._count_memory(user.name, period, now)
        if count > max_calls:
            RATE_LIMITED.labels(endpoint=self.endpoint).inc()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Limite atteinte : {max_calls} requêtes / {period}s sur {self.endpoint}",
                headers={"Retry-After": str(int(period - (now - oldest)) + 1)},
            )

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

    def dependency(self, role: str = "user") -> Callable[[User], User]:
        """Dépendance FastAPI : authentifie, vérifie le rôle, puis compte la requête."""
        check_role = require_role(role, self.endpoint)

        def _check(user: User = Depends(check_role)) -> User:
            self.hit(user)
            return user

        return _check


predict_limiter = RateLimiter("/predict", {
    "user": os.getenv("PREDICT_RATE_LIMIT", "30/minute"),
    "admin": os.getenv("PREDICT_RATE_LIMIT_ADMIN", "600/minute"),
})
retrain_limiter = RateLimiter("/retrain", {
    "user": os.getenv("RETRAIN_RATE_LIMIT", "2/hour"),
    "admin": os.getenv("RETRAIN_RATE_LIMIT", "2/hour"),
})

# compteurs exposés à 0 dès le démarrage : sinon Prometheus découvre la série
# au premier refus et increase() ignore les premières valeurs (alertes)
for _endpoint in ("/predict", "/retrain", "/model/info"):
    RATE_LIMITED.labels(endpoint=_endpoint)
    FORBIDDEN.labels(endpoint=_endpoint)
