"""
Sécurité de l'API : login / mot de passe (HTTP Basic) et limite de requêtes.

  - Identifiants lus dans les variables d'environnement API_USERNAME / API_PASSWORD
    (docker-compose.yml), comparés en temps constant (secrets.compare_digest).
  - Limite de requêtes par utilisateur sur une fenêtre glissante :
      PREDICT_RATE_LIMIT  (défaut "30/minute")
      RETRAIN_RATE_LIMIT  (défaut "2/hour")
    Au-delà : HTTP 429 + en-tête Retry-After.

/health et /metrics restent publics (healthcheck Docker, scrape Prometheus).
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import Counter

AUTH_FAILURES = Counter("auth_failures_total", "Tentatives de connexion refusées")
RATE_LIMITED = Counter("rate_limited_total", "Requêtes refusées (limite atteinte)", ["endpoint"])

# auto_error=False : on gère nous-mêmes l'absence d'identifiants pour la compter
_basic = HTTPBasic(realm="Rakuten API", auto_error=False)

PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def _credentials() -> tuple[str, str]:
    return os.getenv("API_USERNAME", "admin"), os.getenv("API_PASSWORD", "rakuten")


def require_user(credentials: Optional[HTTPBasicCredentials] = Depends(_basic)) -> str:
    """Dépendance FastAPI : 401 si les identifiants manquent ou sont faux."""
    username, password = _credentials()
    valid = credentials is not None and (
        secrets.compare_digest(credentials.username.encode(), username.encode())
        & secrets.compare_digest(credentials.password.encode(), password.encode())
    )
    if not valid:
        AUTH_FAILURES.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiant ou mot de passe incorrect",
            headers={"WWW-Authenticate": 'Basic realm="Rakuten API"'},
        )
    return credentials.username


def parse_limit(spec: str) -> tuple[int, int]:
    """"30/minute" -> (30, 60)."""
    count, period = spec.strip().split("/")
    return int(count), PERIODS[period.strip().rstrip("s")]


class RateLimiter:
    """Au plus `max_calls` appels par utilisateur sur les `period` dernières secondes."""

    def __init__(self, endpoint: str, spec: str):
        self.endpoint = endpoint
        self.max_calls, self.period = parse_limit(spec)
        self._calls: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] >= self.period:
                calls.popleft()
            if len(calls) >= self.max_calls:
                RATE_LIMITED.labels(endpoint=self.endpoint).inc()
                retry_after = int(self.period - (now - calls[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Limite atteinte : {self.max_calls} requêtes / {self.period}s sur {self.endpoint}",
                    headers={"Retry-After": str(retry_after)},
                )
            calls.append(now)

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

    def dependency(self) -> Callable[[str], str]:
        """Dépendance FastAPI : authentifie puis compte la requête."""

        def _check(user: str = Depends(require_user)) -> str:
            self.hit(user)
            return user

        return _check


predict_limiter = RateLimiter("/predict", os.getenv("PREDICT_RATE_LIMIT", "30/minute"))
retrain_limiter = RateLimiter("/retrain", os.getenv("RETRAIN_RATE_LIMIT", "2/hour"))

# compteurs exposés à 0 dès le démarrage : sinon Prometheus découvre la série
# au premier refus et increase() ignore les premières valeurs (alerte RateLimitHit)
for _limiter in (predict_limiter, retrain_limiter):
    RATE_LIMITED.labels(endpoint=_limiter.endpoint)
