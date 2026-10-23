"""Tests de l'API : routes publiques / protégées, validation, limite de requêtes.

Un faux prédicteur remplace les vrais modèles (pas de torch / CLIP à charger).
"""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.security import predict_limiter, retrain_limiter

AUTH = ("test", "secret")


class FakePredictor:
    text_ready = image_ready = fusion_ready = True

    def health(self):
        return True, {"status": "ok", "text_ready": True, "image_ready": True,
                      "fusion_ready": True, "errors": {}}

    def info(self):
        return {"pipelines": {"text": {"ready": True, "f1_macro": 0.78}}}

    def predict(self, text=None, image=None):
        if not text and image is None:
            raise ValueError("Fournir au moins un texte ou une image")
        modality = "fusion" if text and image else ("text" if text else "image")
        return {"prdtypecode": 1560, "category": "Mobilier", "modality": modality,
                "model": "fake", "f1_macro": 0.78}

    def reload(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    predict_limiter.reset()
    retrain_limiter.reset()
    with TestClient(api_main.app) as c:
        yield c


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_is_public(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predict_requests_total" in r.text


@pytest.mark.parametrize("method,path", [("POST", "/predict"), ("GET", "/model/info"), ("POST", "/retrain")])
def test_protected_routes_require_auth(client, method, path):
    r = client.request(method, path)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"].startswith("Basic")


def test_wrong_password_is_rejected(client):
    r = client.post("/predict", data={"text": "chaise"}, auth=("test", "mauvais"))
    assert r.status_code == 401


def test_auth_failures_are_counted(client):
    from api.security import AUTH_FAILURES

    before = AUTH_FAILURES._value.get()
    client.post("/predict", data={"text": "chaise"})                       # sans identifiants
    client.post("/predict", data={"text": "chaise"}, auth=("x", "y"))      # mauvais identifiants
    assert AUTH_FAILURES._value.get() == before + 2


def test_predict_text(client):
    r = client.post("/predict", data={"text": "Chaise scandinave en bois"}, auth=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["prdtypecode"] == 1560
    assert body["modality"] == "text"
    assert body["latency_ms"] >= 0


def test_predict_text_and_image_uses_fusion(client):
    r = client.post("/predict", data={"text": "Chaise"}, files={"image": ("a.jpg", b"fake", "image/jpeg")}, auth=AUTH)
    assert r.status_code == 200
    assert r.json()["modality"] == "fusion"


def test_predict_without_input_is_400(client):
    r = client.post("/predict", data={"text": "   "}, auth=AUTH)
    assert r.status_code == 400


def test_model_info_with_auth(client):
    r = client.get("/model/info", auth=AUTH)
    assert r.status_code == 200
    assert "pipelines" in r.json()


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(predict_limiter, "max_calls", 3)
    codes = [client.post("/predict", data={"text": "chaise"}, auth=AUTH).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = client.post("/predict", data={"text": "chaise"}, auth=AUTH)
    assert int(r.headers["Retry-After"]) > 0
