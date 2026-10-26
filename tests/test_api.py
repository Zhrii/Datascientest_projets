"""Tests de l'API : routes publiques / protégées, rôles, validation, limite de requêtes.

Un faux prédicteur remplace les vrais modèles (pas de torch / CLIP à charger).
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.security import predict_limiter, retrain_limiter

AUTH = ("test", "secret")    # rôle admin
USER = ("alice", "pw")       # rôle user


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
        confidence = 0.05 if text == "ambigu" else 0.9
        return {"prdtypecode": 1560, "category": "Mobilier", "modality": modality,
                "model": "fake", "f1_macro": 0.78, "confidence": confidence, "low_confidence": confidence < 0.2}

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


def test_user_can_predict(client):
    r = client.post("/predict", data={"text": "chaise"}, auth=USER)
    assert r.status_code == 200
    assert r.headers["X-API-Instance"]


@pytest.mark.parametrize("method,path", [("POST", "/retrain"), ("GET", "/model/info")])
def test_user_cannot_use_admin_routes(client, method, path):
    r = client.request(method, path, auth=USER)
    assert r.status_code == 403


def test_predictions_are_logged_for_drift(client):
    from src import state

    before = len(state.recent_predictions())
    client.post("/predict", data={"text": "Coussin velours"}, auth=USER)
    logged = state.recent_predictions()
    assert len(logged) == before + 1
    assert logged[-1]["text"] == "Coussin velours" and logged[-1]["prdtypecode"] == 1560


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
    monkeypatch.setitem(predict_limiter.limits, "user", (3, 60))
    codes = [client.post("/predict", data={"text": "chaise"}, auth=USER).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = client.post("/predict", data={"text": "chaise"}, auth=USER)
    assert int(r.headers["Retry-After"]) > 0


def test_admin_has_a_higher_limit(client, monkeypatch):
    monkeypatch.setitem(predict_limiter.limits, "user", (3, 60))
    codes = {client.post("/predict", data={"text": "chaise"}, auth=AUTH).status_code for _ in range(5)}
    assert codes == {200}


def test_only_one_retrain_at_a_time(client, monkeypatch):
    from src import state

    monkeypatch.setattr(api_main, "_run_retrain", lambda: asyncio.sleep(0))
    assert state.acquire_retrain_lock("autre-replique")   # un retrain tourne déjà ailleurs
    try:
        assert client.post("/retrain", auth=AUTH).status_code == 409
    finally:
        state.release_retrain_lock()


# ---------------------------------------------------------------- confiance et retours
def test_prediction_has_id_and_confidence(client):
    body = client.post("/predict", data={"text": "chaise"}, auth=USER).json()
    assert len(body["prediction_id"]) == 32
    assert body["confidence"] == 0.9 and body["low_confidence"] is False
    body = client.post("/predict", data={"text": "ambigu"}, auth=USER).json()
    assert body["low_confidence"] is True


def predict_id(client):
    return client.post("/predict", data={"text": "chaise"}, auth=USER).json()["prediction_id"]


def test_feedback_is_recorded(client):
    from src import state

    pid = predict_id(client)
    r = client.post("/feedback", json={"prediction_id": pid, "prdtypecode": 1560}, auth=USER)
    assert r.status_code == 200 and r.json()["correct"] is True
    r = client.post("/feedback", json={"prediction_id": predict_id(client), "prdtypecode": 2060}, auth=USER)
    assert r.json()["correct"] is False and r.json()["predicted"] == 1560 and r.json()["true"] == 2060
    assert state.recent_feedback()[-1]["true"] == 2060


def test_feedback_twice_is_409(client):
    pid = predict_id(client)
    assert client.post("/feedback", json={"prediction_id": pid, "prdtypecode": 1560}, auth=USER).status_code == 200
    assert client.post("/feedback", json={"prediction_id": pid, "prdtypecode": 10}, auth=USER).status_code == 409


def test_feedback_unknown_prediction_is_404(client):
    r = client.post("/feedback", json={"prediction_id": "0" * 32, "prdtypecode": 1560}, auth=USER)
    assert r.status_code == 404


def test_feedback_unknown_category_is_400(client):
    r = client.post("/feedback", json={"prediction_id": predict_id(client), "prdtypecode": 9999}, auth=USER)
    assert r.status_code == 400


def test_feedback_requires_auth(client):
    assert client.post("/feedback", json={"prediction_id": "x", "prdtypecode": 10}).status_code == 401
