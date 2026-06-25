"""API behaves: validates input, predicts, honours the penalty hint."""

import pytest
from fastapi.testclient import TestClient

from xg.models.baseline import MODEL_PATH
from xg.scenarios import CLEAR_CHANCE, PENALTY

# The API loads the model at startup; skip the whole module if it isn't trained.
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="no trained model (run python -m xg.models.baseline)"
)


@pytest.fixture(scope="module")
def client():
    from xg.serve.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_clear_chance(client):
    body = CLEAR_CHANCE.state.model_dump()
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    xg = r.json()["xg"]
    assert CLEAR_CHANCE.xg_low <= xg <= CLEAR_CHANCE.xg_high


def test_predict_penalty_uses_constant(client):
    body = PENALTY.state.model_dump()
    r = client.post("/predict?shot_type=penalty", json=body)
    assert r.json()["xg"] == pytest.approx(0.76)


def test_off_pitch_request_rejected(client):
    # x=200 is off the pitch -> schema validation -> 422, never reaches the model.
    r = client.post("/predict", json={"shot_xy": [200.0, 40.0], "players": []})
    assert r.status_code == 422
