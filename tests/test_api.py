import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src.data import generate
from src.models import MODELS
from src.train import split


@pytest.fixture()
def client(tmp_path, monkeypatch):
    inter, items, _ = generate(n_users=80, n_items=60, seed=2)
    train, _ = split(inter)
    for name, cls in MODELS.items():
        joblib.dump(cls().fit(train, items), tmp_path / f"{name}.joblib")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"salt": "x", "control": "popularity",
                               "variants": {"popularity": 0.5, "hybrid": 0.5}}))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("AB_CONFIG", str(cfg))
    from src import service

    monkeypatch.setattr(service, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(service, "EVENTS", tmp_path / "events.jsonl")
    with TestClient(service.app) as c:
        yield c, tmp_path


def test_health(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"


def test_recommend_is_sticky_and_logged(client):
    c, tmp = client
    r1, r2 = c.get("/recommend/5").json(), c.get("/recommend/5").json()
    assert r1["variant"] == r2["variant"] and len(r1["items"]) == 10
    assert len((tmp / "events.jsonl").read_text().splitlines()) == 2


def test_invalid_inputs_rejected(client):
    c, _ = client
    assert c.get("/recommend/1", params={"k": 0}).status_code == 422
    bad = {"user_id": 1, "item_id": 1, "variant": "hybrid", "action": "like"}
    assert c.post("/feedback", json=bad).status_code == 422


def test_metrics_endpoint(client):
    c, _ = client
    c.get("/recommend/3")
    assert "p7_impressions_total" in c.get("/metrics").text
