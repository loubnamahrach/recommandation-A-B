"""Étape 4b - API FastAPI : sert les recommandations selon la variante A/B.

Endpoints : /health, /recommend/{user_id}, /feedback (clic / achat).
Chaque événement est journalisé dans data/events.jsonl (base de l'analyse).
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from src.ab import assign_variant, load_config

EVENTS = Path(os.getenv("EVENTS_PATH", "data/events.jsonl"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

@asynccontextmanager
async def lifespan(_app):
    _load()
    yield


app = FastAPI(title="P7 - Recommandation A/B", lifespan=lifespan)
_state = {"models": {}, "config": None}


def _load():
    _state["config"] = load_config(os.getenv("AB_CONFIG", "config/ab_config.json"))
    for v in _state["config"]["variants"]:
        _state["models"][v] = joblib.load(MODELS_DIR / f"{v}.joblib")


def log_event(event):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as f:
        f.write(json.dumps({**event, "ts": time.time()}) + "\n")


class Feedback(BaseModel):
    user_id: int
    item_id: int
    variant: str
    action: str  # "click" ou "purchase"
    price: float = 0.0


IMPRESSIONS = Counter("p7_impressions_total", "Recommandations servies", ["variant"])
ACTIONS = Counter("p7_feedback_total", "Clics et achats", ["variant", "action"])
LATENCY = Histogram("p7_recommend_latency_seconds", "Latence de /recommend")


@app.get("/metrics")
def metrics():
    """Supervision : format Prometheus (impressions, clics/achats par variante, latence)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "variants": list(_state["models"])}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if k < 1 or k > 50:
        raise HTTPException(422, "k doit être entre 1 et 50")
    start = time.perf_counter()
    variant = assign_variant(user_id, _state["config"])
    model = _state["models"][variant]
    items = model.recommend(user_id, k, exclude=model.seen(user_id))
    LATENCY.observe(time.perf_counter() - start)
    IMPRESSIONS.labels(variant).inc()
    log_event({"type": "impression", "user_id": user_id, "variant": variant, "items": items})
    return {"user_id": user_id, "variant": variant, "items": items}


@app.post("/feedback")
def feedback(fb: Feedback):
    if fb.action not in ("click", "purchase"):
        raise HTTPException(422, "action invalide")
    ACTIONS.labels(fb.variant, fb.action).inc()
    log_event({"type": fb.action, **fb.model_dump()})
    return {"ok": True}
