"""Étape 5 - Simulation du trafic réel.

Des visiteurs virtuels appellent l'API ; leur réaction (clic / achat) dépend de
la VRAIE affinité (vérité terrain générée en étape 1), jamais vue par les modèles.
=> on obtient des événements comme en production, de façon reproductible.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def main(n_visits=6000, seed=7, url=None):
    """url=None : API en mémoire (TestClient). url="http://host:8000" : vraie API déployée (HTTP)."""
    if url is None:
        Path(os.environ.setdefault("EVENTS_PATH", "data/events.jsonl")).unlink(missing_ok=True)
        from src.service import app  # import après avoir fixé EVENTS_PATH

        client_cm = TestClient(app)
    else:
        import httpx

        client_cm = httpx.Client(base_url=url, timeout=10)

    rng = np.random.default_rng(seed)
    affinity = np.load("data/raw/ground_truth_affinity.npy")
    prices = pd.read_csv("data/raw/items.csv").price.values
    n_users = affinity.shape[0]
    with client_cm as client:
        for _ in range(n_visits):
            u = int(rng.integers(0, n_users))
            r = client.get(f"/recommend/{u}", params={"k": 5}).json()
            for pos, item in enumerate(r["items"]):
                p_click = 1 / (1 + np.exp(-(affinity[u, item] * 1.5 - 2.5))) / (1 + 0.15 * pos)
                if rng.random() < p_click:
                    base = {"user_id": u, "item_id": item, "variant": r["variant"]}
                    client.post("/feedback", json={**base, "action": "click"})
                    if rng.random() < 0.3:
                        client.post("/feedback", json={**base, "action": "purchase", "price": float(prices[item])})
    print("événements générés")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    # usage : python -m src.simulate [n_visites] [url_api]
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000, url=sys.argv[2] if len(sys.argv) > 2 else None)
