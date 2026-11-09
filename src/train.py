"""Étape 3 - Entraînement + évaluation OFFLINE + traçage MLflow.

Split "leave-last-out" : la dernière interaction de chaque utilisateur est
mise de côté (test). Métriques : Precision@K, Recall@K, NDCG@K.
Chaque modèle = 1 run MLflow (paramètres, métriques, artefact).
"""
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.models import MODELS

K = 10
MODELS_DIR = Path("models")


def split(inter):
    inter = inter.sort_values(["user_id", "timestamp"])
    last = inter.groupby("user_id").tail(1)
    train = inter.drop(last.index)
    return train, last


def evaluate(model, test, k=K):
    prec, rec, ndcg = [], [], []
    for u, i in zip(test.user_id, test.item_id):
        recs = model.recommend(u, k, exclude=model.seen(u))
        hit = i in recs
        prec.append(hit / k)
        rec.append(float(hit))
        ndcg.append(1 / np.log2(recs.index(i) + 2) if hit else 0.0)
    return {f"precision_at_{k}": np.mean(prec), f"recall_at_{k}": np.mean(rec), f"ndcg_at_{k}": np.mean(ndcg)}


def main():
    inter = pd.read_csv("data/raw/interactions.csv")
    items = pd.read_csv("data/raw/items.csv")
    train, test = split(inter)
    MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("p7-recommandation-offline")
    results = {}
    for name, cls in MODELS.items():
        with mlflow.start_run(run_name=name) as run:
            model = cls().fit(train, items)
            metrics = evaluate(model, test)
            mlflow.log_param("algorithm", name)
            mlflow.log_param("k", K)
            mlflow.log_param("n_train", len(train))
            mlflow.log_metrics(metrics)
            path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, path)
            mlflow.log_artifact(str(path))
            results[name] = {**metrics, "run_id": run.info.run_id}
            print(name, {k: round(v, 4) for k, v in metrics.items()})
    Path("reports").mkdir(exist_ok=True)
    Path("reports/offline_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
