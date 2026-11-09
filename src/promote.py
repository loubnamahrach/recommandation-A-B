"""Étape 7 - Promotion automatique du gagnant dans le Model Registry MLflow.

- Crée/maj le modèle enregistré "p7-recommender" ; nouvelle version = run du gagnant.
- Alias "champion" déplacé vers cette version (rollback = repointer l'alias).
- Écrit models/champion.json lu par le déploiement.
"""
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

NAME = "p7-recommender"


def main():
    result = json.loads(Path("reports/ab_result.json").read_text())
    winner = result["winner"]
    offline = json.loads(Path("reports/offline_metrics.json").read_text())
    run_id = offline[winner]["run_id"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    try:
        client.create_registered_model(NAME)
    except mlflow.exceptions.MlflowException:
        pass  # existe déjà
    mv = client.create_model_version(NAME, source=f"runs:/{run_id}/{winner}.joblib", run_id=run_id,
                                     tags={"ab_winner": winner, "control": result["control"]})
    client.set_registered_model_alias(NAME, "champion", mv.version)
    Path("models").mkdir(exist_ok=True)
    Path("models/champion.json").write_text(json.dumps({"model": winner, "version": mv.version, "run_id": run_id}))
    # Le gagnant devient la nouvelle référence (contrôle) du prochain test A/B.
    cfg_path = Path("config/ab_config.json")
    cfg = json.loads(cfg_path.read_text())
    cfg["control"] = winner
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Promu: {winner} -> {NAME} v{mv.version} (alias champion)")


if __name__ == "__main__":
    main()
