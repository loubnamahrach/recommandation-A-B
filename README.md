# P7 – Recommandation de produits avec test A/B

Objectif : comparer **4 moteurs de recommandation en conditions (simulées) réelles** plutôt que sur un simple score offline, puis **promouvoir automatiquement le meilleur**.

## Architecture

```
data.py ─► train.py ─► models/*.joblib ─► service.py (FastAPI, répartition A/B) ─► events.jsonl
            │ MLflow (runs)                                  ▲                          │
            ▼                                          simulate.py                 analyze.py (CTR, z-test)
        offline_metrics.json                                                           │ rapport CML (PR)
                                                                                       ▼
                                                                 promote.py ─► MLflow Registry (alias champion)
```

| Fichier | Rôle | Concept à retenir |
|---|---|---|
| `src/data.py` | Jeu e-commerce synthétique + vérité terrain | Reproductibilité (seed fixe) |
| `src/models.py` | Popularité (contrôle), CF item-item, contenu, hybride | Interface commune `recommend()` |
| `src/train.py` | Split leave-last-out, Precision/Recall/NDCG@10, runs MLflow | Évaluation **offline** + traçage |
| `src/ab.py` | Affectation par hash(user_id) | Déterministe, pas de contamination |
| `src/service.py` | API `/recommend`, `/feedback`, `/health` | Journalise impressions/clics/achats |
| `src/simulate.py` | Faux visiteurs réagissant selon la vraie affinité | Remplace le trafic réel |
| `src/analyze.py` | CTR, achats, revenu/visite, z-test + Bonferroni | Évaluation **online**, métriques métier |
| `src/promote.py` | Registry MLflow, alias `champion` | Rollback = repointer l'alias |
| `.github/workflows/ci.yml` | lint, tests, pipeline, rapport CML sur PR, promotion sur `main` | CI/CD |

## Relance (une commande)

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux : source .venv/bin/activate
pip install -r requirements.txt
dvc pull            # récupère données/modèles du remote DVC (optionnel)
dvc repro           # rejoue le pipeline : prepare -> train -> simulate -> analyze
python -m src.promote   # promotion du gagnant dans le registre MLflow
pytest -q
mlflow ui --backend-store-uri sqlite:///mlflow.db   # voir les runs et le registre
```
Alternative sans DVC : `python run_pipeline.py --promote`.
> Le venv doit être activé : `dvc repro` appelle `python` du PATH.

Le remote DVC est un dossier local `../dvc_storage_p7` (`dvc remote modify storage url <chemin>` pour le changer).

## Test A/B sur l'API réellement déployée
```bash
docker compose up --build            # API :8000 + Prometheus :9090 (modèles montés depuis ./models)
python -m src.simulate 3000 http://localhost:8000   # trafic HTTP réel
python -m src.analyze
```
Supervision : `http://localhost:8000/metrics` (Prometheus), sonde `/health`.

API : `uvicorn src.service:app` (après `python run_pipeline.py`), ou Docker :
`docker build -t p7 . && docker run -p 8000:8000 -v ./models:/app/models p7`

## Variables d'environnement (optionnelles)
`EVENTS_PATH`, `MODELS_DIR`, `AB_CONFIG`.

## Limites connues
- Données et trafic synthétiques ; MovieLens possible en remplaçant `data.py`.
- Test à horizon fixe (pas de test séquentiel) ; CTR comme critère de décision ; pas de détection de dérive.
