# Rapport technique – P7 : Recommandation de produits avec test A/B

> Version candidate (Markdown). À exporter en PDF : `DEVOPS2026_P7_<Groupe>_rapport.pdf`. Compléter les `[À COMPLÉTER]` avec les chiffres de votre dernière exécution.

## 1. Contexte et objectifs
Une plateforme e-commerce veut comparer plusieurs moteurs de recommandation **en conditions réelles** plutôt que sur un score offline. Périmètre : 4 algorithmes, déploiement simultané avec répartition du trafic, métriques métier, promotion automatique du gagnant. Contraintes assumées : données et trafic **synthétiques** (reproductibilité, absence de données personnelles).

## 2. Architecture de la solution
Flux : `data → train (MLflow) → models → API FastAPI (A/B) → événements → analyse → promotion (Registry)`.
Composants : Git/GitHub (code, PR), DVC (données/artefacts/pipeline), MLflow (expériences, registre), CML (rapport dans les PR), Docker/Compose (service + Prometheus), GitHub Actions (CI/CD).
[Insérer le schéma : voir README, section Architecture.]

## 3. Pipeline et reproductibilité
- `dvc.yaml` : `prepare → train → simulate → analyze` ; données et modèles en cache DVC avec remote.
- Une commande : `dvc repro` (ou `python run_pipeline.py --promote`).
- Seeds fixes, dépendances listées (`requirements.txt`), aucune donnée dans Git ni dans l'image Docker.

## 4. Expérimentation
Modèles : popularité (contrôle), filtrage collaboratif item-item, contenu, hybride (0,6 CF + 0,4 contenu).
Évaluation offline (leave-last-out, @10) puis online (A/B). Tableau comparatif : [À COMPLÉTER depuis `reports/ab_report.md` et MLflow UI].
Test statistique : z-test de deux proportions sur le CTR, correction de Bonferroni (α/3).

## 5. Automatisation et déploiement
- CI : lint (ruff), 13 tests (unitaires, API, validation de modèle), pipeline complet.
- CML : rapport comparatif commenté dans chaque PR.
- Promotion : sur `main`, le gagnant reçoit l'alias `champion` dans le Model Registry MLflow et devient le contrôle suivant. **Retour arrière** : repointer l'alias vers une version précédente.
- Déploiement : `docker compose up` (API + Prometheus).

## 6. Supervision
Endpoint `/metrics` (Prometheus) : `p7_impressions_total{variant}`, `p7_feedback_total{variant,action}`, `p7_recommend_latency_seconds`. Sonde de santé `/health`. Seuils proposés : latence p95 < 100 ms ; alerte si le CTR d'une variante chute de plus de 30 % par rapport à sa moyenne. Dérive : non implémentée (limite).

## 7. Bilan
Choix : hash déterministe pour l'affectation (pas de contamination), z-test + Bonferroni (rigueur statistique), DVC + MLflow (traçabilité).
Difficultés : [À COMPLÉTER, ex. espace disque, installation MLflow, intégration DVC/Git].
Limites : trafic simulé, test à horizon fixe, critère CTR, attributs d'articles pauvres (modèle contenu faible), pas de détection de dérive.
Pistes : test séquentiel, bandits multi-bras, métriques de revenu comme critère, Grafana, données réelles (MovieLens).
