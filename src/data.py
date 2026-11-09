"""Étape 1 - Données : génération d'un jeu e-commerce synthétique (reproductible).

Pourquoi synthétique ? Les directives autorisent MovieLens/Amazon, mais un jeu
généré avec une *vérité terrain* (les goûts réels des utilisateurs) nous permet
ensuite de SIMULER le comportement des visiteurs pendant le test A/B.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
CATEGORIES = ["tech", "mode", "maison", "sport", "beaute", "livres"]


def generate(n_users=600, n_items=300, dim=8, seed=42):
    rng = np.random.default_rng(seed)
    # facteurs latents = goûts réels (cachés des modèles)
    U = rng.normal(size=(n_users, dim))
    V = rng.normal(size=(n_items, dim))
    item_cat = rng.integers(0, len(CATEGORIES), n_items)
    # la catégorie influence aussi le facteur latent -> le contenu est informatif
    cat_dir = rng.normal(size=(len(CATEGORIES), dim))
    V = V * 0.6 + cat_dir[item_cat]
    prices = np.round(rng.lognormal(3.5, 0.6, n_items), 2)
    popularity_bias = rng.normal(0, 0.8, n_items)

    affinity = U @ V.T / np.sqrt(dim) + popularity_bias  # score réel
    prob = 1 / (1 + np.exp(-(affinity - 1.5)))  # proba d'interaction

    rows = []
    for u in range(n_users):
        n = rng.binomial(n_items, prob[u].mean()) + 5
        items = rng.choice(n_items, size=min(n, n_items), replace=False, p=prob[u] / prob[u].sum())
        for rank, i in enumerate(items):
            rows.append((u, int(i), int(rng.integers(0, 10_000_000)) + rank))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])

    items_df = pd.DataFrame(
        {"item_id": np.arange(n_items), "category": [CATEGORIES[c] for c in item_cat], "price": prices}
    )
    return inter, items_df, affinity


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    inter, items, affinity = generate()
    inter.to_csv(RAW / "interactions.csv", index=False)
    items.to_csv(RAW / "items.csv", index=False)
    np.save(RAW / "ground_truth_affinity.npy", affinity)
    print(json.dumps({"interactions": len(inter), "users": inter.user_id.nunique(), "items": len(items)}))


if __name__ == "__main__":
    main()
