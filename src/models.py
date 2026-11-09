"""Étape 2 - Les 4 moteurs de recommandation.

Interface commune : fit(train_df, items_df) puis recommend(user_id, k, exclude).
Les modèles sont de simples classes -> faciles à tester, sérialiser (joblib) et comparer.
"""
import numpy as np
import pandas as pd


class Recommender:
    name = "base"

    def _scores(self, user_id):  # -> vecteur de scores sur tous les items
        raise NotImplementedError

    def recommend(self, user_id, k=10, exclude=None):
        s = np.asarray(self._scores(user_id), dtype=float).copy()
        if exclude:
            s[list(exclude)] = -np.inf
        top = np.argsort(-s)[:k]
        return [int(i) for i in top if np.isfinite(s[i])]

    def seen(self, user_id):
        return set(self.user_items.get(user_id, []))

    def _build_matrix(self, train, n_users, n_items):
        M = np.zeros((n_users, n_items), dtype=np.float32)
        M[train.user_id.values, train.item_id.values] = 1.0
        self.M = M
        self.user_items = train.groupby("user_id").item_id.apply(list).to_dict()


class Popularity(Recommender):
    """Baseline / groupe contrôle : les articles les plus achetés."""
    name = "popularity"

    def fit(self, train, items):
        self._build_matrix(train, train.user_id.max() + 1, len(items))
        self.pop = self.M.sum(axis=0)
        return self

    def _scores(self, user_id):
        return self.pop


class ItemCF(Recommender):
    """Filtrage collaboratif item-item (similarité cosinus)."""
    name = "cf"

    def fit(self, train, items):
        self._build_matrix(train, train.user_id.max() + 1, len(items))
        norm = np.linalg.norm(self.M, axis=0) + 1e-9
        self.sim = (self.M.T @ self.M) / np.outer(norm, norm)
        np.fill_diagonal(self.sim, 0)
        return self

    def _scores(self, user_id):
        if user_id >= self.M.shape[0] or self.M[user_id].sum() == 0:
            return self.M.sum(axis=0)  # cold start -> popularité
        return self.M[user_id] @ self.sim


class ContentBased(Recommender):
    """Basé sur le contenu : profil utilisateur = moyenne des attributs des articles vus."""
    name = "content"

    def fit(self, train, items):
        self._build_matrix(train, train.user_id.max() + 1, len(items))
        cat = pd.get_dummies(items.category).values.astype(np.float32)
        price = np.log1p(items.price.values)[:, None]
        price = (price - price.mean()) / price.std()
        F = np.hstack([cat, price * 0.3])
        self.F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        return self

    def _scores(self, user_id):
        if user_id >= self.M.shape[0] or self.M[user_id].sum() == 0:
            return self.M.sum(axis=0)
        profile = self.M[user_id] @ self.F
        profile /= np.linalg.norm(profile) + 1e-9
        return self.F @ profile


class Hybrid(Recommender):
    """Hybride : combinaison pondérée (scores normalisés) de CF et contenu."""
    name = "hybrid"

    def __init__(self, w_cf=0.6):
        self.w_cf = w_cf

    def fit(self, train, items):
        self.cf = ItemCF().fit(train, items)
        self.cb = ContentBased().fit(train, items)
        self.M, self.user_items = self.cf.M, self.cf.user_items
        return self

    @staticmethod
    def _norm(s):
        s = np.asarray(s, dtype=float)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    def _scores(self, user_id):
        return self.w_cf * self._norm(self.cf._scores(user_id)) + (1 - self.w_cf) * self._norm(
            self.cb._scores(user_id)
        )


MODELS = {c.name: c for c in (Popularity, ItemCF, ContentBased, Hybrid)}
