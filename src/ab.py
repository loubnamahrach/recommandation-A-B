"""Étape 4a - Logique du test A/B : affectation déterministe des utilisateurs.

Hash(salt + user_id) -> nombre dans [0,1[ -> variante selon les parts de trafic.
Un même utilisateur voit TOUJOURS la même variante (pas de contamination).
"""
import hashlib
import json
from pathlib import Path

CONFIG_PATH = Path("config/ab_config.json")


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def assign_variant(user_id, config):
    h = hashlib.sha256(f"{config['salt']}:{user_id}".encode()).hexdigest()
    x = int(h[:12], 16) / 16**12
    cum = 0.0
    for variant, share in config["variants"].items():
        cum += share
        if x < cum:
            return variant
    return list(config["variants"])[-1]  # sécurité arrondis
