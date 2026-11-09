import numpy as np
import pytest

from src.ab import assign_variant
from src.analyze import z_test
from src.data import generate
from src.models import MODELS
from src.train import evaluate, split

CFG = {"salt": "t", "variants": {"a": 0.5, "b": 0.5}}


@pytest.fixture(scope="module")
def data():
    inter, items, _ = generate(n_users=120, n_items=80, seed=1)
    train, test = split(inter)
    return train, test, items


def test_assignment_is_deterministic():
    assert all(assign_variant(u, CFG) == assign_variant(u, CFG) for u in range(100))


def test_assignment_respects_split():
    share = np.mean([assign_variant(u, CFG) == "a" for u in range(10000)])
    assert 0.47 < share < 0.53


def test_z_test_detects_difference_and_null():
    assert z_test(100, 1000, 200, 1000)[1] < 0.001
    assert z_test(100, 1000, 101, 1000)[1] > 0.5


@pytest.mark.parametrize("name", list(MODELS))
def test_model_output_format(data, name):
    train, _, items = data
    m = MODELS[name]().fit(train, items)
    recs = m.recommend(3, k=5, exclude=m.seen(3))
    assert len(recs) == 5 and len(set(recs)) == 5
    assert not set(recs) & m.seen(3)  # jamais un article déjà acheté
    assert all(0 <= i < len(items) for i in recs)


def test_cold_start_user_does_not_crash(data):
    train, _, items = data
    m = MODELS["hybrid"]().fit(train, items)
    assert len(m.recommend(10_000, k=5)) == 5


def test_model_validation_threshold(data):
    """Validation de modèle : le hybride doit battre un tirage aléatoire (k/n_items)."""
    train, test, items = data
    m = MODELS["hybrid"]().fit(train, items)
    recall = evaluate(m, test, k=10)["recall_at_10"]
    assert recall > 10 / len(items)
