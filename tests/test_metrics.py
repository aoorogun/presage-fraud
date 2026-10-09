import numpy as np
from presage.evaluation.metrics import temporal_split, thresholded, expected_cost, best_threshold


def test_temporal_split_is_chronological():
    day = np.repeat(np.arange(100), 5)
    tr, va, te = temporal_split(day)
    assert day[tr].max() < day[va].min() <= day[va].max() < day[te].min()
    assert (tr | va | te).all() and not (tr & va).any() and not (va & te).any()


def test_threshold_selected_on_validation_only_and_cost():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.01).astype(int); p = np.clip(y * 0.5 + rng.random(5000) * 0.5, 0, 1); a = rng.uniform(10, 500, 5000)
    t = best_threshold(y, p, a, "cost")
    c = expected_cost(y, p, a, t)
    assert c["cost"] <= expected_cost(y, p, a, 0.0)["cost"] and c["cost"] >= 0
    assert thresholded(y, p, t)["tp"] + thresholded(y, p, t)["fn"] == y.sum()
