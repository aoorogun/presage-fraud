import numpy as np
from sklearn.metrics import roc_auc_score
from presage.evaluation import stats as S


def test_delong_matches_sklearn_auc_and_detects_difference():
    rng = np.random.default_rng(0); y = (rng.random(3000) < 0.1).astype(int)
    good = y * 1.5 + rng.normal(0, 1, 3000); bad = y * 0.3 + rng.normal(0, 1, 3000)
    a1, a2, z, p = S.delong_test(y, good, bad)
    assert abs(a1 - roc_auc_score(y, good)) < 1e-9 and abs(a2 - roc_auc_score(y, bad)) < 1e-9
    assert p < 1e-6
    assert S.delong_test(y, good, good)[3] == 1.0


def test_holm_and_friedman():
    h = S.holm({"a": 0.01, "b": 0.02, "c": 0.5}); assert h["a"] <= h["b"] <= h["c"] and h["a"] == 0.03
    rng = np.random.default_rng(1); sc = np.column_stack([rng.normal(1, .1, 30), rng.normal(0.5, .1, 30), rng.normal(0, .1, 30)])
    f = S.friedman_nemenyi(sc, ["a", "b", "c"]); assert f["p"] < 0.001 and f["mean_rank"]["a"] < f["mean_rank"]["c"]


def test_paired_bootstrap_sign():
    rng = np.random.default_rng(2); y = (rng.random(4000) < 0.05).astype(int)
    a = y + rng.normal(0, 0.8, 4000); b = rng.normal(0, 1, 4000)
    r = S.paired_bootstrap_diff(y, a, b, n_boot=100); assert r["diff"] > 0 and r["lo"] > 0
