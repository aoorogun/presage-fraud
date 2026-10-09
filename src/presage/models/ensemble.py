"""Rank-average ensembles and the PRESAGE + GBDT hybrid.

Everything that is chosen (here only the hybrid weight) is chosen on the validation split; the test split is only scored.
Rank averaging makes scores from differently calibrated models comparable (PRESAGE and XGBoost are trained with class weights)."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap


def rk(p) -> np.ndarray:
    return rankdata(p) / len(p)


def rank_ensemble(preds: list) -> np.ndarray:
    """Mean of the rank-transformed predictions of several runs (seeds); returned as ranks in (0, 1]."""
    return rk(np.mean([rk(p) for p in preds], 0))


def choose_hybrid_weight(y_val, a_val, b_val, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Weight on model `a` in rk(w*a + (1-w)*b), chosen by validation PR-AUC. Returns (best_w, {w: val_pr_auc})."""
    scores = {w: ap(y_val, rk(w * a_val + (1 - w) * b_val)) for w in grid}
    return max(scores, key=scores.get), scores


def hybrid(a, b, w: float) -> np.ndarray:
    return rk(w * a + (1 - w) * b)


def paired_block_bootstrap(y, days, pairs: dict, n_boot=500, seed=0):
    """Paired bootstrap of PR-AUC differences, resampling whole days. `pairs` maps a label to (scores_a, scores_b)."""
    ud = np.unique(days); rng = np.random.default_rng(seed); d = {k: [] for k in pairs}; where = {u: np.where(days == u)[0] for u in ud}
    for _ in range(n_boot):
        idx = np.concatenate([where[u] for u in rng.choice(ud, len(ud))])
        if y[idx].sum() == 0: continue
        for k, (a, b) in pairs.items(): d[k].append(ap(y[idx], a[idx]) - ap(y[idx], b[idx]))
    return {k: dict(mean_diff=float(np.mean(v)), ci_lo=float(np.percentile(v, 2.5)), ci_hi=float(np.percentile(v, 97.5))) for k, v in d.items()}
