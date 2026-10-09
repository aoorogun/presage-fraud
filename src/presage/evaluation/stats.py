"""Statistical testing helpers: paired block bootstrap, DeLong (ROC-AUC), McNemar, Friedman + Nemenyi, Holm correction, effect sizes."""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score


def paired_bootstrap_diff(y, pa, pb, metric=average_precision_score, block=None, n_boot=500, seed=0):
    """Difference metric(pa)-metric(pb) with percentile CI and a two-sided bootstrap p-value. `block` (e.g. day) resamples whole blocks."""
    rng = np.random.default_rng(seed); y = np.asarray(y)
    if block is None: groups = [np.arange(len(y))]; ub = [0]; idx = {0: groups[0]}
    else:
        ub = np.unique(block); idx = {b: np.where(block == b)[0] for b in ub}
    d = []
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub)); i = np.concatenate([idx[b] for b in pick]) if block is not None else rng.integers(0, len(y), len(y))
        if y[i].sum() == 0: continue
        d.append(metric(y[i], pa[i]) - metric(y[i], pb[i]))
    d = np.array(d); obs = metric(y, pa) - metric(y, pb)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(diff=float(obs), lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)), p=float(max(p, 1 / (len(d) + 1))))


def _midrank(x):
    o = np.argsort(x); z = x[o]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]: j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(n); out[o] = t; return out


def _delong_components(y, p):
    pos, neg = p[y], p[~y]; m, n = len(pos), len(neg)
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
    auc = tz[:m].sum() / m / n - (m + 1.0) / (2.0 * n)
    return auc, (tz[:m] - tx) / n, 1.0 - (tz[m:] - ty) / m


def delong_test(y, p1, p2):
    """Paired DeLong test for two correlated ROC-AUCs (Sun & Xu 2014 fast implementation). Returns auc1, auc2, z, two-sided p."""
    y = np.asarray(y).astype(bool); m, n = y.sum(), (~y).sum()
    a1, v01a, v10a = _delong_components(y, np.asarray(p1, dtype=float)); a2, v01b, v10b = _delong_components(y, np.asarray(p2, dtype=float))
    s01 = np.cov(np.vstack([v01a, v01b])); s10 = np.cov(np.vstack([v10a, v10b]))
    Sm = s01 / m + s10 / n; var = Sm[0, 0] + Sm[1, 1] - 2 * Sm[0, 1]
    if var <= 1e-18: return float(a1), float(a2), 0.0, 1.0
    z = (a1 - a2) / np.sqrt(var); return float(a1), float(a2), float(z), float(2 * stats.norm.sf(abs(z)))


def mcnemar(y, pred_a, pred_b):
    """McNemar's test on correctness of two classifiers (exact binomial when discordant pairs are few)."""
    ca, cb = (pred_a == y), (pred_b == y); b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum()); n = b01 + b10
    if n == 0: return dict(b01=0, b10=0, p=1.0)
    return dict(b01=b01, b10=b10, p=float(stats.binomtest(b01, n, 0.5).pvalue))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def friedman_nemenyi(scores: np.ndarray, names: list[str]):
    """scores [n_blocks, n_models] (higher is better). Returns friedman p, mean ranks and the Nemenyi critical difference (alpha=.05)."""
    n, k = scores.shape; ranks = np.vstack([stats.rankdata(-r) for r in scores]); mr = ranks.mean(0)
    chi = stats.friedmanchisquare(*[scores[:, j] for j in range(k)]); q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}.get(k, 3.164)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return dict(p=float(chi.pvalue), stat=float(chi.statistic), mean_rank=dict(zip(names, mr.round(3))), critical_difference=float(cd), n_blocks=n)


def cohens_d_paired(a, b):
    d = np.asarray(a) - np.asarray(b); return float(d.mean() / (d.std(ddof=1) + 1e-12))
