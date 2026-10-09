"""Evaluation: threshold-free and thresholded metrics, ranking metrics, cost model, bootstrap CIs.

Accuracy is deliberately NOT reported as a headline metric (class prevalence ~0.25%).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef,
                             balanced_accuracy_score, confusion_matrix, ndcg_score)


@dataclass
class CostModel:
    """Monetary assumptions (SIMULATED, GBP). FN loses the payment amount; every alert costs review time;
    a false alert additionally adds customer-friction cost."""
    c_review: float = 8.0
    c_fp: float = 2.0
    recovery_rate: float = 0.0   # share of fraud value recoverable after the fact (conservative: none)


def thresholded(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    beta2 = 4.0
    f2 = (1 + beta2) * pr * rc / (beta2 * pr + rc) if (pr + rc) > 0 else 0.0
    return dict(precision=pr, recall=rc, f1=f1, f2=f2, mcc=matthews_corrcoef(y, yhat) if yhat.any() else 0.0,
                balanced_acc=balanced_accuracy_score(y, yhat), specificity=tn / max(tn + fp, 1), fpr=fp / max(fp + tn, 1),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), alerts=int(yhat.sum()))


def precision_recall_at_k(y: np.ndarray, p: np.ndarray, ks=(50, 100, 500, 1000)) -> dict:
    order = np.argsort(-p)
    out = {}
    pos = max(int(y.sum()), 1)
    for k in ks:
        k = min(k, len(y))
        top = y[order[:k]]
        out[f"precision@{k}"] = top.mean(); out[f"recall@{k}"] = top.sum() / pos
    return out


def expected_cost(y, p, amount, thr, cm: CostModel = CostModel()) -> dict:
    alert = p >= thr
    fn_loss = float(amount[(y == 1) & ~alert].sum()) * (1 - cm.recovery_rate)
    fp_n = int(((y == 0) & alert).sum())
    cost = fn_loss + cm.c_fp * fp_n + cm.c_review * int(alert.sum())
    prevented = float(amount[(y == 1) & alert].sum())
    total_fraud = float(amount[y == 1].sum())
    return dict(cost=cost, fn_loss=fn_loss, prevented_value=prevented, prevented_ratio=prevented / max(total_fraud, 1e-9),
                intervention_rate=float(alert.mean()), cost_per_prevented=(cm.c_review * alert.sum() + cm.c_fp * fp_n) / max(prevented, 1e-9))


def best_threshold(y, p, amount=None, objective="f2", cm: CostModel = CostModel(), grid=400) -> float:
    """Choose a threshold on VALIDATION data only. objective in {'f1','f2','cost'}."""
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.99999, grid)))
    best, bt = -np.inf, qs[-1]
    for t in qs:
        if objective == "cost":
            v = -expected_cost(y, p, amount, t, cm)["cost"]
        else:
            m = thresholded(y, p, t); v = m[objective]
        if v > best: best, bt = v, t
    return float(bt)


def full_report(y, p, amount, thr, cm: CostModel = CostModel()) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    out = dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan)
    out.update(thresholded(y, p, thr)); out.update(precision_recall_at_k(y, p))
    out.update(expected_cost(y, p, amount, thr, cm))
    k = min(1000, len(y))
    out["ndcg@1000"] = ndcg_score(y[None, :], p[None, :], k=k)
    return out


def bootstrap_ci(y, p, metric_fn, n_boot=200, seed=0, alpha=0.05, block: np.ndarray | None = None):
    """Percentile bootstrap CI. If `block` (e.g. day index) is given, resample blocks to respect time dependence."""
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    vals = []
    if block is None:
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            if y[i].sum() == 0: continue
            vals.append(metric_fn(y[i], p[i]))
    else:
        ub = np.unique(block); idx = {b: np.where(block == b)[0] for b in ub}
        for _ in range(n_boot):
            i = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
            if y[i].sum() == 0: continue
            vals.append(metric_fn(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def temporal_split(day: np.ndarray, fracs=(0.6, 0.2, 0.2), horizon: int | None = None):
    """Chronological masks (train, val, test) by day index. No shuffling."""
    horizon = horizon or int(day.max()) + 1
    a, b = int(round(horizon * fracs[0])), int(round(horizon * (fracs[0] + fracs[1])))
    return day < a, (day >= a) & (day < b), day >= b
