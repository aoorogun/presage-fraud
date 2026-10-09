"""Walk-forward engine for drift-adaptation and label-delay experiments.

At the start of each block (default 7 days) a strategy may update its model using ONLY information available at that
moment: rows with ts <= now - delta (delta = label delay) carry final labels; a fraud confirmed later is invisible.
Predictions for the block are then made with the current model. Nothing from the block (or later) is ever used.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DAY = 86400.0


@dataclass
class Task:
    X: np.ndarray            # feature matrix (all rows)
    y: np.ndarray            # fraud labels (ground truth)
    ts: np.ndarray
    amount: np.ndarray
    mask: np.ndarray         # population (customer-initiated payments)
    start_day: int = 60
    end_day: int = 120
    block_days: int = 7
    delay_s: float = 0.0     # label delay (staleness) applied to all labels


def labelled_upto(task: Task, now: float) -> np.ndarray:
    """Rows whose label is final at time `now` under a fixed confirmation delay."""
    return task.mask & (task.ts <= now - task.delay_s)


def blocks(task: Task):
    d = task.start_day
    while d < task.end_day:
        yield d, min(d + task.block_days, task.end_day)
        d += task.block_days


def run_strategy(task: Task, strategy, seed: int = 42) -> dict:
    """`strategy` implements init(task, now) -> state, update(state, task, now, block_idx) -> state, predict(state, X) -> scores."""
    state = strategy.init(task, task.start_day * DAY)
    res = dict(block_start=[], pr_auc=[], recall_at_fpr=[], n_fraud=[], updated=[], retrain_s=0.0)
    for bi, (d0, d1) in enumerate(blocks(task)):
        now = d0 * DAY
        t0 = time.time()
        state, did = strategy.update(state, task, now, bi)
        res["retrain_s"] += time.time() - t0 if did else 0.0
        sel = task.mask & (task.ts >= now) & (task.ts < d1 * DAY)
        p = strategy.predict(state, task.X[sel]); yy = task.y[sel]
        res["block_start"].append(d0); res["n_fraud"].append(int(yy.sum())); res["updated"].append(bool(did))
        res["pr_auc"].append(average_precision_score(yy, p) if yy.sum() > 0 else np.nan)
        neg = np.sort(p[yy == 0]); thr = neg[int(0.99 * len(neg))] if len(neg) else 1.0     # threshold at 1% FPR of the block's legit rows
        res["recall_at_fpr"].append(float((p[yy == 1] > thr).mean()) if yy.sum() > 0 else np.nan)
        if hasattr(strategy, "after_block"): strategy.after_block(state, task, sel, p)
    res["state"] = state
    return res
