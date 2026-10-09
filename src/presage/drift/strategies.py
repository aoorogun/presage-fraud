"""Adaptation strategies for the walk-forward drift experiment (all share one boosted-tree base learner)."""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from presage.drift.walkforward import Task, labelled_upto, DAY
from presage.drift.detectors import make_detector

PARAMS = dict(n_estimators=120, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=20.0, tree_method="hist", n_jobs=2, verbosity=0, eval_metric="aucpr")


def _fit(X, y, w=None, seed=42, base=None, n=None):
    p = dict(PARAMS, random_state=seed)
    if n: p["n_estimators"] = n
    m = xgb.XGBClassifier(**p)
    m.fit(X, y, sample_weight=w, xgb_model=base.get_booster() if base is not None else None)
    return m


def _window(task, now, days=None):
    lab = labelled_upto(task, now)
    if days is not None: lab &= task.ts >= now - task.delay_s - days * DAY
    return np.where(lab)[0]


class Static:
    name = "static"

    def init(self, task, now):
        idx = _window(task, now); self.n_retrain = 1
        return dict(model=_fit(task.X[idx], task.y[idx]), retrains=[task.start_day])

    def update(self, st, task, now, bi): return st, False
    def predict(self, st, X): return st["model"].predict_proba(X)[:, 1]


class Scheduled(Static):
    def __init__(self, every_blocks=2): self.every = every_blocks; self.name = f"scheduled_{every_blocks * 7}d"

    def update(self, st, task, now, bi):
        if bi % self.every == 0 and bi > 0:
            idx = _window(task, now); st = dict(model=_fit(task.X[idx], task.y[idx]), retrains=st["retrains"] + [now / DAY]); return st, True
        return st, False


class Sliding(Static):
    def __init__(self, window_days=30): self.w = window_days; self.name = f"sliding_{window_days}d"

    def init(self, task, now):
        idx = _window(task, now, self.w); return dict(model=_fit(task.X[idx], task.y[idx]), retrains=[task.start_day])

    def update(self, st, task, now, bi):
        if bi == 0: return st, False
        idx = _window(task, now, self.w)
        if task.y[idx].sum() < 5: return st, False
        return dict(model=_fit(task.X[idx], task.y[idx]), retrains=st["retrains"] + [now / DAY]), True


class DriftTriggered(Static):
    """Monitor the miss-rate stream of confirmed frauds (rows whose labels just became final) with a drift detector;
    on alarm retrain on the recent window. Detection can only use delayed labels, so adaptation delay >= label delay."""

    def __init__(self, detector="adwin", window_days=30):
        self.det_name = detector; self.w = window_days; self.name = f"drift_triggered_{detector}"

    def init(self, task, now):
        idx = _window(task, now, self.w); self.det = make_detector(self.det_name); self.thr = 0.5
        m = _fit(task.X[idx], task.y[idx])
        neg = np.sort(m.predict_proba(task.X[idx][task.y[idx] == 0])[:, 1]); self.thr = neg[int(0.99 * len(neg))]
        self.seen = now - task.delay_s
        return dict(model=m, retrains=[task.start_day], alarms=[])

    def update(self, st, task, now, bi):
        lo, hi = self.seen, now - task.delay_s
        new = np.where(task.mask & (task.ts >= lo) & (task.ts < hi) & (task.y == 1))[0]
        self.seen = hi
        fired = False
        if len(new):
            miss = (st["model"].predict_proba(task.X[new])[:, 1] <= self.thr).astype(float)
            for v in miss:
                self.det.update(v)
                if getattr(self.det, "drift_detected", False): fired = True
        if fired:
            idx = _window(task, now, self.w)
            m = _fit(task.X[idx], task.y[idx]); neg = np.sort(m.predict_proba(task.X[idx][task.y[idx] == 0])[:, 1]); self.thr = neg[int(0.99 * len(neg))]
            self.det = make_detector(self.det_name)
            return dict(model=m, retrains=st["retrains"] + [now / DAY], alarms=st["alarms"] + [now / DAY]), True
        return st, False


class Incremental(Static):
    """Warm-started boosting: append trees fitted to the newest labelled block only (online-style, no old data)."""
    name = "incremental"

    def init(self, task, now):
        idx = _window(task, now); self.seen = now - task.delay_s
        return dict(model=_fit(task.X[idx], task.y[idx]), retrains=[task.start_day])

    def update(self, st, task, now, bi):
        lo, hi = self.seen, now - task.delay_s; self.seen = hi
        idx = np.where(task.mask & (task.ts >= lo) & (task.ts < hi))[0]
        if len(idx) == 0 or task.y[idx].sum() < 3: return st, False
        return dict(model=_fit(task.X[idx], task.y[idx], base=st["model"], n=25), retrains=st["retrains"] + [now / DAY]), True


class Replay(Static):
    """Continual learning by (weighted) experience replay: recent window + a reservoir of historical frauds and legit rows."""

    def __init__(self, window_days=21, replay_pos=800, replay_neg=4000, replay_weight=0.6, seed=0):
        self.w, self.rp, self.rn, self.rw = window_days, replay_pos, replay_neg, replay_weight; self.rng = np.random.default_rng(seed); self.name = "replay_buffer"

    def _train(self, task, now):
        rec = _window(task, now, self.w)
        old = np.setdiff1d(_window(task, now), rec)
        pos = old[task.y[old] == 1]; neg = old[task.y[old] == 0]
        pos = self.rng.choice(pos, min(self.rp, len(pos)), replace=False) if len(pos) else pos
        neg = self.rng.choice(neg, min(self.rn, len(neg)), replace=False) if len(neg) else neg
        idx = np.concatenate([rec, pos, neg]); w = np.concatenate([np.ones(len(rec)), np.full(len(pos) + len(neg), self.rw)])
        return _fit(task.X[idx], task.y[idx], w)

    def init(self, task, now): return dict(model=self._train(task, now), retrains=[task.start_day])

    def update(self, st, task, now, bi):
        if bi == 0: return st, False
        return dict(model=self._train(task, now), retrains=st["retrains"] + [now / DAY]), True


def all_strategies():
    return [Static(), Scheduled(2), Sliding(30), DriftTriggered("adwin"), DriftTriggered("ddm"), Incremental(), Replay()]
