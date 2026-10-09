"""Label-delay study: supervised learning under confirmation delays and three mitigations
(pseudo-labelling, weak supervision by rules, analyst-in-the-loop active learning)."""
from __future__ import annotations
import json, time
import numpy as np, pandas as pd
from presage.drift.walkforward import Task, run_strategy, DAY
from presage.drift.strategies import _fit, _window, Static
from presage.utils.common import ROOT, get_logger, log_experiment

log = get_logger("labeldelay")
DELAYS = {"1h": 3600.0, "1d": DAY, "3d": 3 * DAY, "7d": 7 * DAY, "30d": 30 * DAY}


class DelayStrategy(Static):
    """Retrain every block on the labelled window (60 d) with an optional label-delay mitigation."""

    def __init__(self, kind="supervised", window=60, rule_cols=None, budget=40):
        self.kind, self.w, self.rule_cols, self.budget = kind, window, rule_cols, budget; self.name = kind; self.rng = np.random.default_rng(0)

    def _train(self, task, now):
        idx = _window(task, now, self.w); X, y, w = task.X[idx], task.y[idx].astype(float), np.ones(len(idx))
        m = _fit(X, y, w)
        if self.kind == "supervised": return m
        recent = np.where(task.mask & (task.ts > now - task.delay_s) & (task.ts < now))[0]       # rows whose labels have NOT arrived
        if len(recent) == 0: return m
        p = m.predict_proba(task.X[recent])[:, 1]
        if self.kind == "pseudo":
            pos = recent[p > 0.9]; neg = recent[(p < 0.01)]; neg = self.rng.choice(neg, min(len(neg), 5 * max(len(pos), 20)), replace=False) if len(neg) else neg
            sel = np.concatenate([pos, neg]); ya = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            return _fit(np.concatenate([X, task.X[sel]]), np.concatenate([y, ya]), np.concatenate([w, np.full(len(sel), 0.5)]))
        if self.kind == "weak":
            c = self.rule_cols; r = task.X[recent]
            lf = (r[:, c["fanin24"]] >= 3) & (r[:, c["pass24"]] > 0.5) & (r[:, c["newben"]] > 0)       # labelling function: classic mule-beneficiary pattern
            pos = recent[lf]; neg = recent[~lf & (p < 0.01)]; neg = self.rng.choice(neg, min(len(neg), 5 * max(len(pos), 20)), replace=False) if len(neg) else neg
            sel = np.concatenate([pos, neg]); ya = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            return _fit(np.concatenate([X, task.X[sel]]), np.concatenate([y, ya]), np.concatenate([w, np.full(len(sel), 0.5)]))
        if self.kind == "active":
            unc = np.abs(p - 0.5); cand = np.argsort(-(p * (1 - p) + 1e-3 * p))[: self.budget]       # uncertainty+score sampling; analyst returns TRUE labels
            sel = recent[cand]
            return _fit(np.concatenate([X, task.X[sel]]), np.concatenate([y, task.y[sel]]), np.concatenate([w, np.full(len(sel), 3.0)]))

    def init(self, task, now): return dict(model=self._train(task, now), retrains=[task.start_day])

    def update(self, st, task, now, bi):
        if bi == 0: return st, False
        return dict(model=self._train(task, now), retrains=st["retrains"] + [now / DAY]), True


def run_label_delay(df, acc, X0, g, features_for_delay, seed=42):
    cust = df.src_is_customer.values == 1; y = df.is_app_fraud.values; ts = df.ts.values; amt = df.amount.values
    tab = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"]; allc = tab + g["graph"]
    rc = dict(fanin24=allc.index("dst_fanin_new_24h"), pass24=allc.index("dst_pass_through_24h"), newben=allc.index("new_beneficiary"))
    rows = []
    for dn, ds in DELAYS.items():
        M = features_for_delay(ds)
        task = Task(M, y, ts, amt, cust, start_day=60, end_day=120, block_days=7, delay_s=ds)
        for kind in (["supervised", "pseudo", "weak", "active"] if ds >= DAY else ["supervised"]):
            t0 = time.time(); r = run_strategy(task, DelayStrategy(kind, rule_cols=rc), seed); pa = np.array(r["pr_auc"])
            row = dict(delay=dn, variant=kind, mean_pr_auc=np.nanmean(pa), recall_at_1pct_fpr=np.nanmean(r["recall_at_fpr"]), post_drift_pr_auc=np.nanmean(pa[np.array(r["block_start"]) >= 98]))
            rows.append(row); log.info("delay %s %s PR-AUC %.3f recall@1%%FPR %.2f (%.0fs)", dn, kind, row["mean_pr_auc"], row["recall_at_1pct_fpr"], time.time() - t0)
            log_experiment("label_delay", f"{kind}|{dn}", "sim_v1", seed, {k: float(v) for k, v in row.items() if isinstance(v, (float, np.floating)) and np.isfinite(v)})
            pd.DataFrame(rows).to_csv(ROOT / "reports/tables/label_delay.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
