"""Investigator-facing explanations for a boosted-tree surrogate of the PRESAGE feature set.

Levels  : transaction, behaviour, beneficiary, graph (feature groups map to levels)
Methods : SHAP (TreeExplainer) for attribution; template reasons computed from raw feature values; greedy counterfactuals over
          semantically coherent edits; a deletion-based faithfulness test; symbolic graph evidence (not GNNExplainer).
Caveats : SHAP attributions describe the MODEL, not causal truth; counterfactuals are model-based what-ifs, not guarantees; attention
          weights are not used as explanations anywhere in this repository.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap

# Edits that move a flagged payment toward a legitimate profile. Each edit changes a coherent set of features together.
EDITS = {
    "beneficiary account older than 90 days": {"dst_age_days": 400.0, "dst_age_lt30d": 0.0},
    "beneficiary previously paid by this customer": {"new_beneficiary": 0.0, "prior_pays_to_dst": 3.0},
    "beneficiary not receiving from many new senders": {"dst_fanin_new_1h": 0.0, "dst_fanin_new_24h": 1.0, "dst_fanin_new_7d": 3.0},
    "beneficiary not forwarding funds rapidly": {"dst_pass_through_1h": 0.0, "dst_pass_through_24h": 0.0, "dst_pass_through_7d": 0.1, "dst_out_sum_1h": 0.0, "dst_out_sum_24h": 0.0},
    "no confirmed mule accounts near the beneficiary": {"dg_conf1": 0.0, "dg_conf2": 0.0, "dg_self_conf": 0.0},
    "amount in line with customer history": {"amt_to_hist_mean": 1.0, "amt_to_hist_max": 0.6, "amt_z": 0.0},
    "small share of balance": {"pct_balance": 0.05, "pct_to_hist_max_pct": 0.8},
    "no failed beneficiary additions / normal session": {"failed_ben_adds": 0.0, "session_z": 0.0},
    "trusted device and IP": {"new_device": 0.0, "new_ip": 0.0},
}
LEVEL = None  # filled from feature groups


def level_of(feat: str, groups: dict) -> str:
    if feat in groups.get("graph", []): return "graph"
    if feat in groups.get("beneficiary", []): return "beneficiary"
    if feat in groups.get("behaviour", []): return "behaviour"
    return "transaction" if feat in groups.get("txn", []) else "device"


def reason_text(feat: str, v: float, row: pd.Series) -> str | None:
    f = {
        "amt_to_hist_mean": lambda: f"amount is {v:.1f}x this customer's historical mean payment",
        "amt_to_hist_max": lambda: f"amount is {v:.1f}x the customer's largest previous payment",
        "pct_balance": lambda: f"payment moves {100 * v:.0f}% of the account balance",
        "pct_to_hist_max_pct": lambda: f"share of balance moved is {v:.1f}x the customer's previous maximum share",
        "new_beneficiary": lambda: "first payment to this beneficiary" if v > 0 else None,
        "dst_age_days": lambda: f"beneficiary account is only {v:.0f} days old" if v < 90 else None,
        "dst_fanin_new_24h": lambda: f"beneficiary received from {v:.0f} previously unseen senders in the last 24h" if v >= 2 else None,
        "dst_fanin_new_7d": lambda: f"beneficiary received from {v:.0f} new senders in 7 days" if v >= 3 else None,
        "dst_pass_through_24h": lambda: f"beneficiary forwarded {100 * v:.0f}% of its 24h inflow onward" if v > 0.3 else None,
        "dst_avg_holding_s": lambda: f"beneficiary typically holds funds only {v / 3600:.1f} h before sending on" if 0 <= v < 6 * 3600 else None,
        "dst_fanout_total": lambda: f"beneficiary has paid {v:.0f} distinct accounts onward" if v >= 3 else None,
        "dg_conf1": lambda: f"{v:.0f} confirmed mule account(s) directly connected to the beneficiary" if v > 0 else None,
        "dg_conf2": lambda: f"{v:.0f} confirmed-mule connection paths within two hops of the beneficiary" if v > 0 else None,
        "dg_self_conf": lambda: "beneficiary was already reported as a mule account" if v > 0 else None,
        "failed_ben_adds": lambda: f"{v:.0f} failed beneficiary additions in this session" if v > 0 else None,
        "session_z": lambda: f"session length is {v:.1f} standard deviations above this customer's norm" if v > 1.5 else None,
        "new_device": lambda: "payment made from a device not previously seen for this customer" if v > 0 else None,
        "hour_dev": lambda: f"payment time is {v:.1f} h away from the customer's usual hour" if v > 5 else None,
        "inactive_30d": lambda: "customer was inactive for more than 30 days" if v > 0 else None,
        "device_shared_accounts": lambda: f"device is shared by {v:.0f} accounts" if v >= 3 else None,
    }.get(feat)
    return f() if f else None


@dataclass
class Explainer:
    model: object                      # fitted xgboost classifier (surrogate on the PRESAGE feature set)
    cols: list[str]
    groups: dict
    background: pd.DataFrame           # legit reference rows (for medians / SHAP baseline)
    threshold: float

    def __post_init__(self):
        self.tree = shap.TreeExplainer(self.model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return self.tree.shap_values(X[self.cols].values)

    def explain(self, row: pd.Series, top_k: int = 5) -> dict:
        x = row[self.cols].values.astype(float)[None, :]
        sv = self.tree.shap_values(x)[0]; p = float(self.model.predict_proba(x)[0, 1])
        order = np.argsort(-sv); reasons = []
        for j in order[:12]:
            if sv[j] <= 0: break
            t = reason_text(self.cols[j], float(x[0, j]), row)
            if t: reasons.append(dict(level=level_of(self.cols[j], self.groups), feature=self.cols[j], shap=float(sv[j]), text=t))
            if len(reasons) >= top_k: break
        return dict(probability=p, reasons=reasons, counterfactual=self.counterfactual(row))

    def counterfactual(self, row: pd.Series, max_edits: int = 3) -> dict:
        """Greedy forward selection over EDITS: pick the edit that lowers risk most, until risk < threshold or max_edits reached."""
        x = row[self.cols].values.astype(float).copy(); p0 = float(self.model.predict_proba(x[None, :])[0, 1])
        applied, cur, p = [], x.copy(), p0
        for _ in range(max_edits):
            best = None
            for name, ed in EDITS.items():
                if name in [a for a in applied]: continue
                z = cur.copy()
                for f, val in ed.items():
                    if f in self.cols: z[self.cols.index(f)] = val
                pz = float(self.model.predict_proba(z[None, :])[0, 1])
                if best is None or pz < best[1]: best = (name, pz, z)
            if best is None or best[1] >= p - 1e-4: break
            applied.append(best[0]); cur, p = best[2], best[1]
            if p < self.threshold: break
        return dict(p_before=p0, p_after=p, flipped=bool(p < self.threshold <= p0), edits=applied,
                    text=(f"Had {' and '.join(applied)}, estimated risk would fall from {p0:.2f} to {p:.2f}." if applied else "No single coherent edit lowers the estimated risk."))

    def faithfulness(self, X: pd.DataFrame, k: int = 5, n: int = 300, seed: int = 0) -> dict:
        """Deletion test: replace the top-k SHAP features of flagged rows by legit medians vs k random features; compare mean probability drop."""
        rng = np.random.default_rng(seed); Xs = X[self.cols].sample(min(n, len(X)), random_state=seed)
        sv = self.tree.shap_values(Xs.values); med = self.background[self.cols].median().values
        p0 = self.model.predict_proba(Xs.values)[:, 1]; top = Xs.values.copy(); rnd = Xs.values.copy()
        for i in range(len(Xs)):
            t = np.argsort(-sv[i])[:k]; top[i, t] = med[t]
            r = rng.choice(len(self.cols), k, replace=False); rnd[i, r] = med[r]
        return dict(mean_drop_top_shap=float((p0 - self.model.predict_proba(top)[:, 1]).mean()), mean_drop_random=float((p0 - self.model.predict_proba(rnd)[:, 1]).mean()), k=k, n=len(Xs))


def graph_evidence(df: pd.DataFrame, acc: pd.DataFrame, txn_row: pd.Series, conf_mules: set, ts_cut: float, max_items: int = 5) -> dict:
    """Symbolic subgraph evidence available strictly before the payment: confirmed mules within two hops, shared-device partners, recent senders."""
    past = df[df.ts < ts_cut]; dst = int(txn_row.dst)
    nb1 = set(past.src[past.dst == dst]) | set(past.dst[past.src == dst])
    nb2 = set(past.src[past.dst.isin(nb1)]) | set(past.dst[past.src.isin(nb1)]) if len(nb1) < 200 else set()
    dev = set(past.src[past.device_id.isin(past.device_id[past.src == dst])]) - {dst}
    return dict(confirmed_mules_1hop=sorted(nb1 & conf_mules)[:max_items], confirmed_mules_2hop=sorted((nb2 - nb1) & conf_mules)[:max_items],
                shared_device_accounts=sorted(dev)[:max_items], n_distinct_recent_senders=int(past.src[(past.dst == dst) & (past.ts > ts_cut - 2 * 86400)].nunique()))
