"""Incremental (O(1) amortised) feature service for real-time scoring.

Computes a COMPACT feature set whose values are provably identical to the batch implementation in
presage.features.temporal (parity is enforced by tests/test_online_parity.py). Semantics match the batch code: a payment sees only
events from strictly earlier whole seconds, so events within the current second are held in a pending buffer until the clock advances.
"""
from __future__ import annotations
import math
from collections import deque, defaultdict
import numpy as np

H1, H24, D7 = 3600, 86400, 7 * 86400
SERVING_FEATURES = [
    "amount", "log_amount", "balance_before", "pct_balance", "failed_ben_adds", "is_mobile", "session_s", "hour",
    "snd_cnt_1h", "snd_cnt_24h", "snd_sum_24h", "snd_newben_24h", "snd_newben_7d", "hist_cnt", "amt_to_hist_mean", "amt_to_hist_max", "amt_z",
    "secs_since_last_txn", "new_beneficiary", "prior_pays_to_dst", "dst_age_days", "dst_age_lt30d", "new_device", "device_age_days",
    "dst_in_cnt_24h", "dst_in_sum_24h", "dst_fanin_new_24h", "dst_fanin_new_7d", "dst_fanin_total", "dst_in_cnt_total",
    "dst_out_sum_24h", "dst_pass_through_24h", "dst_fanout_total", "dst_since_last_in_s", "dst_since_last_out_s",
]


class _Win:
    """Sliding sum/count over (ts, value) with strict upper bound handled by the caller's committed state."""

    def __init__(self, w): self.w, self.q, self.s = w, deque(), 0.0

    def add(self, t, v): self.q.append((t, v)); self.s += v

    def stats(self, t):
        while self.q and self.q[0][0] < t - self.w: self.s -= self.q.popleft()[1]
        return float(len(self.q)), self.s


class OnlineFeatureStore:
    def __init__(self):
        self.snd = defaultdict(lambda: dict(n=0, mean=0.0, m2=0.0, max=0.0, last=None, w1=_Win(H1), w24=_Win(H24), nb24=_Win(H24), nb7=_Win(D7),
                                            pairs=defaultdict(int), dev_first={}, devs=set()))
        self.rcv = defaultdict(lambda: dict(n=0, senders=set(), w24=_Win(H24), amt24=_Win(H24), new24=_Win(H24), new7=_Win(D7), last=None))
        self.out = defaultdict(lambda: dict(amt24=_Win(H24), receivers=set(), last=None))
        self.pending: list = []; self.cur_sec = None

    # ---- feature computation (reads committed state only) ----
    def features(self, e: dict) -> dict:
        t = float(e["ts"]); self._advance(t)
        s, r, o = self.snd[e["src"]], self.rcv[e["dst"]], self.out[e["dst"]]
        amt = float(e["amount"]); n = s["n"]; ts_ = math.floor(t)
        std = math.sqrt(max(s["m2"] / n, 0.0)) if n > 1 else float("nan")
        std = 0.0 if (std == std and std * std < 1e-4) else std
        c1, _ = s["w1"].stats(ts_); c24, s24 = s["w24"].stats(ts_); _, nb24 = s["nb24"].stats(ts_); _, nb7 = s["nb7"].stats(ts_)
        bal = e.get("balance_before"); bal = float("nan") if bal is None else float(bal)
        pct = amt / bal if bal and bal > 0 else float("nan")
        new_pair = 1.0 if s["pairs"][e["dst"]] == 0 else 0.0
        dev_first = s["dev_first"].get(e["device_id"])
        new_dev = 1.0 if (dev_first is None and n > 0) else 0.0
        in24c, _ = r["w24"].stats(ts_); _, in24s = r["amt24"].stats(ts_); _, fn24 = r["new24"].stats(ts_); _, fn7 = r["new7"].stats(ts_); _, out24 = o["amt24"].stats(ts_)
        gap = -1.0 if s["last"] is None else ts_ - s["last"]
        f = {
            "amount": amt, "log_amount": math.log1p(max(amt, 0)), "balance_before": 0.0 if bal != bal else bal, "pct_balance": 0.0 if pct != pct else pct,
            "failed_ben_adds": float(e.get("failed_ben_adds", 0)), "is_mobile": 1.0 if e.get("channel") == "mobile" else 0.0, "session_s": float(e.get("session_s", 0.0)),
            "hour": (t % 86400) / 3600.0, "snd_cnt_1h": c1, "snd_cnt_24h": c24, "snd_sum_24h": s24, "snd_newben_24h": nb24, "snd_newben_7d": nb7, "hist_cnt": float(n),
            "amt_to_hist_mean": amt / max(s["mean"], 1.0) if n > 0 else 0.0, "amt_to_hist_max": amt / max(s["max"], 1.0) if n > 0 else 0.0,
            "amt_z": (amt - s["mean"]) / max(std, 1.0) if n > 1 else 0.0, "secs_since_last_txn": gap, "new_beneficiary": new_pair, "prior_pays_to_dst": float(s["pairs"][e["dst"]]),
            "dst_age_days": float(e["dst_age_days"]), "dst_age_lt30d": 1.0 if e["dst_age_days"] < 30 else 0.0, "new_device": new_dev,
            "device_age_days": 0.0 if dev_first is None else (ts_ - dev_first) / 86400.0,
            "dst_in_cnt_24h": in24c, "dst_in_sum_24h": in24s, "dst_fanin_new_24h": fn24, "dst_fanin_new_7d": fn7, "dst_fanin_total": float(len(r["senders"])), "dst_in_cnt_total": float(r["n"]),
            "dst_out_sum_24h": out24, "dst_pass_through_24h": out24 / max(in24s, 1.0) if in24s > 0 else 0.0, "dst_fanout_total": float(len(o["receivers"])),
            "dst_since_last_in_s": -1.0 if r["last"] is None else ts_ - r["last"], "dst_since_last_out_s": -1.0 if o["last"] is None else ts_ - o["last"],
        }
        return f

    # ---- state updates ----
    def update(self, e: dict):
        """Queue the event; it becomes visible to later payments once the wall clock moves past its second."""
        self._advance(float(e["ts"])); self.pending.append(e); self.cur_sec = math.floor(float(e["ts"]))

    def _advance(self, t):
        sec = math.floor(t)
        if self.pending and sec > self.cur_sec:
            for e in self.pending: self._commit(e)
            self.pending = []

    def _commit(self, e):
        ts_ = math.floor(float(e["ts"])); amt = float(e["amount"]); s, r, o = self.snd[e["src"]], self.rcv[e["dst"]], self.out[e["src"]]
        new_pair = 1.0 if s["pairs"][e["dst"]] == 0 else 0.0
        s["n"] += 1; d = amt - s["mean"]; s["mean"] += d / s["n"]; s["m2"] += d * (amt - s["mean"]); s["max"] = max(s["max"], amt); s["last"] = ts_
        s["w1"].add(ts_, 1.0); s["w24"].add(ts_, amt); s["nb24"].add(ts_, new_pair); s["nb7"].add(ts_, new_pair); s["pairs"][e["dst"]] += 1
        s["dev_first"].setdefault(e["device_id"], ts_)
        r["n"] += 1; r["senders"].add(e["src"]); r["w24"].add(ts_, 1.0); r["amt24"].add(ts_, amt); r["new24"].add(ts_, new_pair); r["new7"].add(ts_, new_pair); r["last"] = ts_
        o["amt24"].add(ts_, amt); o["receivers"].add(e["dst"]); o["last"] = ts_
