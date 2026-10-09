"""Leakage-safe behavioural, beneficiary and device features for each payment.

Every feature for a payment at time t uses only events with time < t (see windows.py).
Feature groups (used for ablations) are exposed in FEATURE_GROUPS.

Input  : transactions DataFrame from presage.simulation.app_world.simulate (or any frame
         with the same core columns) and the accounts table.
Output : DataFrame indexed like the input, columns = features (float32) plus `txn_id`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from presage.features.windows import EventIndex, INF
from presage.utils.common import get_logger

log = get_logger("features")
WINDOWS = {"10m": 600, "1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}


def _first_flag(keys: list[np.ndarray]) -> np.ndarray:
    """1 where this row is the first (chronological) occurrence of the key tuple. Causal by construction."""
    return (~pd.DataFrame({i: k for i, k in enumerate(keys)}).duplicated(keep="first")).astype(np.float64).values


def build_features(df: pd.DataFrame, acc: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Compute all temporal features. `df` must be sorted by ts (ties by txn_id)."""
    assert df.ts.is_monotonic_increasing, "transactions must be time-sorted"
    n = len(df)
    src, dst, ts, amt = df.src.values, df.dst.values, df.ts.values, df.amount.values
    F: dict[str, np.ndarray] = {}
    groups: dict[str, list[str]] = {"txn": [], "behaviour": [], "beneficiary": [], "device": []}

    def add(name, arr, grp):
        F[name] = np.asarray(arr, dtype=np.float32)
        groups[grp].append(name)

    # ---- per-row causal flags -------------------------------------------------
    new_pair = _first_flag([src, dst])
    prior_to_dst = df.groupby(["src", "dst"]).cumcount().values.astype(float)
    dev_first = _first_flag([src, df.device_id.values])
    ip_first = _first_flag([src, df.ip_id.values])
    sender_prior = df.groupby("src").cumcount().values.astype(float)
    pct_bal = amt / np.where(df.balance_before.values > 0, df.balance_before.values, np.nan)
    hour = (ts % 86400) / 3600.0
    ang = 2 * np.pi * hour / 24
    first_dev_ts = df.groupby(["src", "device_id"]).ts.transform("min").values
    dev_acct_first = _first_flag([df.device_id.values, src])
    ip_acct_first = _first_flag([df.ip_id.values, src])
    recv_pair_first = new_pair  # receiver view: first time this sender pays this dst

    # ---- sender event index ----------------------------------------------------
    S = EventIndex(src, ts, dict(amt=amt, newben=new_pair, pct=np.nan_to_num(pct_bal, nan=0.0), sess=df.session_s.values,
                                 sin=np.sin(ang), cos=np.cos(ang), dage=df.dst_age_days.values, ones=np.ones(n)))
    for wn, w in WINDOWS.items():
        cnt = S.count(src, ts, w)
        s = S.sum("amt", src, ts, w)
        mean, std, _ = S.mean_std("amt", src, ts, w)
        add(f"snd_cnt_{wn}", cnt, "behaviour")
        add(f"snd_sum_{wn}", s, "behaviour")
        add(f"snd_mean_{wn}", np.nan_to_num(mean, nan=0.0), "behaviour")
        add(f"snd_std_{wn}", np.nan_to_num(std, nan=0.0), "behaviour")
        add(f"snd_max_{wn}", np.nan_to_num(S.max("amt", src, ts, w), nan=0.0), "behaviour")
        add(f"snd_newben_{wn}", S.sum("newben", src, ts, w), "behaviour")
    h_mean, h_std, h_cnt = S.mean_std("amt", src, ts, INF)
    h_max = S.max("amt", src, ts, INF)
    add("hist_cnt", h_cnt, "behaviour")
    add("amount", amt, "txn")
    add("log_amount", np.log1p(np.maximum(amt, 0)), "txn")
    add("amt_to_hist_mean", np.where(h_cnt > 0, amt / np.maximum(h_mean, 1.0), 0.0), "behaviour")
    add("amt_to_hist_max", np.where(h_cnt > 0, amt / np.maximum(h_max, 1.0), 0.0), "behaviour")
    add("amt_z", np.where(h_cnt > 1, (amt - h_mean) / np.maximum(h_std, 1.0), 0.0), "behaviour")
    add("balance_before", np.nan_to_num(df.balance_before.values, nan=0.0), "txn")
    add("pct_balance", np.nan_to_num(pct_bal, nan=0.0), "txn")
    hp_max = S.max("pct", src, ts, INF)
    add("pct_to_hist_max_pct", np.where(h_cnt > 0, np.nan_to_num(pct_bal, nan=0.0) / np.maximum(np.nan_to_num(hp_max, nan=0.0), 0.02), 0.0), "behaviour")
    gap = S.last_time_gap(src, ts)
    add("secs_since_last_txn", np.nan_to_num(gap, nan=-1.0), "behaviour")
    add("inactive_30d", (np.nan_to_num(gap, nan=0) > 30 * 86400).astype(float), "behaviour")
    # time-of-day deviation vs the customer's circular mean hour
    ms, _, c = S.mean_std("sin", src, ts, INF); mc, _, _ = S.mean_std("cos", src, ts, INF)
    mean_ang = np.arctan2(np.nan_to_num(ms), np.nan_to_num(mc))
    dev = np.abs(np.angle(np.exp(1j * (ang - mean_ang)))) * 24 / (2 * np.pi)
    add("hour", hour, "txn"); add("hour_dev", np.where(c > 2, dev, 0.0), "behaviour")
    add("dow", ((ts // 86400) % 7), "txn")
    sm, ss, sc = S.mean_std("sess", src, ts, INF)
    add("session_s", df.session_s.values, "txn")
    add("session_z", np.where(sc > 2, (df.session_s.values - sm) / np.maximum(ss, 5.0), 0.0), "behaviour")
    add("failed_ben_adds", df.failed_ben_adds.values, "txn")
    add("is_mobile", (df.channel.values == "mobile").astype(float), "txn")
    am, _, ac = S.mean_std("dage", src, ts, INF)
    add("avg_hist_ben_age_days", np.nan_to_num(am, nan=0.0), "behaviour")
    # recipient novelty
    add("new_beneficiary", new_pair, "beneficiary")
    add("prior_pays_to_dst", prior_to_dst, "beneficiary")
    add("dst_age_days", df.dst_age_days.values, "beneficiary")
    add("dst_age_lt30d", (df.dst_age_days.values < 30).astype(float), "beneficiary")
    add("cross_bank", df.cross_bank.values, "beneficiary")
    # device / ip
    add("new_device", dev_first * (sender_prior > 0), "device")
    add("device_age_days", (ts - first_dev_ts) / 86400, "device")
    add("new_ip", ip_first * (sender_prior > 0), "device")
    D = EventIndex(df.device_id.values, ts, dict(first=dev_acct_first))
    add("device_shared_accounts", D.sum("first", df.device_id.values, ts, INF), "device")
    I = EventIndex(df.ip_id.values, ts, dict(first=ip_acct_first))
    add("ip_shared_accounts", I.sum("first", df.ip_id.values, ts, INF), "device")

    # ---- beneficiary as receiver ------------------------------------------------
    R = EventIndex(dst, ts, dict(amt=amt, first=recv_pair_first, ones=np.ones(n)))
    for wn in ("1h", "24h", "7d"):
        w = WINDOWS[wn]
        add(f"dst_in_cnt_{wn}", R.count(dst, ts, w), "beneficiary")
        add(f"dst_in_sum_{wn}", R.sum("amt", dst, ts, w), "beneficiary")
        add(f"dst_fanin_new_{wn}", R.sum("first", dst, ts, w), "beneficiary")   # distinct new senders
    add("dst_fanin_total", R.sum("first", dst, ts, INF), "beneficiary")
    add("dst_in_cnt_total", R.count(dst, ts, INF), "beneficiary")
    # beneficiary as sender (outflow) -> pass-through, fan-out, holding time
    # lag since the sender's own last receipt, computed at each of its sends
    lag_in = R.last_time_gap(src, ts)
    SO = EventIndex(src, ts, dict(amt=amt, first=new_pair, lag=np.nan_to_num(lag_in, nan=np.nan)))
    for wn in ("1h", "24h", "7d"):
        w = WINDOWS[wn]
        out = SO.sum("amt", dst, ts, w); inn = R.sum("amt", dst, ts, w)
        add(f"dst_out_sum_{wn}", out, "beneficiary")
        add(f"dst_fanout_new_{wn}", SO.sum("first", dst, ts, w), "beneficiary")
        add(f"dst_pass_through_{wn}", np.where(inn > 0, out / np.maximum(inn, 1.0), 0.0), "beneficiary")
    add("dst_fanout_total", SO.sum("first", dst, ts, INF), "beneficiary")
    lm, _, lc = SO.mean_std("lag", dst, ts, INF)
    add("dst_avg_holding_s", np.where(lc > 0, np.nan_to_num(lm, nan=-1.0), -1.0), "beneficiary")
    add("dst_since_last_in_s", np.nan_to_num(R.last_time_gap(dst, ts), nan=-1.0), "beneficiary")
    add("dst_since_last_out_s", np.nan_to_num(SO.last_time_gap(dst, ts), nan=-1.0), "beneficiary")
    # rapid dispersion indicator: many distinct receivers within 24h relative to inflow events
    add("dst_rapid_dispersion", F["dst_fanout_new_24h"] / np.maximum(F["dst_in_cnt_24h"], 1.0), "beneficiary")

    out = pd.DataFrame(F, index=df.index)
    out.insert(0, "txn_id", df.txn_id.values)
    log.info("built %d temporal features for %d rows", out.shape[1] - 1, n)
    return out, groups
