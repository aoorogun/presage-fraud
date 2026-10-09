"""Temporal heterogeneous graph snapshots.

Node types : account (all customers, payees and mules; type is NOT given to models), device, ip
Edge types : account -pays-> account (attrs: count, value, recency, age)      [+ reverse]
             account -uses-> device                                          [+ reverse]
             account -from-> ip                                              [+ reverse]
Snapshot k contains only edges whose timestamp is strictly before cutoff c_k = k * step. A payment at time t in
[c_k, c_{k+1}) is therefore scored with a graph that cannot contain itself or any later edge.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from presage.utils.common import get_logger

log = get_logger("snapshots")
DAY = 86400.0


@dataclass
class Snapshot:
    k: int
    cutoff: float
    x: torch.Tensor               # [N_acc, d] account node features
    pay_ei: torch.Tensor          # [2, E] directed payer -> payee
    pay_ea: torch.Tensor          # [E, 4]
    dev_ei: torch.Tensor          # [2, Ed] account -> device
    ip_ei: torch.Tensor           # [2, Ei] account -> ip
    n_dev: int
    n_ip: int


def _unique_edges(a: np.ndarray, b: np.ndarray, n_b: int):
    key = a.astype(np.int64) * n_b + b
    u, first = np.unique(key, return_index=True)
    return u // n_b, u % n_b


def build_snapshots(df: pd.DataFrame, acc: pd.DataFrame, step_days: float = 4.0, label_delay_s: float | None = None,
                    use_labels: bool = True) -> tuple[list[Snapshot], np.ndarray]:
    n = len(acc)
    ts, src, dst, amt = df.ts.values, df.src.values, df.dst.values, df.amount.values
    dev_codes, dev_idx = np.unique(df.device_id.values, return_inverse=True); n_dev = len(dev_codes)
    ip_codes, ip_idx = np.unique(df.ip_id.values, return_inverse=True); n_ip = len(ip_codes)
    n_snap = int(np.ceil((ts.max() / DAY + 1) / step_days))
    cut_idx = np.searchsorted(ts, np.arange(n_snap) * step_days * DAY, side="left")
    bank = acc.bank.values; age0 = acc.created_ts.values
    bank_oh = np.eye(int(bank.max()) + 1, dtype=np.float32)[bank]
    # delayed confirmed-mule labels (same rule as features/network.py)
    fr = df.is_app_fraud.values == 1
    delay = df.report_delay_s.values if label_delay_s is None else np.full(len(df), float(label_delay_s))
    conf_time = np.full(n, np.inf)
    if use_labels:
        np.minimum.at(conf_time, dst[fr], ts[fr] + np.nan_to_num(delay[fr], nan=30 * DAY))
    snaps = []
    for k in range(n_snap):
        m = cut_idx[k]; c = k * step_days * DAY
        if m == 0:
            x = np.zeros((n, bank_oh.shape[1] + 6), dtype=np.float32); x[:, :bank_oh.shape[1]] = bank_oh
            snaps.append(Snapshot(k, c, torch.tensor(x), torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4), torch.zeros(2, 0, dtype=torch.long),
                                  torch.zeros(2, 0, dtype=torch.long), n_dev, n_ip)); continue
        s_, d_, t_, a_ = src[:m], dst[:m], ts[:m], amt[:m]
        key = s_.astype(np.int64) * n + d_
        order = np.argsort(key, kind="stable"); ks = key[order]
        uk, start = np.unique(ks, return_index=True)
        cnt = np.diff(np.append(start, len(ks))).astype(np.float64)
        val = np.add.reduceat(a_[order], start); last = np.maximum.reduceat(t_[order], start); first = np.minimum.reduceat(t_[order], start)
        pe = np.stack([uk // n, uk % n]); ea = np.stack([np.log1p(cnt), np.log1p(val), np.log1p((c - last) / DAY), np.log1p((c - first) / DAY)], 1).astype(np.float32)
        indeg = np.bincount(pe[1], minlength=n); outdeg = np.bincount(pe[0], minlength=n)
        win = np.bincount(pe[1], weights=val, minlength=n); wout = np.bincount(pe[0], weights=val, minlength=n)
        age = np.clip((c - age0) / DAY, 0, None)
        x = np.concatenate([bank_oh, np.stack([np.log1p(age), np.log1p(indeg), np.log1p(outdeg), np.log1p(win), np.log1p(wout),
                                               (conf_time <= c).astype(np.float32) if use_labels else np.zeros(n)], 1)], 1).astype(np.float32)
        da, db = _unique_edges(s_, dev_idx[:m], n_dev); ia, ib = _unique_edges(s_, ip_idx[:m], n_ip)
        snaps.append(Snapshot(k, c, torch.tensor(x), torch.tensor(pe, dtype=torch.long), torch.tensor(ea), torch.tensor(np.stack([da, db]), dtype=torch.long),
                              torch.tensor(np.stack([ia, ib]), dtype=torch.long), n_dev, n_ip))
    snap_of = np.minimum((ts // (step_days * DAY)).astype(int), n_snap - 1)
    log.info("built %d snapshots (%d accounts, %d devices, %d ips)", n_snap, n, n_dev, n_ip)
    return snaps, snap_of
