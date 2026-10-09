"""Leakage-safe graph-derived features computed on rolling snapshots.

Design: the account-to-account payment graph is rebuilt every `step_days`. A payment in
[c_k, c_{k+1}) is described by node statistics computed from edges with ts < c_k only, so
graph context is stale by at most `step_days` and can never contain the row's own edge or
any later edge. Label-derived features (neighbouring *confirmed* mules) additionally honour
label delay: a mule is 'confirmed' at cutoff c only if a victim report has arrived by c
(payment ts + report delay <= c).

Features per endpoint (prefix `dg_` = beneficiary, `sg_` = sender):
  in/out unique degree, weighted degree (amount), pagerank (+delta), clustering coeff,
  component size, approx. betweenness, 2/3-hop forward path counts (dispersion / chain depth proxy),
  confirmed-mule neighbours at 1 and 2 hops.
Pair features: reciprocal edge, common neighbours.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from presage.utils.common import get_logger

log = get_logger("netfeat")
DAY = 86400.0


def _pagerank(A: sp.csr_matrix, d: float = 0.85, it: int = 30) -> np.ndarray:
    n = A.shape[0]
    out = np.asarray(A.sum(1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
    P = sp.diags(inv) @ A                       # row-stochastic
    r = np.full(n, 1.0 / n)
    for _ in range(it):
        r = (1 - d) / n + d * (P.T @ r + r[out == 0].sum() / n)
    return r * n                                  # normalised so the mean is 1


def _approx_betweenness(U: sp.csr_matrix, k: int = 24, seed: int = 0) -> np.ndarray:
    """Sampled Brandes betweenness (k random pivots) on the undirected graph."""
    import networkx as nx
    G = nx.from_scipy_sparse_array(U)
    b = nx.betweenness_centrality(G, k=min(k, G.number_of_nodes()), seed=seed, normalized=True)
    out = np.zeros(U.shape[0])
    for i, v in b.items(): out[i] = v
    return out * 1e3


def build_network_features(df: pd.DataFrame, acc: pd.DataFrame, step_days: float = 4.0, betweenness_every: int = 2,
                           label_delay_s: float | None = None, use_labels: bool = True) -> pd.DataFrame:
    """Return per-transaction graph features (indexed like df). `label_delay_s` overrides per-fraud report delays."""
    n_nodes = len(acc)
    ts = df.ts.values; src = df.src.values; dst = df.dst.values; amt = df.amount.values
    days = int(np.ceil(ts.max() / DAY)) + 1
    n_snap = int(np.ceil(days / step_days))
    # confirmed-mule label stream: fraud payment i makes its beneficiary 'known' at ts + delay
    fr = df.is_app_fraud.values == 1
    conf_time = np.full(n_nodes, np.inf)
    delay = df.report_delay_s.values if label_delay_s is None else np.full(len(df), float(label_delay_s))
    if use_labels:
        ct = ts[fr] + np.nan_to_num(delay[fr], nan=30 * DAY)
        np.minimum.at(conf_time, dst[fr], ct)
    cols = ["in_deg", "out_deg", "w_in", "w_out", "pr", "pr_delta", "clus", "comp", "btw", "paths2", "paths3", "conf1", "conf2", "self_conf"]
    node_feats = [np.zeros((n_nodes, len(cols)), dtype=np.float32) for _ in range(n_snap)]
    snap_edges_cut = []
    prev_pr = np.ones(n_nodes)
    prev_in = np.zeros(n_nodes)
    btw_cache = np.zeros(n_nodes)
    cut_idx = np.searchsorted(ts, np.arange(n_snap) * step_days * DAY, side="left")  # edges strictly before cutoff
    for k in range(1, n_snap):
        m = cut_idx[k]
        if m == 0: continue
        s_, d_, a_ = src[:m], dst[:m], amt[:m]
        A = sp.csr_matrix((np.ones(m), (s_, d_)), shape=(n_nodes, n_nodes)); A.sum_duplicates()
        W = sp.csr_matrix((a_, (s_, d_)), shape=(n_nodes, n_nodes))
        Ab = A.copy(); Ab.data[:] = 1.0
        U = ((Ab + Ab.T) > 0).astype(np.float32).tocsr(); U.setdiag(0); U.eliminate_zeros()
        deg = np.asarray(U.sum(1)).ravel()
        tri = np.asarray(U.multiply(U @ U).sum(1)).ravel()
        clus = np.divide(tri, deg * (deg - 1), out=np.zeros(n_nodes), where=deg > 1)
        _, lab = connected_components(U, directed=False)
        comp = np.bincount(lab)[lab]
        pr = _pagerank(Ab)
        if k % betweenness_every == 0 or k == 1:
            btw_cache = _approx_betweenness(U, seed=k)
        one = np.ones(n_nodes); p2 = Ab @ (Ab @ one); p3 = Ab @ p2
        conf = (conf_time <= k * step_days * DAY).astype(np.float64)
        c1 = U @ conf; c2 = U @ c1
        in_deg = np.asarray(Ab.sum(0)).ravel()
        f = node_feats[k]
        f[:, 0] = in_deg; f[:, 1] = np.asarray(Ab.sum(1)).ravel()
        f[:, 2] = np.asarray(W.sum(0)).ravel(); f[:, 3] = np.asarray(W.sum(1)).ravel()
        f[:, 4] = pr; f[:, 5] = pr - prev_pr; f[:, 6] = clus; f[:, 7] = comp; f[:, 8] = btw_cache
        f[:, 9] = np.log1p(p2); f[:, 10] = np.log1p(p3); f[:, 11] = c1; f[:, 12] = c2; f[:, 13] = conf
        prev_pr = pr
        snap_edges_cut.append((k, Ab, U))
    # map transactions to snapshots
    snap_of = np.minimum((ts // (step_days * DAY)).astype(int), n_snap - 1)
    out = {}
    for prefix, nodes in (("dg_", dst), ("sg_", src)):
        arr = np.zeros((len(df), len(cols)), dtype=np.float32)
        for k in range(n_snap):
            idx = np.where(snap_of == k)[0]
            if len(idx): arr[idx] = node_feats[k][nodes[idx]]
        for j, c in enumerate(cols): out[prefix + c] = arr[:, j]
    # pair features (reciprocity, common neighbours) from the snapshot graph
    recip = np.zeros(len(df), dtype=np.float32); common = np.zeros(len(df), dtype=np.float32)
    for k, Ab, U in snap_edges_cut:
        idx = np.where(snap_of == k)[0]
        if not len(idx): continue
        recip[idx] = np.asarray(Ab[dst[idx], src[idx]]).ravel()
        common[idx] = np.asarray(U[src[idx]].multiply(U[dst[idx]]).sum(1)).ravel()
    out["pair_reciprocal"] = recip; out["pair_common_nbrs"] = common
    res = pd.DataFrame(out, index=df.index); res.insert(0, "txn_id", df.txn_id.values)
    log.info("built %d network features (%d snapshots, step %.1fd)", res.shape[1] - 1, n_snap, step_days)
    return res
