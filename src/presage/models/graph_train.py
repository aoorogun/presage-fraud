"""Snapshot-wise training/evaluation of payment-level graph models (EdgeClassifier)."""
from __future__ import annotations
import time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import average_precision_score
from presage.models.gnn import EdgeClassifier
from presage.utils.common import set_seed

torch.set_num_threads(2)


def prep_tab(M: np.ndarray, tr_mask: np.ndarray) -> np.ndarray:
    """Signed-log transform then z-score with TRAIN statistics only (no leakage)."""
    Z = np.sign(M) * np.log1p(np.abs(M))
    mu, sd = Z[tr_mask].mean(0), Z[tr_mask].std(0) + 1e-6
    return np.clip((Z - mu) / sd, -6, 6).astype(np.float32)


def fit_edge_model(kind, snaps, snap_of, src, dst, tab, y, tr, va, te, epochs=8, seed=42, lr=2e-3, pos_w=20.0, h=64, neg_keep=0.4, verbose=True):
    """Train on snapshots containing train rows; model-select on validation PR-AUC; return val/test predictions + timings."""
    set_seed(seed)
    mdl = EdgeClassifier(kind, snaps[0].x.shape[1], tab.shape[1], h)
    opt = torch.optim.AdamW(mdl.parameters(), lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    T = torch.tensor(tab); Y = torch.tensor(y, dtype=torch.float32); S = torch.tensor(src); D = torch.tensor(dst)
    ks = sorted(set(snap_of[tr | va | te]))
    rows_by_k = {k: np.where(snap_of == k)[0] for k in ks}
    rng = np.random.default_rng(seed)
    temporal = kind == "evolvegcn"

    def run_eval(mask):
        mdl.eval(); out = np.full(len(y), np.nan, dtype=np.float32)
        if temporal: mdl.enc.reset()
        with torch.no_grad():
            for k in range(min(ks), max(ks) + 1):
                idx = rows_by_k.get(k, np.array([], dtype=int))
                Z = mdl.enc(snaps[k]) if (kind != "none" and (temporal or (len(idx) and mask[idx].any()))) else None
                sel = idx[mask[idx]] if len(idx) else idx
                if len(sel): out[sel] = torch.sigmoid(mdl(snaps[k], S[sel], D[sel], T[sel], Z)).numpy()
        return out

    best, best_state, t_train = -1, None, 0.0
    for ep in range(epochs):
        t0 = time.time(); mdl.train()
        if temporal: mdl.enc.reset()
        order = [k for k in range(min(ks), max(ks) + 1)]
        tot = 0.0
        for k in order:
            idx = rows_by_k.get(k, np.array([], dtype=int))
            sel = idx[tr[idx]] if len(idx) else idx
            if temporal and (len(sel) == 0):                       # advance the evolving state through quiet snapshots
                with torch.no_grad(): mdl.enc(snaps[k])
                continue
            if len(sel) == 0: continue
            keep = (y[sel] == 1) | (rng.random(len(sel)) < neg_keep)
            sel = sel[keep]
            logit = mdl(snaps[k], S[sel], D[sel], T[sel])
            w = torch.where(Y[sel] == 1, 1.0, 1.0 / neg_keep)       # undo negative subsampling
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, Y[sel], reduction="none", pos_weight=torch.tensor(pos_w)) * w).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mdl.parameters(), 2.0); opt.step(); tot += float(loss)
        t_train += time.time() - t0
        pv = run_eval(va)
        ap = average_precision_score(y[va], pv[va])
        if verbose: print(f"  [{kind}] ep{ep+1} loss {tot:.3f} val PR-AUC {ap:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if ap > best: best, best_state = ap, {k_: v.clone() for k_, v in mdl.state_dict().items()}
    mdl.load_state_dict(best_state)
    t0 = time.time(); pv = run_eval(va); pt = run_eval(te); t_inf = (time.time() - t0) / max((va | te).sum(), 1)
    return mdl, pv, pt, dict(train_s=t_train, infer_s=t_inf, val_ap=best)
