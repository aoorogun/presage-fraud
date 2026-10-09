"""Training / inference loop for PRESAGE (stateful over chronological graph snapshots)."""
from __future__ import annotations
import time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import average_precision_score
from presage.models.presage import Presage, PresageConfig
from presage.utils.common import set_seed

torch.set_num_threads(2)


def make_aux_labels(df, acc):
    """Task B: beneficiary is a mule account. Task C: beneficiary belongs to the fraud network
    (is a mule OR has directly transacted with a mule). Targets only; never used as input features."""
    is_mule = acc.is_mule.values.astype(bool)
    linked = np.zeros(len(acc), dtype=bool)
    m_src, m_dst = is_mule[df.src.values], is_mule[df.dst.values]
    linked[df.dst.values[m_src]] = True; linked[df.src.values[m_dst]] = True
    y_mule = is_mule[df.dst.values].astype(np.float32)
    y_net = (is_mule | linked)[df.dst.values].astype(np.float32)
    return y_mule, y_net


def _enable_dropout(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout): mod.train()


def fit_presage(cfg: PresageConfig, snaps, snap_of, A: dict, tr, va, te, ben_idx, epochs=10, seed=42, lr=2e-3, pos_w=20.0, neg_keep=0.4,
                mc_samples=20, verbose=True, label_mask_fn=None, cosine=False):
    """A holds arrays aligned with df rows: src,dst,tab,seq,mask,y_app,y_mule,y_net.
    `label_mask_fn(k, idx)` may return a boolean mask of rows whose label is *available* when training at snapshot k
    (used by the delayed-label experiments); default: all train rows are labelled."""
    set_seed(seed)
    mdl = Presage(cfg, snaps[0].x.shape[1], A["tab"].shape[1], A["seq"].shape[2], A["seq"].shape[1], ben_idx)
    opt = torch.optim.AdamW(mdl.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05) if cosine else None
    T, SEQ, MASK = torch.tensor(A["tab"]), torch.tensor(A["seq"]), torch.tensor(A["mask"])
    S, D = torch.tensor(A["src"]), torch.tensor(A["dst"])
    Ya, Ym, Yn = (torch.tensor(A[k]) for k in ("y_app", "y_mule", "y_net"))
    y = A["y_app"]
    ks = sorted(set(snap_of[tr | va | te])); k0, k1 = min(ks), max(ks)
    rows_by_k = {k: np.where(snap_of == k)[0] for k in range(k0, k1 + 1)}
    rng = np.random.default_rng(seed)
    lam = cfg.lam if cfg.multitask else (1.0, 0.0, 0.0)
    pw = torch.tensor(pos_w)

    def predict(mask, mc=False):
        out = {k: np.full(len(y), np.nan, dtype=np.float32) for k in ("app", "mule", "net", "unc")}
        mdl.eval(); mdl.graph.reset()
        with torch.no_grad():
            for k in range(k0, k1 + 1):
                idx = rows_by_k[k]; sel = idx[mask[idx]]
                Z = mdl.encode_graph(snaps[k])
                if not len(sel): continue
                args = (snaps[k], S[sel], D[sel], T[sel], SEQ[sel], MASK[sel], Z)
                o = mdl(*args)
                out["app"][sel] = torch.sigmoid(o["app"]).numpy(); out["net"][sel] = torch.sigmoid(o["net"]).numpy()
                if o["mule"] is not None: out["mule"][sel] = torch.sigmoid(o["mule"]).numpy()
                if mc:
                    _enable_dropout(mdl)
                    ps = torch.stack([torch.sigmoid(mdl(*args)["app"]) for _ in range(mc_samples)])
                    out["unc"][sel] = ps.std(0).numpy(); mdl.eval()
        return out

    best, best_state, t_train = -1, None, 0.0
    for ep in range(epochs):
        t0 = time.time(); mdl.train(); mdl.graph.reset(); tot = 0.0
        for k in range(k0, k1 + 1):
            idx = rows_by_k[k]; sel = idx[tr[idx]]
            if label_mask_fn is not None and len(sel): sel = sel[label_mask_fn(k, sel)]
            if len(sel) == 0:
                with torch.no_grad(): mdl.encode_graph(snaps[k])
                continue
            sel = sel[(y[sel] == 1) | (rng.random(len(sel)) < neg_keep)]
            Z = mdl.encode_graph(snaps[k])
            o = mdl(snaps[k], S[sel], D[sel], T[sel], SEQ[sel], MASK[sel], Z)
            w = torch.where(Ya[sel] == 1, 1.0, 1.0 / neg_keep)
            loss = lam[0] * (F.binary_cross_entropy_with_logits(o["app"], Ya[sel], reduction="none", pos_weight=pw) * w).mean()
            if cfg.multitask and o["mule"] is not None:
                loss = loss + lam[1] * (F.binary_cross_entropy_with_logits(o["mule"], Ym[sel], reduction="none", pos_weight=torch.tensor(5.0)) * w).mean()
            if cfg.multitask:
                loss = loss + lam[2] * (F.binary_cross_entropy_with_logits(o["net"], Yn[sel], reduction="none", pos_weight=torch.tensor(5.0)) * w).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mdl.parameters(), 2.0); opt.step(); tot += float(loss.detach())
        if sched is not None: sched.step()
        t_train += time.time() - t0
        pv = predict(va)
        ap = average_precision_score(y[va], pv["app"][va])
        if verbose: print(f"  [presage] ep{ep+1} loss {tot:.2f} val PR-AUC {ap:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if ap > best: best, best_state = ap, {k_: v.clone() for k_, v in mdl.state_dict().items()}
    mdl.load_state_dict(best_state)
    t0 = time.time(); pv = predict(va); pt = predict(te, mc=True); t_inf = (time.time() - t0) / max((va | te).sum(), 1)
    return mdl, pv, pt, dict(train_s=t_train, infer_s=t_inf, val_ap=best)
