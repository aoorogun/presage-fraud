"""Model Family C: customer-transaction sequence encoders (LSTM / GRU / Transformer).

For payment i by sender s at time t the input is the sender's previous n payments plus the current one:
    x_{t-n}, ..., x_{t-1}, x_t       (earlier steps are zero-padded with a mask)
Only payments strictly before i in (ts, txn_id) order are used for steps < t, so no future data enters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from presage.utils.common import get_device, set_seed

SEQ_FEATS = ["log_amount", "pct_balance", "log_gap", "hour_sin", "hour_cos", "new_ben", "is_mobile", "log_dst_age", "session_log", "failed", "new_device"]


def step_features(df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Per-payment step vectors (computed from that payment's own fields and its own causal features)."""
    h = (df.ts.values % 86400) / 3600.0
    gap = np.log1p(np.maximum(X["secs_since_last_txn"].values, 0))
    out = np.stack([X["log_amount"].values, np.clip(X["pct_balance"].values, 0, 1.5), gap, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                    X["new_beneficiary"].values, X["is_mobile"].values, np.log1p(np.maximum(df.dst_age_days.values, 0)),
                    np.log1p(df.session_s.values), np.minimum(X["failed_ben_adds"].values, 5), X["new_device"].values], 1).astype(np.float32)
    return out


def build_sequences(df: pd.DataFrame, steps: np.ndarray, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (seq [N, n+1, d], mask [N, n+1]) aligned with df rows. Last position is the current payment."""
    N, d = steps.shape
    order = np.lexsort((df.txn_id.values, df.ts.values, df.src.values))      # by sender then time
    src_sorted = df.src.values[order]
    start = np.searchsorted(src_sorted, src_sorted, side="left")
    pos = np.arange(N) - start                                              # rank within sender
    seq = np.zeros((N, n + 1, d), dtype=np.float32); mask = np.zeros((N, n + 1), dtype=bool)
    seq[order, n] = steps[order]; mask[order, n] = True
    for k in range(1, n + 1):
        ok = pos >= k
        rows = order[ok]; prev = order[np.arange(N)[ok] - k]
        seq[rows, n - k] = steps[prev]; mask[rows, n - k] = True
    return seq, mask


class SeqEncoder(nn.Module):
    def __init__(self, d_in: int, kind: str = "gru", h: int = 64, n: int = 11, out: int = 32):
        super().__init__()
        self.kind = kind
        if kind in ("gru", "lstm"):
            self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(d_in, h, batch_first=True, num_layers=1)
        else:
            self.inp = nn.Linear(d_in, h); self.pos = nn.Parameter(torch.randn(1, n, h) * 0.02)
            layer = nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.1, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
        self.proj = nn.Sequential(nn.Linear(h, out), nn.ReLU())

    def forward(self, x, mask):
        if self.kind in ("gru", "lstm"):
            o, _ = self.rnn(x * mask.unsqueeze(-1))
            return self.proj(o[:, -1])
        z = self.inp(x) + self.pos
        z = self.tf(z, src_key_padding_mask=~mask)
        return self.proj(z[:, -1])


class SeqClassifier(nn.Module):
    def __init__(self, d_in, kind="gru", n=11, tab_dim: int = 0, h=64):
        super().__init__()
        self.enc = SeqEncoder(d_in, kind, h, n)
        self.tab = nn.Sequential(nn.Linear(tab_dim, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU()) if tab_dim else None
        self.head = nn.Sequential(nn.Linear(32 + (32 if tab_dim else 0), 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask, tab=None):
        z = self.enc(x, mask)
        if self.tab is not None: z = torch.cat([z, self.tab(tab)], 1)
        return self.head(z).squeeze(-1)


def train_seq(kind, seq, mask, y, tr, va, tab=None, epochs=6, seed=42, bs=1024, lr=2e-3, pos_w=30.0, verbose=True):
    """Train on `tr`, early-stop on validation PR-AUC. Returns (model, predict_fn)."""
    from sklearn.metrics import average_precision_score
    set_seed(seed); dev = get_device()
    mdl = SeqClassifier(seq.shape[2], kind, seq.shape[1], tab.shape[1] if tab is not None else 0).to(dev)
    opt = torch.optim.AdamW(mdl.parameters(), lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))
    Xs, Ms, Ys = torch.tensor(seq), torch.tensor(mask), torch.tensor(y, dtype=torch.float32)
    Ts = torch.tensor(tab, dtype=torch.float32) if tab is not None else None
    tri = np.where(tr)[0]; vai = np.where(va)[0]

    def predict(idx):
        mdl.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 8192):
                b = idx[i:i + 8192]
                out.append(torch.sigmoid(mdl(Xs[b].to(dev), Ms[b].to(dev), Ts[b].to(dev) if Ts is not None else None)).cpu().numpy())
        return np.concatenate(out)

    best, best_state = -1, None
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        mdl.train(); perm = rng.permutation(tri)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            logit = mdl(Xs[b].to(dev), Ms[b].to(dev), Ts[b].to(dev) if Ts is not None else None)
            loss = lossf(logit, Ys[b].to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mdl.parameters(), 2.0); opt.step()
        ap = average_precision_score(y[vai], predict(vai))
        if verbose: print(f"  [{kind}] epoch {ep+1} val PR-AUC {ap:.3f}", flush=True)
        if ap > best: best, best_state = ap, {k: v.clone() for k, v in mdl.state_dict().items()}
    mdl.load_state_dict(best_state)
    return mdl, predict
