"""Horizontal federated learning simulation across banks: FedAvg, FedProx, personalised FL (local fine-tuning),
client-level differential privacy (clipped, noised updates with an RDP epsilon estimate) and a secure-aggregation simulation.

Honest scope: this is a SIMULATION in one process. Secure aggregation is demonstrated by pairwise additive masks that cancel in the
sum (no cryptographic key exchange). Differential privacy here is client-level with only a handful of clients, so the formal
guarantee is weak; epsilon values are reported for the privacy-utility trade-off, not as a deployment claim. Federated learning
by itself does NOT guarantee privacy (model updates can leak information).
"""
from __future__ import annotations
import copy, math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)


class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def signed_log(x): return np.sign(x) * np.log1p(np.abs(x))


def local_train(model, X, y, epochs=1, lr=2e-3, bs=512, pos_w=20.0, mu=0.0, global_params=None, seed=0):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float32); n = len(Xt); pw = torch.tensor(pos_w)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(model(Xt[b]), yt[b], pos_weight=pw)
            if mu > 0 and global_params is not None:                    # FedProx proximal term
                loss = loss + 0.5 * mu * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), global_params))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def _flat(m): return torch.cat([p.detach().flatten() for p in m.parameters()])


def _set_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


def secure_aggregate(updates: list[torch.Tensor], seed=0) -> torch.Tensor:
    """Pairwise-mask secure aggregation simulation: client i adds sum_j s_ij*mask_ij (s_ij=+1 if i<j else -1); masks cancel in the sum."""
    n = len(updates); rng = torch.Generator().manual_seed(seed); masked = [u.clone() for u in updates]
    for i in range(n):
        for j in range(i + 1, n):
            m = torch.randn(updates[0].shape, generator=rng) * 10.0
            masked[i] += m; masked[j] -= m
    return torch.stack(masked).sum(0)


def rdp_epsilon(sigma: float, rounds: int, delta: float = 1e-5) -> float:
    """Epsilon of `rounds` compositions of the Gaussian mechanism (sensitivity 1, noise multiplier sigma) via RDP -> (eps, delta)."""
    if sigma <= 0: return float("inf")
    best = float("inf")
    for a in np.concatenate([np.linspace(1.01, 10, 60), np.arange(11, 256)]):
        eps = rounds * a / (2 * sigma ** 2) + math.log(1 / delta) / (a - 1)
        best = min(best, eps)
    return float(best)


def fed_train(clients: list[dict], d_in: int, rounds=15, local_epochs=1, algo="fedavg", mu=0.01, dp_sigma=0.0, clip=2.0, secure=False, seed=0,
              eval_fn=None, personalise_epochs=0):
    """clients: [{'X','y'}...]. Returns (global_model, personalised_models or None, log dict with bytes communicated and eps)."""
    torch.manual_seed(seed); np.random.seed(seed)
    gm = MLP(d_in); sizes = np.array([len(c["y"]) for c in clients], dtype=float); wts = sizes / sizes.sum()
    bytes_up = 0; nparam = _flat(gm).numel(); hist = []
    for r in range(rounds):
        gv = _flat(gm); gparams = [p.detach().clone() for p in gm.parameters()]; deltas = []
        for ci, c in enumerate(clients):
            lm = copy.deepcopy(gm)
            local_train(lm, c["X"], c["y"], local_epochs, mu=mu if algo == "fedprox" else 0.0, global_params=gparams if algo == "fedprox" else None, seed=seed + r * 17 + ci)
            d = _flat(lm) - gv
            if dp_sigma > 0:
                d = d * min(1.0, clip / (d.norm().item() + 1e-12))                                  # clip update to norm <= clip
            deltas.append(d); bytes_up += nparam * 4
        if secure:
            agg = secure_aggregate([w * d for w, d in zip(wts, deltas)], seed=seed + r)               # server only sees the masked sum
        else:
            agg = sum(w * d for w, d in zip(wts, deltas))
        if dp_sigma > 0:
            agg = agg + torch.randn_like(agg) * dp_sigma * clip / len(clients) * float(np.max(wts) * len(clients))   # noise calibrated to max weighted contribution
        _set_flat(gm, gv + agg)
        if eval_fn is not None: hist.append(eval_fn(gm))
    pers = None
    if personalise_epochs:
        pers = []
        for ci, c in enumerate(clients):
            lm = copy.deepcopy(gm); local_train(lm, c["X"], c["y"], personalise_epochs, lr=1e-3, seed=seed + 999 + ci); pers.append(lm)
    info = dict(bytes_up=bytes_up, bytes_down=bytes_up, eps=rdp_epsilon(dp_sigma, rounds) if dp_sigma > 0 else float("inf"), hist=hist)
    return gm, pers, info


@torch.no_grad()
def predict(model, X, bs=16384):
    model.eval(); out = []
    for i in range(0, len(X), bs): out.append(torch.sigmoid(model(torch.tensor(X[i:i + bs]))).numpy())
    return np.concatenate(out)
