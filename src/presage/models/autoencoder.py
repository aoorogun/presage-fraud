"""Autoencoder / VAE anomaly scorers (reconstruction error). Trained only on training-period rows."""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn
from presage.utils.common import get_device, set_seed


class _AE(nn.Module):
    def __init__(self, d: int, h: int = 32, z: int = 8, variational: bool = False):
        super().__init__()
        self.var = variational
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, h), nn.ReLU())
        self.mu = nn.Linear(h, z); self.lv = nn.Linear(h, z) if variational else None
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        h = self.enc(x); mu = self.mu(h)
        if self.var:
            lv = self.lv(h); zz = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            return self.dec(zz), mu, lv
        return self.dec(mu), mu, None


class _Scorer:
    def __init__(self, net, dev): self.net, self.dev = net.eval(), dev

    @torch.no_grad()
    def score(self, Z: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(Z), 8192):
            x = torch.tensor(Z[i:i + 8192], dtype=torch.float32, device=self.dev)
            r, *_ = self.net(x)
            out.append(((r - x) ** 2).mean(1).cpu().numpy())
        return np.concatenate(out)


def fit_autoencoder(Z: np.ndarray, variational: bool = False, seed: int = 42, epochs: int = 25) -> _Scorer:
    set_seed(seed); dev = get_device()
    net = _AE(Z.shape[1], variational=variational).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    X = torch.tensor(Z, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 512):
            x = X[perm[i:i + 512]]
            r, mu, lv = net(x)
            loss = ((r - x) ** 2).mean()
            if variational: loss = loss + 1e-3 * (-0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp()))
            opt.zero_grad(); loss.backward(); opt.step()
    return _Scorer(net, dev)
