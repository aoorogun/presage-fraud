"""PRESAGE: multi-stream fusion model for pre-payment APP fraud and mule detection.

Streams
  z_txn  : MLP over leakage-safe transaction / velocity / beneficiary-novelty features
  z_beh  : sequence encoder (GRU or Transformer) over the sender's previous payments
  z_graph: temporal heterogeneous graph encoder (account/device/ip; HGT over recency-split payment relations
           + per-account GRU memory carried across snapshots) -> [z_src, z_dst, z_src*z_dst]
  z_mule : mule-risk module on the beneficiary's graph embedding + beneficiary features
Fusion  : concat | gated | attention (token attention over the four streams; attention is NOT used as an explanation)
Heads   : APP-fraud logit (task A), beneficiary-mule logit (B), fraud-network membership logit (C),
          epistemic uncertainty (D) via MC-dropout at inference.
Each stream can be switched off for ablations (its embedding becomes zeros).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import HGTConv


@dataclass
class PresageConfig:
    h: int = 64
    use_txn: bool = True
    use_beh: bool = True
    use_graph: bool = True
    use_mule: bool = True
    temporal: bool = True            # recency-split relations + GRU memory across snapshots
    multitask: bool = True
    fusion: str = "gated"            # concat | gated | attention
    seq_kind: str = "gru"            # gru | transformer
    dropout: float = 0.15
    lam: tuple = (1.0, 0.5, 0.3)     # task weights (A, B, C)
    recent_days: float = 7.0


class TemporalHeteroEncoder(nn.Module):
    """HGT over account/device/ip with recency-aware relations and a node-level GRU memory over snapshots."""

    def __init__(self, d_in: int, h: int, temporal: bool, recent_days: float = 7.0, layers: int = 2):
        super().__init__()
        self.temporal, self.h, self.recent = temporal, h, recent_days
        self.inp = nn.Linear(d_in, h)
        self.dev0 = nn.Parameter(torch.randn(1, h) * 0.1); self.ip0 = nn.Parameter(torch.randn(1, h) * 0.1)
        pay = [("account", "pays_recent", "account"), ("account", "paid_by_recent", "account"), ("account", "pays_old", "account"), ("account", "paid_by_old", "account")] \
            if temporal else [("account", "pays", "account"), ("account", "paid_by", "account")]
        rel = pay + [("account", "uses", "device"), ("device", "used_by", "account"), ("account", "from", "ip"), ("ip", "of", "account")]
        self.meta = (["account", "device", "ip"], rel)
        self.convs = nn.ModuleList([HGTConv(h, h, self.meta, heads=4) for _ in range(layers)])
        self.gru = nn.GRUCell(h, h) if temporal else None
        self.state = None

    def reset(self):
        self.state = None

    def _edges(self, s):
        if self.temporal:
            rec = s.pay_ea[:, 2] <= torch.log1p(torch.tensor(self.recent))     # attr[2] = log1p(days since last payment on that edge)
            r, o = s.pay_ei[:, rec], s.pay_ei[:, ~rec]
            pay = {("account", "pays_recent", "account"): r, ("account", "paid_by_recent", "account"): r.flip(0),
                   ("account", "pays_old", "account"): o, ("account", "paid_by_old", "account"): o.flip(0)}
        else:
            pay = {("account", "pays", "account"): s.pay_ei, ("account", "paid_by", "account"): s.pay_ei.flip(0)}
        pay.update({("account", "uses", "device"): s.dev_ei, ("device", "used_by", "account"): s.dev_ei.flip(0),
                    ("account", "from", "ip"): s.ip_ei, ("ip", "of", "account"): s.ip_ei.flip(0)})
        return pay

    def forward(self, s):
        x = {"account": F.relu(self.inp(s.x)), "device": self.dev0.expand(s.n_dev, -1), "ip": self.ip0.expand(s.n_ip, -1)}
        ed = self._edges(s)
        for c in self.convs:
            out = c(x, ed)
            x = {k: (F.relu(out[k]) + x[k]) if out.get(k) is not None else x[k] for k in x}
        z = x["account"]
        if self.temporal:
            prev = self.state if self.state is not None else torch.zeros_like(z)
            z = self.gru(z, prev)
            self.state = z.detach()
        return z


class Fusion(nn.Module):
    def __init__(self, kind: str, h: int, n_streams: int = 4, p: float = 0.15):
        super().__init__()
        self.kind = kind
        if kind == "concat":
            self.out = nn.Sequential(nn.Linear(n_streams * h, 2 * h), nn.ReLU(), nn.Dropout(p), nn.Linear(2 * h, h), nn.ReLU())
        elif kind == "gated":
            self.gate = nn.Linear(n_streams * h, n_streams); self.proj = nn.Linear(h, h)
            self.out = nn.Sequential(nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU())
        else:
            self.q = nn.Parameter(torch.randn(1, 1, h) * 0.1); self.k = nn.Linear(h, h); self.v = nn.Linear(h, h)
            self.out = nn.Sequential(nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU())

    def forward(self, zs: list[torch.Tensor]):
        if self.kind == "concat": return self.out(torch.cat(zs, 1)), None
        if self.kind == "gated":
            g = torch.softmax(self.gate(torch.cat(zs, 1)), 1)                                  # [B, S] stream importances
            z = sum(g[:, i:i + 1] * self.proj(zi) for i, zi in enumerate(zs))
            return self.out(z), g
        T = torch.stack(zs, 1)                                                                 # [B, S, h]
        att = torch.softmax((self.q * self.k(T)).sum(-1) / T.shape[-1] ** 0.5, 1)
        return self.out((att.unsqueeze(-1) * self.v(T)).sum(1)), att


class Presage(nn.Module):
    def __init__(self, cfg: PresageConfig, d_node: int, d_tab: int, d_seq: int, n_seq: int, ben_idx: list[int]):
        super().__init__()
        from presage.models.sequence import SeqEncoder
        self.cfg = cfg; h = cfg.h
        self.register_buffer("ben_idx", torch.tensor(ben_idx, dtype=torch.long))
        self.tab = nn.Sequential(nn.Linear(d_tab, 128), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(128, h), nn.ReLU())
        self.seq = SeqEncoder(d_seq, cfg.seq_kind, 64, n_seq, out=h)
        self.graph = TemporalHeteroEncoder(d_node, h, cfg.temporal, cfg.recent_days)
        self.gproj = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU())
        self.mule_net = nn.Sequential(nn.Linear(h + len(ben_idx), h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, h), nn.ReLU())
        self.mule_head = nn.Linear(h, 1)
        self.fusion = Fusion(cfg.fusion, h, 4, cfg.dropout)
        self.head_app = nn.Linear(h, 1); self.head_net = nn.Linear(h, 1)

    def encode_graph(self, snap):
        return self.graph(snap) if self.cfg.use_graph else None

    def forward(self, snap, src, dst, tab, seq, mask, Z=None):
        c = self.cfg; B = tab.shape[0]; h = c.h
        zero = torch.zeros(B, h)
        z_txn = self.tab(tab) if c.use_txn else zero
        z_beh = self.seq(seq, mask) if c.use_beh else zero
        z_graph, z_mule, mule_logit = zero, zero, None
        if c.use_graph:
            Z = self.graph(snap) if Z is None else Z
            zs, zd = Z[src], Z[dst]
            z_graph = self.gproj(torch.cat([zs, zd, zs * zd], 1))
            if c.use_mule:
                z_mule = self.mule_net(torch.cat([zd, tab[:, self.ben_idx]], 1)); mule_logit = self.mule_head(z_mule).squeeze(-1)
        elif c.use_mule:   # mule module without a graph: beneficiary features only
            z_mule = self.mule_net(torch.cat([zero, tab[:, self.ben_idx]], 1)); mule_logit = self.mule_head(z_mule).squeeze(-1)
        z, w = self.fusion([z_txn, z_beh, z_graph, z_mule])
        return dict(app=self.head_app(z).squeeze(-1), mule=mule_logit, net=self.head_net(z).squeeze(-1), weights=w)
