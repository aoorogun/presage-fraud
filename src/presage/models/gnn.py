"""Graph encoders (Model Families D, E, F-snapshot) on top of presage.graph.snapshots.Snapshot.

Every encoder maps a Snapshot -> account embeddings [N, h].
  static homogeneous : gcn, sage, gat                 (payment edges only, treated as undirected)
  heterogeneous      : rgcn, hgt, han                 (account/device/ip nodes, typed edges)
  temporal           : tgat (time-encoded attention), evolvegcn (GRU-evolved weights across snapshots)
"""
from __future__ import annotations
import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, RGCNConv, HGTConv, HANConv, TransformerConv


def _undirected(ei):
    return torch.cat([ei, ei.flip(0)], 1)


class _Static(nn.Module):
    def __init__(self, kind, d_in, h, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, h)
        mk = {"gcn": lambda: GCNConv(h, h), "sage": lambda: SAGEConv(h, h, aggr="mean"), "gat": lambda: GATv2Conv(h, h // 4, heads=4)}[kind]
        self.convs = nn.ModuleList([mk() for _ in range(layers)])

    def forward(self, s):
        x = F.relu(self.inp(s.x)); ei = _undirected(s.pay_ei)
        for c in self.convs: x = F.relu(c(x, ei)) + x
        return x


class _Hetero(nn.Module):
    """R-GCN / HGT / HAN over account, device, ip nodes."""

    def __init__(self, kind, d_in, h, layers=2):
        super().__init__()
        self.kind, self.h = kind, h
        self.inp = nn.Linear(d_in, h)
        self.dev0 = nn.Parameter(torch.zeros(1, h)); self.ip0 = nn.Parameter(torch.zeros(1, h))
        nn.init.normal_(self.dev0, std=0.1); nn.init.normal_(self.ip0, std=0.1)
        self.meta = (["account", "device", "ip"], [("account", "pays", "account"), ("account", "paid_by", "account"), ("account", "uses", "device"),
                                                   ("device", "used_by", "account"), ("account", "from", "ip"), ("ip", "of", "account")])
        if kind == "rgcn":
            self.convs = nn.ModuleList([RGCNConv(h, h, num_relations=6, num_bases=4) for _ in range(layers)])
        elif kind == "hgt":
            self.convs = nn.ModuleList([HGTConv(h, h, self.meta, heads=4) for _ in range(layers)])
        else:
            self.convs = nn.ModuleList([HANConv(h, h, self.meta, heads=4) for _ in range(layers)])

    def _hdata(self, s):
        return {("account", "pays", "account"): s.pay_ei, ("account", "paid_by", "account"): s.pay_ei.flip(0),
                ("account", "uses", "device"): s.dev_ei, ("device", "used_by", "account"): s.dev_ei.flip(0),
                ("account", "from", "ip"): s.ip_ei, ("ip", "of", "account"): s.ip_ei.flip(0)}

    def forward(self, s):
        n, nd, ni = s.x.shape[0], s.n_dev, s.n_ip
        xa = F.relu(self.inp(s.x)); xd = self.dev0.expand(nd, -1); xi = self.ip0.expand(ni, -1)
        if self.kind == "rgcn":
            x = torch.cat([xa, xd, xi], 0)
            ei = torch.cat([s.pay_ei, s.pay_ei.flip(0), torch.stack([s.dev_ei[0], s.dev_ei[1] + n]), torch.stack([s.dev_ei[1] + n, s.dev_ei[0]]),
                            torch.stack([s.ip_ei[0], s.ip_ei[1] + n + nd]), torch.stack([s.ip_ei[1] + n + nd, s.ip_ei[0]])], 1)
            et = torch.cat([torch.zeros(s.pay_ei.shape[1]), torch.ones(s.pay_ei.shape[1]), 2 * torch.ones(s.dev_ei.shape[1]), 3 * torch.ones(s.dev_ei.shape[1]),
                            4 * torch.ones(s.ip_ei.shape[1]), 5 * torch.ones(s.ip_ei.shape[1])]).long()
            for c in self.convs: x = F.relu(c(x, ei, et)) + x
            return x[:n]
        xd_ = {"account": xa, "device": xd, "ip": xi}
        ed = self._hdata(s)
        for c in self.convs:
            out = c(xd_, ed)
            xd_ = {k: (F.relu(out[k]) + xd_[k]) if out.get(k) is not None else xd_[k] for k in xd_}
        return xd_["account"]


class Time2Vec(nn.Module):
    def __init__(self, d=8):
        super().__init__(); self.w = nn.Parameter(torch.randn(d) * 0.5); self.b = nn.Parameter(torch.zeros(d))

    def forward(self, t):  # t: [E] (log-days)
        return torch.cos(t.unsqueeze(-1) * self.w + self.b)


class _TGAT(nn.Module):
    """Time-aware attention: edge attributes + Time2Vec encoding of edge recency feed a TransformerConv (TGAT-style)."""

    def __init__(self, d_in, h, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, h); self.t2v = Time2Vec(8)
        self.convs = nn.ModuleList([TransformerConv(h, h // 4, heads=4, edge_dim=4 + 8) for _ in range(layers)])

    def forward(self, s):
        x = F.relu(self.inp(s.x))
        ei = _undirected(s.pay_ei); ea = s.pay_ea
        ea = torch.cat([ea, ea], 0)
        ea = torch.cat([ea, self.t2v(ea[:, 2])], 1)
        for c in self.convs: x = F.relu(c(x, ei, ea)) + x
        return x


class _EvolveGCN(nn.Module):
    """EvolveGCN-O (Pareja et al., AAAI 2020): GCN weights are evolved by a GRU along the snapshot sequence."""

    def __init__(self, d_in, h, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, h); self.h = h
        self.W0 = nn.ParameterList([nn.Parameter(torch.eye(h) + 0.01 * torch.randn(h, h)) for _ in range(layers)])
        self.grus = nn.ModuleList([nn.GRUCell(h, h) for _ in range(layers)])
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, s):
        W = self.state if self.state is not None else [w for w in self.W0]
        x = F.relu(self.inp(s.x)); ei = _undirected(s.pay_ei); n = x.shape[0]
        newW = []
        for i, gru in enumerate(self.grus):
            Wi = gru(W[i].detach() if not self.training else W[i], W[i])   # rows of W are the GRU batch, input = previous weights (EvolveGCN-O)
            newW.append(Wi)
            deg = torch.zeros(n).index_add_(0, ei[1], torch.ones(ei.shape[1])).clamp(min=1)
            agg = torch.zeros_like(x).index_add_(0, ei[1], x[ei[0]]) / deg.unsqueeze(-1)
            x = F.relu((agg + x) @ Wi) + x
        self.state = [w.detach() for w in newW]
        return x


def make_encoder(kind: str, d_in: int, h: int = 64) -> nn.Module:
    if kind in ("gcn", "sage", "gat"): return _Static(kind, d_in, h)
    if kind in ("rgcn", "hgt", "han"): return _Hetero(kind, d_in, h)
    if kind == "tgat": return _TGAT(d_in, h)
    if kind == "evolvegcn": return _EvolveGCN(d_in, h)
    raise ValueError(kind)


class EdgeClassifier(nn.Module):
    """Payment-level scorer: [z_src, z_dst, z_src*z_dst, MLP(tabular)] -> logit. kind='none' gives the tabular-only twin
    (identical head, no graph), enabling a like-for-like measurement of graph contribution."""

    def __init__(self, kind: str, d_node: int, d_tab: int, h: int = 64):
        super().__init__()
        self.kind = kind
        self.enc = make_encoder(kind, d_node, h) if kind != "none" else None
        self.tab = nn.Sequential(nn.Linear(d_tab, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU())
        d = 64 + (3 * h if self.enc is not None else 0)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, snap, src, dst, tab, Z=None):
        parts = [self.tab(tab)]
        if self.enc is not None:
            Z = self.enc(snap) if Z is None else Z
            zs, zd = Z[src], Z[dst]
            parts += [zs, zd, zs * zd]
        return self.head(torch.cat(parts, 1)).squeeze(-1)
