"""Detailed architecture figure for PRESAGE (drawn from the implementation in src/presage/models/presage.py,
presage_train.py, graph/snapshots.py, features/sequence step features and evaluation/policy.py).
Panels: (a) leakage-safe temporal protocol, (b) PRESAGE forward pass, (c) temporal heterogeneous graph encoder,
(d) fusion variants, (e) objective and cost-optimised decision."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch
from matplotlib.colors import to_rgb
from figstyle import C, INK, MUTED, W2, save

TXN, SEQ, GRA, MUL = C["blue"], C["aqua"], C["violet"], C["orange"]
FUS, HEAD, POL, GREY = "#444b54", C["red"], C["yellow"], "#8a9099"


def tint(c, a):
    r = np.array(to_rgb(c)); return tuple(1 - a * (1 - r))


def draw_architecture(name="fig11_presage_architecture"):
    fig, ax = plt.subplots(figsize=(W2, W2 * 1.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 150); ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, lines=(), c=GREY, tfs=6.4, fs=5.6, fill=0.10, tc=None, ha="center", lw=0.8, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=tint(c, fill), ec=c, lw=lw, ls=ls, zorder=2))
        tx = x + w / 2 if ha == "center" else x + 1.2
        if title: ax.text(tx, y + h - 1.0, title, ha=ha, va="top", fontsize=tfs, fontweight="bold", color=tc or tint(c, 1.0) if tc else INK, zorder=3)
        off = 3.2 if title else 1.2
        for i, l in enumerate(lines): ax.text(tx, y + h - off - 1.62 * i, l, ha=ha, va="top", fontsize=fs, color=INK, zorder=3)

    def arrow(pts, c=GREY, lw=0.8, ls="-", head=True, z=1):
        pts = np.array(pts, float)
        if len(pts) > 2: ax.plot(pts[:-1, 0], pts[:-1, 1], color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2] if len(pts) > 2 else pts[0], arrowprops=dict(arrowstyle="-|>" if head else "-", lw=lw, color=c, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=6), zorder=z)

    def vec(x, y, c, n=6, h=1.35, w=3.0, label=None, seed=0, horizontal=False):
        rng = np.random.default_rng(seed); a = 0.25 + 0.7 * rng.random(n)
        for i in range(n):
            ax.add_patch(Rectangle((x + (i * w / n if horizontal else 0), y - (0 if horizontal else i * h)), (w / n if horizontal else w), h, fc=tint(c, a[i]), ec="white", lw=0.3, zorder=3))
        if label: ax.text(x + w / 2, y + 0.9 if horizontal else y + 0.9, label, ha="center", va="bottom", fontsize=5.8, color=INK, zorder=3)

    def pan(x, y, s, t): 
        ax.text(x, y, s, fontsize=10, fontweight="bold", va="top", ha="left"); ax.text(x + 3.4, y - 0.7, t, fontsize=7, fontweight="bold", va="top", ha="left", color=INK)

    # ============================================================ (a) leakage-safe temporal protocol
    pan(0.5, 149.5, "a", "Leakage-safe temporal protocol")
    X = lambda d: 6 + d * (88 / 120.0); y0 = 137.0
    ax.add_patch(Rectangle((X(0), y0 - 0.3), X(80) - X(0), 5.2, fc=tint(GRA, 0.10), ec="none", zorder=0))
    ax.text(X(40), y0 + 3.9, r"snapshot $G_k$: only edges with timestamp $< c_k$", ha="center", fontsize=5.8, color=GRA, va="center")
    ax.plot([X(0), X(120)], [y0, y0], color=INK, lw=0.9, zorder=2)
    for d in range(0, 121, 4): ax.plot([X(d), X(d)], [y0 - (0.9 if d in (80, 84) else 0.5), y0 + 0.5], color=GRA if d in (80, 84) else GREY, lw=1.0 if d in (80, 84) else 0.4, zorder=2)
    ax.text(X(80) - 0.4, y0 - 1.5, r"$c_k$", ha="right", va="top", fontsize=6.4, color=GRA, fontweight="bold"); ax.text(X(84) + 0.4, y0 - 1.5, r"$c_{k+1}$", ha="left", va="top", fontsize=6.4, color=GRA, fontweight="bold")
    ax.text(X(0), y0 + 0.9, "day 0", fontsize=5.6, color=MUTED, ha="left", va="bottom"); ax.text(X(120), y0 + 0.9, "120", fontsize=5.6, color=MUTED, ha="right", va="bottom")
    ax.text(X(100), y0 + 0.9, "snapshot cutoffs every 4 days", fontsize=5.6, color=MUTED, ha="center", va="bottom")
    ax.scatter([X(82)], [y0], s=22, color=MUL, zorder=4, ec="white", lw=0.5)
    arrow([(X(82), y0 + 1.0), (X(82), y0 + 5.2), (X(80) - 0.8, y0 + 5.2)], c=MUL, lw=0.7)
    ax.text(X(82) + 1.0, y0 + 4.6, r"payment at $t\in[c_k,c_{k+1})$ is scored with $G_k$", fontsize=5.6, color=MUL, va="center", ha="left")
    # label delay lanes
    def lane(y, d0, dr, ok, txt):
        ax.scatter([X(d0)], [y], marker="x", s=14, color=C["red"], lw=1.0, zorder=4); ax.scatter([X(dr)], [y], s=16, fc="white", ec=C["red"], lw=0.9, zorder=4)
        ax.plot([X(d0), X(dr)], [y, y], color=C["red"], lw=0.7, ls=(0, (2, 1.5)), zorder=3)
        cut = int(np.ceil(dr / 4.0) * 4); ax.plot([X(cut), X(cut)], [y - 0.9, y + 0.9], color=GRA if ok else MUTED, lw=1.2, zorder=3)
        ax.text(X(cut) + 1.2, y, txt, fontsize=5.5, color=INK if ok else MUTED, va="center", ha="left")
    lane(132.9, 56, 63, True, r"report $d$ days later; flag enters $G_k$ at first cutoff $\geq t_0+d$")
    lane(130.2, 74, 83, False, r"flag not yet visible in $G_k$ (label delay respected)")
    ax.scatter([X(5)], [132.9], marker="x", s=14, color=C["red"], lw=1.0); ax.text(X(5) + 1.5, 132.9, r"scam payment $t_0$", fontsize=5.5, va="center")
    ax.scatter([X(5)], [130.2], s=16, fc="white", ec=C["red"], lw=0.9); ax.text(X(5) + 1.5, 130.2, "victim report (delayed)", fontsize=5.5, va="center")
    for (a_, b_, col, t) in [(0, 72, TXN, "train, days 0 to 71"), (72, 96, SEQ, "validation, 72 to 95"), (96, 120, MUL, "test, 96 to 119")]:
        ax.add_patch(Rectangle((X(a_), 126.4), X(b_) - X(a_), 2.4, fc=tint(col, 0.35), ec="white", lw=0.6)); ax.text((X(a_) + X(b_)) / 2, 127.6, t, ha="center", va="center", fontsize=5.6, color=INK)
    ax.plot([X(100)] * 2, [125.9, 129.3], color=C["red"], lw=1.0); ax.text(X(100) + 0.8, 125.1, "sudden drift (day 100)", fontsize=5.4, color=C["red"], va="top")
    ax.text(X(0), 125.1, "chronological split; no resampling", fontsize=5.4, color=MUTED, va="top")

    # ============================================================ (b) forward pass
    pan(0.5, 123.5, "b", "PRESAGE forward pass for one payment (hidden size 64)")
    box(1, 107.5, 27, 12.5, "Payment features (90)", ["transaction 9, behaviour 46,", "beneficiary 30, device 5", "signed-log, z-score (train stats)"], TXN)
    box(1, 91.5, 27, 13.0, "Sender history", ["last 10 payments + current,", "11 step features each: log amount,", "balance share, gap, hour (sin, cos),", "new payee, mobile, payee age, ..."], SEQ)
    box(1, 71.5, 27, 16.0, "Graph snapshot $G_k$", ["account, device, IP nodes", "payment edges split by recency", "delayed confirmed-mule flag as", "a node feature (label delay kept)"], GRA)
    box(1, 56.0, 27, 12.5, "Beneficiary profile (30)", ["age, fan-in, pass-through,", "holding time (subset of the 90)"], MUL)
    box(33, 107.5, 28, 12.5, "Transaction MLP", ["Linear 90 to 128, ReLU, dropout", "Linear 128 to 64, ReLU"], TXN)
    box(33, 91.5, 28, 13.0, "Sequence encoder", ["GRU (1 layer, 64 units)", "last state, Linear 64, ReLU", "(Transformer variant in ablation)"], SEQ)
    box(33, 71.5, 28, 16.0, "Temporal graph encoder", ["HGT with recency-split relations", "and per-account GRU memory", "(detail in panel c)", r"output $Z_k\in\mathbb{R}^{N\times 64}$"], GRA)
    box(33, 56.0, 28, 12.5, "Mule module", [r"MLP on $[z_{dst},\,$beneficiary feats$]$", "Linear to 64, ReLU, dropout, Linear 64"], MUL)
    for yy in (113.5, 98.0, 79.5, 62.0): arrow([(28, yy), (33, yy)], c=GREY)
    arrow([(28, 79.5), (28, 79.5)], c=GREY, head=False)
    # destination row of Z_k feeds mule module
    arrow([(47, 71.5), (47, 68.5)], c=GRA, lw=0.8)
    ax.text(48.0, 70.0, r"$z_{dst}$", fontsize=5.4, color=GRA, va="center")
    # gather
    box(63.0, 71.5, 5.0, 16.0, "", [], GRA, fill=0.04, lw=0.5, ls="--")
    ax.text(65.5, 79.5, "gather src, dst", ha="center", va="center", fontsize=5.0, rotation=90, color=GRA)
    arrow([(61, 79.5), (63.0, 79.5)], c=GRA)
    arrow([(68.0, 79.5), (70.0, 79.5)], c=GRA)
    vec(63.0, 119.0, TXN, label=r"$z_{txn}$", seed=1); vec(63.0, 103.0, SEQ, label=r"$z_{beh}$", seed=2); vec(70.2, 84.5, GRA, label=r"$z_{graph}$", seed=3); vec(63.0, 66.3, MUL, label=r"$z_{mule}$", seed=4)
    arrow([(61, 113.5), (63, 113.5)], c=TXN); arrow([(61, 98.0), (63, 98.0)], c=SEQ); arrow([(61, 62.0), (63, 62.0)], c=MUL)
    # fusion
    box(76.5, 74.0, 8.0, 46.0, "Fusion", [], FUS, fill=0.12, tc=FUS)
    ax.text(80.5, 97.0, "concat\ngated\nattention\n\n4 x 64\nto 64", ha="center", va="center", fontsize=5.4, color=INK, zorder=3)
    arrow([(66.2, 114.0), (72.0, 114.0), (72.0, 110.0), (76.5, 110.0)], c=TXN)
    arrow([(66.2, 98.5), (76.5, 98.5)], c=SEQ)
    arrow([(73.5, 79.5), (74.8, 79.5), (74.8, 84.0), (76.5, 84.0)], c=GRA)
    # mule branch: to fusion and to head B
    arrow([(66.2, 60.5), (80.5, 60.5), (80.5, 74.0)], c=MUL)
    ax.plot([72.0, 72.0], [60.5, 66.0], color=MUL, lw=0.8, zorder=1)
    arrow([(72.0, 66.0), (88.5, 66.0)], c=MUL)
    # heads
    box(88.5, 111.0, 11, 11.0, "Head A", ["APP fraud", r"logit $p_A$"], HEAD)
    box(88.5, 95.0, 11, 11.0, "MC dropout", ["20 passes at", r"inference: $u=\mathrm{sd}(p_A)$"], C["magenta"], fs=5.0)
    box(88.5, 79.0, 11, 11.0, "Head C", ["fraud-network", "member"], HEAD)
    box(88.5, 60.0, 11, 11.0, "Head B", ["beneficiary", "is a mule"], HEAD)
    arrow([(84.5, 116.5), (88.5, 116.5)], c=FUS); arrow([(84.5, 84.5), (88.5, 84.5)], c=FUS)
    arrow([(94.0, 111.0), (94.0, 106.0)], c=C["magenta"], lw=0.7)
    ax.text(0.5, 54.2, r"$z_{graph}=\mathrm{ReLU}(W[z_s,\,z_d,\,z_s\odot z_d])$ for payer $s$ and payee $d$. Heads B and C use labels only as training targets, never as inputs.", fontsize=5.2, color=MUTED, va="top")

    # ============================================================ (c) temporal heterogeneous graph encoder
    pan(0.5, 51.3, "c", "Temporal heterogeneous graph encoder (per snapshot)")
    acc = [(5, 40.0), (10, 43.5), (15, 40.5), (8, 34.5), (14, 35.5), (19, 38.0)]; dev = [(4, 30.0), (13, 29.5)]; ip = [(20, 31.5)]
    def e(p, q, col, ls, lw): ax.plot([p[0], q[0]], [p[1], q[1]], color=col, ls=ls, lw=lw, zorder=1)
    for p, q in [(0, 1), (1, 2), (3, 4), (0, 3), (2, 5), (4, 5)]: e(acc[p], acc[q], GRA, "-", 1.5)
    for p, q in [(1, 3), (2, 4), (0, 4)]: e(acc[p], acc[q], GREY, (0, (2, 1.5)), 0.7)
    e(acc[3], dev[0], C["aqua"], "-", 0.9); e(acc[4], dev[1], C["aqua"], "-", 0.9); e(acc[5], ip[0], C["orange"], "-", 0.9); e(acc[2], ip[0], C["orange"], "-", 0.9)
    for p in acc: ax.add_patch(Circle(p, 1.15, fc=GRA, ec="white", lw=0.5, zorder=3))
    for p in dev: ax.add_patch(Rectangle((p[0] - 1.1, p[1] - 1.1), 2.2, 2.2, fc=C["aqua"], ec="white", lw=0.5, zorder=3))
    for p in ip: ax.add_patch(Polygon([(p[0] - 1.3, p[1] - 1.0), (p[0] + 1.3, p[1] - 1.0), (p[0], p[1] + 1.4)], fc=C["orange"], ec="white", lw=0.5, zorder=3))
    ax.text(2.0, 45.2, "account", fontsize=5.3, color=GRA); ax.text(2.0, 27.9, "device", fontsize=5.3, color=C["aqua"], va="top"); ax.text(17.5, 27.9, "IP", fontsize=5.3, color=C["orange"], va="top")
    box(25.0, 34.2, 33.5, 12.2, "Relations (8)", [], GRA, fill=0.05)
    ax.text(25.5, 31.2, "node shapes: circle = account, square = device, triangle = IP", fontsize=5.0, color=MUTED, va="center")
    items = [(GRA, "-", 1.6, "pays / paid_by, last 7 days"), (GREY, (0, (2, 1.5)), 0.8, "pays / paid_by, older"), (C["aqua"], "-", 0.9, "uses / used_by (account, device)"), (C["orange"], "-", 0.9, "from / of (account, IP)")]
    for i, (col, ls, lw, t) in enumerate(items):
        yy = 41.9 - 2.0 * i; ax.plot([26.2, 29.5], [yy, yy], color=col, ls=ls, lw=lw); ax.text(30.5, yy, t, fontsize=5.4, va="center")
    box(1, 17.5, 57.5, 9.0, "Node inputs", [r"account (11): bank one-hot (5), log age, log in/out degree, log in/out value,", "delayed confirmed-mule flag.  Device and IP: one learned type vector each."], GRA, fill=0.05, ha="left", fs=5.3)
    arrow([(29.5, 17.5), (29.5, 15.8)], c=GRA)
    box(1, 7.0, 57.5, 8.8, "2 x HGTConv (4 heads, hidden 64)", ["ReLU and residual per node type; attention over relation-specific edges;", "recency split separates fresh from stale payment links."], GRA, fill=0.10, ha="left", fs=5.3)
    arrow([(29.5, 7.0), (29.5, 5.3)], c=GRA)
    box(1, 0.3, 57.5, 5.0, "", [], GRA, fill=0.18)
    ax.text(29.7, 2.8, r"GRUCell memory:  $Z_k=\mathrm{GRU}(H_k,\,Z_{k-1})$;  state detached and carried to $G_{k+1}$", fontsize=5.4, ha="center", va="center", color=INK, zorder=3)

    # ============================================================ (d) fusion variants
    pan(61.5, 51.3, "d", "Fusion variants (4 streams)")
    cols = [TXN, SEQ, GRA, MUL]
    def sq(x, y, hs=None):
        for i, c in enumerate(cols):
            ax.add_patch(Rectangle((x + i * 2.7, y), 2.2, 2.2, fc=tint(c, 0.75), ec="white", lw=0.3, zorder=3))
            if hs is not None: ax.add_patch(Rectangle((x + i * 2.7 + 0.5, y + 2.5), 1.2, 3.2 * hs[i], fc=FUS, ec="none", zorder=3, alpha=0.85))
    rows = [(42.5, "concat"), (35.0, "gated (default)"), (27.5, "attention")]
    for y, nme in rows: ax.text(62.0, y + 1.1, nme, fontsize=5.8, fontweight="bold", va="center")
    sq(76.0, 41.4); arrow([(87.2, 42.5), (89.0, 42.5)], c=GREY); ax.text(94.0, 42.5, "MLP 256, 128, 64\nReLU, dropout", fontsize=5.0, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc=tint(FUS, 0.1), ec=FUS, lw=0.5))
    sq(76.0, 33.2, hs=[0.45, 0.6, 1.0, 0.3]); ax.text(87.2, 36.2, r"$g=\mathrm{softmax}(W[z_1..z_4])$", fontsize=4.9, va="center"); ax.text(87.2, 34.0, r"$\Sigma_i\; g_i W_i z_i$, then MLP", fontsize=4.9, va="center")
    sq(76.0, 25.9); ax.add_patch(Circle((89.5, 31.0), 1.1, fc=FUS, ec="none", zorder=3)); ax.text(91.2, 31.0, "query q", fontsize=4.9, va="center")
    for i, c in enumerate(cols): ax.plot([76.0 + i * 2.7 + 1.1, 89.5], [28.1, 31.0], color=tint(c, 0.9), lw=0.6, zorder=2)
    ax.text(76.0, 24.1, r"$a=\mathrm{softmax}(q\cdot K(z)/\sqrt{h})$, output $\Sigma_i\; a_i V z_i$", fontsize=4.9, va="center")
    ax.plot([62.0, 99.0], [23.2, 23.2], color=GREY, lw=0.4)

    # ============================================================ (e) objective and decision
    pan(61.5, 22.6, "e", "Objective and decision policy")
    box(62.0, 6.9, 37.0, 12.6, "Multi-task training", [r"$\mathcal{L}=1.0\,\mathcal{L}_A+0.5\,\mathcal{L}_B+0.3\,\mathcal{L}_C$ (weighted BCE)", "positive weight 20 (A), 5 (B, C); negatives kept with", "probability 0.4 and reweighted; AdamW 2e-3;", "snapshots in time order; best validation PR-AUC kept"], FUS, fill=0.06, ha="left", fs=5.1)
    steps = [("approve", C["aqua"]), ("soft\nwarning", C["yellow"]), ("strong warn\n+ confirm", C["orange"]), ("hold +\nreview", C["red"])]
    xs = [62.0, 71.0, 80.0, 90.0]; ws = [8.2, 8.2, 9.2, 9.0]
    for (t, c), x, w in zip(steps, xs, ws):
        ax.add_patch(FancyBboxPatch((x, 0.0), w, 4.3, boxstyle="round,pad=0,rounding_size=0.6", fc=tint(c, 0.25), ec=c, lw=0.8, zorder=2)); ax.text(x + w / 2, 2.15, t, fontsize=4.6, ha="center", va="center", zorder=3)
    for x, t in zip([70.6, 79.6, 89.5], [r"$T_1$", r"$T_2$", r"$T_3$"]): ax.text(x, 4.6, t, fontsize=5.2, ha="center", va="bottom") if False else None
    ax.text(62.0, 5.6, r"$p_A$, $u$ drive the policy; $T_1\leq T_2\leq T_3$ minimise validation cost", fontsize=4.8, color=MUTED, va="center")
    return fig, ax
