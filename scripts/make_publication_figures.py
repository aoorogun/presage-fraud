"""Publication-quality figures built ONLY from result files produced by the experiment scripts.
Outputs vector PDF (fonts embedded) and 600 dpi PNG to reports/figures/publication/.
Every APP/mule figure is labelled as synthetic data. Usage: PYTHONPATH=src python3 scripts/make_publication_figures.py"""
import json, glob, os, warnings
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, average_precision_score
warnings.filterwarnings("ignore")
from presage.data.loaders import load_sim
from presage.utils.common import ROOT

T, R = ROOT / "reports/tables", ROOT / "reports/results"
REQUIRED = ["reports/tables/baselines.csv", "reports/tables/graph_models.csv", "reports/tables/seed_variance.csv", "reports/tables/drift_adaptation.csv", "reports/tables/label_delay.csv",
            "reports/tables/federated.csv", "reports/tables/adversarial_robustness.csv", "reports/tables/real_datasets.csv", "reports/tables/presage_full_s42.csv", "reports/results/drift_curves.json",
            "reports/results/explainability_metrics.json", "reports/results/test_index.npy", "reports/results/presage_concat_s42.npz"]
MAKE = {"baselines": "make train-baselines", "graph_models": "make train-graph", "seed_variance": "python3 scripts/run_seed_variance.py --train", "drift": "make drift-experiment", "label_delay": "make label-delay",
        "federated": "make federated-experiment", "adversarial": "make adversarial", "real_datasets": "python3 scripts/run_real_datasets.py", "presage": "make train-presage ablation (variants full, concat, attention, transformer_seq, no_*)", "explainability": "make explain"}
missing = [f for f in REQUIRED if not (ROOT / f).exists()]
if missing:
    raise SystemExit("Missing inputs for figures:\n  " + "\n  ".join(missing) + "\nProduce them with: " + "; ".join(f"{k}: {v}" for k, v in MAKE.items()))
OUT = ROOT / "reports/figures/publication"; OUT.mkdir(parents=True, exist_ok=True)
# validated categorical slots (blue, orange, aqua, violet, red, magenta) + neutral ink
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#4a3aa7", red="#e34948", magenta="#e87ba4", yellow="#c98500")
INK, MUTED, GRID = "#1a1a1a", "#6b6b66", "#e4e3de"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
W1, W2 = 3.4, 7.1                                   # single / double column width in inches
SYN = "Synthetic data (sim_v1). Chronological test split."


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig); print("saved", name)


def tag(fig, t="Synthetic data (sim_v1)"): fig.text(0.995, 1.0, t, fontsize=6.5, color=MUTED, style="italic", ha="right", va="bottom")


def panel(ax, s): ax.text(-0.02, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def ci_barh(ax, labels, val, lo, hi, colors, ref=None, xlim=(0.4, 1.0)):
    y = np.arange(len(labels))[::-1]
    for yi, v, l, h, c in zip(y, val, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=1.6, solid_capstyle="round"); ax.plot(v, yi, "o", color=c, mec="white", mew=0.8, ms=5.5)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(*xlim); ax.grid(axis="y", visible=False)
    if ref is not None: ax.axvline(ref, color=MUTED, lw=0.8, ls="--")


df, acc = load_sim(); te_idx = np.load(R / "test_index.npy"); yt = df.is_app_fraud.values[te_idx]

# ---------------------------------------------------------------- Fig 1: feature staircase + PR curves
B = pd.read_csv(T / "baselines.csv"); order = ["txn_only", "txn+behaviour", "all_tabular", "tabular+graph_struct", "tabular+graph_full"]
lab = ["Trans-\naction", "+ Behav-\niour", "+ Benefi-\nciary, device", "+ Graph\nstructure", "+ Mule\nlabels"]
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
x = np.arange(len(order)); w = 0.36
for k, (m, c, nm) in enumerate([("xgboost", C["blue"], "XGBoost"), ("catboost", C["orange"], "CatBoost")]):
    s = B[B.model == m].set_index("feature_set").loc[order]
    ax[0].bar(x + (k - 0.5) * w, s.pr_auc, w * 0.92, color=c, label=nm, yerr=[s.pr_auc - s.pr_auc_lo, s.pr_auc_hi - s.pr_auc], error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK))
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=6.3); ax[0].set_ylabel("PR-AUC (test)"); ax[0].set_ylim(0, 1.05); ax[0].grid(axis="x", visible=False)
ax[0].legend(loc="upper left", ncol=2); ax[0].set_title("Cumulative feature groups"); panel(ax[0], "a")
curves = [("xgboost_txn_only", "XGBoost, transaction only", C["magenta"], ":"), ("xgboost_all_tabular", "XGBoost, tabular", C["aqua"], "-."), ("xgboost_tabular+graph_full", "XGBoost, tabular + graph", C["blue"], "-"),
          ("graph_hgt", "HGT (graph encoder)", C["violet"], "--"), ("PRESAGE", "PRESAGE (concat fusion)", C["orange"], "-")]
for key, nm, c, ls in curves:
    p = np.load(R / "presage_concat_s42.npz")["app"] if key == "PRESAGE" else np.load(R / f"pred_{key}.npy")
    pr, rc, _ = precision_recall_curve(yt, p); ax[1].plot(rc, pr, color=c, ls=ls, label=f"{nm} ({average_precision_score(yt, p):.3f})")
ax[1].axhline(yt.mean(), color=MUTED, lw=0.7, ls=":"); ax[1].text(0.02, yt.mean() + 0.025, f"no-skill (prevalence {yt.mean()*100:.2f}%)", fontsize=6.3, color=MUTED)
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1.02); ax[1].legend(loc="center left", bbox_to_anchor=(0.0, 0.72), fontsize=6.0, handlelength=2.2); ax[1].set_title("Precision-recall (PR-AUC in brackets)"); panel(ax[1], "b")
tag(fig); save(fig, "fig1_features_and_pr_curves")

# ---------------------------------------------------------------- Fig 2: seed variance (the central honest comparison)
S = pd.read_csv(T / "seed_variance.csv"); names = [("presage_concat", "PRESAGE, concat fusion", C["orange"]), ("presage_full", "PRESAGE, default fusion", C["red"]),
                                                   ("xgboost_graph_full", "XGBoost, tabular + graph", C["blue"]), ("catboost_graph_full", "CatBoost, tabular + graph", C["aqua"])]
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), sharey=True)
for a, col, ttl, p in zip(ax, ["test_pr_auc", "val_pr_auc"], ["Test", "Validation"], "ab"):
    for i, (k, nm, c) in enumerate(names[::-1]):
        v = S[S.model == k][col].values; a.scatter(v, np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), color=c, s=22, zorder=3, ec="white", lw=0.6)
        a.plot([v.mean() - v.std(ddof=1)] * 1 + [v.mean() + v.std(ddof=1)], [i, i], color=c, lw=3, alpha=0.35, solid_capstyle="butt"); a.plot(v.mean(), i, "|", color=INK, ms=12, mew=1.4)
    a.set_yticks(range(len(names))); a.set_yticklabels([n for _, n, _ in names[::-1]]); a.set_xlabel("PR-AUC"); a.set_title(ttl); a.grid(axis="y", visible=False); a.set_xlim(0.89, 1.0); panel(a, p)
fig.legend([Line2D([], [], marker="o", color="w", mfc=MUTED, ms=5), Line2D([], [], color=MUTED, lw=3, alpha=0.4), Line2D([], [], marker="|", color=INK, ms=10, ls="")],
           ["one seed (42, 1, 2)", "mean $\\pm$ 1 s.d.", "mean"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
tag(fig); save(fig, "fig2_seed_variance_presage_vs_gbdt")

# ---------------------------------------------------------------- Fig 3: graph encoders
G = pd.read_csv(T / "graph_models.csv"); nm = dict(none="Tabular twin (no graph)", gcn="GCN", sage="GraphSAGE", gat="GAT", rgcn="R-GCN", hgt="HGT", han="HAN", tgat="TGAT-style", evolvegcn="EvolveGCN-O")
G = G.sort_values("pr_auc"); kind = {"none": MUTED, "gcn": C["blue"], "sage": C["blue"], "gat": C["blue"], "rgcn": C["aqua"], "hgt": C["aqua"], "han": C["aqua"], "tgat": C["violet"], "evolvegcn": C["violet"]}
fig, ax = plt.subplots(figsize=(W1 + 0.6, 2.6)); ci_barh(ax, [nm[m] for m in G.model], G.pr_auc, G.pr_auc_lo, G.pr_auc_hi, [kind[m] for m in G.model], ref=0.977, xlim=(0.78, 1.0))
ax.set_xlabel("PR-AUC (test), 95% block-bootstrap CI"); ax.set_title("Graph encoders")
ax.legend([Line2D([], [], marker="o", color=c, ls="") for c in (MUTED, C["blue"], C["aqua"], C["violet"])] + [Line2D([], [], color=MUTED, ls="--")], ["no graph", "homogeneous", "heterogeneous", "temporal", "XGBoost tabular+graph (0.977)"], loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6.3)
tag(fig); save(fig, "fig3_graph_encoders")

# ---------------------------------------------------------------- Fig 4: PRESAGE ablation and fusion (single seed)
A = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(T / "presage_*_s42.csv")))]).drop_duplicates("variant")      # every PRESAGE variant run at seed 42
A = A[A.variant != "transformer_seq"]
lab4 = dict(full="Full (default fusion)", concat="Concat fusion", attention="Attention fusion", no_graph="No graph branch", no_temporal="No temporal memory", no_mule="No mule module", no_multitask="No multi-task heads",
            no_beh="No behaviour features", no_txn="No transaction branch", no_benfeat="No beneficiary features")
A = A.sort_values("pr_auc"); col = [C["red"] if v == "full" else (C["orange"] if v in ("concat", "attention") else C["blue"]) for v in A.variant]
fig, ax = plt.subplots(figsize=(W1 + 0.7, 2.9)); ci_barh(ax, [lab4[v] for v in A.variant], A.pr_auc, A.pr_auc_lo, A.pr_auc_hi, col, ref=float(A[A.variant == "full"].pr_auc.iloc[0]), xlim=(0.78, 1.0))
ax.set_xlabel("PR-AUC (test), 95% block-bootstrap CI"); ax.set_title("PRESAGE ablation and fusion"); ax.legend([Line2D([], [], marker="o", color=c, ls="") for c in (C["red"], C["orange"], C["blue"])], ["default (reference line)", "fusion variant", "component removed"], loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=6.3, columnspacing=1.0, handletextpad=0.2)
tag(fig); save(fig, "fig4_presage_ablation")

# ---------------------------------------------------------------- Fig 5: drift adaptation
D = json.load(open(R / "drift_curves.json")); DS = pd.read_csv(T / "drift_adaptation.csv").set_index("strategy")
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(width_ratios=[1.4, 1], wspace=0.75))
sty = [("static", "Static", MUTED, "-"), ("scheduled_14d", "Scheduled (14 d)", C["blue"], "-"), ("sliding_30d", "Sliding window (30 d)", C["orange"], "--"), ("replay_buffer", "Replay buffer", C["aqua"], "-."), ("incremental", "Incremental", C["violet"], ":")]
for k, nm_, c, ls in sty: ax[0].plot(np.array(D[k]["block_start"]), D[k]["pr_auc"], color=c, ls=ls, marker="o", ms=2.8, label=nm_)
ax[0].axvline(100, color=C["red"], lw=0.9); ax[0].text(100.6, ax[0].get_ylim()[0] + 0.005, "sudden drift", color=C["red"], fontsize=6.5, rotation=90, va="bottom")
ax[0].set_xlabel("Day (start of 7-day block)"); ax[0].set_ylabel("PR-AUC"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.3, ncol=3); ax[0].set_title("Walk-forward performance"); panel(ax[0], "a")
ks = [k for k, _, _, _ in sty]; y = np.arange(len(ks))[::-1]
ax[1].barh(y + 0.19, DS.loc[ks, "pre_drift_pr_auc"], 0.36, color=C["blue"], label="before day 98"); ax[1].barh(y - 0.19, DS.loc[ks, "post_drift_pr_auc"], 0.36, color=C["orange"], label="after day 98")
ax[1].set_yticks(y); ax[1].set_yticklabels([n for _, n, _, _ in sty]); ax[1].set_xlim(0.9, 1.0); ax[1].set_xlabel("Mean PR-AUC"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2, fontsize=6.3); ax[1].grid(axis="y", visible=False); ax[1].set_title("Before vs after drift"); panel(ax[1], "b")
tag(fig); save(fig, "fig5_concept_drift")

# ---------------------------------------------------------------- Fig 6: label delay
L = pd.read_csv(T / "label_delay.csv"); dl = ["1h", "1d", "3d", "7d", "30d"]
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
for v, nm_, c, mk in [("supervised", "Supervised", MUTED, "o"), ("pseudo", "Pseudo-labelling", C["blue"], "s"), ("weak", "Weak supervision", C["red"], "^"), ("active", "Active labelling", C["aqua"], "D")]:
    s = L[L.variant == v].set_index("delay").reindex([d for d in dl if d in set(L[L.variant == v].delay)])
    for a, colname in zip(ax, ["mean_pr_auc", "recall_at_1pct_fpr"]): a.plot([dl.index(d) for d in s.index], s[colname], color=c, marker=mk, label=nm_)
for a, yl, p in zip(ax, ["Mean PR-AUC", "Recall at 1% false-positive rate"], "ab"):
    a.set_xticks(range(len(dl))); a.set_xticklabels(dl); a.set_xlabel("Fraud-label delay"); a.set_ylabel(yl); panel(a, p)
ax[0].legend(loc="lower left"); ax[0].set_title("Accuracy vs label delay"); ax[1].set_title("Detection at fixed FPR")
tag(fig); save(fig, "fig6_label_delay")

# ---------------------------------------------------------------- Fig 7: federated learning
F = pd.read_csv(T / "federated.csv").set_index("method")
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
rows = [("local_only", "Local only", MUTED), ("fedavg", "FedAvg", C["blue"]), ("fedavg_personalised", "FedAvg, personalised", C["blue"]), ("fedprox(mu=0.05)", "FedProx", C["blue"]),
        ("fedavg+secure_agg(sim)", "FedAvg + secure agg. (simulated)", C["blue"]), ("centralised_pooled(local feats)", "Pooled, local-view features", C["aqua"]), ("centralised_full_visibility(oracle)", "Oracle, full visibility", C["violet"])]
yy = np.arange(len(rows))[::-1]; ax[0].barh(yy, [F.loc[k, "pr_auc"] for k, _, _ in rows], 0.62, color=[c for _, _, c in rows]); ax[0].set_yticks(yy); ax[0].set_yticklabels([n for _, n, _ in rows])
for yi, (k, _, _) in zip(yy, rows): ax[0].text(F.loc[k, "pr_auc"] + 0.01, yi, f"{F.loc[k, 'pr_auc']:.2f}", va="center", fontsize=6.5)
ax[0].set_xlim(0, 1.05); ax[0].set_xlabel("PR-AUC (test)"); ax[0].grid(axis="y", visible=False); ax[0].set_title("Five-bank federation"); panel(ax[0], "a")
dp = F[F.index.str.startswith("fedavg+DP")].copy(); dp = dp.sort_values("eps", ascending=False)
ax[1].plot(range(len(dp)), dp.pr_auc, "o-", color=C["red"], label="FedAvg + client-level DP"); ax[1].axhline(F.loc["fedavg", "pr_auc"], color=C["blue"], ls="--", lw=1, label="FedAvg without DP")
ax[1].set_xticks(range(len(dp))); ax[1].set_xticklabels([f"{e:.0f}" for e in dp.eps]); ax[1].set_xlim(-0.4, len(dp) - 0.6)
ax[1].set_xlabel("$\\varepsilon$ (RDP accountant); smaller = more private"); ax[1].set_ylabel("PR-AUC (test)"); ax[1].set_ylim(0, 1); ax[1].legend(loc="center right", fontsize=6.3); ax[1].set_title("Privacy-utility trade-off"); panel(ax[1], "b")
tag(fig); save(fig, "fig7_federated")

# ---------------------------------------------------------------- Fig 8: adversarial robustness
Ad = pd.read_csv(T / "adversarial_robustness.csv"); clean = float(Ad[Ad.experiment == "clean"].pr_auc.iloc[0])
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4), sharey=True, gridspec_kw=dict(wspace=0.12))
ev = Ad[Ad.experiment == "evasion_mimicry"].copy(); ev["grp"] = ev.setting.str.split("@").str[0]; ev["al"] = ev.setting.str.extract(r"alpha=([\d.]+)").astype(float)
for g, nm_, c, ls in [("victim_coached", "Victim-side", C["blue"], "-"), ("mule_camouflage", "Mule-side", C["aqua"], "--"), ("both", "Both", C["red"], "-")]:
    s = ev[ev.grp == g]; ax[0].plot(np.r_[0, s.al], np.r_[clean, s.pr_auc], color=c, ls=ls, marker="o", label=nm_)
ax[0].set_xlabel("Mimicry strength $\\alpha$"); ax[0].set_ylabel("PR-AUC"); ax[0].legend(loc="lower left", fontsize=6.3); ax[0].set_title("Feature evasion"); panel(ax[0], "a")
po = Ad[Ad.experiment == "label_poisoning"]; pv_ = [0] + [int(s.split("%")[0]) for s in po.setting]; ax[1].plot(pv_, [clean] + po.pr_auc.tolist(), color=C["orange"], marker="s"); ax[1].set_xlabel("Train frauds relabelled legitimate (%)"); ax[1].set_title("Label poisoning"); panel(ax[1], "b")
fd = Ad[Ad.experiment == "feature_dropout"]; dv = [0] + [int(s.split("%")[0]) for s in fd.setting]; ax[2].plot(dv, [clean] + fd.pr_auc.tolist(), color=C["violet"], marker="^"); ax[2].set_xlabel("Features zeroed at test time (%)"); ax[2].set_title("Missing features"); panel(ax[2], "c")
for a in ax: a.set_ylim(0.6, 1.0)
tag(fig); save(fig, "fig8_adversarial_robustness")

# ---------------------------------------------------------------- Fig 9: explainability
E = json.load(open(R / "explainability_metrics.json")); fg = E["faithfulness"]
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1, 1.2]))
ax[0].bar([0, 1], [fg["mean_drop_top_shap"], fg["mean_drop_random"]], 0.55, color=[C["blue"], MUTED]); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Top-5 SHAP\nfeatures", "5 random\nfeatures"])
for i, v in enumerate([fg["mean_drop_top_shap"], fg["mean_drop_random"]]): ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
ax[0].set_ylabel("Mean drop in fraud score\nafter removing features"); ax[0].set_ylim(0, 1); ax[0].grid(axis="x", visible=False); ax[0].set_title(f"Deletion faithfulness (n = {fg['n']})"); panel(ax[0], "a")
cov = E["share_of_flagged_frauds_with_reason_at_level"]; ks = sorted(cov, key=cov.get); ax[1].barh(range(len(ks)), [cov[k] for k in ks], 0.55, color=C["aqua"]); ax[1].set_yticks(range(len(ks))); ax[1].set_yticklabels([k.capitalize() for k in ks])
for i, k in enumerate(ks): ax[1].text(cov[k] + 0.015, i, f"{cov[k]*100:.0f}%", va="center", fontsize=7)
ax[1].set_xlim(0, 1); ax[1].set_xlabel("Share of flagged frauds with a top reason at this level"); ax[1].grid(axis="y", visible=False); ax[1].set_title("Reason coverage"); panel(ax[1], "b")
tag(fig); save(fig, "fig9_explainability")

# ---------------------------------------------------------------- Fig 10: real benchmark datasets (NOT APP)
RD = pd.read_csv(T / "real_datasets.csv"); RD["lab"] = RD.dataset.str.replace(r"\(enriched subsample\)", "", regex=True).str.replace("-HI-Small", " HI-Small") + "\n" + RD.feature_set.replace({"txn_only": "transaction only", "txn+behaviour+beneficiary": "+ behaviour, beneficiary", "txn+behaviour+beneficiary+graph": "+ behaviour, beneficiary, graph"})
fig, ax = plt.subplots(figsize=(W1 + 1.2, 3.0)); RD = RD.iloc[::-1]
ci_barh(ax, RD.lab.tolist(), RD.pr_auc, RD.pr_auc_lo, RD.pr_auc_hi, [{"PaySim": C["blue"], "Elliptic": C["violet"]}.get(d, C["aqua"]) for d in RD.dataset.str.split("-").str[0].str.split("(").str[0]], xlim=(0.5, 1.02))
ax.tick_params(axis="y", labelsize=6); ax.set_xlabel("PR-AUC (test), 95% CI"); ax.set_title("Real public benchmarks (not APP fraud)")
tag(fig, "Real public data"); save(fig, "fig10_real_benchmarks")
print("done ->", OUT)
