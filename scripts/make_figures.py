"""Research figures. Each figure answers a specific question; missing inputs are skipped (not faked)."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from presage.data.loaders import load_sim
from presage.evaluation.metrics import temporal_split
from presage.utils.common import ROOT

R, T, F = ROOT / "reports/results", ROOT / "reports/tables", ROOT / "reports/figures"; F.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
df, acc = load_sim(); cust = df.src_is_customer.values == 1; _, _, te0 = temporal_split(df.day.values); te = te0 & cust; yt = df.is_app_fraud.values[te]


def save(name): plt.tight_layout(); plt.savefig(F / name); plt.close(); print("saved", name)


# 1. PR curves of representative models
try:
    pick = {"XGBoost txn-only": "xgboost_txn_only", "XGBoost + behaviour": "xgboost_txn+behaviour", "XGBoost tabular": "xgboost_all_tabular", "XGBoost + graph": "xgboost_tabular+graph_full",
            "HGT edge model": "graph_hgt", "Tabular-only NN twin": "graph_none"}
    z = np.load(R / "presage_full_s42.npz") if (R / "presage_full_s42.npz").exists() else None
    plt.figure(figsize=(5.5, 4))
    for lab, key in pick.items():
        f = R / f"pred_{key}.npy"
        if f.exists(): p = np.load(f); pr, rc, _ = precision_recall_curve(yt, p); plt.plot(rc, pr, label=f"{lab} (AP {average_precision_score(yt, p):.2f})")
    if z is not None: pr, rc, _ = precision_recall_curve(yt, z["app"]); plt.plot(rc, pr, "k", lw=2, label=f"PRESAGE (AP {average_precision_score(yt, z['app']):.2f})")
    plt.axhline(yt.mean(), ls=":", c="grey"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Chronological test period, customer payments (simulated)"); plt.legend(fontsize=7); save("fig_pr_curves.png")
except Exception as e: print("pr curves skipped", e)
# 2. feature-group staircase
try:
    b = pd.read_csv(T / "baselines.csv"); b = b[b.model.isin(["xgboost", "catboost", "random_forest"])]
    pv = b.pivot_table(index="feature_set", columns="model", values="pr_auc").reindex(["txn_only", "txn+behaviour", "all_tabular", "tabular+graph_struct", "tabular+graph_full"])
    pv.plot(kind="bar", figsize=(6, 3.6)); plt.ylabel("PR-AUC (test)"); plt.title("RQ1: information added to the feature set"); plt.xticks(rotation=20, ha="right"); save("fig_feature_staircase.png")
except Exception as e: print("staircase skipped", e)
# 3. graph models
try:
    g = pd.read_csv(T / "graph_models.csv").sort_values("pr_auc"); plt.figure(figsize=(5.5, 3.6)); plt.barh(g.model, g.pr_auc, xerr=[g.pr_auc - g.pr_auc_lo, g.pr_auc_hi - g.pr_auc], color="#3b6ea5")
    plt.xlabel("PR-AUC (test, 95% block-bootstrap CI)"); plt.title("Graph encoders vs tabular-only twin ('none')"); save("fig_graph_models.png")
except Exception as e: print("graph models skipped", e)
# 4. drift curves
try:
    c = json.load(open(R / "drift_curves.json")); plt.figure(figsize=(7, 4))
    for k, v in c.items(): plt.plot(v["block_start"], v["pr_auc"], marker="o", ms=3, label=k)
    plt.axvline(100, c="r", ls="--", lw=1); plt.text(100.5, 0.02, "sudden drift", color="r", fontsize=7); plt.axvspan(90, 105, color="orange", alpha=0.1)
    plt.xlabel("walk-forward block start (day)"); plt.ylabel("PR-AUC in block"); plt.legend(fontsize=6, ncol=2); plt.title("Adaptation strategies under drift (3-day label delay)"); save("fig_drift_curves.png")
except Exception as e: print("drift skipped", e)
# 5. drift monitors
try:
    d = pd.read_csv(T / "drift_detectors.csv"); d = d[d.weekly_values.notna()]; plt.figure(figsize=(7, 3.6)); wk = list(range(60, 120, 7))
    for _, r in d.iterrows():
        if r.detector.startswith("psi"): plt.plot(wk, json.loads(r.weekly_values), marker="o", ms=3, label=r.detector)
    plt.axhline(0.2, c="grey", ls=":"); plt.axvline(100, c="r", ls="--", lw=1); plt.ylabel("PSI vs days 0-59"); plt.xlabel("week start (day)"); plt.legend(fontsize=7); plt.title("Feature-distribution monitors"); save("fig_drift_monitors.png")
except Exception as e: print("monitors skipped", e)
# 6. label delay
try:
    l = pd.read_csv(T / "label_delay.csv"); order = ["1h", "1d", "3d", "7d", "30d"]; plt.figure(figsize=(5.5, 3.6))
    for v, s in l.groupby("variant"): s = s.set_index("delay").reindex(order); plt.plot(order, s.mean_pr_auc, marker="o", label=v)
    plt.ylabel("mean walk-forward PR-AUC"); plt.xlabel("label confirmation delay"); plt.legend(); plt.title("Effect of label delay and mitigations"); save("fig_label_delay.png")
except Exception as e: print("label delay skipped", e)
# 7. federated
try:
    f = pd.read_csv(T / "federated.csv"); fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); f2 = f.set_index("method")
    f2[["pr_auc", "crossbank_pr_auc"]].plot(kind="barh", ax=ax[0]); ax[0].set_title("Cross-bank detection"); ax[0].legend(fontsize=7)
    dp = f[f.method.str.contains("DP|fedavg$", regex=True)].copy(); dp = dp[dp.method.str.startswith("fedavg")]
    ax[1].scatter(dp.eps.replace(np.inf, np.nan), dp.pr_auc); [ax[1].annotate(m, (e, p), fontsize=6) for m, e, p in zip(dp.method, dp.eps.replace(np.inf, np.nan), dp.pr_auc) if np.isfinite(e)]
    ax[1].set_xlabel("epsilon (client-level, delta=1e-5)"); ax[1].set_ylabel("PR-AUC"); ax[1].set_title("Privacy-utility"); save("fig_federated.png")
except Exception as e: print("federated skipped", e)
# 8. ablation
try:
    rows = []
    for f_ in glob.glob(str(T / "presage_*_s42.csv")): rows.append(pd.read_csv(f_))
    a = pd.concat(rows).drop_duplicates("variant").sort_values("pr_auc"); plt.figure(figsize=(5.5, 3.8)); plt.barh(a.variant, a.pr_auc, xerr=[a.pr_auc - a.pr_auc_lo, a.pr_auc_hi - a.pr_auc], color="#2a9d8f")
    plt.xlabel("PR-AUC (test)"); plt.title("PRESAGE ablations / fusion variants (single seed)"); save("fig_ablation.png")
except Exception as e: print("ablation skipped", e)
# 9. victim -> mule -> mule -> cash-out flow for the most active network
try:
    import networkx as nx
    s = df[df.is_app_fraud == 1].groupby("net_id").size().sort_values(ascending=False); n = int(s.index[0])
    sub = df[(df.net_id == n)]; G = nx.DiGraph(); pos = {}; col = {}
    tiers = acc.set_index("account").mule_tier
    for r in sub.itertuples():
        a, b = int(r.src), int(r.dst); G.add_edge(a, b, w=r.amount)
    xs = {}
    for v in G.nodes:
        t = tiers.get(v, 0); is_mule = acc.set_index("account").is_mule.get(v, 0)
        if r_ := (df.is_app_fraud[(df.src == v) & (df.net_id == n)].any()): xs[v] = 0
        elif is_mule and t > 0: xs[v] = int(t)
        else: xs[v] = 4
    levels = {}
    for v, x in xs.items(): levels.setdefault(x, []).append(v)
    for x, vs in levels.items():
        for i, v in enumerate(sorted(vs)): pos[v] = (x, -i + len(vs) / 2)
    plt.figure(figsize=(8, 5)); cmap = {0: "#e76f51", 1: "#f4a261", 2: "#e9c46a", 3: "#2a9d8f", 4: "#8d99ae"}
    w = np.array([G[u][v]["w"] for u, v in G.edges]); nx.draw_networkx_edges(G, pos, width=0.3 + 3 * w / w.max(), alpha=0.5, arrows=True, arrowsize=6)
    nx.draw_networkx_nodes(G, pos, node_color=[cmap[xs[v]] for v in G.nodes], node_size=40)
    for x, lab in {0: "victims", 1: "tier-1 collection", 2: "tier-2 layering", 3: "tier-3 cash-out", 4: "cash-out sinks"}.items(): plt.text(x, len(levels[0]) / 2 + 1.5, lab, ha="center", fontsize=8)
    plt.axis("off"); plt.title(f"Mule network {n}: victim -> mule -> mule -> cash-out (edge width ~ amount; simulated)"); save("fig_mule_network.png")
except Exception as e: print("mule network skipped", e)
# 10. fan-in / pass-through distributions (mule beneficiaries vs legit)
try:
    X = pd.read_parquet(ROOT / "data/processed/sim_v1_features.parquet"); m = (df.src_is_customer == 1)
    isf = df.is_app_fraud.values == 1; fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a_, col, lab in zip(ax, ["dst_fanin_new_24h", "dst_pass_through_24h", "dst_age_days"], ["new senders into beneficiary (24h)", "beneficiary pass-through ratio (24h)", "beneficiary age (days)"]):
        for flag, name, c in [(False, "legitimate payments", "#8d99ae"), (True, "scam payments", "#e76f51")]:
            v = X.loc[m & (isf == flag), col].clip(upper=X[col].quantile(0.995)); a_.hist(v, bins=30, alpha=0.6, density=True, color=c, label=name)
        a_.set_xlabel(lab); a_.set_yscale("log")
    ax[0].legend(fontsize=7); plt.suptitle("Beneficiary-side signals available before settlement (simulated)", fontsize=9); save("fig_beneficiary_signals.png")
except Exception as e: print("beneficiary signals skipped", e)
# 11. uncertainty vs error
try:
    z = np.load(R / "presage_full_s42.npz"); p, u = z["app"], z["unc"]; fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    q = pd.qcut(u, 10, duplicates="drop"); err = pd.Series(np.abs(p - yt)); ax[0].bar(range(len(q.categories)), err.groupby(q, observed=True).mean().values); ax[0].set_xlabel("uncertainty decile"); ax[0].set_ylabel("mean |p - y|"); ax[0].set_title("Uncertainty vs error")
    pos = yt == 1; ax[1].hist(u[~pos], bins=40, alpha=.6, density=True, label="legit"); ax[1].hist(u[pos], bins=40, alpha=.6, density=True, label="fraud"); ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_xlabel("MC-dropout std"); save("fig_uncertainty.png")
except Exception as e: print("uncertainty skipped", e)
