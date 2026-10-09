"""Advanced publication figures (11-19), generated from the codebase and the result files:
 11 architecture, 12 mule-network growth, 13 operating characteristics, 14 early mule detection, 15 uncertainty and selective prediction,
 16 critical-difference diagram, 17 drift monitors heatmap, 18 feature importance by group, 19 error analysis and per-bank check, 20 ensembles (needs scripts/run_ensemble.py).
Each figure is built independently; a figure whose inputs are missing is skipped with a message.
Usage: PYTHONPATH=src python3 scripts/make_advanced_figures.py"""
import sys, json, textwrap, warnings, traceback, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
from figstyle import C, INK, MUTED, GRID, W1, W2, save, panel, tag, ROOT
from presage.data.loaders import load_sim, load_features
from presage.evaluation.metrics import temporal_split, expected_cost, CostModel

T, R = ROOT / "reports/tables", ROOT / "reports/results"
df, acc = load_sim(); DAY = 86400.0
cust = df.src_is_customer.values == 1; tr0, va0, te0 = temporal_split(df.day.values); te = te0 & cust
te_idx = np.load(R / "test_index.npy"); yt = df.is_app_fraud.values[te_idx]; dte = df.iloc[te_idx]
PRED = {"PRESAGE (concat)": lambda: np.load(R / "presage_concat_s42.npz")["app"], "PRESAGE (gated, default)": lambda: np.load(R / "presage_full_s42.npz")["app"],
        "XGBoost, tabular + graph": lambda: np.load(R / "pred_xgboost_tabular+graph_full.npy"), "XGBoost, tabular": lambda: np.load(R / "pred_xgboost_all_tabular.npy"),
        "HGT": lambda: np.load(R / "pred_graph_hgt.npy")}
COL = {"PRESAGE (concat)": C["orange"], "PRESAGE (gated, default)": C["red"], "XGBoost, tabular + graph": C["blue"], "XGBoost, tabular": C["aqua"], "HGT": C["violet"]}
LS = {"PRESAGE (concat)": "-", "PRESAGE (gated, default)": "--", "XGBoost, tabular + graph": "-", "XGBoost, tabular": "-.", "HGT": ":"}


def pred(k):
    try: return PRED[k]()
    except Exception: return None


def run(fn):
    try: fn()
    except Exception as e: print(f"SKIPPED {fn.__name__}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)


# ------------------------------------------------------------------ 11 architecture
def fig11():
    from arch_figure import draw_architecture
    fig, ax = draw_architecture(); save(fig, "fig11_presage_architecture")


# ------------------------------------------------------------------ 12 mule network growth
def fig12():
    m = acc[acc.is_mule == 1]; netc = m.mule_net.value_counts(); nid = netc.index[0]
    members = set(m[m.mule_net == nid].account); tier = acc.set_index("account").mule_tier
    e = df[(df.src.isin(members) | df.dst.isin(members)) & ((df.is_app_fraud == 1) | (df.src.isin(members) & df.dst.isin(members)))].copy()
    e["kind"] = np.where(e.is_app_fraud == 1, "victim", "mule")
    t0, t1 = e.ts.min(), e.ts.max(); cuts = [t0 + f * (t1 - t0) for f in (0.34, 0.67, 1.0)]
    nodes = {a: tier[a] for a in members}; vict = sorted(set(e[e.kind == "victim"].src)); xs = {0: 0, 1: 1, 2: 2, 3: 3}
    rng = np.random.default_rng(1); pos = {}
    for t in (1, 2, 3):
        ms = sorted([a for a, k in nodes.items() if k == t]); [pos.__setitem__(a, (t, (i + 0.5) / max(len(ms), 1))) for i, a in enumerate(ms)]
    for i, v in enumerate(vict): pos[v] = (0, (i + 0.5) / max(len(vict), 1))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.9), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.04))
    tc = {1: C["orange"], 2: C["violet"], 3: C["red"]}; titles = ["Early (first third)", "Middle (two thirds)", "Complete"]
    for a_, cut, ttl, p in zip(ax, cuts, titles, "abc"):
        s = e[e.ts <= cut]; a_.axis("off"); a_.grid(False)
        for r in s.itertuples():
            if r.src in pos and r.dst in pos:
                (x0, y0), (x1, y1) = pos[r.src], pos[r.dst]; a_.plot([x0, x1], [y0, y1], color=C["magenta"] if r.kind == "victim" else MUTED, lw=0.25 + 0.5 * np.log10(max(r.amount, 10)) / 3, alpha=0.5, zorder=1)
        act = set(s.src) | set(s.dst)
        for v in vict:
            if v in act: a_.scatter(*pos[v], s=9, color=C["magenta"], zorder=3, lw=0)
        for a, t in nodes.items():
            if a in act: a_.scatter(*pos[a], s=18, color=tc[t], zorder=4, ec="white", lw=0.4)
        a_.set_title(f"{ttl}\n{s.kind.eq('victim').sum()} victim payments", fontsize=7.5); panel(a_, p)
    ax[0].set_xlim(-0.25, 3.25); ax[0].set_ylim(-0.03, 1.03)
    for x, t in [(0, "victims"), (1, "tier 1"), (2, "tier 2"), (3, "tier 3")]: ax[0].text(x, -0.07, t, ha="center", fontsize=6.5, color=MUTED, transform=ax[0].transData, clip_on=False)
    fig.legend([Line2D([], [], marker="o", color="w", mfc=c, ms=5) for c in (C["magenta"], C["orange"], C["violet"], C["red"])] + [Line2D([], [], color=C["magenta"], lw=1), Line2D([], [], color=MUTED, lw=1)],
               ["victim account", "tier-1 mule (receives scam payments)", "tier-2 mule", "tier-3 mule (cash-out)", "scam payment", "mule-to-mule transfer"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=6.5)
    tag(fig, f"Synthetic data (sim_v1): network {nid}, {len(members)} mule accounts"); save(fig, "fig12_mule_network_growth")


# ------------------------------------------------------------------ 13 operating characteristics
def fig13():
    ks = ["PRESAGE (concat)", "XGBoost, tabular + graph", "HGT", "XGBoost, tabular"]; P = {k: pred(k) for k in ks}; P = {k: v for k, v in P.items() if v is not None}
    amt = dte.amount.values; cm = CostModel(); fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.5))
    fr = np.logspace(-3.4, -1.3, 40)
    for k, p in P.items():
        rec = []; cost = []
        for f in fr:
            thr = np.quantile(p, 1 - f); rec.append(((p >= thr) & (yt == 1)).sum() / yt.sum()); cost.append(expected_cost(yt, p, amt, thr, cm)["cost"])
        ax[0].plot(fr * 100, rec, color=COL[k], ls=LS[k], label=k); ax[2].plot(fr * 100, np.array(cost) / 1000, color=COL[k], ls=LS[k])
    for a_ in (ax[0], ax[2]): a_.set_xscale("log"); a_.set_xlabel("Alert rate (% of payments)"); a_.axvline(yt.mean() * 100, color=MUTED, lw=0.7, ls=":")
    ax[0].set_ylabel("Recall"); ax[0].set_ylim(0, 1.02); ax[0].set_title("Recall vs alert budget"); panel(ax[0], "a"); ax[0].legend(loc="lower right", fontsize=6)
    ax[2].set_ylabel("Cost (thousand GBP, simulated)"); ax[2].set_title("Cost vs alert budget"); panel(ax[2], "c"); ax[2].set_yscale("log")
    for k, p in P.items():                                                                  # reliability on log-log axes (events are rare)
        q = pd.cut(pd.Series(p), np.logspace(-4, 0, 11), include_lowest=True); gb = pd.DataFrame({"p": p, "y": yt}).groupby(q, observed=True).agg(mp=("p", "mean"), ob=("y", "mean"), n=("y", "size")); gb = gb[(gb.n >= 30) & (gb.ob > 0)]
        ax[1].plot(gb.mp, gb.ob, "o-", color=COL[k], ls=LS[k], ms=3, label=k)
    ax[1].plot([1e-4, 1], [1e-4, 1], color=MUTED, lw=0.7); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(1e-4, 1.2); ax[1].set_ylim(1e-4, 1.2); ax[1].set_xticks([1e-3, 1e-1]); ax[1].set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax[1].set_xlabel("Predicted probability (log bins)"); ax[1].set_ylabel("Observed fraud rate"); ax[1].set_title("Reliability"); panel(ax[1], "b")
    tag(fig); save(fig, "fig13_operating_characteristics")


# ------------------------------------------------------------------ 14 early mule detection
def fig14():
    ks = ["PRESAGE (concat)", "XGBoost, tabular + graph", "HGT", "XGBoost, tabular"]; P = {k: pred(k) for k in ks}; P = {k: v for k, v in P.items() if v is not None}
    d = dte.reset_index(drop=True); scam = d[d.is_app_fraud == 1]; g = scam.groupby("dst").agg(t0=("ts", "min"), delay=("report_delay_s", "min")); g["confirm"] = g.t0 + g.delay
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.85)); grid = np.linspace(-5, 30, 141); bars = {}
    for k, p in P.items():
        thr = np.quantile(p, 1 - yt.mean() * 1.3); fl = d[p >= thr]; tf = fl.groupby("dst").ts.min(); j = g.join(tf.rename("tf"), how="left")
        lead = ((j.confirm - j.tf) / DAY).values; lead = np.where(np.isnan(lead), -np.inf, lead)
        ax[0].plot(grid, [(lead >= x).mean() for x in grid], color=COL[k], ls=LS[k], label=k); bars[k] = ((j.tf <= j.t0).mean(), j.tf.notna().mean())
    ax[0].axvline(0, color=MUTED, lw=0.7, ls=":"); ax[0].set_xlabel("Lead time before report confirmation (days)"); ax[0].set_ylabel("Share of mule accounts warned")
    ax[0].set_ylim(0, 1.02); ax[0].set_title("Early detection of mule accounts"); ax[0].legend(loc="center right", fontsize=6); panel(ax[0], "a")
    y = np.arange(len(bars)); ax[1].barh(y + 0.19, [v[0] for v in bars.values()], 0.36, color=C["blue"], label="flagged by first victim payment"); ax[1].barh(y - 0.19, [v[1] for v in bars.values()], 0.36, color=C["aqua"], label="flagged at any time")
    ax[1].set_yticks(y); ax[1].set_yticklabels(list(bars), fontsize=6.5); ax[1].set_xlim(0, 1.05); ax[1].invert_yaxis(); ax[1].grid(axis="y", visible=False); ax[1].set_xlabel("Share of mule accounts"); ax[1].set_title("Detection coverage"); panel(ax[1], "b")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.0, -0.22), fontsize=6.3); tag(fig, f"Synthetic data (sim_v1): {len(g)} mule accounts, alert budget 1.3x prevalence"); save(fig, "fig14_early_mule_detection")


# ------------------------------------------------------------------ 15 uncertainty
def fig15():
    z = np.load(R / "presage_concat_s42.npz"); p, u = z["app"], z["unc"]; rng = np.random.default_rng(0); fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(wspace=0.3))
    keep = np.linspace(0.3, 1.0, 22); order = np.argsort(u); n = len(p)
    cur = [average_precision_score(yt[order[:int(k * n)]], p[order[:int(k * n)]]) if yt[order[:int(k * n)]].sum() else np.nan for k in keep]
    rnd = np.array([[average_precision_score(yt[s], p[s]) for s in (rng.permutation(n)[:int(k * n)] for k in keep)] for _ in range(20)])
    ax[0].fill_between(keep * 100, np.nanpercentile(rnd, 2.5, 0), np.nanpercentile(rnd, 97.5, 0), color=MUTED, alpha=0.25, lw=0); ax[0].plot(keep * 100, np.nanmean(rnd, 0), color=MUTED, ls="--", label="random abstention (95% band)")
    ax[0].plot(keep * 100, cur, color=C["orange"], marker="o", ms=3, label="abstain on highest MC-dropout uncertainty"); ax[0].set_xlabel("Payments retained (%)"); ax[0].set_ylabel("PR-AUC on retained payments")
    ax[0].legend(loc="lower left", fontsize=6.3); ax[0].set_title("Selective prediction"); panel(ax[0], "a")
    bins = np.logspace(np.log10(max(u.min(), 1e-5)), np.log10(u.max()), 40)
    for lab, m, c in [("legitimate", yt == 0, MUTED), ("fraud", yt == 1, C["red"])]: ax[1].hist(u[m], bins=bins, alpha=0.6, color=c, label=lab)
    ax[1].set_xscale("log"); ax[1].set_xlabel("MC-dropout standard deviation of fraud score"); ax[1].set_ylabel("Payments (count)"); ax[1].legend(fontsize=6.5); ax[1].set_title("Uncertainty by class"); panel(ax[1], "b")
    tag(fig, "Synthetic data (sim_v1): PRESAGE concat, seed 42"); save(fig, "fig15_uncertainty_selective_prediction")


# ------------------------------------------------------------------ 16 critical difference
def fig16():
    f = json.load(open(T / "friedman_nemenyi.json")); rk = f["mean_rank"]; cd = f["critical_difference"]; nm = {"presage:full": "PRESAGE (gated, seed 42)", "presage:concat": "PRESAGE (concat)", "xgboost_tabular+graph_full": "XGBoost tab.+graph",
                                                                                    "catboost_tabular+graph_full": "CatBoost tab.+graph", "presage:no_beh": "PRESAGE no behaviour", "graph_hgt": "HGT"}
    items = sorted(rk.items(), key=lambda kv: kv[1]); k = len(items); lo, hi = 1, k; fig, ax = plt.subplots(figsize=(W2, 2.2)); ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    x = lambda r: r; ax.plot([lo, hi], [0.62, 0.62], color=INK, lw=0.8)
    for r in range(lo, hi + 1): ax.plot([r, r], [0.62, 0.66], color=INK, lw=0.8); ax.text(r, 0.7, str(r), ha="center", fontsize=7)
    ax.text((lo + hi) / 2, 0.97, "Mean rank over test days (lower is better)", ha="center", fontsize=7.5); ax.plot([lo, lo + cd], [0.8, 0.8], color=C["red"], lw=1.6); ax.text(lo + cd / 2, 0.83, f"CD = {cd:.2f}", ha="center", fontsize=6.5, color=C["red"])
    half = k // 2; left, right = items[:half], items[half:]
    for i, (n, r) in enumerate(left): y = 0.5 - 0.1 * i; ax.plot([r, r, lo - 0.25], [0.62, y, y], color=MUTED, lw=0.6); ax.text(lo - 0.28, y, f"{nm.get(n, n)} ({r:.2f})", ha="right", va="center", fontsize=6.8)
    for i, (n, r) in enumerate(right[::-1]): y = 0.5 - 0.1 * i; ax.plot([r, r, hi + 0.25], [0.62, y, y], color=MUTED, lw=0.6); ax.text(hi + 0.28, y, f"({r:.2f}) {nm.get(n, n)}", ha="left", va="center", fontsize=6.8)
    rs = [r for _, r in items]; yb = 0.57; used = []
    for i in range(k):
        j = max([jj for jj in range(i, k) if rs[jj] - rs[i] < cd], default=i)
        if j > i and not any(a <= i and j <= b for a, b in used): ax.plot([rs[i] - 0.03, rs[j] + 0.03], [yb, yb], color=INK, lw=2.2, solid_capstyle="round"); used.append((i, j)); yb -= 0.035
    ax.set_xlim(lo - 1.9, hi + 1.9); tag(fig, f"Synthetic data (sim_v1): Friedman p = {f['p']:.1e}, {f['n_blocks']} test days; bars join models not significantly different (Nemenyi)"); save(fig, "fig16_critical_difference")


# ------------------------------------------------------------------ 17 drift monitors heatmap
def fig17():
    D = pd.read_csv(T / "drift_detectors.csv"); D = D[D.weekly_values.notna()].copy(); wk = list(range(60, 120, 7)); fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    nice = {"amount": "Amount", "pct_balance": "Balance share", "is_mobile": "Mobile share", "failed_ben_adds": "Failed payee adds"}
    for a_, (met, ttl, p) in zip(ax, [("psi", "PSI", "a"), ("js", "Jensen-Shannon", "b"), ("wasserstein", "Wasserstein", "c")]):
        rows = D[D.detector.str.startswith(met + "(")]; M = np.array([json.loads(v) for v in rows.weekly_values]); names = [nice[d.split("(")[1][:-1]] for d in rows.detector]
        im = a_.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(M.max(), 1e-3)); a_.set_xticks(range(len(wk))); a_.set_xticklabels(wk, fontsize=6.3); a_.set_yticks(range(len(names))); a_.set_yticklabels(names); a_.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): a_.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.3, color="white" if M[i, j] > 0.6 * M.max() else INK)
        a_.axvline(5.5, color=C["red"], lw=1.0); a_.set_title(ttl); a_.set_xlabel("Week starting day"); panel(a_, p)
    tag(fig, "Synthetic data (sim_v1): reference days 0-59; red line = sudden drift (day 100)"); save(fig, "fig17_drift_monitors_heatmap")


# ------------------------------------------------------------------ 18 feature importance by group
def fig18():
    from presage.models.tabular import make_supervised
    X = load_features(); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json")); cols = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"] + g["graph"]
    grp = {c: k for k in ["txn", "behaviour", "beneficiary", "device", "graph"] for c in g[k]}; tr = tr0 & cust; y = df.is_app_fraud.values
    Mx = np.nan_to_num(X[cols].values.astype(np.float32)); m = make_supervised("xgboost", (1 - y[tr].mean()) / y[tr].mean(), 42).fit(Mx[tr], y[tr]); imp = pd.Series(m.feature_importances_, index=cols); imp = imp / imp.sum()
    gc = {"txn": C["magenta"], "behaviour": C["blue"], "beneficiary": C["orange"], "device": C["yellow"], "graph": C["violet"]}; gl = {"txn": "Transaction", "behaviour": "Sender behaviour", "beneficiary": "Beneficiary", "device": "Device", "graph": "Graph"}
    top = imp.sort_values(ascending=False).head(14)[::-1]; fig, ax = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.55))
    ax[0].barh(range(len(top)), top.values, 0.65, color=[gc[grp[c]] for c in top.index]); ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels([c.replace("_", " ") for c in top.index], fontsize=6.5)
    ax[0].set_xlabel("Normalised XGBoost gain importance"); ax[0].set_title("Top 14 features"); ax[0].grid(axis="y", visible=False); panel(ax[0], "a")
    gs = imp.groupby(imp.index.map(grp)).sum().sort_values(); ax[1].barh(range(len(gs)), gs.values, 0.6, color=[gc[k] for k in gs.index]); ax[1].set_yticks(range(len(gs))); ax[1].set_yticklabels([gl[k] for k in gs.index])
    for i, v in enumerate(gs.values): ax[1].text(v + 0.01, i, f"{v*100:.0f}%", va="center", fontsize=7)
    ax[1].set_xlim(0, gs.max() * 1.2); ax[1].set_xlabel("Share of total gain"); ax[1].set_title("By feature group"); ax[1].grid(axis="y", visible=False); panel(ax[1], "b")
    tag(fig, "Synthetic data (sim_v1): XGBoost, tabular + graph, seed 42"); save(fig, "fig18_feature_importance")


# ------------------------------------------------------------------ 19 error analysis and banks
def fig19():
    E = pd.read_csv(R / "error_analysis.csv"); B = pd.read_csv(T / "subgroup_by_bank.csv"); fig, ax = plt.subplots(1, 2, figsize=(W2, 3.3), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.45))
    fn = E[(E.error_type == "false_negative") & (E["count"] > 0)].sort_values("count"); fp = E[(E.error_type == "false_positive") & (E["count"] > 0)].sort_values("count")
    rows = list(fp.itertuples()) + [None] + list(fn.itertuples()); y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        if r is None: continue
        c = C["red"] if r.error_type == "false_negative" else C["blue"]; ax[0].barh(yi, r.count, 0.65, color=c); ax[0].text(r.count + 0.5, yi, str(int(r.count)), va="center", fontsize=6.5)
    ax[0].set_yticks(y); ax[0].set_yticklabels(["" if r is None else textwrap.fill(r.category, 44) for r in rows], fontsize=5.8); ax[0].grid(axis="y", visible=False)
    ax[0].set_xlabel("Count in test period (categories overlap)"); ax[0].set_title("Error analysis"); panel(ax[0], "a"); ax[0].legend([plt.Rectangle((0, 0), 1, 1, color=C["blue"]), plt.Rectangle((0, 0), 1, 1, color=C["red"])], ["false positives", "missed scams"], loc="lower right", fontsize=6.3)
    ax[1].scatter(B.fpr * 100, B.recall, s=20 + B.fraud * 1.2, color=C["orange"], ec="white", lw=0.6, zorder=3)
    for r in B.itertuples(): ax[1].annotate(f"bank {r.bank}\n(n={int(r.fraud)} frauds)", (r.fpr * 100, r.recall), textcoords="offset points", xytext=(6, -13), fontsize=5.8)
    ax[1].set_xlabel("False-positive rate (%)"); ax[1].set_ylabel("Recall"); ax[1].set_ylim(0.8, 1.02); ax[1].set_xlim(0.15, B.fpr.max() * 100 + 0.12); ax[1].set_title("Operational check by bank"); panel(ax[1], "b")
    tag(fig, "Synthetic data (sim_v1): PRESAGE gated, seed 42; not a demographic fairness analysis"); save(fig, "fig19_error_analysis_and_banks")


# ------------------------------------------------------------------ 20 ensembles
def fig20():
    from sklearn.metrics import precision_recall_curve
    Z = np.load(R / "ensemble_preds.npz"); E = pd.read_csv(T / "ensemble_results.csv"); Bt = pd.read_csv(T / "ensemble_paired_bootstrap.csv"); yy = Z["y"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.8), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.25], wspace=0.8))
    fam = [("presage", "PRESAGE (concat)", C["orange"], "-"), ("xgboost", "XGBoost", C["blue"], "-"), ("catboost", "CatBoost", C["aqua"], "-."), ("hybrid", "Hybrid", C["violet"], "-")]
    for k, lab, c, ls in fam:
        pr, rc, _ = precision_recall_curve(yy, Z[k]); ax[0].plot(rc, pr, color=c, ls=ls, label=lab + (" ensemble" if k != "hybrid" else ""))
    ax[0].set_xlim(0.6, 1.0); ax[0].set_ylim(0.6, 1.01); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title("Test PR curves"); ax[0].legend(loc="lower left", fontsize=5.8); panel(ax[0], "a")
    names = {"presage": "PRESAGE", "xgboost": "XGBoost", "catboost": "CatBoost", "hybrid_presage_xgboost": "Hybrid"}
    for i, (k, c) in enumerate([("presage", C["orange"]), ("xgboost", C["blue"]), ("catboost", C["aqua"])]):
        sg = E[E.model == f"{k}_single"].test_pr_auc.values; ens = E[E.model == f"{k}_ensemble"].test_pr_auc.iloc[0]
        ax[1].scatter(np.full(len(sg), i) + np.linspace(-0.12, 0.12, len(sg)), sg, s=10, color=c, alpha=0.6, lw=0); ax[1].scatter(i, ens, s=36, marker="D", color=c, ec="white", lw=0.5, zorder=3)
    hv = E[E.model == "hybrid_presage_xgboost"].test_pr_auc.iloc[0]; ax[1].scatter(3, hv, s=36, marker="D", color=C["violet"], ec="white", lw=0.5, zorder=3)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["PRESAGE", "XGB", "CatB.", "Hybrid"], fontsize=6.3); ax[1].set_ylabel("Test PR-AUC (dots: seeds, diamonds: ensembles)", fontsize=6.3); ax[1].set_title("Seeds and ensembles"); ax[1].grid(axis="x", visible=False); panel(ax[1], "b")
    y_ = np.arange(len(Bt))[::-1]; ax[2].axvline(0, color=MUTED, lw=0.7); ax[2].errorbar(Bt.mean_diff, y_, xerr=[Bt.mean_diff - Bt.ci_lo, Bt.ci_hi - Bt.mean_diff], fmt="o", color=C["blue"], ms=3.5, lw=1, capsize=2)
    ax[2].set_yticks(y_); ax[2].set_yticklabels([c.replace(" - ", "\nminus ") for c in Bt.comparison], fontsize=6); ax[2].set_xlabel("PR-AUC difference (95% day-block CI)"); ax[2].set_title("Paired differences"); ax[2].grid(axis="y", visible=False); panel(ax[2], "c")
    tag(fig, "Synthetic data (sim_v1): 3 seeds, rank-average ensembles, hybrid weight chosen on validation"); save(fig, "fig20_ensembles")


for f in (fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19, fig20): run(f)
