"""Phases 9-10: static / heterogeneous / temporal graph encoders vs the tabular-only twin (same head)."""
import sys, json, argparse, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from presage.data.loaders import load_sim, load_features
from presage.graph.snapshots import build_snapshots
from presage.models.graph_train import fit_edge_model, prep_tab
from presage.evaluation.metrics import temporal_split, full_report, best_threshold, CostModel, bootstrap_ci
from presage.utils.common import ROOT, log_experiment, get_logger

log = get_logger("graphmodels")
ap = argparse.ArgumentParser(); ap.add_argument("--kinds", default="none,gcn,sage,gat,rgcn,hgt,han,tgat,evolvegcn"); ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
df, acc = load_sim(); X = load_features(); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json"))
tab_cols = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"]
cust = df.src_is_customer.values == 1
tr0, va0, te0 = temporal_split(df.day.values); tr, va, te = tr0 & cust, va0 & cust, te0 & cust
y = df.is_app_fraud.values; amt = df.amount.values
tab = prep_tab(np.nan_to_num(X[tab_cols].values.astype(np.float32)), tr)
snaps, snap_of = build_snapshots(df, acc)
cm = CostModel(); rows = []
for kind in a.kinds.split(","):
    t0 = time.time()
    mdl, pv, pt, info = fit_edge_model(kind, snaps, snap_of, df.src.values, df.dst.values, tab, y, tr, va, te, epochs=a.epochs, seed=a.seed)
    thr = best_threshold(y[va], pv[va], amt[va], "f2"); thr_c = best_threshold(y[va], pv[va], amt[va], "cost", cm)
    r = full_report(y[te], pt[te], amt[te], thr, cm); rc = full_report(y[te], pt[te], amt[te], thr_c, cm)
    lo, hi = bootstrap_ci(y[te], pt[te], average_precision_score, n_boot=100, block=df.day.values[te])
    row = dict(model=kind, pr_auc=r["pr_auc"], pr_auc_lo=lo, pr_auc_hi=hi, roc_auc=r["roc_auc"], recall=r["recall"], precision=r["precision"], f1=r["f1"], f2=r["f2"],
               mcc=r["mcc"], fpr=r["fpr"], cost_costthr=rc["cost"], prevented_ratio=rc["prevented_ratio"], train_s=info["train_s"], latency_ms=info["infer_s"] * 1000,
               n_params=sum(p.numel() for p in mdl.parameters()))
    rows.append(row); log.info("%s PR-AUC %.3f [%.3f,%.3f] R %.2f P %.2f (%.0fs)", kind, row["pr_auc"], lo, hi, row["recall"], row["precision"], time.time() - t0)
    np.save(ROOT / "reports/results" / f"pred_graph_{kind}.npy", pt[te])
    log_experiment("graph_models", kind, "sim_v1", a.seed, {k: v for k, v in row.items() if isinstance(v, float)}, n_features=len(tab_cols), train_seconds=info["train_s"], infer_seconds=info["infer_s"])
    pd.DataFrame(rows).to_csv(ROOT / "reports/tables/graph_models.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
