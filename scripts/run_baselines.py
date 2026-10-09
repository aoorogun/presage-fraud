"""Phase 6: tabular + anomaly baselines with chronological split. Usage: python scripts/run_baselines.py [--quick]"""
import sys, time, json, argparse
import numpy as np, pandas as pd
from presage.data.loaders import load_sim, load_features
from presage.features.temporal import build_features
from presage.evaluation.metrics import temporal_split, full_report, best_threshold, CostModel, bootstrap_ci
from presage.models.tabular import make_supervised, AnomalyModel
from presage.utils.common import set_seed, log_experiment, get_logger, ROOT
from sklearn.metrics import average_precision_score

log = get_logger("baselines")
ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args(); set_seed(args.seed)

df, acc = load_sim(); X = load_features()
groups = json.load(open(ROOT / "data" / "processed" / "sim_v1_feature_groups.json"))
tab = groups["txn"] + groups["behaviour"] + groups["beneficiary"] + groups["device"]
cols = {"txn_only": groups["txn"], "txn+behaviour": groups["txn"] + groups["behaviour"], "all_tabular": tab,
        "tabular+graph_struct": tab + groups["graph_structure"], "tabular+graph_full": tab + groups["graph"]}
# Task-A population: payments initiated by customers. Splits are chronological (60/20/20 of the horizon).
cust = df.src_is_customer.values == 1
tr0, va0, te0 = temporal_split(df.day.values)
tr, va, te = tr0 & cust, va0 & cust, te0 & cust
y = df.is_app_fraud.values; amt = df.amount.values
np.save(ROOT / "reports" / "results" / "test_index.npy", np.where(te)[0])
log.info("train/val/test rows %d/%d/%d; frauds %d/%d/%d", tr.sum(), va.sum(), te.sum(), y[tr].sum(), y[va].sum(), y[te].sum())
cm = CostModel(); rows = []
fs_list = ["txn_only", "all_tabular"] if args.quick else list(cols)
import os
models = [m for m in ["logreg", "random_forest", "xgboost", "lightgbm", "catboost"] if m not in os.environ.get("PRESAGE_SKIP", "").split(",")]   # e.g. PRESAGE_SKIP=lightgbm
for fs in fs_list:
    M = X[cols[fs]].values.astype(np.float32); M = np.nan_to_num(M)
    pw = (1 - y[tr].mean()) / y[tr].mean()
    for m in models:
        t0 = time.time(); clf = make_supervised(m, pw, args.seed).fit(M[tr], y[tr]); tt = time.time() - t0
        t0 = time.time(); pv = clf.predict_proba(M[va])[:, 1]; pt = clf.predict_proba(M[te])[:, 1]; ti = (time.time() - t0) / (va.sum() + te.sum())
        thr = best_threshold(y[va], pv, amt[va], "f2")
        thr_c = best_threshold(y[va], pv, amt[va], "cost", cm)
        r = full_report(y[te], pt, amt[te], thr, cm); rc = full_report(y[te], pt, amt[te], thr_c, cm)
        lo, hi = bootstrap_ci(y[te], pt, average_precision_score, n_boot=100, block=df.day.values[te])
        row = dict(feature_set=fs, model=m, pr_auc=r["pr_auc"], pr_auc_lo=lo, pr_auc_hi=hi, roc_auc=r["roc_auc"], recall=r["recall"], precision=r["precision"],
                   f1=r["f1"], f2=r["f2"], mcc=r["mcc"], fpr=r["fpr"], cost_f2thr=r["cost"], cost_costthr=rc["cost"], prevented_ratio=rc["prevented_ratio"],
                   latency_ms=ti * 1000, train_s=tt)
        rows.append(row); log.info("%s %s PR-AUC %.3f [%.3f,%.3f] R %.2f P %.2f", fs, m, row["pr_auc"], lo, hi, row["recall"], row["precision"])
        log_experiment("baselines", f"{m}|{fs}", "sim_v1", args.seed, {k: v for k, v in row.items() if isinstance(v, float)}, n_features=len(cols[fs]), train_seconds=tt, infer_seconds=ti)
        np.save(ROOT / "reports" / "results" / f"pred_{m}_{fs}.npy", pt)
    if fs == "tabular+graph_full":
        for a in (["isolation_forest", "autoencoder"] if args.quick else ["isolation_forest", "lof", "ocsvm", "autoencoder", "vae"]):
            t0 = time.time(); am = AnomalyModel(a, args.seed).fit(M[tr & (y == 0)]); tt = time.time() - t0
            pv, pt = am.score(M[va]), am.score(M[te]); thr = best_threshold(y[va], pv, amt[va], "f2")
            r = full_report(y[te], pt, amt[te], thr, cm)
            rows.append(dict(feature_set=fs, model=a, pr_auc=r["pr_auc"], roc_auc=r["roc_auc"], recall=r["recall"], precision=r["precision"], f1=r["f1"], f2=r["f2"], mcc=r["mcc"], fpr=r["fpr"], cost_f2thr=r["cost"], train_s=tt))
            log.info("anomaly %s PR-AUC %.3f ROC %.3f", a, r["pr_auc"], r["roc_auc"])
res = pd.DataFrame(rows); res.to_csv(ROOT / "reports" / "tables" / "baselines.csv", index=False); print(res.round(3).to_string())
