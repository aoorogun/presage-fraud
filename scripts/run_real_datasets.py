"""Phase 3/9/10 on REAL public datasets: PaySim (txn fraud), IBM AML HI-Small (laundering), Elliptic (illicit-node detection).
Same leakage-safe pipeline: chronological 60/20/20 split, features from strictly earlier data, class-weighted XGBoost, graph features on/off.
These datasets are NOT APP-labelled: results are reported as transaction-fraud / AML benchmarks only."""
import json, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from presage.data.adapters import load_paysim, load_ibm_aml, load_elliptic
from presage.features.temporal import build_features
from presage.features.network import build_network_features
from presage.evaluation.metrics import temporal_split, best_threshold, thresholded, bootstrap_ci
from presage.utils.common import ROOT, get_logger, log_experiment, set_seed

log = get_logger("real"); set_seed(42); rows = []
which = sys.argv[1] if len(sys.argv) > 1 else "all"
P = dict(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, verbosity=0, random_state=0)


def evaluate(name, fs, y, pv, pt, yv, yt, day_t=None):
    thr = best_threshold(yv, pv, np.ones_like(pv), "f2"); m = thresholded(yt, pt, thr)
    lo, hi = bootstrap_ci(yt, pt, average_precision_score, n_boot=100, block=day_t)
    r = dict(dataset=name, feature_set=fs, pr_auc=average_precision_score(yt, pt), pr_auc_lo=lo, pr_auc_hi=hi, roc_auc=roc_auc_score(yt, pt), recall=m["recall"], precision=m["precision"], f1=m["f1"], f2=m["f2"], mcc=m["mcc"],
             fpr=m["fpr"], test_rows=len(yt), test_positives=int(yt.sum()), test_prevalence=float(yt.mean()))
    rows.append(r); log.info("%s | %s PR-AUC %.3f [%.3f,%.3f] ROC %.3f R %.2f P %.2f (prev %.4f)", name, fs, r["pr_auc"], lo, hi, r["roc_auc"], r["recall"], r["precision"], r["test_prevalence"])
    log_experiment("real_datasets", f"{name}|{fs}", name, 42, {k: float(v) for k, v in r.items() if isinstance(v, (float, np.floating)) and np.isfinite(v)})
    pd.DataFrame(rows).to_csv(ROOT / "reports/tables/real_datasets.csv", index=False)


def run_ledger(name, df, acc, use_network=True):
    y = df.is_app_fraud.values; tr, va, te = temporal_split(df.day.values)
    log.info("%s: %d rows, %d positives (%.4f%%), days %d, split pos %d/%d/%d", name, len(df), y.sum(), 100 * y.mean(), df.day.max() + 1, y[tr].sum(), y[va].sum(), y[te].sum())
    X, g = build_features(df, acc)
    sets = {"txn_only": g["txn"], "txn+behaviour+beneficiary": g["txn"] + g["behaviour"] + g["beneficiary"]}
    if use_network:
        N = build_network_features(df, acc, step_days=1.0 if df.day.max() < 40 else 2.0, use_labels=False, betweenness_every=10**6)
        X = X.merge(N, on="txn_id"); gn = [c for c in N.columns if c != "txn_id" and "conf" not in c]
        sets["txn+behaviour+beneficiary+graph"] = sets["txn+behaviour+beneficiary"] + gn
    for fs, cols in sets.items():
        spw = min(((1 - y[tr].mean()) / max(y[tr].mean(), 1e-6)) ** 0.5, 50)
        m = xgb.XGBClassifier(scale_pos_weight=spw, **P).fit(np.nan_to_num(X.loc[tr, cols].values), y[tr])
        pv, pt = (m.predict_proba(np.nan_to_num(X.loc[k, cols].values))[:, 1] for k in (va, te))
        evaluate(name, fs, y, pv, pt, y[va], y[te], df.day.values[te])
    return X, g


if which in ("all", "paysim"):
    df, acc = load_paysim(); run_ledger("PaySim", df, acc, use_network=True)
if which in ("all", "ibm"):
    df, acc = load_ibm_aml(); run_ledger("IBM-AML-HI-Small(enriched subsample)", df, acc, use_network=True)
if which in ("all", "elliptic"):
    X, lab, ts, ei = load_elliptic(); log.info("Elliptic: %d nodes, %d edges", len(lab), ei.shape[1])
    lbl = lab >= 0; tr = (ts <= 34) & lbl; te = (ts > 34) & lbl; va = (ts > 27) & (ts <= 34) & lbl; tr_fit = (ts <= 27) & lbl
    # neighbour-aggregated features (mean of in/out neighbours' local features, computed within the same time step graph = no future edges)
    import scipy.sparse as sp
    n = len(lab); A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    deg_in = np.asarray(A.sum(0)).ravel(); deg_out = np.asarray(A.sum(1)).ravel()
    inagg = (A.T @ X[:, :20]) / np.maximum(deg_in, 1)[:, None]; outagg = (A @ X[:, :20]) / np.maximum(deg_out, 1)[:, None]
    Xg = np.concatenate([X, inagg, outagg, np.stack([deg_in, deg_out], 1)], 1)
    for fs, M in [("local features (166)", X), ("local + 1-hop aggregated + degree", Xg)]:
        spw = (1 - lab[tr_fit].mean()) / max(lab[tr_fit].mean(), 1e-6)
        m = xgb.XGBClassifier(scale_pos_weight=min(spw, 20), **P).fit(M[tr_fit], lab[tr_fit])
        evaluate("Elliptic", fs, lab, m.predict_proba(M[va])[:, 1], m.predict_proba(M[te])[:, 1], lab[va], lab[te], ts[te])
print(pd.DataFrame(rows).round(3).to_string())
