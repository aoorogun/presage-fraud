"""Multi-seed comparison of PRESAGE (full, concat) against XGBoost/CatBoost on tabular+graph features.
Writes reports/tables/seed_variance.csv, which Fig. 2 reads. With --train, missing PRESAGE seeds are trained first
(about 25 minutes per seed on a laptop CPU). Usage: PYTHONPATH=src python3 scripts/run_seed_variance.py [--train] [--seeds 42 1 2]"""
import argparse, json, subprocess, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from presage.data.loaders import load_sim, load_features
from presage.evaluation.metrics import temporal_split
from presage.models.tabular import make_supervised
from presage.utils.common import ROOT

a = argparse.ArgumentParser(); a.add_argument("--train", action="store_true"); a.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2]); a.add_argument("--epochs", type=int, default=12)
a = a.parse_args(); R = ROOT / "reports/results"
for s in a.seeds:
    if not all((R / f"presage_{v}_s{s}.npz").exists() for v in ("full", "concat")):
        if not a.train: sys.exit(f"missing PRESAGE results for seed {s}; rerun with --train or run: python3 scripts/run_presage.py --variants full,concat --epochs {a.epochs} --seed {s}")
        subprocess.run([sys.executable, str(ROOT / "scripts/run_presage.py"), "--variants", "full,concat", "--epochs", str(a.epochs), "--seed", str(s)], check=True)
df, acc = load_sim(); X = load_features(); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json"))
cols = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"] + g["graph"]
cust = df.src_is_customer.values == 1; tr0, va0, te0 = temporal_split(df.day.values); tr, va, te = tr0 & cust, va0 & cust, te0 & cust; y = df.is_app_fraud.values
M = np.nan_to_num(X[cols].values.astype(np.float32)); pw = (1 - y[tr].mean()) / y[tr].mean(); rows = []
for s in a.seeds:
    for m in ("xgboost", "catboost"):
        c = make_supervised(m, pw, s).fit(M[tr], y[tr]); rows.append(dict(model=m + "_graph_full", seed=s, test_pr_auc=ap(y[te], c.predict_proba(M[te])[:, 1]), val_pr_auc=ap(y[va], c.predict_proba(M[va])[:, 1])))
    for v in ("full", "concat"):
        z = np.load(R / f"presage_{v}_s{s}.npz"); rows.append(dict(model="presage_" + v, seed=s, test_pr_auc=ap(y[te], z["app"]), val_pr_auc=ap(y[va], z["val_app"])))
D = pd.DataFrame(rows); D.to_csv(ROOT / "reports/tables/seed_variance.csv", index=False); print(D.round(3).to_string(index=False)); print(D.groupby("model")[["test_pr_auc", "val_pr_auc"]].agg(["mean", "std"]).round(3))
