"""Seed ensembles and the PRESAGE + XGBoost hybrid, selected on VALIDATION; the test split is scored once.
Compares like with like: 3-seed PRESAGE ensemble vs 3-seed XGBoost / CatBoost ensembles (tabular + graph features).
Needs reports/results/presage_concat_s{42,1,2}.npz (make seed-variance trains them).
Writes reports/tables/ensemble_results.csv, ensemble_paired_bootstrap.csv, ensemble_weight_sweep.csv and reports/results/ensemble_preds.npz (read by Fig. 20).
Usage: PYTHONPATH=src python3 scripts/run_ensemble.py [--seeds 42 1 2]"""
import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from presage.data.loaders import load_sim, load_features
from presage.evaluation.metrics import temporal_split
from presage.models.tabular import make_supervised
from presage.models.ensemble import rank_ensemble, choose_hybrid_weight, hybrid, paired_block_bootstrap
from presage.utils.common import ROOT

a = argparse.ArgumentParser(); a.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2]); SEEDS = a.parse_args().seeds
R, T = ROOT / "reports/results", ROOT / "reports/tables"
df, acc = load_sim(); X = load_features(); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json"))
cols = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"] + g["graph"]
cust = df.src_is_customer.values == 1; tr0, va0, te0 = temporal_split(df.day.values); tr, va, te = tr0 & cust, va0 & cust, te0 & cust; y = df.is_app_fraud.values
M = np.nan_to_num(X[cols].values.astype(np.float32)); pw = (1 - y[tr].mean()) / y[tr].mean()
P = {"presage": ([], []), "xgboost": ([], []), "catboost": ([], [])}
for s in SEEDS:
    z = np.load(R / f"presage_concat_s{s}.npz"); assert (np.where(te)[0] == z["te_index"]).all(), "test rows differ"
    P["presage"][0].append(z["val_app"]); P["presage"][1].append(z["app"])
    for m in ("xgboost", "catboost"):
        c = make_supervised(m, pw, s).fit(M[tr], y[tr]); P[m][0].append(c.predict_proba(M[va])[:, 1]); P[m][1].append(c.predict_proba(M[te])[:, 1])
E = {k: (rank_ensemble(v[0]), rank_ensemble(v[1])) for k, v in P.items()}
w, sweep = choose_hybrid_weight(y[va], E["presage"][0], E["xgboost"][0])
H = (hybrid(E["presage"][0], E["xgboost"][0], w), hybrid(E["presage"][1], E["xgboost"][1], w))
rows = []
def add(name, pv, pt, **kw): rows.append(dict(model=name, val_pr_auc=ap(y[va], pv), test_pr_auc=ap(y[te], pt), **kw))
for k in P:
    for i, s in enumerate(SEEDS): add(f"{k}_single", P[k][0][i], P[k][1][i], seed=s)
    add(f"{k}_ensemble", *E[k])
add(f"hybrid_presage_xgboost", *H, note=f"w_presage={w}, chosen on validation")
D = pd.DataFrame(rows); D.to_csv(T / "ensemble_results.csv", index=False); print(D.round(4).to_string(index=False))
pd.DataFrame([dict(w_presage=k, val_pr_auc=v) for k, v in sweep.items()]).to_csv(T / "ensemble_weight_sweep.csv", index=False)
B = paired_block_bootstrap(y[te], df.day.values[te], {"PRESAGE ens - XGBoost ens": (E["presage"][1], E["xgboost"][1]), "hybrid - XGBoost ens": (H[1], E["xgboost"][1]),
                                                       "hybrid - PRESAGE ens": (H[1], E["presage"][1]), "hybrid - CatBoost ens": (H[1], E["catboost"][1])})
O = pd.DataFrame([dict(comparison=k, **v) for k, v in B.items()]); O.to_csv(T / "ensemble_paired_bootstrap.csv", index=False); print(O.round(4).to_string(index=False))
np.savez_compressed(R / "ensemble_preds.npz", te_index=np.where(te)[0], y=y[te], presage=E["presage"][1], xgboost=E["xgboost"][1], catboost=E["catboost"][1], hybrid=H[1], w=w)
