"""Phase 17: adversarial robustness (feature-space, XGBoost on tabular+graph_full, simulated data).
Experiments (all on the chronological test fraud rows; models trained on train only):
  1. evasion by mimicry: attacker-controllable features are moved a fraction alpha toward the legitimate train median.
     two controllable sets: 'victim_coached' (amount, pct_balance, hour, session, failed adds) and
     'mule_camouflage' (pass-through, fan-out, holding time, dispersion), and both.
  2. label poisoning: a fraction of TRAIN frauds relabelled legitimate, model retrained.
  3. feature dropout: random features zeroed at test time (missing feed / partial outage).
Caveat: these are feature-space perturbations, not constrained to be jointly realisable; they bound sensitivity, not real attacker success."""
import json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from presage.data.loaders import load_sim, load_features
from presage.evaluation.metrics import temporal_split
from presage.models.tabular import make_supervised
from presage.utils.common import ROOT, set_seed, get_logger, log_experiment

log = get_logger("adversarial"); set_seed(42)
df, acc = load_sim(); X = load_features(); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json"))
cols = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"] + g["graph"]
cust = df.src_is_customer.values == 1; tr0, va0, te0 = temporal_split(df.day.values); tr, te = tr0 & cust, te0 & cust
y = df.is_app_fraud.values; M = np.nan_to_num(X[cols].values.astype(np.float32))
pw = (1 - y[tr].mean()) / y[tr].mean(); clf = make_supervised("xgboost", pw, 42).fit(M[tr], y[tr])
Mt, yt = M[te], y[te]; base = average_precision_score(yt, clf.predict_proba(Mt)[:, 1])
fr = yt == 1; ci = {c: i for i, c in enumerate(cols)}
victim = [c for c in ["amount", "log_amount", "pct_balance", "hour", "session_s", "failed_ben_adds", "amt_to_hist_mean", "amt_z", "pct_to_hist_max_pct"] if c in ci]
mule = [c for c in cols if c.startswith(("dst_pass_through", "dst_fanout", "dst_rapid", "dst_avg_holding", "dst_since_last_out", "dst_out_sum"))]
legit_med = np.median(M[tr & (y == 0)], axis=0)
rows = [dict(experiment="clean", setting="none", pr_auc=base, recall_at_1pct_fpr=np.nan)]
thr = np.quantile(clf.predict_proba(M[tr & (y == 0)])[:, 1], 0.99)
def evaluate(Mx):
    p = clf.predict_proba(Mx)[:, 1]; return average_precision_score(yt, p), float((p[fr] > thr).mean())
rows[0]["recall_at_1pct_fpr"] = evaluate(Mt)[1]
for name, fs in [("victim_coached", victim), ("mule_camouflage", mule), ("both", victim + mule)]:
    idx = [ci[c] for c in fs]
    for al in (0.25, 0.5, 0.75, 1.0):
        Mx = Mt.copy(); Mx[np.ix_(fr, idx)] = (1 - al) * Mx[np.ix_(fr, idx)] + al * legit_med[idx]
        ap, rc = evaluate(Mx); rows.append(dict(experiment="evasion_mimicry", setting=f"{name}@alpha={al}", n_features_changed=len(idx), pr_auc=ap, recall_at_1pct_fpr=rc)); log.info("%s a=%.2f PR-AUC %.3f recall@1%%FPR %.2f", name, al, ap, rc)
rng = np.random.default_rng(0); tri = np.where(tr)[0]; fi = tri[y[tri] == 1]
for fl in (0.1, 0.3, 0.5):
    yp = y[tri].copy(); pos = np.where(yp == 1)[0]; yp[rng.choice(pos, int(fl * len(pos)), replace=False)] = 0
    c2 = make_supervised("xgboost", (1 - yp.mean()) / yp.mean(), 42).fit(M[tri], yp); p = c2.predict_proba(Mt)[:, 1]
    t2 = np.quantile(c2.predict_proba(M[tr & (y == 0)])[:, 1], 0.99)
    rows.append(dict(experiment="label_poisoning", setting=f"{int(fl*100)}% train frauds relabelled legit", pr_auc=average_precision_score(yt, p), recall_at_1pct_fpr=float((p[fr] > t2).mean())))
    log.info("poison %.0f%% PR-AUC %.3f", fl * 100, rows[-1]["pr_auc"])
for fd in (0.05, 0.1, 0.2, 0.4):
    pr = []
    for s in range(5):
        r2 = np.random.default_rng(s); Mx = Mt.copy(); Mx[:, r2.random(len(cols)) < fd] = 0; pr.append(evaluate(Mx))
    rows.append(dict(experiment="feature_dropout", setting=f"{int(fd*100)}% of features zeroed (mean of 5 draws)", pr_auc=float(np.mean([a for a, _ in pr])), recall_at_1pct_fpr=float(np.mean([b for _, b in pr]))))
    log.info("dropout %.2f PR-AUC %.3f", fd, rows[-1]["pr_auc"])
R = pd.DataFrame(rows); R.to_csv(ROOT / "reports/tables/adversarial_robustness.csv", index=False); print(R.round(3).to_string())
log_experiment("adversarial", "xgboost_graph_full", "sim_v1", 42, {"clean_pr_auc": float(base)})
