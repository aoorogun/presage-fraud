"""Train the real-time serving models on features produced by the ONLINE store (guaranteeing train/serve parity).
Outputs models/serving/{app_s*.json, mule.json, policy.json, meta.json}."""
import json, time, pickle
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from presage.data.loaders import load_sim
from presage.serving.online_features import OnlineFeatureStore, SERVING_FEATURES
from presage.evaluation.metrics import temporal_split
from presage.evaluation.policy import optimise_policy, PolicyCosts
from presage.models.presage_train import make_aux_labels
from presage.utils.common import ROOT, get_logger

log = get_logger("serving-train")
df, acc = load_sim(); y = df.is_app_fraud.values; cust = df.src_is_customer.values == 1
ym, _ = make_aux_labels(df, acc)
store = OnlineFeatureStore(); F = np.full((len(df), len(SERVING_FEATURES)), np.nan, dtype=np.float32); lat = []
cols = ["ts", "src", "dst", "amount", "device_id", "session_s", "failed_ben_adds", "channel", "dst_age_days", "balance_before"]
recs = df[cols].to_dict("records")
t0 = time.time()
for i, e in enumerate(recs):
    if cust[i]:
        s = time.perf_counter(); f = store.features(e); lat.append(time.perf_counter() - s)
        F[i] = [f[c] for c in SERVING_FEATURES]
    store.update(e)
log.info("replayed %d events in %.0fs; online feature latency p50 %.0f us p99 %.0f us", len(recs), time.time() - t0, 1e6 * np.percentile(lat, 50), 1e6 * np.percentile(lat, 99))
np.save(ROOT / "data/processed/serving_features.npy", F)
F = np.nan_to_num(F)
tr0, va0, te0 = temporal_split(df.day.values); tr, va, te = tr0 & cust, va0 & cust, te0 & cust
P = dict(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=20, tree_method="hist", n_jobs=2, verbosity=0)
models = []
for sd in range(3):
    m = xgb.XGBClassifier(random_state=sd, **P).fit(F[tr], y[tr]); m.save_model(str(ROOT / f"models/serving/app_s{sd}.json")); models.append(m)
pv = np.mean([m.predict_proba(F[va])[:, 1] for m in models], 0); pt = np.mean([m.predict_proba(F[te])[:, 1] for m in models], 0)
log.info("serving model val PR-AUC %.3f test PR-AUC %.3f", average_precision_score(y[va], pv), average_precision_score(y[te], pt))
dst_cols = [c for c in SERVING_FEATURES if c.startswith("dst_")]; di = [SERVING_FEATURES.index(c) for c in dst_cols]
mm = xgb.XGBClassifier(random_state=0, **dict(P, scale_pos_weight=5)).fit(F[tr][:, di], ym[tr]); mm.save_model(str(ROOT / "models/serving/mule.json"))
log.info("mule model test PR-AUC %.3f", average_precision_score(ym[te], mm.predict_proba(F[te][:, di])[:, 1]))
pol = optimise_policy(pv, y[va], df.amount.values[va]); json.dump(dict(t1=pol.t1, t2=pol.t2, t3=pol.t3), open(ROOT / "models/serving/policy.json", "w"))
ev = pol.expected_cost(pt, y[te], df.amount.values[te]); log.info("policy %s -> test expected cost %.0f, prevented %.0f, intervention rate %.4f", (round(pol.t1, 4), round(pol.t2, 4), round(pol.t3, 4)), ev["total"], ev["prevented"], ev["intervention_rate"])
no_action = optimise_policy.__globals__["Policy"](2, 2, 2).expected_cost(pt, y[te], df.amount.values[te])
log.info("no-intervention test cost %.0f", no_action["total"])
json.dump(dict(features=SERVING_FEATURES, mule_features=dst_cols, val_pr_auc=float(average_precision_score(y[va], pv)), test_pr_auc=float(average_precision_score(y[te], pt)),
               online_latency_us_p50=float(1e6 * np.percentile(lat, 50)), online_latency_us_p99=float(1e6 * np.percentile(lat, 99)),
               test_policy_cost=ev["total"], test_no_action_cost=no_action["total"], test_prevented=ev["prevented"]), open(ROOT / "models/serving/meta.json", "w"), indent=1)
# reference distribution for monitoring (train period)
np.save(ROOT / "models/serving/reference_features.npy", F[tr][:20000])
