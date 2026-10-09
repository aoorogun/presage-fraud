"""Phases 12-13: drift detectors benchmark, adaptation strategies, label-delay study (walk-forward, XGBoost base learner)."""
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from presage.data.loaders import load_sim
from presage.features.network import build_network_features
from presage.drift.walkforward import Task, run_strategy, DAY
from presage.drift.strategies import all_strategies, _fit, Static, Sliding
from presage.drift.detectors import run_detector, psi, js_div, wasserstein
from presage.utils.common import ROOT, get_logger, log_experiment

log = get_logger("drift")
ap = argparse.ArgumentParser(); ap.add_argument("--delay-days", type=float, default=3.0); ap.add_argument("--part", default="all"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
df, acc = load_sim(); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json"))
X0 = pd.read_parquet(ROOT / "data/processed/sim_v1_features.parquet")
tab = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"]; netc = g["graph"]


def features_for_delay(delay_s):
    """Graph label features depend on label delay, so rebuild them for each delay (cached)."""
    f = ROOT / "data/processed" / f"net_delay_{int(delay_s)}.parquet"
    if f.exists(): N = pd.read_parquet(f)
    else: N = build_network_features(df, acc, label_delay_s=delay_s); N.to_parquet(f)
    M = np.concatenate([X0[tab].values, N[netc].values], 1).astype(np.float32)
    return np.nan_to_num(M)


cust = df.src_is_customer.values == 1; y = df.is_app_fraud.values; ts = df.ts.values; amt = df.amount.values
out = ROOT / "reports/results"
if a.part in ("all", "adapt"):
    M = features_for_delay(a.delay_days * DAY)
    task = Task(M, y, ts, amt, cust, start_day=60, end_day=120, block_days=7, delay_s=a.delay_days * DAY)
    summ = []; curves = {}
    probe = np.where(cust & (df.day.values >= 20) & (df.day.values < 55))[0]       # fixed early-period probe for forgetting
    init_ap = None
    for s in all_strategies():
        t0 = time.time(); r = run_strategy(task, s, a.seed)
        pa = np.array(r["pr_auc"]); bs = np.array(r["block_start"]); post = bs >= 98
        fm = r["state"]["model"]; fp = average_precision_score(y[probe], fm.predict_proba(M[probe])[:, 1])
        rec = r["state"]["retrains"]; alarms = r["state"].get("alarms", [])
        row = dict(strategy=s.name, mean_pr_auc=np.nanmean(pa), pre_drift_pr_auc=np.nanmean(pa[~post]), post_drift_pr_auc=np.nanmean(pa[post]), recall_at_1pct_fpr=np.nanmean(r["recall_at_fpr"]),
                   n_retrains=len(rec) - 1, retrain_seconds=r["retrain_s"], probe_pr_auc_final=fp, first_alarm_day=(alarms[0] if alarms else np.nan), n_alarms=len(alarms))
        summ.append(row); curves[s.name] = dict(block_start=bs.tolist(), pr_auc=pa.tolist(), retrains=rec)
        log.info("%s mean %.3f pre %.3f post %.3f recall@1%%FPR %.2f retrains %d (%.0fs)", s.name, row["mean_pr_auc"], row["pre_drift_pr_auc"], row["post_drift_pr_auc"], row["recall_at_1pct_fpr"], row["n_retrains"], time.time() - t0)
        log_experiment("drift_adaptation", s.name, "sim_v1", a.seed, {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)}, params=dict(delay_days=a.delay_days))
    S = pd.DataFrame(summ)
    base = S.loc[S.strategy == "static", "probe_pr_auc_final"].iloc[0]
    S.to_csv(ROOT / "reports/tables/drift_adaptation.csv", index=False); json.dump(curves, open(out / "drift_curves.json", "w"))
    print(S.round(3).to_string())

if a.part in ("all", "detect"):
    # --- detector benchmark on the miss stream of a STATIC model; ground-truth sudden drift at day 100 (config), gradual drift throughout
    M = features_for_delay(3 * DAY)
    tr_idx = np.where(cust & (df.day.values < 60))[0]; m = _fit(M[tr_idx], y[tr_idx])
    thr = np.sort(m.predict_proba(M[tr_idx][y[tr_idx] == 0])[:, 1])[int(0.99 * (y[tr_idx] == 0).sum())]
    fr = np.where(cust & (y == 1) & (df.day.values >= 60))[0]; fr = fr[np.argsort(ts[fr])]
    miss = (m.predict_proba(M[fr])[:, 1] <= thr).astype(float); t_conf = ts[fr] + df.report_delay_s.values[fr]
    order = np.argsort(t_conf); stream, tt = miss[order], t_conf[order] / DAY          # alarms are raised when labels CONFIRM
    rows = []
    for d in ["adwin", "ddm", "eddm", "page_hinkley", "kswin"]:
        al = np.array(run_detector(d, stream, tt)); drift_day = 100.0
        false_a = int((al < drift_day).sum()); after = al[al >= drift_day]
        rows.append(dict(detector=d, n_alarms=len(al), false_alarms_before_drift=false_a, first_alarm_after_drift_day=float(after[0]) if len(after) else np.nan,
                         detection_delay_days=float(after[0] - drift_day) if len(after) else np.nan, first_alarm_day=float(al[0]) if len(al) else np.nan))
    # --- distribution monitors on raw features (no labels needed): reference = days 0-59, compare weekly windows
    ref = cust & (df.day.values < 60)
    for fname in ["amount", "pct_balance", "is_mobile", "failed_ben_adds"]:
        col = X0[fname].values
        for metric, fn in [("psi", psi), ("js", js_div), ("wasserstein", wasserstein)]:
            vals = []
            for d0 in range(60, 120, 7):
                cur = cust & (df.day.values >= d0) & (df.day.values < d0 + 7)
                vals.append(fn(col[ref], col[cur]))
            thr_m = {"psi": 0.2, "js": 0.05, "wasserstein": 0.2}[metric]     # conventional / assumed alert levels
            days_w = list(range(60, 120, 7)); hit = [d for d, v in zip(days_w, vals) if v > thr_m]
            pre = [v for d, v in zip(days_w, vals) if d < 98]; post = [v for d, v in zip(days_w, vals) if d >= 98]
            rows.append(dict(detector=f"{metric}({fname})", n_alarms=len(hit), false_alarms_before_drift=sum(d < 98 for d in hit), first_alarm_after_drift_day=next((d for d in hit if d >= 98), np.nan),
                             detection_delay_days=np.nan, first_alarm_day=hit[0] if hit else np.nan, mean_pre_drift=float(np.mean(pre)), mean_post_drift=float(np.mean(post)),
                             weekly_values=json.dumps([round(v, 4) for v in vals])))
    pd.DataFrame(rows).to_csv(ROOT / "reports/tables/drift_detectors.csv", index=False); print(pd.DataFrame(rows).drop(columns=["weekly_values"], errors="ignore").round(2).to_string())

if a.part in ("all", "labeldelay"):
    from presage.drift.label_delay import run_label_delay
    run_label_delay(df, acc, X0, g, features_for_delay, a.seed)
