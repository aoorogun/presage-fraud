"""Phase 16/20: master tables, paired statistical tests, APP-specific metrics, error analysis, fairness note.
Reads predictions written by the run_* scripts (all aligned to the chronological test rows of customer payments)."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from presage.data.prepare import prepare
from presage.evaluation.metrics import best_threshold, thresholded, CostModel, full_report
from presage.evaluation.policy import optimise_policy, Policy
from presage.evaluation import stats as S
from presage.utils.common import ROOT

R = ROOT / "reports" / "results"; T = ROOT / "reports" / "tables"
df, acc = __import__("presage.data.loaders", fromlist=["x"]).load_sim()
cust = df.src_is_customer.values == 1
from presage.evaluation.metrics import temporal_split
tr0, va0, te0 = temporal_split(df.day.values); va, te = va0 & cust, te0 & cust
y = df.is_app_fraud.values; amt = df.amount.values; yt = y[te]; day_t = df.day.values[te]
preds = {}
for f in glob.glob(str(R / "pred_*.npy")):
    n = os.path.basename(f)[5:-4]; p = np.load(f)
    if len(p) == te.sum(): preds[n] = p
for f in glob.glob(str(R / "presage_*_s42.npz")):
    z = np.load(f); v = os.path.basename(f)[8:-8]; preds["presage:" + v] = z["app"]
print("models with aligned test predictions:", len(preds))
cm = CostModel(); rows = []
# val predictions are only stored for PRESAGE; for the others use a quantile threshold calibrated so the alert rate on test equals the PRESAGE F2 alert rate (reported separately)
for n, p in preds.items():
    r = full_report(yt, p, amt[te], np.quantile(p, 1 - yt.mean() * 1.3), cm)         # operating point: alert on top 1.3x prevalence (a fixed-budget comparison)
    ci = __import__("presage.evaluation.metrics", fromlist=["x"]).bootstrap_ci(yt, p, average_precision_score, n_boot=200, block=day_t)
    rows.append(dict(model=n, pr_auc=r["pr_auc"], pr_auc_ci_lo=ci[0], pr_auc_ci_hi=ci[1], roc_auc=r["roc_auc"], recall_budget=r["recall"], precision_budget=r["precision"], f1_budget=r["f1"], mcc_budget=r["mcc"],
                     fpr_budget=r["fpr"], precision_at_100=r["precision@100"], recall_at_500=r["recall@500"], ndcg_at_1000=r["ndcg@1000"], expected_cost_budget=r["cost"], prevented_ratio=r["prevented_ratio"]))
M = pd.DataFrame(rows).sort_values("pr_auc", ascending=False); M.to_csv(T / "master_results.csv", index=False)
try: md = M.round(3).to_markdown(index=False)
except ImportError: md = M.round(3).to_string(index=False)      # tabulate not installed
open(T / "master_results.md", "w").write(md)
print(M.round(3).head(15).to_string(index=False))

# ---- paired tests of the headline model against the strongest baselines
ref = "presage:full"
if ref in preds:
    cands = [m for m in M.model.tolist() if m != ref][:8]; res = []; pv = {}
    for c in cands:
        b = S.paired_bootstrap_diff(yt, preds[ref], preds[c], block=day_t, n_boot=400); a1, a2, z, pd_ = S.delong_test(yt, preds[ref], preds[c])
        thr_r, thr_c = np.quantile(preds[ref], 1 - yt.mean() * 1.3), np.quantile(preds[c], 1 - yt.mean() * 1.3)
        mc = S.mcnemar(yt, (preds[ref] >= thr_r).astype(int), (preds[c] >= thr_c).astype(int))
        res.append(dict(vs=c, delta_pr_auc=b["diff"], ci_lo=b["lo"], ci_hi=b["hi"], p_bootstrap=b["p"], auc_ref=a1, auc_other=a2, p_delong=pd_, mcnemar_p=mc["p"])); pv[c] = b["p"]
    hp = S.holm(pv)
    for r in res: r["p_bootstrap_holm"] = hp[r["vs"]]
    pd.DataFrame(res).to_csv(T / "stat_tests_vs_presage.csv", index=False); print(pd.DataFrame(res).round(4).to_string(index=False))
    # Friedman / Nemenyi over test days with >=1 fraud
    days = [d for d in np.unique(day_t) if yt[day_t == d].sum() >= 2]; names = [ref] + cands[:5]
    sc = np.array([[average_precision_score(yt[day_t == d], preds[m][day_t == d]) for m in names] for d in days])
    fr = S.friedman_nemenyi(sc, names); json.dump(fr, open(T / "friedman_nemenyi.json", "w"), indent=1); print("Friedman", fr)

# ---- APP-specific table: pre-payment detection, early mule detection, warning lead time, simulated money prevented
if ref in preds:
    p = preds[ref]; z = np.load(R / "presage_full_s42.npz"); pv_ = z["val_app"]
    yv = y[va]; pol = optimise_policy(pv_, yv, amt[va], grid=25); dec = pol.decide(p)
    flagged = dec >= 1; fr_idx = np.where(te)[0]
    d = df.iloc[fr_idx].copy(); d["score"] = p; d["flag"] = flagged; d["action"] = dec
    scam = d[d.is_app_fraud == 1]
    caught = scam[scam.flag]
    # mule early detection: first flagged payment to a mule vs its first victim-report confirmation
    first_scam = scam.groupby("dst").agg(t0=("ts", "min"), delay=("report_delay_s", "min")); first_scam["confirm"] = first_scam.t0 + first_scam.delay
    first_flag = d[d.flag & (d.dst.isin(first_scam.index))].groupby("dst").ts.min()
    jm = first_scam.join(first_flag.rename("t_flag"), how="left"); jm["early"] = jm.t_flag.notna() & (jm.t_flag <= jm.t0)       # flagged at or before the first victim payment
    jm["lead_days"] = (jm.confirm - jm.t_flag) / 86400
    app_row = dict(model=ref, scam_payments=int(len(scam)), scam_detected_pre_payment=int(len(caught)), recall_pre_payment=len(caught) / max(len(scam), 1),
                   mule_accounts=int(len(jm)), mule_flagged_before_first_victim_payment=int(jm.early.sum()), mule_flagged_before_report_confirmation=int((jm.t_flag.notna() & (jm.t_flag < jm.confirm)).sum()),
                   mean_warning_lead_days=float(jm.lead_days[jm.t_flag.notna() & (jm.t_flag < jm.confirm)].mean()), simulated_gbp_prevented_expected=pol.expected_cost(p, y[te], amt[te])["prevented"],
                   simulated_gbp_total_scam_value=float(scam.amount.sum()), policy_thresholds=json.dumps([pol.t1, pol.t2, pol.t3]), intervention_rate=float((dec > 0).mean()), hold_rate=float((dec == 3).mean()),
                   policy_cost=pol.expected_cost(p, y[te], amt[te])["total"], no_action_cost=Policy(2, 2, 2).expected_cost(p, y[te], amt[te])["total"])
    pd.DataFrame([app_row]).to_csv(T / "app_specific_metrics.csv", index=False); print(json.dumps(app_row, indent=1, default=float))
    # ---- error analysis
    X = pd.read_parquet(ROOT / "data/processed/sim_v1_features.parquet").set_index("txn_id").loc[d.txn_id.values]
    d = d.assign(new_ben=X.new_beneficiary.values, pct=X.pct_balance.values, dst_in=X.dst_in_cnt_total.values, fanin24=X.dst_fanin_new_24h.values, new_dev=X.new_device.values)
    acc_c = acc.set_index("account"); d["dst_is_customer"] = (acc_c.account_type.reindex(d.dst.values).values == "customer")
    fn = d[(d.is_app_fraud == 1) & ~d.flag]; fp = d[(d.is_app_fraud == 0) & d.flag]
    cats = []
    def add(kind, cat, sub, note):
        cats.append(dict(error_type=kind, category=cat, count=len(sub), share=len(sub) / max(1, len(fn) if kind == "false_negative" else len(fp)), example_txn_ids=" ".join(map(str, sub.txn_id.head(5))), note=note))
    add("false_negative", "scam to recruited genuine customer account (old, normal history)", fn[fn.dst_is_customer], "mule hides inside a real account")
    add("false_negative", "scam beneficiary with limited history (<=2 prior inbound)", fn[(fn.dst_in <= 2) & ~fn.dst_is_customer], "cold-start beneficiary")
    add("false_negative", "post sudden-drift structuring (day>=100, small split payments)", fn[fn.day >= 100], "strategy change")
    add("false_negative", "cross-bank scam payment", fn[fn.cross_bank == 1], "partial visibility")
    add("false_negative", "low-value purchase scam", fn[fn.scam_type == "purchase"], "small amounts blend in")
    add("false_negative", "romance scam (repeat moderate payments)", fn[fn.scam_type == "romance"], "payments resemble habitual transfers")
    add("false_negative", "other", fn[~fn.index.isin(fn[fn.dst_is_customer | ((fn.dst_in <= 2) & ~fn.dst_is_customer) | (fn.day >= 100) | (fn.cross_bank == 1) | fn.scam_type.isin(["purchase", "romance"])].index)], "")
    add("false_positive", "legitimate large first-time transfer (new payee, >30% of balance)", fp[(fp.new_ben == 1) & (fp.pct > 0.3)], "house deposit / family transfer pattern")
    add("false_positive", "legitimate payment to a young beneficiary account", fp[(fp.dst_age_days < 30)], "new business / neo-bank account")
    add("false_positive", "payment from a new device", fp[fp.new_dev == 1], "device upgrade")
    add("false_positive", "payee with many new senders (marketplace-like)", fp[fp.fanin24 >= 3], "legitimate high fan-in")
    add("false_positive", "mule-account background activity labelled legitimate", fp[fp.dst.isin(acc[acc.is_mule == 1].account)], "payment to mule account that was not a scam payment (label noise by design)")
    E = pd.DataFrame(cats); E.to_csv(R / "error_analysis.csv", index=False); print(E.to_string(index=False))
    # ---- subgroup analysis by institution (NOT demographic fairness)
    g = []
    for b in range(int(d.bank_src.max()) + 1):
        s = d[d.bank_src == b]
        g.append(dict(bank=b, n=len(s), fraud=int(s.is_app_fraud.sum()), recall=float(s.flag[s.is_app_fraud == 1].mean()) if s.is_app_fraud.sum() else np.nan, fpr=float(s.flag[s.is_app_fraud == 0].mean()), intervention_rate=float(s.flag.mean())))
    pd.DataFrame(g).to_csv(T / "subgroup_by_bank.csv", index=False)
open(R / "fairness_note.md", "w").write("# Fairness note\n\nThe simulated world contains no demographic attributes and none were invented. A demographic fairness analysis is therefore NOT performed. "
    "`reports/tables/subgroup_by_bank.csv` compares false-positive rate, recall and intervention rate across the five simulated institutions; this is an operational subgroup check, not a fairness audit. "
    "Any deployment must evaluate error-rate parity on legitimately held protected or proxy attributes under the operator's legal basis.\n")

# ---- benchmark matrix: every compared method is a RE-IMPLEMENTATION run on the same split; no numbers are copied from papers.
REF = {"logreg": "Logistic regression (baseline)", "random_forest": "Breiman 2001, Machine Learning 45", "xgboost": "Chen & Guestrin 2016, KDD", "lightgbm": "Ke et al. 2017, NeurIPS",
       "catboost": "Prokhorenkova et al. 2018, NeurIPS", "graph_gcn": "Kipf & Welling 2017, ICLR", "graph_sage": "Hamilton et al. 2017, NeurIPS", "graph_gat": "Velickovic et al. 2018, ICLR",
       "graph_rgcn": "Schlichtkrull et al. 2018, ESWC", "graph_hgt": "Hu et al. 2020, WWW", "graph_han": "Wang et al. 2019, WWW", "graph_tgat": "Xu et al. 2020, ICLR (TGAT-style snapshot variant)",
       "graph_evolvegcn": "Pareja et al. 2020, AAAI (EvolveGCN-O)", "graph_none": "Tabular-only twin of the GNN head (control)", "presage:full": "This work (PRESAGE)"}
bm = M[["model", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi", "roc_auc", "recall_budget", "precision_budget", "expected_cost_budget"]].copy()
bm["reference"] = bm.model.map(lambda n: next((v for k, v in REF.items() if n == k or n.startswith(k + "_") or n.startswith("presage:full")), "")); 
bm["reference"] = bm.model.map(lambda n: REF.get(n) or next((v for k, v in REF.items() if n.startswith(k + "_")), "this work (ablation/variant)" if n.startswith("presage:") else ""))
bm["note"] = "re-implementation, same chronological split and feature set where applicable; published numbers are not comparable and are not reproduced"
bm.to_csv(T / "benchmark_matrix.csv", index=False); print("benchmark matrix rows:", len(bm))
