"""Phase 11 + 16: PRESAGE full model, fusion variants and component ablations (single seed here; see run_seeds.py)."""
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from presage.data.prepare import prepare
from presage.models.presage import PresageConfig
from presage.models.presage_train import fit_presage
from presage.evaluation.metrics import full_report, best_threshold, CostModel, bootstrap_ci
from presage.utils.common import ROOT, log_experiment, get_logger

log = get_logger("presage")
ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="full"); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--cosine", action="store_true")
a = ap.parse_args()
VARIANTS = {
    "full": {}, "concat": dict(fusion="concat"), "attention": dict(fusion="attention"), "transformer_seq": dict(seq_kind="transformer"),
    "no_txn": dict(use_txn=False), "no_beh": dict(use_beh=False), "no_graph": dict(use_graph=False), "no_mule": dict(use_mule=False),
    "no_temporal": dict(temporal=False), "no_multitask": dict(multitask=False), "no_benfeat": {},
    # candidate improvements (pre-specified; selected on VALIDATION only): same inputs as the GBDT graph_full baseline
    "concat_net": dict(fusion="concat", _netfeat=True), "concat_long": dict(fusion="concat"), "gated_net": dict(_netfeat=True),
}
D = prepare(); df, tr, va, te = D["df"], D["tr"], D["va"], D["te"]; y = D["A"]["y_app"]; amt = df.amount.values
cm = CostModel(); rows = []
for v in a.variants.split(","):
    t0 = time.time(); A = D["A"]; ben_idx = D["ben_idx"]
    if v == "no_benfeat":
        keep = [i for i in range(A["tab"].shape[1]) if i not in set(ben_idx)]
        A = dict(A, tab=A["tab"][:, keep]); ben_idx = []
    vd = dict(VARIANTS[v]); netfeat = vd.pop("_netfeat", False)
    if netfeat:
        D2 = prepare(tab_groups=("txn", "behaviour", "beneficiary", "device", "graph")); A, ben_idx = D2["A"], D2["ben_idx"]
    cfg = PresageConfig(**vd)
    mdl, pv, pt, info = fit_presage(cfg, D["snaps"], D["snap_of"], A, tr, va, te, ben_idx, epochs=a.epochs, seed=a.seed, cosine=a.cosine)
    thr = best_threshold(y[va], pv["app"][va], amt[va], "f2"); thr_c = best_threshold(y[va], pv["app"][va], amt[va], "cost", cm)
    r = full_report(y[te], pt["app"][te], amt[te], thr, cm); rc = full_report(y[te], pt["app"][te], amt[te], thr_c, cm)
    lo, hi = bootstrap_ci(y[te], pt["app"][te], average_precision_score, n_boot=100, block=df.day.values[te])
    ym = A["y_mule"]; mule_ap = average_precision_score(ym[te], pt["mule"][te]) if np.isfinite(pt["mule"][te]).all() else np.nan
    row = dict(variant=v, pr_auc=r["pr_auc"], pr_auc_lo=lo, pr_auc_hi=hi, roc_auc=r["roc_auc"], recall=r["recall"], precision=r["precision"], f1=r["f1"], f2=r["f2"], mcc=r["mcc"],
               fpr=r["fpr"], cost_costthr=rc["cost"], prevented_ratio=rc["prevented_ratio"], mule_pr_auc=mule_ap, train_s=info["train_s"], latency_ms=info["infer_s"] * 1000,
               n_params=sum(p.numel() for p in mdl.parameters()))
    rows.append(row); log.info("%s PR-AUC %.3f [%.3f,%.3f] R %.2f P %.2f mule-AP %.3f (%.0fs)", v, row["pr_auc"], lo, hi, row["recall"], row["precision"], mule_ap, time.time() - t0)
    np.savez_compressed(ROOT / "reports/results" / f"presage_{v}_s{a.seed}{'_cos' if a.cosine else ''}.npz", te_index=np.where(te)[0], app=pt["app"][te], mule=pt["mule"][te], net=pt["net"][te], unc=pt["unc"][te], val_app=pv["app"][va], val_index=np.where(va)[0])
    log_experiment("presage", v, "sim_v1", a.seed, {k: x for k, x in row.items() if isinstance(x, float)}, params=cfg.__dict__, train_seconds=info["train_s"], infer_seconds=info["infer_s"])
    pd.DataFrame(rows).to_csv(ROOT / "reports/tables" / f"presage_{'_'.join(a.variants.split(',')[:3])}_s{a.seed}{'_cos' if a.cosine else ''}.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
