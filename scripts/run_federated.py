"""Phase 14: cross-bank experiment. Each bank only sees transactions where it originates or receives the payment.

Compared: local-only | FedAvg | FedProx | personalised FL | FedAvg+DP (sweep) | FedAvg+secure-agg | centralised pooled (local-view features)
| centralised full-visibility (global features, privacy-violating upper bound).
Cross-bank lift = recall/PR-AUC on payments whose beneficiary sits at a DIFFERENT bank than the sender.
"""
import json, time, argparse
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from presage.data.loaders import load_sim
from presage.features.temporal import build_features
from presage.features.network import build_network_features
from presage.federated.fl import fed_train, local_train, predict, MLP, signed_log, rdp_epsilon
from presage.evaluation.metrics import temporal_split, best_threshold, thresholded
from presage.utils.common import ROOT, get_logger, log_experiment, set_seed

log = get_logger("federated")
ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=15); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
set_seed(a.seed)
df, acc = load_sim(); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json"))
X0 = pd.read_parquet(ROOT / "data/processed/sim_v1_features.parquet")
tabc = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"]; netc = g["graph_structure"]
nb = int(df.bank_src.max()) + 1; y = df.is_app_fraud.values; cust = df.src_is_customer.values == 1
tr0, va0, te0 = temporal_split(df.day.values)


def local_view(b):
    """Bank b's visibility: payments it originates or receives. Features computed on that sub-ledger only."""
    cache = ROOT / "data/processed" / f"fed_bank{b}.parquet"
    vis = ((df.bank_src.values == b) | (df.bank_dst.values == b))
    sub = df[vis].copy()
    if cache.exists(): F = pd.read_parquet(cache)
    else:
        T, _ = build_features(sub, acc); N = build_network_features(sub, acc, use_labels=False)
        F = T.merge(N, on="txn_id"); F.to_parquet(cache)
    return sub, F


clients_tr, clients_va, clients_te, views = [], [], [], []
for b in range(nb):
    sub, F = local_view(b)
    own = (sub.bank_src.values == b) & (sub.src_is_customer.values == 1)                       # bank b scores its own customers' payments
    M = signed_log(np.nan_to_num(F[tabc + netc].values.astype(np.float32))).astype(np.float32)
    idx_all = sub.index.values; d = sub.day.values
    views.append(dict(M=M, idx=idx_all, own=own, day=d, y=sub.is_app_fraud.values, cross=(sub.bank_src.values != sub.bank_dst.values)))
    log.info("bank %d: %d visible payments, %d own-customer rows, %d own frauds", b, len(sub), own.sum(), sub.is_app_fraud.values[own].sum())
tr_d, va_d = int(df.day.max() * 0.6) + 1 - 1, int(df.day.max() * 0.8) + 1 - 1
def split(v):
    tr = v["own"] & (v["day"] < 72); va = v["own"] & (v["day"] >= 72) & (v["day"] < 96); te = v["own"] & (v["day"] >= 96); return tr, va, te
S = [split(v) for v in views]
D_in = views[0]["M"].shape[1]
clients = [dict(X=v["M"][s[0]], y=v["y"][s[0]]) for v, s in zip(views, S)]
Xv = np.concatenate([v["M"][s[1]] for v, s in zip(views, S)]); yv = np.concatenate([v["y"][s[1]] for v, s in zip(views, S)])
Xt = np.concatenate([v["M"][s[2]] for v, s in zip(views, S)]); yt = np.concatenate([v["y"][s[2]] for v, s in zip(views, S)])
ct = np.concatenate([v["cross"][s[2]] for v, s in zip(views, S)]); bt = np.concatenate([np.full(s[2].sum(), i) for i, s in enumerate(S)])
amt_t = np.concatenate([df.amount.values[v["idx"][s[2]]] for v, s in zip(views, S)]); amt_v = np.concatenate([df.amount.values[v["idx"][s[1]]] for v, s in zip(views, S)])
log.info("test rows %d frauds %d (cross-bank frauds %d)", len(yt), yt.sum(), yt[ct].sum())
rows = []


def report(name, pt, pv, extra=None):
    thr = best_threshold(yv, pv, amt_v, "f2")
    m = thresholded(yt, pt, thr)
    cross_pos = (yt == 1) & ct
    row = dict(method=name, pr_auc=average_precision_score(yt, pt), recall=m["recall"], precision=m["precision"], f1=m["f1"], fpr=m["fpr"],
               crossbank_pr_auc=average_precision_score(yt[ct], pt[ct]) if yt[ct].sum() else np.nan, crossbank_recall=float(((pt >= thr) & cross_pos).sum() / max(cross_pos.sum(), 1)))
    row.update(extra or {}); rows.append(row)
    log.info("%-26s PR-AUC %.3f  recall %.2f prec %.2f | cross-bank PR-AUC %.3f recall %.2f", name, row["pr_auc"], row["recall"], row["precision"], row["crossbank_pr_auc"], row["crossbank_recall"])
    log_experiment("federated", name, "sim_v1", a.seed, {k: float(v) for k, v in row.items() if isinstance(v, (float, np.floating)) and np.isfinite(v)})
    pd.DataFrame(rows).to_csv(ROOT / "reports/tables/federated.csv", index=False)


def pred_all(models):  # models: list per bank OR single
    pv, pt = [], []
    for i, (v, s) in enumerate(zip(views, S)):
        m = models[i] if isinstance(models, list) else models
        pv.append(predict(m, v["M"][s[1]])); pt.append(predict(m, v["M"][s[2]]))
    return np.concatenate(pv), np.concatenate(pt)


# 1) local-only
loc = []
for i, c in enumerate(clients):
    m = MLP(D_in); local_train(m, c["X"], c["y"], epochs=a.rounds, seed=i); loc.append(m)
pv, pt = pred_all(loc); report("local_only", pt, pv, dict(bytes_up=0, eps=np.nan))
# 2-4) federated
t0 = time.time(); gm, pers, info = fed_train(clients, D_in, a.rounds, 1, "fedavg", personalise_epochs=3, seed=a.seed)
pv, pt = pred_all(gm); report("fedavg", pt, pv, dict(bytes_up=info["bytes_up"] + info["bytes_down"]))
pv, pt = pred_all(pers); report("fedavg_personalised", pt, pv, dict(bytes_up=info["bytes_up"] + info["bytes_down"]))
gm, _, info = fed_train(clients, D_in, a.rounds, 1, "fedprox", mu=0.05, seed=a.seed)
pv, pt = pred_all(gm); report("fedprox(mu=0.05)", pt, pv, dict(bytes_up=info["bytes_up"] + info["bytes_down"]))
# 5) secure aggregation (functionally identical aggregate, server sees masked sums)
gm, _, info = fed_train(clients, D_in, a.rounds, 1, "fedavg", secure=True, seed=a.seed)
pv, pt = pred_all(gm); report("fedavg+secure_agg(sim)", pt, pv, dict(bytes_up=info["bytes_up"] + info["bytes_down"]))
# 6) differential privacy sweep
for sg in (0.5, 1.0, 2.0):
    gm, _, info = fed_train(clients, D_in, a.rounds, 1, "fedavg", dp_sigma=sg, seed=a.seed)
    pv, pt = pred_all(gm); report(f"fedavg+DP(sigma={sg})", pt, pv, dict(bytes_up=info["bytes_up"] + info["bytes_down"], eps=info["eps"]))
# 7) centralised pooled with LOCAL-view features (raw data pooling, but each bank's features computed from its own ledger)
pool = dict(X=np.concatenate([c["X"] for c in clients]), y=np.concatenate([c["y"] for c in clients]))
cm_ = MLP(D_in); local_train(cm_, pool["X"], pool["y"], epochs=a.rounds, seed=1)
pv, pt = pred_all(cm_); report("centralised_pooled(local feats)", pt, pv)
# 8) centralised FULL-visibility: features from the global ledger (oracle for cross-bank graph sharing)
GM = signed_log(np.nan_to_num(X0[tabc + netc].values.astype(np.float32))).astype(np.float32)
trg = np.concatenate([v["idx"][s[0]] for v, s in zip(views, S)]); vag = np.concatenate([v["idx"][s[1]] for v, s in zip(views, S)]); teg = np.concatenate([v["idx"][s[2]] for v, s in zip(views, S)])
gm2 = MLP(D_in); local_train(gm2, GM[trg], y[trg], epochs=a.rounds, seed=2)
report("centralised_full_visibility(oracle)", predict(gm2, GM[teg]), predict(gm2, GM[vag]))
S_ = pd.DataFrame(rows); print(S_.round(3).to_string())
