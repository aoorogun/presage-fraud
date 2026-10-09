"""Adapters mapping real public datasets into the common ledger schema used by presage.features.*.

The schema carries the columns the feature code needs (ts, src, dst, amount, balance_before, device/ip/session placeholders, bank ids).
Fields a dataset does not provide are filled with NEUTRAL constants and the corresponding feature groups are simply uninformative; they are never fabricated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from presage.utils.common import ROOT

RAW = ROOT / "data" / "raw"
DAY = 86400.0


def _finish(df: pd.DataFrame, n_acc: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["ts"], kind="stable").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df))
    first = pd.Series(df.ts.values).groupby(df.dst.values).transform("min").values        # as-of account age: time since first appearance as a beneficiary
    df["dst_age_days"] = (df.ts.values - first) / DAY
    first_s = pd.Series(df.ts.values).groupby(df.src.values).transform("min").values
    df["src_age_days"] = (df.ts.values - first_s) / DAY
    df["day"] = (df.ts // DAY).astype(int)
    for c, v in dict(device_id=0, ip_id=0, session_s=0.0, failed_ben_adds=0, channel="mobile", report_delay_s=np.nan, scam_type="", campaign_id=-1, net_id=-1).items():
        if c not in df: df[c] = v
    df["device_id"] = df.src.values            # placeholder: one 'device' per account (no sharing information)
    df["ip_id"] = df.src.values
    df["customer_id"] = df.src.values
    df["cross_bank"] = (df.bank_src != df.bank_dst).astype(int)
    acc = pd.DataFrame(dict(account=np.arange(n_acc), account_type="customer", bank=0, created_ts=0.0, is_mule=0, mule_tier=0, mule_net=-1))
    return df, acc


def load_paysim(frac: float = 0.35, seed: int = 0, types=("TRANSFER", "CASH_OUT")):
    """PaySim restricted to TRANSFER/CASH_OUT (the only types carrying fraud), subsampled by sender hash to keep each sender's ledger intact."""
    p = pd.read_csv(RAW / "paysim.csv")
    p = p[p.type.isin(types)].copy()
    keep = (pd.util.hash_pandas_object(p.nameOrig, index=False).values % 1000) < int(frac * 1000)
    p = p[keep]
    codes, uniq = pd.factorize(pd.concat([p.nameOrig, p.nameDest]).values)
    src, dst = codes[:len(p)], codes[len(p):]
    df = pd.DataFrame(dict(ts=p.step.values * 3600.0 + (np.arange(len(p)) % 3600), src=src, dst=dst, amount=p.amount.values, balance_before=p.oldbalanceOrg.values,
                           txn_type=p.type.values, is_app_fraud=p.isFraud.values, bank_src=0, bank_dst=0, src_is_customer=1))
    df.loc[df.balance_before <= 0, "balance_before"] = np.nan
    return _finish(df, len(uniq))


def load_ibm_aml(n_extra_frac: float = 0.06, seed: int = 0, max_rows: int = 1_200_000):
    """IBM AML HI-Small: induced subgraph on (all accounts touching laundering) + a random fraction of other accounts.
    Prevalence therefore rises above the original 0.10%; it is reported with the results."""
    d = pd.read_csv(RAW / "HI-Small_Trans.csv")
    d.columns = ["ts", "bank_src", "src_a", "bank_dst", "dst_a", "amt_recv", "cur_recv", "amount", "cur_pay", "fmt", "label"]
    d["ts"] = (pd.to_datetime(d.ts) - pd.Timestamp("2022-09-01")).dt.total_seconds()
    d["src_k"] = d.bank_src.astype(str) + ":" + d.src_a; d["dst_k"] = d.bank_dst.astype(str) + ":" + d.dst_a
    laund_acc = pd.unique(pd.concat([d.src_k[d.label == 1], d.dst_k[d.label == 1]]))
    rng = np.random.default_rng(seed); others = pd.unique(pd.concat([d.src_k, d.dst_k])); others = others[~np.isin(others, laund_acc)]
    S = set(laund_acc) | set(rng.choice(others, int(n_extra_frac * len(others)), replace=False))
    m = d.src_k.isin(S) & d.dst_k.isin(S); d = d[m]
    if len(d) > max_rows: d = d[(d.label == 1) | (rng.random(len(d)) < (max_rows - d.label.sum()) / len(d))]
    codes, uniq = pd.factorize(pd.concat([d.src_k, d.dst_k]).values)
    df = pd.DataFrame(dict(ts=d.ts.values, src=codes[:len(d)], dst=codes[len(d):], amount=d.amount.values, balance_before=np.nan, txn_type=d.fmt.values,
                           is_app_fraud=d.label.values, bank_src=d.bank_src.values, bank_dst=d.bank_dst.values, src_is_customer=1))
    return _finish(df, len(uniq))


def load_elliptic():
    """Elliptic: node (transaction) classification. Returns features, labels (1 illicit, 0 licit, -1 unknown), timestep, directed edge list."""
    f = pd.read_csv(RAW / "elliptic_txs_features.csv", header=None)
    c = pd.read_csv(RAW / "elliptic_txs_classes.csv"); e = pd.read_csv(RAW / "elliptic_txs_edgelist.csv")
    ids = f[0].values; ts = f[1].values.astype(int); X = f.iloc[:, 2:].values.astype(np.float32)
    lab = c.set_index("txId")["class"].reindex(ids).map({"1": 1, "2": 0, "unknown": -1}).fillna(-1).astype(int).values
    idx = {t: i for i, t in enumerate(ids)}
    src = e.iloc[:, 0].map(idx).values; dst = e.iloc[:, 1].map(idx).values; ok = ~(np.isnan(src.astype(float)) | np.isnan(dst.astype(float)))
    return X, lab, ts, np.stack([src[ok].astype(int), dst[ok].astype(int)])
