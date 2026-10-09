"""Assemble all model inputs (tabular, sequences, snapshots, labels) from the simulated world."""
from __future__ import annotations
import json
import numpy as np
from presage.data.loaders import load_sim, load_features
from presage.evaluation.metrics import temporal_split
from presage.graph.snapshots import build_snapshots
from presage.models.graph_train import prep_tab
from presage.models.presage_train import make_aux_labels
from presage.models.sequence import step_features, build_sequences
from presage.utils.common import ROOT


def prepare(name="sim_v1", step_days=4.0, seq_len=10, label_delay_s=None, drop_groups=(), tab_groups=("txn", "behaviour", "beneficiary", "device")):
    df, acc = load_sim(name); X = load_features(name)
    g = json.load(open(ROOT / "data/processed" / f"{name}_feature_groups.json"))
    tab_cols = [c for grp in tab_groups if grp not in drop_groups for c in g[grp]]
    ben_cols_all = g["beneficiary"]
    cust = df.src_is_customer.values == 1
    tr0, va0, te0 = temporal_split(df.day.values); tr, va, te = tr0 & cust, va0 & cust, te0 & cust
    tab = prep_tab(np.nan_to_num(X[tab_cols].values.astype(np.float32)), tr)
    ben_idx = [i for i, c in enumerate(tab_cols) if c in set(ben_cols_all)]
    steps = step_features(df, X)
    mu, sd = steps[tr].mean(0), steps[tr].std(0) + 1e-6
    seq, mask = build_sequences(df, (steps - mu) / sd, seq_len)
    snaps, snap_of = build_snapshots(df, acc, step_days, label_delay_s)
    y_mule, y_net = make_aux_labels(df, acc)
    A = dict(src=df.src.values, dst=df.dst.values, tab=tab, seq=seq, mask=mask, y_app=df.is_app_fraud.values.astype(np.float32), y_mule=y_mule, y_net=y_net)
    return dict(df=df, acc=acc, X=X, groups=g, tab_cols=tab_cols, ben_idx=ben_idx, A=A, snaps=snaps, snap_of=snap_of, tr=tr, va=va, te=te, cust=cust)
