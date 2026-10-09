import numpy as np, pytest
from presage.features.temporal import build_features
from presage.graph.snapshots import build_snapshots
from presage.models.graph_train import fit_edge_model, prep_tab
from presage.evaluation.metrics import temporal_split


@pytest.mark.parametrize("kind", ["none", "gcn", "sage", "gat", "rgcn", "hgt", "han", "tgat", "evolvegcn"])
def test_graph_encoders_run(small_world, kind):
    df, acc = small_world
    X, g = build_features(df, acc)
    cols = g["txn"] + g["behaviour"][:10]
    cust = df.src_is_customer.values == 1
    tr0, va0, te0 = temporal_split(df.day.values); tr, va, te = tr0 & cust, va0 & cust, te0 & cust
    tab = prep_tab(np.nan_to_num(X[cols].values.astype(np.float32)), tr)
    snaps, snap_of = build_snapshots(df, acc, step_days=5.0)
    mdl, pv, pt, info = fit_edge_model(kind, snaps, snap_of, df.src.values, df.dst.values, tab, df.is_app_fraud.values, tr, va, te, epochs=1, verbose=False)
    assert np.isfinite(pt[te]).all() and (pt[te] >= 0).all() and (pt[te] <= 1).all()


def test_snapshot_edges_strictly_before_cutoff(small_world):
    df, acc = small_world
    snaps, _ = build_snapshots(df, acc, step_days=5.0)
    # recency attr = log1p((cutoff - last_ts)/day) must be >= 0, i.e. no edge timestamp at/after the cutoff
    for s in snaps:
        if s.pay_ea.shape[0]: assert float(s.pay_ea[:, 2].min()) >= 0.0
