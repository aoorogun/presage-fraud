"""Future-data leakage tests. The strongest is prefix invariance: features for the first T seconds must be
identical whether or not later data exists. Any feature using future rows would change."""
import numpy as np
from presage.features.temporal import build_features
from presage.features.network import build_network_features


def test_temporal_features_prefix_invariant(small_world):
    df, acc = small_world
    X, _ = build_features(df, acc)
    cut = df.ts.quantile(0.6)
    pre = df[df.ts < cut].copy()
    Xp, _ = build_features(pre, acc)
    a = X.iloc[: len(pre)].drop(columns="txn_id").values
    b = Xp.drop(columns="txn_id").values
    bad = [c for j, c in enumerate(X.columns[1:]) if not np.allclose(a[:, j], b[:, j], equal_nan=True, atol=1e-3)]
    assert not bad, f"future leakage in: {bad}"


def test_network_features_prefix_invariant(small_world):
    df, acc = small_world
    X = build_network_features(df, acc, step_days=4.0)
    cut = 20 * 86400.0  # aligned to a snapshot boundary so both builds share the snapshot grid
    pre = df[df.ts < cut].copy()
    Xp = build_network_features(pre, acc, step_days=4.0)
    a = X.iloc[: len(pre)].drop(columns="txn_id").values
    b = Xp.drop(columns="txn_id").values
    bad = [c for j, c in enumerate(X.columns[1:]) if not np.allclose(a[:, j], b[:, j], equal_nan=True, atol=1e-3)]
    assert not bad, f"future leakage in graph features: {bad}"


def test_labels_do_not_leak_into_features(small_world):
    """Flip every future fraud label: temporal features must not change (they never read labels);
    network features may read labels only through delayed confirmation, tested by changing labels after the cutoff."""
    df, acc = small_world
    X1 = build_network_features(df, acc, step_days=4.0)
    cut = 20 * 86400.0
    df2 = df.copy()
    fut = df2.ts >= cut
    df2.loc[fut, "is_app_fraud"] = 1 - df2.loc[fut, "is_app_fraud"]
    X2 = build_network_features(df2, acc, step_days=4.0)
    early = (df.ts < cut).values
    assert np.allclose(X1[early].drop(columns="txn_id").values, X2[early].drop(columns="txn_id").values, atol=1e-4)


def test_window_is_strictly_before(small_world):
    from presage.features.windows import EventIndex
    ev = EventIndex(np.array([1, 1, 1]), np.array([10.0, 20.0, 20.0]), dict(v=np.array([1.0, 2.0, 4.0])))
    # query at t=20 must see only the t=10 event (not same-second events)
    assert ev.sum("v", np.array([1]), np.array([20.0]), np.inf)[0] == 1.0
    assert ev.count(np.array([1]), np.array([10.0]), np.inf)[0] == 0.0
