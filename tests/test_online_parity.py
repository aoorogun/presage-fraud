"""Train/serve parity: the incremental feature store must reproduce the batch features (up to same-second tie effects)."""
import numpy as np
from presage.features.temporal import build_features
from presage.serving.online_features import OnlineFeatureStore, SERVING_FEATURES


def test_online_equals_batch(small_world):
    df, acc = small_world
    X, _ = build_features(df, acc)
    store = OnlineFeatureStore(); out = np.zeros((len(df), len(SERVING_FEATURES)))
    recs = df[["ts", "src", "dst", "amount", "device_id", "session_s", "failed_ben_adds", "channel", "dst_age_days", "balance_before"]].to_dict("records")
    for i, e in enumerate(recs):
        f = store.features(e); out[i] = [f[c] for c in SERVING_FEATURES]; store.update(e)
    batch = X[SERVING_FEATURES].values.astype(float)
    ok = np.isclose(out, batch, rtol=1e-3, atol=1e-2, equal_nan=True)
    frac = ok.mean(axis=0)
    bad = {c: float(f) for c, f in zip(SERVING_FEATURES, frac) if f < 0.995}
    assert not bad, f"online/batch mismatch >0.5% rows in: {bad}"
