import numpy as np


def test_simulation_basic(small_world):
    df, acc = small_world
    assert df.ts.is_monotonic_increasing
    assert 0 < df.is_app_fraud.mean() < 0.05
    assert set(df.txn_type) >= {"legit", "scam", "mule_forward"}
    assert (df.amount > 0).all()
    mules = set(acc.account[acc.is_mule == 1])
    assert set(df.dst[df.is_app_fraud == 1]).issubset(mules), "every scam payment must land on a mule account"


def test_no_pii_columns(small_world):
    df, _ = small_world
    assert not any(c in df.columns for c in ("name", "email", "phone", "address"))
