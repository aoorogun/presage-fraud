"""Dataset access helpers: simulated world (primary), features, chronological splits."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from presage.utils.common import ROOT


def load_sim(name: str = "sim_v1"):
    d = ROOT / "data"
    df = pd.read_parquet(d / "simulated" / f"{name}_txns.parquet")
    acc = pd.read_parquet(d / "simulated" / f"{name}_accounts.parquet")
    return df, acc


def load_features(name: str = "sim_v1") -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / f"{name}_features.parquet")
