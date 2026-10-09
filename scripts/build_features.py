"""Phase 5: build leakage-safe temporal + network features for the simulated world and store them with their group map."""
import json, time, argparse
import pandas as pd
from presage.data.loaders import load_sim
from presage.features.temporal import build_features
from presage.features.network import build_network_features
from presage.utils.common import ROOT, get_logger

log = get_logger("build_features")
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="sim_v1"); ap.add_argument("--step-days", type=float, default=4.0)
ap.add_argument("--label-delay", type=float, default=None, help="override per-fraud report delay (seconds) for graph label features")
a = ap.parse_args()
df, acc = load_sim(a.name)
t = time.time(); X, g = build_features(df, acc); log.info("temporal %.1fs", time.time() - t)
t = time.time(); N = build_network_features(df, acc, step_days=a.step_days, label_delay_s=a.label_delay); log.info("network %.1fs", time.time() - t)
g["graph"] = [c for c in N.columns if c != "txn_id"]
g["graph_label"] = [c for c in N.columns if "conf" in c]         
g["graph_structure"] = [c for c in g["graph"] if c not in g["graph_label"]]
full = X.merge(N, on="txn_id")
full.to_parquet(ROOT / "data" / "processed" / f"{a.name}_features.parquet")
json.dump(g, open(ROOT / "data" / "processed" / f"{a.name}_feature_groups.json", "w"))
log.info("saved %d features", full.shape[1] - 1)
