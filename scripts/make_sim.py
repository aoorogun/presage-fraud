"""Generate the SYNTHETIC APP/mule world (sim_v1). Everything it produces is simulated; see docs/simulation.md."""
from presage.simulation.app_world import simulate, SimConfig
from presage.utils.common import ROOT
df, acc = simulate(SimConfig())
(ROOT / "data/simulated").mkdir(parents=True, exist_ok=True)
df.to_parquet(ROOT / "data/simulated/sim_v1_txns.parquet"); acc.to_parquet(ROOT / "data/simulated/sim_v1_accounts.parquet")
