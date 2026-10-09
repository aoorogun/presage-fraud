import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
# Import order matters on macOS: load torch first so its OpenMP runtime is the one the other libraries share.
try:
    import torch; torch.set_num_threads(1)
except Exception:
    pass
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
from presage.simulation.app_world import simulate, SimConfig


@pytest.fixture(scope="session")
def small_world():
    cfg = SimConfig(n_customers=600, n_payees=1200, days=40, n_mule_networks=12, scam_rate_per_day=2.0, recurring_windows=((10, 15),), sudden_drift_day=30)
    return simulate(cfg)
