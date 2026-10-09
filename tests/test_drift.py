import numpy as np
from presage.drift.detectors import run_detector, psi


def test_detectors_fire_on_abrupt_change():
    rng = np.random.default_rng(0)
    stream = np.concatenate([(rng.random(1500) < 0.1), (rng.random(1500) < 0.7)]).astype(float)
    t = np.arange(len(stream), dtype=float)
    for d in ["adwin", "ddm", "page_hinkley"]:
        al = run_detector(d, stream, t)
        assert any(a >= 1500 for a in al), d
        assert sum(a < 1500 for a in al) <= 2, f"{d} fired too early"


def test_psi_zero_for_same_and_positive_for_shift():
    rng = np.random.default_rng(1); a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.2
