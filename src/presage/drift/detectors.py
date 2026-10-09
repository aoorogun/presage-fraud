"""Concept-drift detectors: stream detectors (River) and distribution-distance monitors (PSI, JS, Wasserstein)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def make_detector(name: str):
    from river import drift
    return {"adwin": lambda: drift.ADWIN(delta=0.002), "ddm": lambda: drift.binary.DDM(), "eddm": lambda: drift.binary.EDDM(),
            "page_hinkley": lambda: drift.PageHinkley(min_instances=30, delta=0.005, threshold=20.0), "kswin": lambda: drift.KSWIN(alpha=0.005, window_size=500, stat_size=150, seed=0)}[name]()


def run_detector(name: str, stream: np.ndarray, times: np.ndarray):
    """Feed a 0/1 error (or real-valued) stream; return the list of times at which drift was signalled."""
    det = make_detector(name); alarms = []
    for v, t in zip(stream, times):
        det.update(float(v))
        if getattr(det, "drift_detected", False): alarms.append(float(t))
    return alarms


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with reference-quantile bins (rule of thumb: >0.2 significant shift)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + eps; b = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((b - a) * np.log(b / a)))


def js_div(ref, cur, bins=30):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] + 1e-9; b = np.histogram(cur, edges)[0] + 1e-9
    return float(jensenshannon(a / a.sum(), b / b.sum()) ** 2)


def wasserstein(ref, cur):
    s = np.std(ref) + 1e-9
    return float(wasserstein_distance(ref / s, cur / s))
