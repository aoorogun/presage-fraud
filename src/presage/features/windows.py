"""Vectorised, strictly-causal rolling statistics.

Core primitive: for every *query* (group g, time t) compute aggregates over *events* of
the same group with event_time < t (strictly earlier, whole-second resolution) and
event_time >= t - window. Because the upper bound is strict, a row can never see itself
or anything that happens at or after its own timestamp, which is the no-leakage rule
x(t) may only use data from t' < t.

Implementation: events are sorted by a composite integer key g*K + floor(ts). Window
bounds are found with np.searchsorted; sums use prefix sums; maxima use a sparse table.
"""
from __future__ import annotations
import numpy as np

K = np.int64(20_000_000)  # > max timestamp in seconds (here <= 1.04e7); keeps group blocks disjoint
INF = np.inf


class EventIndex:
    """Sorted event table for one grouping (e.g. events keyed by sender account)."""

    def __init__(self, group: np.ndarray, ts: np.ndarray, vals: dict[str, np.ndarray]):
        g = np.asarray(group, dtype=np.int64)
        t = np.floor(np.asarray(ts, dtype=np.float64)).astype(np.int64)
        order = np.argsort(g * K + t, kind="stable")
        self.key = (g * K + t)[order]
        self.ts = t[order]
        self.vals = {k: np.asarray(v, dtype=np.float64)[order] for k, v in vals.items()}
        # Numerically stable moments: centre each value on its group's FIRST event value (a causal constant:
        # it is the earliest observation of that group), then take prefix sums of centred values.
        gsorted = (self.key // K)
        first_idx = np.searchsorted(self.key, gsorted * K, side="left")
        self.shift = {k: np.nan_to_num(v)[first_idx] for k, v in self.vals.items()}
        cen = {k: np.nan_to_num(v) - self.shift[k] for k, v in self.vals.items()}
        self.cs = {k: np.concatenate([[0.0], np.cumsum(c)]) for k, c in cen.items()}
        self.cs2 = {k: np.concatenate([[0.0], np.cumsum(c ** 2)]) for k, c in cen.items()}
        self._sparse: dict[str, list[np.ndarray]] = {}

    def _bounds(self, qg: np.ndarray, qt: np.ndarray, window: float):
        qg = np.asarray(qg, dtype=np.int64)
        qt = np.floor(np.asarray(qt, dtype=np.float64)).astype(np.int64)
        q = qg * K + qt
        hi = np.searchsorted(self.key, q, side="left")          # strict: events with key < q
        gstart = np.searchsorted(self.key, qg * K, side="left")
        if np.isinf(window):
            lo = gstart
        else:
            lo = np.maximum(np.searchsorted(self.key, q - np.int64(window), side="left"), gstart)
        return lo, np.maximum(hi, lo)

    def count(self, qg, qt, window):
        lo, hi = self._bounds(qg, qt, window)
        return (hi - lo).astype(np.float64)

    def sum(self, name, qg, qt, window):
        lo, hi = self._bounds(qg, qt, window)
        n = hi - lo
        sh = np.where(n > 0, self.shift[name][np.minimum(lo, len(self.key) - 1)], 0.0)
        return self.cs[name][hi] - self.cs[name][lo] + n * sh

    def mean_std(self, name, qg, qt, window):
        lo, hi = self._bounds(qg, qt, window)
        n = (hi - lo).astype(np.float64)
        s = self.cs[name][hi] - self.cs[name][lo]
        s2 = self.cs2[name][hi] - self.cs2[name][lo]
        sh = np.where(n > 0, self.shift[name][np.minimum(lo, len(self.key) - 1)], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mc = s / np.maximum(n, 1)
            mean = np.where(n > 0, mc + sh, np.nan)
            var = np.where(n > 1, np.maximum(s2 / np.maximum(n, 1) - mc ** 2, 0.0), np.nan)
            var = np.where(var < 1e-4, 0.0, var)   # floor float noise
        return mean, np.sqrt(var), n

    def max(self, name, qg, qt, window):
        if name not in self._sparse:
            v = np.nan_to_num(self.vals[name], nan=-np.inf)
            tab = [v]
            j = 1
            while (1 << j) <= len(v):
                prev = tab[-1]
                tab.append(np.maximum(prev[:-(1 << (j - 1))], prev[(1 << (j - 1)):]))
                j += 1
            self._sparse[name] = tab
        tab = self._sparse[name]
        lo, hi = self._bounds(qg, qt, window)
        n = hi - lo
        out = np.full(len(lo), np.nan)
        m = n > 0
        if m.any():
            k = np.floor(np.log2(n[m])).astype(int)
            res = np.empty(m.sum())
            lo_m, hi_m = lo[m], hi[m]
            for kk in np.unique(k):
                sel = k == kk
                a = tab[kk][lo_m[sel]]
                b = tab[kk][hi_m[sel] - (1 << kk)]
                res[sel] = np.maximum(a, b)
            res[~np.isfinite(res)] = np.nan
            out[m] = res
        return out

    def last_time_gap(self, qg, qt):
        """Seconds since the most recent earlier event of the group (NaN if none)."""
        lo, hi = self._bounds(qg, qt, INF)
        qt = np.floor(np.asarray(qt, dtype=np.float64))
        out = np.full(len(lo), np.nan)
        m = hi > lo
        out[m] = qt[m] - self.ts[hi[m] - 1]
        return out
