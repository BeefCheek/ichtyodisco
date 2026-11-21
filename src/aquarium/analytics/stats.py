"""Online statistics that *forget*: the backbone of an aquarium model that keeps learning.

Everything here is O(1) per update, JSON-serialisable and time-aware (decay is driven by
elapsed time, not by sample count, so the learnt baseline is independent of the FPS).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np


def _decay(dt: float, half_life: float) -> float:
    if half_life <= 0 or dt <= 0:
        return 1.0
    return 0.5 ** (dt / half_life)


class DecayingStats:
    """Exponentially time-weighted mean / variance.

    With a half-life of 72 h, data from three days ago weighs half as much as now: the
    baseline follows seasonal drift, new tank mates, a new feeding schedule...
    """

    __slots__ = ("half_life", "w", "sx", "sxx", "last_t", "n")

    def __init__(self, half_life: float):
        self.half_life = half_life
        self.w = 0.0
        self.sx = 0.0
        self.sxx = 0.0
        self.last_t: float | None = None
        self.n = 0

    def update(self, x: float, t: float, weight: float = 1.0) -> None:
        if not math.isfinite(x):
            return
        if self.last_t is not None:
            d = _decay(t - self.last_t, self.half_life)
            self.w *= d
            self.sx *= d
            self.sxx *= d
        self.last_t = t if self.last_t is None else max(self.last_t, t)
        self.w += weight
        self.sx += weight * x
        self.sxx += weight * x * x
        self.n += 1

    @property
    def mean(self) -> float:
        return self.sx / self.w if self.w > 0 else 0.0

    @property
    def var(self) -> float:
        if self.w <= 0:
            return 0.0
        m = self.mean
        return max(self.sxx / self.w - m * m, 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def zscore(self, x: float, min_std: float = 1e-6) -> float:
        return (x - self.mean) / max(self.std, min_std)

    def to_dict(self) -> dict[str, Any]:
        return {"hl": self.half_life, "w": self.w, "sx": self.sx, "sxx": self.sxx,
                "t": self.last_t, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecayingStats:
        s = cls(d["hl"])
        s.w, s.sx, s.sxx, s.last_t, s.n = d["w"], d["sx"], d["sxx"], d["t"], d.get("n", 0)
        return s


class DecayingHistogram:
    """Fixed-bin histogram with time decay, for adaptive quantile thresholds.

    Used e.g. to decide what "resting" or "bursting" means *for this tank*: a betta and a
    shoal of danios have very different speed distributions.
    """

    def __init__(self, lo: float, hi: float, bins: int = 64, half_life: float = 3600.0,
                 log: bool = False):
        self.lo, self.hi, self.bins, self.half_life, self.log = lo, hi, bins, half_life, log
        if log:
            self.edges = np.geomspace(lo, hi, bins + 1)
        else:
            self.edges = np.linspace(lo, hi, bins + 1)
        self.counts = np.zeros(bins, dtype=np.float64)
        self.last_t: float | None = None

    def update(self, x: float, t: float) -> None:
        if not math.isfinite(x):
            return
        if self.last_t is not None:
            self.counts *= _decay(t - self.last_t, self.half_life)
        self.last_t = t if self.last_t is None else max(self.last_t, t)
        idx = int(np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, self.bins - 1))
        self.counts[idx] += 1.0

    @property
    def total(self) -> float:
        return float(self.counts.sum())

    def quantile(self, q: float, default: float | None = None) -> float | None:
        total = self.total
        if total <= 0:
            return default
        cdf = np.cumsum(self.counts) / total
        idx = int(np.searchsorted(cdf, q))
        idx = min(idx, self.bins - 1)
        lo_c = cdf[idx - 1] if idx > 0 else 0.0
        frac = 0.0 if cdf[idx] == lo_c else (q - lo_c) / (cdf[idx] - lo_c)
        a, b = self.edges[idx], self.edges[idx + 1]
        return float(a + np.clip(frac, 0, 1) * (b - a))

    def to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "bins": self.bins, "hl": self.half_life,
                "log": self.log, "counts": self.counts.tolist(), "t": self.last_t}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecayingHistogram:
        h = cls(d["lo"], d["hi"], d["bins"], d["hl"], d.get("log", False))
        h.counts = np.asarray(d["counts"], dtype=np.float64)
        h.last_t = d["t"]
        return h


class CircadianBaseline:
    """Learns what is normal for each metric *at each hour of the day*.

    Fish are strongly circadian (lights on/off, feeding times). An activity level that is
    perfectly normal at 18:00 (dinner) can be alarming at 03:00. Each metric keeps 24
    hourly ``DecayingStats`` plus a global one used until an hour has enough data.
    """

    def __init__(self, half_life_s: float, min_weight: float = 120.0):
        self.half_life_s = half_life_s
        self.min_weight = min_weight
        self.metrics: dict[str, dict[str, Any]] = {}

    def _entry(self, metric: str) -> dict[str, Any]:
        e = self.metrics.get(metric)
        if e is None:
            e = {
                "global": DecayingStats(self.half_life_s),
                "hours": [DecayingStats(self.half_life_s) for _ in range(24)],
            }
            self.metrics[metric] = e
        return e

    @staticmethod
    def hour_of(t_wall: float) -> int:
        return datetime.fromtimestamp(t_wall).hour

    def update(self, metric: str, value: float, t_wall: float) -> None:
        e = self._entry(metric)
        e["global"].update(value, t_wall)
        e["hours"][self.hour_of(t_wall)].update(value, t_wall)

    def stats_for(self, metric: str, t_wall: float) -> DecayingStats | None:
        e = self.metrics.get(metric)
        if e is None:
            return None
        hourly: DecayingStats = e["hours"][self.hour_of(t_wall)]
        if hourly.w >= self.min_weight:
            return hourly
        glob: DecayingStats = e["global"]
        return glob if glob.w >= min(self.min_weight, 30.0) else None

    def zscore(self, metric: str, value: float, t_wall: float, min_std: float = 1e-3) -> float:
        s = self.stats_for(metric, t_wall)
        if s is None:
            return 0.0
        return s.zscore(value, min_std=min_std)

    def hourly_profile(self, metric: str) -> list[dict[str, float]]:
        e = self.metrics.get(metric)
        if e is None:
            return [{"hour": h, "mean": 0.0, "std": 0.0, "weight": 0.0} for h in range(24)]
        return [
            {"hour": h, "mean": s.mean, "std": s.std, "weight": s.w}
            for h, s in enumerate(e["hours"])
        ]

    def coverage(self, metric: str) -> int:
        """Number of hours of the day for which the baseline is considered learnt."""
        e = self.metrics.get(metric)
        if e is None:
            return 0
        return sum(1 for s in e["hours"] if s.w >= self.min_weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "half_life_s": self.half_life_s,
            "min_weight": self.min_weight,
            "metrics": {
                k: {"global": v["global"].to_dict(), "hours": [s.to_dict() for s in v["hours"]]}
                for k, v in self.metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CircadianBaseline:
        b = cls(d["half_life_s"], d.get("min_weight", 120.0))
        for k, v in d.get("metrics", {}).items():
            b.metrics[k] = {
                "global": DecayingStats.from_dict(v["global"]),
                "hours": [DecayingStats.from_dict(s) for s in v["hours"]],
            }
        return b
