"""Where do the fish spend their time? Decaying occupancy maps and named zones."""

from __future__ import annotations

from typing import Any

import numpy as np

from aquarium.analytics.stats import _decay

VERTICAL = ("surface", "milieu", "fond")
DEPTH = ("avant", "centre", "arrière")
HORIZONTAL = ("gauche", "centre", "droite")


def zone_of(x: float, y: float, z: float | None) -> dict[str, str]:
    """x, y in 0..1 of the frame, z in 0..1 of tank depth."""
    v = VERTICAL[0 if y < 0.25 else (2 if y > 0.75 else 1)]
    h = HORIZONTAL[0 if x < 1 / 3 else (2 if x > 2 / 3 else 1)]
    out = {"vertical": v, "horizontal": h}
    if z is not None:
        out["depth"] = DEPTH[0 if z < 1 / 3 else (2 if z > 2 / 3 else 1)]
    return out


class OccupancyMap:
    """Front-view (x, y) heatmap + top-view (x, z) heatmap with time decay."""

    def __init__(self, size: tuple[int, int] = (48, 27), half_life_s: float = 3600.0):
        self.w, self.h = size
        self.half_life_s = half_life_s
        self.front = np.zeros((self.h, self.w), dtype=np.float64)
        self.top = np.zeros((max(8, self.h // 2), self.w), dtype=np.float64)
        self.last_t: float | None = None

    def _decay_to(self, t: float) -> None:
        if self.last_t is not None:
            d = _decay(t - self.last_t, self.half_life_s)
            if d < 0.999999:
                self.front *= d
                self.top *= d
        self.last_t = t if self.last_t is None else max(self.last_t, t)

    def add(self, points: list[tuple[float, float, float | None]], t: float, weight: float = 1.0):
        self._decay_to(t)
        for x, y, z in points:
            ix = int(np.clip(x * self.w, 0, self.w - 1))
            iy = int(np.clip(y * self.h, 0, self.h - 1))
            self.front[iy, ix] += weight
            if z is not None:
                iz = int(np.clip(z * self.top.shape[0], 0, self.top.shape[0] - 1))
                self.top[iz, ix] += weight

    def hotspot(self) -> tuple[float, float] | None:
        if self.front.sum() <= 0:
            return None
        iy, ix = np.unravel_index(int(np.argmax(self.front)), self.front.shape)
        return (ix + 0.5) / self.w, (iy + 0.5) / self.h

    def vertical_profile(self) -> list[float]:
        s = self.front.sum()
        if s <= 0:
            return [0.0] * len(VERTICAL)
        rows = self.front.sum(axis=1) / s
        bands = np.array_split(rows, [int(self.h * 0.25), int(self.h * 0.75)])
        return [float(b.sum()) for b in bands]

    def depth_profile(self) -> list[float]:
        s = self.top.sum()
        if s <= 0:
            return [0.0] * len(DEPTH)
        rows = self.top.sum(axis=1) / s
        return [float(b.sum()) for b in np.array_split(rows, 3)]

    def to_dict(self) -> dict[str, Any]:
        return {"w": self.w, "h": self.h, "hl": self.half_life_s, "t": self.last_t,
                "front": self.front.round(4).tolist(), "top": self.top.round(4).tolist()}

    def load_dict(self, d: dict[str, Any]) -> None:
        front = np.asarray(d["front"], dtype=np.float64)
        top = np.asarray(d["top"], dtype=np.float64)
        if front.shape == self.front.shape and top.shape == self.top.shape:
            self.front, self.top, self.last_t = front, top, d.get("t")
