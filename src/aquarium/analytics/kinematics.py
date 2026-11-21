"""Per-fish trajectory buffer and motion features."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Kinematics:
    speed: float = 0.0  # tank-width units / s (3D when depth is known)
    speed_bls: float = 0.0  # body lengths / s, the biologically meaningful one
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    accel: float = 0.0  # tank-width units / s^2
    turn_rate: float = 0.0  # rad / s in the image plane
    tortuosity: float = 1.0  # path length / net displacement over the window (>= 1)
    body_length: float = 0.0  # tank-width units


class FishTrack:
    """Ring buffer of positions in *tank units* for one identity.

    Positions use a single metric across axes: x and y are divided by the frame width (so
    a vertical pixel is worth the same as a horizontal one), z (0..1 of tank depth) is
    multiplied by ``depth_ratio`` = tank depth / tank width.
    """

    def __init__(self, fish_id: int, history_s: float = 20.0, window_s: float = 0.6,
                 gap_reset_s: float = 0.25):
        self.fish_id = fish_id
        self.gap_reset_s = gap_reset_s
        self.history_s = history_s
        self.window_s = window_s
        self.t: deque[float] = deque()
        self.pos: deque[tuple[float, float, float]] = deque()
        self.zknown: deque[bool] = deque()
        self.lengths: deque[float] = deque(maxlen=90)
        self.first_seen: float | None = None
        self.last_seen: float = 0.0
        self.visible_s = 0.0
        self.kin = Kinematics()
        self._prev_v: tuple[float, float, float, float] | None = None  # t, vx, vy, vz
        self._prev_heading: tuple[float, float] | None = None

    def add(self, t: float, x: float, y: float, z: float | None, body_len: float) -> None:
        if self.first_seen is None:
            self.first_seen = t
        if self.t and 0 < t - self.t[-1] < 1.0:
            self.visible_s += t - self.t[-1]
        if self.t and t - self.t[-1] > self.gap_reset_s:
            # The fish was lost (occlusion, re-identification): start a new segment so the
            # jump between "last seen here" and "found there" is not read as a sprint.
            self.t.clear()
            self.pos.clear()
            self.zknown.clear()
            self._prev_v = None
            self._prev_heading = None
        self.t.append(t)
        self.pos.append((x, y, 0.0 if z is None else z))
        self.zknown.append(z is not None)
        if body_len > 0:
            self.lengths.append(body_len)
        self.last_seen = t
        while self.t and t - self.t[0] > self.history_s:
            self.t.popleft()
            self.pos.popleft()
            self.zknown.popleft()
        self._compute()

    @property
    def position(self) -> tuple[float, float, float] | None:
        return self.pos[-1] if self.pos else None

    def recent(self, seconds: float) -> list[tuple[float, float, float]]:
        if not self.t:
            return []
        t_end = self.t[-1]
        out = []
        for ti, p in zip(reversed(self.t), reversed(self.pos), strict=True):
            if t_end - ti > seconds:
                break
            out.append(p)
        out.reverse()
        return out

    def _window(self) -> tuple[np.ndarray, np.ndarray, bool]:
        t_end = self.t[-1]
        n = 0
        for ti in reversed(self.t):
            if t_end - ti > self.window_s:
                break
            n += 1
        n = max(n, 2)
        ts = np.fromiter(list(self.t)[-n:], dtype=np.float64)
        ps = np.asarray(list(self.pos)[-n:], dtype=np.float64)
        zk = all(list(self.zknown)[-n:])
        return ts, ps, zk

    def _compute(self) -> None:
        k = self.kin
        if self.lengths:
            k.body_length = float(np.median(self.lengths))
        if len(self.t) < 2:
            return
        ts, ps, z_ok = self._window()
        dt = ts[-1] - ts[0]
        if dt <= 1e-6:
            return
        # Least-squares slope = velocity robust to per-frame detection jitter.
        tc = ts - ts.mean()
        denom = float((tc * tc).sum())
        v = (tc[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / max(denom, 1e-9)
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2]) if z_ok else 0.0
        k.vx, k.vy, k.vz = vx, vy, vz
        k.speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        k.speed_bls = k.speed / k.body_length if k.body_length > 1e-4 else 0.0

        t_now = ts[-1]
        if self._prev_v is not None:
            pt, pvx, pvy, pvz = self._prev_v
            ddt = t_now - pt
            if ddt > 1e-3:
                k.accel = math.dist((vx, vy, vz), (pvx, pvy, pvz)) / ddt
        self._prev_v = (t_now, vx, vy, vz)

        if math.hypot(vx, vy) > 1e-3:
            heading = math.atan2(vy, vx)
            if self._prev_heading is not None:
                ph, pt = self._prev_heading
                d = (heading - ph + math.pi) % (2 * math.pi) - math.pi
                if t_now - pt > 1e-3:
                    k.turn_rate = abs(d) / (t_now - pt)
            self._prev_heading = (heading, t_now)

        recent = self.recent(3.0)
        if len(recent) >= 3:
            arr = np.asarray(recent)
            path = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
            net = float(np.linalg.norm(arr[-1] - arr[0]))
            k.tortuosity = path / net if net > 1e-3 else (1.0 if path < 1e-3 else 10.0)
