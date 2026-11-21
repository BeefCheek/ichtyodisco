"""A synthetic 3D aquarium: run and demo the whole pipeline with no camera and no GPU.

It renders fish swimming in a box with perspective (far fish are smaller and bluer),
produces ground-truth tracks (with optional id switches to exercise re-identification) and
a ground-truth depth map, and stages behaviours: shoaling, resting at night, chases and a
feeding frenzy at the surface every few minutes.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from aquarium.types import TrackedFish

PALETTE = [  # BGR body colours: distinct enough for colour-histogram re-id
    (40, 120, 255), (60, 220, 255), (230, 120, 40), (80, 200, 80), (200, 80, 220),
    (240, 240, 240), (40, 60, 200), (180, 200, 60), (120, 90, 255), (20, 160, 230),
]


@dataclass
class SimFish:
    fid: int
    pos: np.ndarray  # x, y, z in 0..1
    vel: np.ndarray
    color: tuple[int, int, int]
    length: float  # fraction of tank width
    temperament: float  # 0 calm .. 1 restless
    rest_until: float = 0.0


class SimulatedAquarium:
    def __init__(self, n_fish: int = 7, size: tuple[int, int] = (1280, 720), fps: float = 30.0,
                 seed: int = 7, feeding_every_s: float = 180.0, id_switch_rate: float = 0.002,
                 realtime: bool = True):
        self.w, self.h = size
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        self.feeding_every_s = feeding_every_s
        self.id_switch_rate = id_switch_rate
        self.realtime = realtime
        self.fish = [
            SimFish(
                fid=i,
                pos=self.rng.uniform([0.15, 0.2, 0.1], [0.85, 0.8, 0.9]),
                vel=self.rng.normal(0, 0.05, 3),
                color=PALETTE[i % len(PALETTE)],
                length=float(self.rng.uniform(0.05, 0.09)),
                temperament=float(self.rng.uniform(0, 1)),
            )
            for i in range(n_fish)
        ]
        self._track_ids = {f.fid: f.fid + 1 for f in self.fish}
        self._next_track = n_fish + 1
        self.t = 0.0
        self._background = self._make_background()
        self._frame: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._tracks: list[TrackedFish] = []
        self._lock = threading.Condition()
        self._seq = 0
        self._consumed = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._times: list[float] = []
        self._served: tuple[np.ndarray | None, list[TrackedFish]] = (None, [])

    # -------------------------------------------------------------- world
    def _make_background(self) -> np.ndarray:
        h, w = self.h, self.w
        grad = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        top, bottom = np.array([120, 90, 30], np.float32), np.array([70, 40, 10], np.float32)
        bg = (top * (1 - grad[..., None]) + bottom * grad[..., None]).repeat(w, axis=1)
        sand_y = int(h * 0.88)
        bg[sand_y:] = np.array([110, 170, 200], np.float32) * 0.8
        noise = self.rng.normal(0, 6, (h, w, 1)).astype(np.float32)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
        for _ in range(9):  # plants
            x = int(self.rng.uniform(0.03, 0.97) * w)
            height = int(self.rng.uniform(0.2, 0.55) * h)
            for k in range(0, height, 6):
                sway = int(8 * math.sin(k / 25.0 + x))
                cv2.circle(bg, (x + sway, sand_y - k), int(6 + 3 * math.sin(k)), (40, 120, 40), -1)
        return bg

    def _feeding(self) -> bool:
        if self.feeding_every_s <= 0:
            return False
        first = 45.0  # first meal shortly after start, so demos show it
        return self.t > first and ((self.t - first) % self.feeding_every_s) < 20.0

    def _night(self) -> bool:
        hour = time.localtime().tm_hour
        return hour >= 23 or hour < 7

    def step(self, dt: float) -> None:
        self.t += dt
        feeding = self._feeding()
        centroid = np.mean([f.pos for f in self.fish], axis=0)
        mean_v = np.mean([f.vel for f in self.fish], axis=0)
        for i, f in enumerate(self.fish):
            if f.rest_until > self.t and not feeding:
                f.vel *= 0.85
            else:
                base = 0.10 + 0.12 * f.temperament
                acc = self.rng.normal(0, 0.25 + 0.4 * f.temperament, 3) * np.array([1, 0.6, 0.6])
                acc += (centroid - f.pos) * 0.35  # cohesion
                acc += (mean_v - f.vel) * 0.4  # alignment
                for g in self.fish:  # separation
                    if g is not f:
                        d = f.pos - g.pos
                        dist = np.linalg.norm(d) + 1e-6
                        if dist < 0.06:
                            acc += d / dist * 0.8
                if feeding:
                    target = np.array([0.5 + 0.2 * math.sin(i), 0.06, 0.3])
                    acc += (target - f.pos) * 3.0
                    base *= 3.0
                if i == 1 and not feeding and (self.t % 90) < 6:  # periodic chase 1 -> 0
                    acc += (self.fish[0].pos - f.pos) * 6.0
                    base *= 3.0
                    self.fish[0].vel += (self.fish[0].pos - f.pos) * 0.4
                f.vel += acc * dt
                speed = np.linalg.norm(f.vel) + 1e-9
                limit = base * (0.3 if self._night() else 1.0)
                f.vel *= min(1.0, limit / speed) if speed > limit else 1.0
                if self.rng.random() < 0.002 * (1.2 - f.temperament):
                    f.rest_until = self.t + self.rng.uniform(3, 15)
            f.pos += f.vel * dt
            for ax, (lo, hi) in enumerate([(0.05, 0.95), (0.05, 0.85), (0.05, 0.95)]):
                if f.pos[ax] < lo:
                    f.pos[ax], f.vel[ax] = lo, abs(f.vel[ax])
                elif f.pos[ax] > hi:
                    f.pos[ax], f.vel[ax] = hi, -abs(f.vel[ax])
            if self.rng.random() < self.id_switch_rate:  # simulate a tracker id switch
                self._track_ids[f.fid] = self._next_track
                self._next_track += 1

    def render(self) -> tuple[np.ndarray, np.ndarray, list[TrackedFish]]:
        frame = self._background.copy()
        depth = np.full((self.h // 4, self.w // 4), 1.0, np.float32)
        # the sand floor runs from the front glass (bottom rows, near) to the back wall
        sand0 = int(depth.shape[0] * 0.88)
        depth[sand0:] = np.linspace(1.0, 0.0, depth.shape[0] - sand0)[:, None]
        tracks: list[TrackedFish] = []
        vp = np.array([0.5, 0.45])  # vanishing point: perspective towards the back wall
        for f in sorted(self.fish, key=lambda g: -g.pos[2]):  # far first
            scale = 1.0 - 0.45 * f.pos[2]
            sx = vp[0] + (f.pos[0] - vp[0]) * scale
            sy = vp[1] + (f.pos[1] - vp[1]) * scale
            cx, cy = int(sx * self.w), int(sy * self.h)
            length = f.length * scale * self.w
            ang = math.degrees(math.atan2(f.vel[1], f.vel[0] if abs(f.vel[0]) > 1e-4 else 1e-4))
            fade = 1.0 - 0.15 * f.pos[2]  # mild underwater colour attenuation
            col = tuple(int(c * fade + 60 * (1 - fade) * (k == 0)) for k, c in enumerate(f.color))
            axes = (max(3, int(length / 2)), max(2, int(length / 5)))
            cv2.ellipse(frame, (cx, cy), axes, ang, 0, 360, col, -1, cv2.LINE_AA)
            tail_dir = -np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
            tail = (np.array([cx, cy]) + tail_dir * length * 0.55).astype(int)
            perp = np.array([-tail_dir[1], tail_dir[0]]) * length * 0.2
            pts = np.array([tail + tail_dir * 1, tail + tail_dir * length * 0.25 + perp,
                            tail + tail_dir * length * 0.25 - perp], dtype=np.int32)
            cv2.fillConvexPoly(frame, pts, col, cv2.LINE_AA)
            dx, dy = int(cx / 4), int(cy / 4)
            cv2.ellipse(depth, (dx, dy), (max(1, axes[0] // 4), max(1, axes[1] // 4)), ang, 0, 360,
                        float(f.pos[2]), -1)
            half_w = abs(math.cos(math.radians(ang))) * axes[0] + abs(math.sin(math.radians(ang))) * axes[1]
            half_h = abs(math.sin(math.radians(ang))) * axes[0] + abs(math.cos(math.radians(ang))) * axes[1]
            jitter = self.rng.normal(0, 1.2, 4)
            tracks.append(TrackedFish(
                track_id=self._track_ids[f.fid],
                bbox=(cx - half_w - length * 0.2 + jitter[0], cy - half_h + jitter[1],
                      cx + half_w + length * 0.2 + jitter[2], cy + half_h + jitter[3]),
                confidence=0.9,
            ))
        return frame, depth, tracks

    # -------------------------------------------------------------- FrameSource API
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sim-aquarium")
        self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            self.step(period)
            frame, depth, tracks = self.render()
            with self._lock:
                self._frame, self._depth, self._tracks = frame, depth, tracks
                self._seq += 1
                self._lock.notify_all()
                if not self.realtime:
                    while self._consumed < self._seq and not self._stop.is_set():
                        self._lock.wait(0.1)
            now = time.monotonic()
            self._times = self._times[-59:] + [now]
            if self.realtime:
                next_t += period
                delay = next_t - now
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = now

    def wait_for_new_frame(self, timeout_s: float | None = 10.0) -> np.ndarray | None:
        with self._lock:
            ok = self._lock.wait_for(lambda: self._seq > self._consumed, timeout=timeout_s)
            if not ok or self._frame is None:
                return None
            self._consumed = self._seq
            self._served = (self._depth, self._tracks)  # ground truth of *this* frame
            self._lock.notify_all()
            return self._frame

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def fps_actual(self) -> float:
        ts = self._times
        return 0.0 if len(ts) < 2 or ts[-1] <= ts[0] else (len(ts) - 1) / (ts[-1] - ts[0])

    def ground_truth(self) -> tuple[np.ndarray | None, list[TrackedFish]]:
        with self._lock:
            depth, tracks = self._served
            return depth, [TrackedFish(t.track_id, t.bbox, t.confidence) for t in tracks]


class SimTracker:
    """Stands in for YOLO: returns the simulator's ground-truth boxes for the current frame."""

    def __init__(self, sim: SimulatedAquarium):
        self.sim = sim

    def update(self, frame: np.ndarray) -> list[TrackedFish]:
        return self.sim.ground_truth()[1]


class SimDepthEstimator:
    """Stands in for Depth Pro: returns the simulator's depth buffer (+ noise), larger = farther."""

    name = "sim-depth"

    def __init__(self, sim: SimulatedAquarium, noise: float = 0.03, latency_s: float = 0.05):
        self.sim = sim
        self.noise = noise
        self.latency_s = latency_s
        self.rng = np.random.default_rng(1)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        time.sleep(self.latency_s)
        depth, _ = self.sim.ground_truth()
        if depth is None:
            return np.ones(frame_bgr.shape[:2], np.float32)
        return (depth + self.rng.normal(0, self.noise, depth.shape)).astype(np.float32)
