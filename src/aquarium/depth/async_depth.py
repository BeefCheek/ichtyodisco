"""Run a slow depth model off the hot path.

Tracking runs every frame (~30 fps). Depth Pro takes ~0.3 s on a datacentre GPU and
seconds on CPU; Depth Anything V2 Small is ~10x faster. Fish change depth slowly, so we
compute depth at a capped rate on the latest frame and every frame reuses the newest map.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from aquarium.depth.base import DepthEstimator
from aquarium.depth.sampling import TankDepthCalibration
from aquarium.types import DepthMap

log = logging.getLogger(__name__)


class AsyncDepthWorker:
    def __init__(
        self,
        estimator: DepthEstimator,
        calibration: TankDepthCalibration,
        max_rate_hz: float = 4.0,
    ):
        self.estimator = estimator
        self.calibration = calibration
        self.min_period = 1.0 / max_rate_hz if max_rate_hz > 0 else 0.0
        self._pending: np.ndarray | None = None
        self._latest: DepthMap | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="depth-worker")
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Offer the newest frame; older pending frames are dropped (no backlog)."""
        with self._lock:
            self._pending = frame
        self._wake.set()

    def latest(self) -> DepthMap | None:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.25):
                continue
            self._wake.clear()
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            t0 = time.monotonic()
            try:
                raw = self.estimator.predict(frame)
                self.calibration.update(raw)
                norm = self.calibration.normalise(raw)
            except Exception:  # keep streaming even if the model hiccups
                self.errors += 1
                log.exception("depth inference failed")
                time.sleep(1.0)
                continue
            latency = time.monotonic() - t0
            with self._lock:
                self._latest = DepthMap(
                    data=norm,
                    timestamp=time.time(),
                    source_shape=frame.shape[:2],
                    model=self.estimator.name,
                    latency_s=latency,
                )
            rest = self.min_period - latency
            if rest > 0:
                self._stop.wait(rest)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
