"""Frame sources sharing the `WebcamCapture` surface (start / wait_for_new_frame / stop)."""

from __future__ import annotations

import threading
import time
from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    """Anything the pipeline can pull frames from. ``wait_for_new_frame`` never repeats a frame."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def wait_for_new_frame(self, timeout_s: float | None = 10.0) -> np.ndarray | None: ...
    def get_frame(self) -> np.ndarray | None: ...

    @property
    def fps_actual(self) -> float: ...


class VideoFileCapture:
    """Plays a video file (or RTSP/HTTP stream) at its native pace on a background thread.

    Useful to develop and test the whole pipeline offline on recorded footage, and to run
    a "rerun" livestream from a recording.
    """

    def __init__(self, uri: str, loop: bool = True, realtime: bool = True):
        self.uri = uri
        self.loop = loop
        self.realtime = realtime
        self._cap = cv2.VideoCapture(uri)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {uri}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._native_fps = fps if fps and fps > 0 else 30.0
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._consumed_seq = -1
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._times: list[float] = []
        self.finished = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="video-source")
        self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / self._native_fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if self.loop and self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    continue
                self.finished = True
                with self._cond:
                    self._cond.notify_all()
                break
            with self._cond:
                self._latest = frame
                self._seq += 1
                self._cond.notify_all()
            now = time.monotonic()
            self._times = [t for t in self._times[-59:]] + [now]
            if self.realtime:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
            else:
                # Offline mode: wait for the consumer so no frame is skipped.
                with self._cond:
                    while self._consumed_seq < self._seq and not self._stop.is_set():
                        self._cond.wait(0.1)
        self._cap.release()

    def get_frame(self) -> np.ndarray | None:
        with self._cond:
            if self._latest is None:
                return None
            self._consumed_seq = self._seq
            self._cond.notify_all()
            return self._latest.copy()

    def wait_for_new_frame(self, timeout_s: float | None = 10.0) -> np.ndarray | None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cond:
            while self._seq <= self._consumed_seq and not self.finished:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._latest is None or (self.finished and self._seq <= self._consumed_seq):
                return None
            self._consumed_seq = self._seq
            self._cond.notify_all()
            return self._latest.copy()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def fps_actual(self) -> float:
        times = self._times
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])
