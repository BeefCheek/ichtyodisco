import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import cv2


class WebcamCapture:
    """
    Threaded wrapper around OpenCV's VideoCapture that shields the rest of the
    pipeline from slow I/O, jittery frame rates, or transient hardware faults.

    Responsibilities handled here (and expected to remain here):
        * Create and manage the dedicated capture thread so `start()` "arms" the
          camera without blocking callers.
        * Maintain a small circular buffer instead of exposing mutable frame
          references; `get_frame()` always serves the freshest snapshot.
        * Handle automatic reconnect attempts whenever the webcam drops, and
          back off briefly before re-trying to avoid tight failure loops.
        * Provide the future hooks for ML preprocessing (e.g. downscaled frames,
          FPS monitoring) while keeping the threading concerns isolated.
    """

    def __init__(
        self,
        webcam_index: int = 0,
        buffer_size: int = 5,
        reconnect_backoff_s: float = 0.5,
        capture_resolution: Tuple[int, int] = (4096, 2160),
        inference_resolution: Tuple[int, int] = (640, 360),
        target_fps: float = 30.0,
        fps_window: int = 60,
    ):
        self.webcam_index = webcam_index
        self.buffer_size = max(1, buffer_size)
        self.reconnect_backoff_s = reconnect_backoff_s
        self.target_fps = target_fps

        self.cap = cv2.VideoCapture(webcam_index)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._frame_buffer: Deque = deque(maxlen=self.buffer_size)
        self._buffer_lock = threading.Lock()
        self._frame_available = threading.Event()
        self._frame_times: Deque[float] = deque(maxlen=max(2, fps_window))
        self._fps_lock = threading.Lock()

        self._capture_resolution = capture_resolution
        self._native_resolution: Tuple[int, int] = (0, 0)
        self._inference_resolution = inference_resolution

    def start(self):
        """
        Start the capture thread if it is not already running.

        Expectations:
            * Idempotent: repeated calls must be safe and should not spawn extra threads.
            * Must leave `_stop_event` cleared so the thread can run until `stop()`.
            * Should not block the caller; all heavy work lives inside `_capture_loop`.
        """
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """
        Continuously pull frames from the webcam on a dedicated thread.

        Expectations:
            * Reconnect quietly if the webcam handle is closed or `read()` fails.
            * Append frames to the bounded deque while holding `_buffer_lock`.
            * Never raise out of the loop unless `stop()` signals `_stop_event`.
            * Release the capture handle exactly once when the loop exits.
        """
        while not self._stop_event.is_set():
            # Ensure the camera handle is open; reconnect when unplugged/replugged.
            if not self.cap.isOpened():
                opened = self.cap.open(self.webcam_index)
                if not opened:
                    time.sleep(self.reconnect_backoff_s)
                    continue
                self._apply_capture_settings()

            ok, frame = self.cap.read()
            if not ok:
                # Release before retrying to avoid a wedged handle.
                self.cap.release()
                time.sleep(self.reconnect_backoff_s)
                continue

            with self._buffer_lock:
                self._frame_buffer.append(frame)
                self._frame_available.set()

            with self._fps_lock:
                self._frame_times.append(time.monotonic())

        # Clean up once the stop event is triggered.
        self.cap.release()

    def get_frame(self):
        """
        Return the freshest frame captured by the background thread.

        Expectations:
            * Non-blocking: returns immediately even if no frame is ready (None).
            * Returns a copy to prevent callers from mutating the internal buffer.
            * Future extension point for delivering multiple resolutions (e.g.,
              `get_frame_for_inference()` can share locking/logic here).
        """
        if not self._frame_available.is_set():
            return None

        with self._buffer_lock:
            latest = self._frame_buffer[-1]
            return latest.copy()

    def get_frame_for_inference(self):
        """
        Return a resized frame suited for downstream ML.

        Expectations:
            * Reuses the threading-safe access pattern from `get_frame`.
            * Downscales (or upscales) to the configured inference resolution,
              defaulting to INTER_AREA for better shrinking results.
            * Returns None if no frame is available yet.
        """
        frame = self.get_frame()
        if frame is None:
            return None

        target_w, target_h = self._inference_resolution
        if target_w <= 0 or target_h <= 0:
            return frame

        native_w, native_h = self._native_resolution
        if (target_w, target_h) == (native_w, native_h) or (native_w == 0 and native_h == 0):
            return frame

        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def stop(self):
        """
        Signal the capture thread to stop and wait for it to exit.

        Expectations:
            * Safe to call multiple times; after the first call there should be no
              lingering threads or open camera handles.
            * Responsible for resetting `_thread` to None so `start()` can be
              invoked again later.
        """
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        else:
            # Ensure capture handle is closed even if start() was never called.
            self.cap.release()

    def _apply_capture_settings(self):
        """Configure width, height, and FPS, then update native resolution info."""
        width, height = self._capture_resolution
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if self.target_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._native_resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def native_resolution(self) -> Tuple[int, int]:
        """Resolution actually reported by the webcam after configuration."""
        return self._native_resolution

    @property
    def capture_resolution(self) -> Tuple[int, int]:
        """Resolution we request from the webcam (may differ from native)."""
        return self._capture_resolution

    @capture_resolution.setter
    def capture_resolution(self, resolution: Tuple[int, int]):
        """Update requested capture resolution (applies on next (re)start)."""
        self._capture_resolution = resolution
        if self.cap.isOpened():
            self._apply_capture_settings()

    @property
    def inference_resolution(self) -> Tuple[int, int]:
        """Resolution used when resizing frames for inference."""
        return self._inference_resolution

    @inference_resolution.setter
    def inference_resolution(self, resolution: Tuple[int, int]):
        self._inference_resolution = resolution

    @property
    def fps_actual(self) -> float:
        """Rolling FPS measurement computed from the capture thread timestamps."""
        with self._fps_lock:
            if len(self._frame_times) < 2:
                return 0.0
            duration = self._frame_times[-1] - self._frame_times[0]
            if duration <= 0:
                return 0.0
            return (len(self._frame_times) - 1) / duration
