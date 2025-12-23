import threading
import time
from collections import deque
from typing import Deque, Optional

import cv2


class WebcamCapture:
    """
    Threaded wrapper around OpenCV's VideoCapture.

    Responsibilities handled here:
        * Creating and managing the background capture thread.
        * Keeping a small circular buffer of frames instead of a single mutable reference.
        * Providing non-blocking read access to the most recent frame.
        * Attempting to reopen the webcam if the stream temporarily fails.
    """

    def __init__(
        self,
        webcam_index: int = 0,
        buffer_size: int = 5,
        reconnect_backoff_s: float = 0.5,
    ):
        self.webcam_index = webcam_index
        self.buffer_size = max(1, buffer_size)
        self.reconnect_backoff_s = reconnect_backoff_s

        self.cap = cv2.VideoCapture(webcam_index)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._frame_buffer: Deque = deque(maxlen=self.buffer_size)
        self._buffer_lock = threading.Lock()
        self._frame_available = threading.Event()

    def start(self):
        """Start capture thread if it is not already running."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Continuously pull frames from the webcam on a dedicated thread."""
        while not self._stop_event.is_set():
            # Ensure the camera handle is open; reconnect when unplugged/replugged.
            if not self.cap.isOpened():
                opened = self.cap.open(self.webcam_index)
                if not opened:
                    time.sleep(self.reconnect_backoff_s)
                    continue

            ok, frame = self.cap.read()
            if not ok:
                # Release before retrying to avoid a wedged handle.
                self.cap.release()
                time.sleep(self.reconnect_backoff_s)
                continue

            with self._buffer_lock:
                self._frame_buffer.append(frame)
                self._frame_available.set()

        # Clean up once the stop event is triggered.
        self.cap.release()

    def get_frame(self):
        """
        Return the most recent frame (copy) captured by the background thread.
        None is returned when no frame has been captured yet.
        """
        if not self._frame_available.is_set():
            return None

        with self._buffer_lock:
            latest = self._frame_buffer[-1]
            return latest.copy()

    def stop(self):
        """Signal the capture thread to stop and wait for it to exit."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        else:
            # Ensure capture handle is closed even if start() was never called.
            self.cap.release()
