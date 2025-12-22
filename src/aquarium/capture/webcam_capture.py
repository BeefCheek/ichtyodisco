import threading
import cv2


class WebcamCapture:
    def __init__(self, webcam_index: int = 0):
        self.webcam_index = webcam_index
        self.cap = cv2.VideoCapture(webcam_index)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        if not self.cap.isOpened():
            self.cap.open(self.webcam_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break  # placeholder
            self._latest_frame = frame

    def get_frame(self):
        return self._latest_frame

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
        self.cap.release()
