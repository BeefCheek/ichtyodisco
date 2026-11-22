import time

import cv2
import numpy as np

from aquarium.capture.video_source import VideoFileCapture


def _clip(path, n=5):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()


def test_waits_for_the_first_frame_instead_of_returning_at_once(tmp_path):
    path = tmp_path / "clip.mp4"
    _clip(path)
    src = VideoFileCapture(str(path), loop=False, realtime=False)
    t = time.monotonic()
    assert src.wait_for_new_frame(timeout_s=0.3) is None  # not started: nothing to read yet
    assert time.monotonic() - t >= 0.25  # ... but it did wait (it used to spin)
    src.start()
    frames = []
    while (f := src.wait_for_new_frame(timeout_s=2.0)) is not None:
        frames.append(f)
    src.stop()
    assert len(frames) == 5 and src.finished
