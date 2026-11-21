import numpy as np

from aquarium.depth.async_depth import AsyncDepthWorker
from aquarium.depth.sampling import FishDepthSmoother, TankDepthCalibration, sample_bbox_depth


def test_sample_bbox_depth_prefers_fish_over_background():
    depth = np.ones((100, 200), np.float32)  # back wall everywhere
    depth[40:60, 80:120] = 0.3  # the fish body
    z = sample_bbox_depth(depth, (70, 35, 130, 65), frame_shape=(100, 200))
    assert abs(z - 0.3) < 1e-6


def test_sample_bbox_handles_resolution_mismatch():
    depth = np.ones((50, 100), np.float32)
    depth[20:30, 40:60] = 0.2
    z = sample_bbox_depth(depth, (80, 40, 120, 60), frame_shape=(100, 200))  # frame is 2x larger
    assert abs(z - 0.2) < 1e-6


def test_calibration_normalises_to_tank_volume():
    cal = TankDepthCalibration(near_percentile=0, far_percentile=100, smoothing=1.0)
    raw = np.linspace(2.0, 4.0, 101, dtype=np.float32).reshape(1, -1)
    cal.update(raw)
    norm = cal.normalise(raw)
    assert abs(norm.min()) < 1e-6 and abs(norm.max() - 1) < 1e-6


def test_smoother_reduces_jitter_and_forgets():
    sm = FishDepthSmoother(alpha=0.3, forget_after_s=1.0)
    vals = [sm.update(1, 0.5 + (0.2 if i % 2 else -0.2), i * 0.03) for i in range(40)]
    assert max(vals[20:]) - min(vals[20:]) < 0.25
    assert sm.update(1, None, 1.25) is not None  # keeps last value briefly
    assert sm.update(1, 0.9, 10.0) == 0.9  # long gap: restart from scratch


class _FakeEstimator:
    name = "fake"

    def predict(self, frame):
        return np.full(frame.shape[:2], 2.0, np.float32) + np.linspace(0, 1, frame.shape[1])


def test_async_worker_produces_normalised_maps():
    w = AsyncDepthWorker(_FakeEstimator(), TankDepthCalibration(), max_rate_hz=50)
    w.start()
    try:
        w.submit(np.zeros((40, 60, 3), np.uint8))
        import time

        for _ in range(100):
            if w.latest() is not None:
                break
            time.sleep(0.02)
        dm = w.latest()
        assert dm is not None and dm.model == "fake"
        assert dm.data.min() >= 0 and dm.data.max() <= 1
    finally:
        w.stop()
