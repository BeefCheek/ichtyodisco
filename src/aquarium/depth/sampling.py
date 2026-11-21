"""Turning a raw depth map into a per-fish ``z`` in [0 (front glass), 1 (back wall)]."""

from __future__ import annotations

import numpy as np


class TankDepthCalibration:
    """Maps raw depth values to the tank volume using robust, slowly-adapting planes.

    Monocular depth is only defined up to scale (and refraction through the glass bends
    it further), so absolute metres are meaningless here. Instead we track two planes:
    the *near* plane (low percentile ~ front glass / closest fish) and the *far* plane
    (high percentile ~ back wall) inside the tank ROI, smoothed over time so a single fish
    passing in front of the camera does not rescale the whole aquarium.
    """

    def __init__(
        self,
        near_percentile: float = 3.0,
        far_percentile: float = 97.0,
        smoothing: float = 0.05,
        roi: tuple[float, float, float, float] | None = None,  # normalised x1,y1,x2,y2
    ):
        self.near_percentile = near_percentile
        self.far_percentile = far_percentile
        self.smoothing = smoothing
        self.roi = roi
        self.near: float | None = None
        self.far: float | None = None

    def update(self, raw: np.ndarray) -> None:
        region = raw
        if self.roi is not None:
            h, w = raw.shape[:2]
            x1, y1, x2, y2 = self.roi
            region = raw[int(y1 * h) : max(int(y2 * h), int(y1 * h) + 1),
                         int(x1 * w) : max(int(x2 * w), int(x1 * w) + 1)]
        finite = region[np.isfinite(region)]
        if finite.size == 0:
            return
        near, far = np.percentile(finite, [self.near_percentile, self.far_percentile])
        if self.near is None or self.far is None:
            self.near, self.far = float(near), float(far)
        else:
            a = self.smoothing
            self.near = (1 - a) * self.near + a * float(near)
            self.far = (1 - a) * self.far + a * float(far)

    def normalise(self, raw: np.ndarray) -> np.ndarray:
        if self.near is None or self.far is None:
            self.update(raw)
        assert self.near is not None and self.far is not None
        span = max(self.far - self.near, 1e-6)
        return np.clip((raw - self.near) / span, 0.0, 1.0).astype(np.float32)


def sample_bbox_depth(
    depth: np.ndarray,
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int],
    inner: float = 0.5,
    percentile: float = 35.0,
) -> float | None:
    """Robust depth of a fish from the central part of its bounding box.

    The box contains background (farther) around the fish body, so we keep the inner
    ``inner`` fraction and take a *low* percentile: the fish is the closest thing in it.
    """
    fh, fw = frame_shape
    dh, dw = depth.shape[:2]
    sx, sy = dw / fw, dh / fh
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    ix1 = int(np.clip((cx - hw) * sx, 0, dw - 1))
    ix2 = int(np.clip((cx + hw) * sx, ix1 + 1, dw))
    iy1 = int(np.clip((cy - hh) * sy, 0, dh - 1))
    iy2 = int(np.clip((cy + hh) * sy, iy1 + 1, dh))
    patch = depth[iy1:iy2, ix1:ix2]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return None
    return float(np.percentile(patch, percentile))


class FishDepthSmoother:
    """Per-identity EWMA of z, so depth jitter does not show up as fake swimming speed."""

    def __init__(self, alpha: float = 0.35, forget_after_s: float = 10.0):
        self.alpha = alpha
        self.forget_after_s = forget_after_s
        self._state: dict[int, tuple[float, float]] = {}  # id -> (z, last_t)

    def update(self, fish_id: int, z: float | None, t: float) -> float | None:
        prev = self._state.get(fish_id)
        if prev is not None and t - prev[1] > self.forget_after_s:
            prev = None
        if z is None:
            return prev[0] if prev else None
        value = z if prev is None else (1 - self.alpha) * prev[0] + self.alpha * z
        self._state[fish_id] = (value, t)
        return value
