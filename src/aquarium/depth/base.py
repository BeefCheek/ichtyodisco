from __future__ import annotations

from typing import Protocol

import numpy as np


class DepthEstimator(Protocol):
    """Returns a raw depth-like map where *larger means farther* (HxW float32)."""

    name: str

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray: ...
