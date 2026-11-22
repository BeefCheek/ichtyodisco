"""Apple Depth Pro 2 (Nov 2026) in live mode.

Unlike single-image models, Depth Pro 2 has an *unbuffered streaming* mode: each frame is
predicted from itself and the few frames before it (a KV cache), so depth is steady from frame
to frame instead of flickering. That is what a live aquarium needs: less jitter means fewer
fake depth movements, so per-fish z (and the 3D speeds) are cleaner.

Output is metric depth in metres along the optical axis (larger = farther), which the tank
calibration downstream maps to 0 (front glass) .. 1 (back wall).

Not on PyPI. Install it yourself (see README), e.g.:
    uv pip install "git+https://github.com/apple-aiml-research/ml-depth-pro-2"
Weights download on first use from Apple's CDN (S 97 MB, B 386 MB, L 1.3 GB, H+ 3.5 GB).
Licence: research / non-commercial only.
"""

from __future__ import annotations

import cv2
import numpy as np

SIZES = {"depth-pro-2": "s", "depth-pro-2-s": "s", "depth-pro-2-b": "b",
         "depth-pro-2-l": "l", "depth-pro-2-h": "h"}


class DepthPro2Estimator:
    def __init__(self, model: str = "depth-pro-2", device: str | None = None,
                 input_width: int = 640, window: int = 4, f_px: float | None = None):
        from depth_pro_2 import create_model_and_transforms

        size = SIZES.get(model, "s")
        self.name = f"depth-pro-2-{size}"
        self.input_width = input_width
        self.window = window
        self.f_px = f_px  # focal length in pixels of the *resized* frame, if known
        self.model, self.transform = create_model_and_transforms(f"depthpro2-{size}", device=device)
        self.device = str(next(self.model.parameters()).device)
        self._stream = None
        self._shape: tuple[int, int] | None = None

    def _open_stream(self, shape: tuple[int, int]) -> None:
        # A stream belongs to one video size: reopen it if the camera resolution changes.
        self._stream = self.model.stream(window=self.window, f_px=self.f_px, mode="unbuffered")
        self._shape = shape

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.input_width and w > self.input_width:
            scale = self.input_width / w
            frame_bgr = cv2.resize(frame_bgr, (self.input_width, max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
        shape = frame_bgr.shape[:2]
        if self._stream is None or shape != self._shape:
            self._open_stream(shape)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preds = self._stream.step(self.transform(rgb))
        if not preds:  # unbuffered mode returns one prediction per frame; be defensive anyway
            preds = self._stream.flush()
        p = preds[-1]
        if p.get("depth") is not None:
            depth = p["depth"]
        else:  # focal length not estimated yet: inverse depth still orders near / far
            depth = 1.0 / p["inverse_depth"].clamp_min(1e-6)
        out = depth.float().cpu().numpy().astype(np.float32)
        out[~np.isfinite(out)] = np.nan  # "infinitely far" pixels: ignored by the calibration
        return out
