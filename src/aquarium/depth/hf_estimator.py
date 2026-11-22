"""Monocular depth with Hugging Face ``transformers`` (Depth Pro, Depth Anything V2, ...).

Two families of outputs exist and are unified here into "larger = farther":
    * metric depth (Apple Depth Pro): already distance-like.
    * relative inverse depth / disparity (Depth Anything V2): larger = closer -> inverted.

Licences matter for a public stream: Apple Depth Pro weights are research-only (Apple
ASCL), Depth Anything V2 *Small* is Apache-2.0 while Base/Large are CC-BY-NC-4.0.
"""

from __future__ import annotations

import cv2
import numpy as np

PRESETS: dict[str, tuple[str, bool]] = {
    # name: (hf checkpoint, output is inverse depth)
    "depth-pro": ("apple/DepthPro-hf", False),
    "depth-anything-v2-small": ("depth-anything/Depth-Anything-V2-Small-hf", True),
    "depth-anything-v2-base": ("depth-anything/Depth-Anything-V2-Base-hf", True),
    "depth-anything-v2-large": ("depth-anything/Depth-Anything-V2-Large-hf", True),
}


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFDepthEstimator:
    def __init__(self, model: str, device: str | None = None, input_width: int = 518):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        checkpoint, inverse = PRESETS.get(model, (model, "depth-anything" in model.lower()))
        self.name = model
        self.inverse = inverse
        self.input_width = input_width
        self.device = device or _auto_device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint, dtype=dtype)
        self.model.to(self.device).eval()
        self._torch = torch
        self._dtype = dtype

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.input_width and w > self.input_width:
            scale = self.input_width / w
            frame_bgr = cv2.resize(
                frame_bgr, (self.input_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {
            k: (v.to(self.device, self._dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }
        with self._torch.inference_mode():
            out = self.model(**inputs)
        depth = out.predicted_depth.squeeze().float().cpu().numpy()
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.inverse:
            depth = 1.0 / np.maximum(depth - depth.min() + 1e-3, 1e-3)
        return depth.astype(np.float32)
