"""Extract varied frames from aquarium footage to annotate a fish detector.

    uv run scripts/extract_frames.py recording.mp4 datasets/fish/images --every 2.0

Then annotate (Label Studio, CVAT or Roboflow, YOLO format, one class "fish") and train:

    uv run yolo detect train model=yolo11n.pt data=datasets/fish/data.yaml imgsz=640 epochs=80
    cp runs/detect/train/weights/best.pt models/fish.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between candidates")
    ap.add_argument("--min-diff", type=float, default=6.0, help="skip near-duplicate frames")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps * a.every))
    prev, idx, saved = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
            if prev is None or float(np.abs(small - prev).mean()) >= a.min_diff:
                cv2.imwrite(str(out / f"{Path(a.video).stem}_{idx:07d}.jpg"), frame)
                saved += 1
                prev = small
        idx += 1
    print(f"saved {saved} frames to {out}")


if __name__ == "__main__":
    main()
