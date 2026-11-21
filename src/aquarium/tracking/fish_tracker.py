"""Multi-object tracking of fish with Ultralytics YOLO + ByteTrack/BoT-SORT."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from aquarium.config import TrackingConfig
from aquarium.types import TrackedFish


class Tracker(Protocol):
    def update(self, frame: np.ndarray) -> list[TrackedFish]: ...


class FishTracker:
    """Thin wrapper around ``YOLO.track(persist=True)``.

    Ultralytics keeps the tracker state between calls when ``persist=True``, so this
    object must be fed consecutive frames of a single stream.

    Generic COCO weights don't know "fish" (closest classes are random animals), so for a
    real tank fine-tune a detector on a few hundred annotated frames of *your* aquarium
    (see README, section "Entraîner le détecteur").
    """

    def __init__(self, cfg: TrackingConfig):
        from ultralytics import YOLO  # heavy import, done lazily

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.names: dict[int, str] = dict(getattr(self.model, "names", {}) or {})

    def update(self, frame: np.ndarray) -> list[TrackedFish]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=self.cfg.classes,
            device=self.cfg.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []  # detections without a confirmed track id yet
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.int().cpu().numpy()
        return [
            TrackedFish(
                track_id=int(tid),
                bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                confidence=float(c),
                class_id=int(k),
                class_name=self.names.get(int(k), "fish"),
            )
            for b, tid, c, k in zip(xyxy, ids, confs, clss, strict=True)
        ]
