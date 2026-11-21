"""Shared data structures passed between pipeline stages.

Coordinates convention (used everywhere downstream of the tracker):
    * ``x`` : 0 = left wall, 1 = right wall (normalised by frame width)
    * ``y`` : 0 = surface, 1 = bottom (normalised by frame height)
    * ``z`` : 0 = front glass, 1 = back wall (normalised tank depth), ``None`` if unknown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrackedFish:
    """One fish detection associated to a tracker identity for a single frame."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "fish"
    z: float | None = None  # filled by the depth stage
    resident_id: int | None = None  # stable identity across tracker id switches / sessions

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1), max(0.0, y2 - y1)

    @property
    def identity(self) -> int:
        """Resident id when known, otherwise the raw tracker id."""
        return self.resident_id if self.resident_id is not None else self.track_id


@dataclass(slots=True)
class DepthMap:
    """Dense relative depth, normalised so 0 = near (front glass) and 1 = far (back wall)."""

    data: np.ndarray  # float32, HxW (may differ from the frame size)
    timestamp: float
    source_shape: tuple[int, int]  # (H, W) of the frame it was computed on
    model: str = "unknown"
    latency_s: float = 0.0


@dataclass(slots=True)
class AquariumEvent:
    """A notable behavioural event detected by the analytics engine."""

    kind: str
    timestamp: float
    severity: float  # 0..1, how unusual / important
    message: str
    fish: list[int] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "severity": round(self.severity, 3),
            "message": self.message,
            "fish": self.fish,
            "data": self.data,
        }
