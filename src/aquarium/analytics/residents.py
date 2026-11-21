"""Stable identities ("residents") on top of short-lived tracker ids.

ByteTrack ids change every time a fish is occluded behind a plant or another fish, and
reset on every restart. To build per-fish statistics over days we re-identify fish with a
light appearance signature (HSV colour histogram of the body), matched against residents
that are not currently on screen.

Limits, stated honestly: identical-looking fish (a shoal of neon tetras) cannot be told
apart by colour alone; they end up as a few "residents" sharing statistics. Distinctive
fish (a betta, a pleco, a couple of guppies with different tails) work well. For a
stronger re-id, swap ``descriptor()`` for a learnt embedding (e.g. BoT-SORT's ReID model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from aquarium.analytics.stats import DecayingStats
from aquarium.types import TrackedFish

LAB_WEIGHTS = np.array([0.5, 1.0, 1.0] * 3, dtype=np.float32)  # lightness varies with depth


def descriptor(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Colour signature of the fish *body*, with the surrounding water masked out.

    The background colour is estimated from the border of the bounding box (mostly water)
    and only pixels that differ clearly from it are kept, so the signature does not change
    when the fish turns and its box holds more or less water. The signature is the 25/50/75th
    percentiles of those pixels in CIELAB (9 numbers): robust with the few dozen pixels a
    small fish covers, and it works for white / grey / black fish that have no hue.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    lab = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:2].reshape(-1, 3), lab[-2:].reshape(-1, 3),
                             lab[:, :2].reshape(-1, 3), lab[:, -2:].reshape(-1, 3)])
    diff = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    body = lab[diff > max(18.0, float(np.percentile(diff, 50)))]
    if len(body) < 12:
        return None
    q = np.percentile(body, [25, 50, 75], axis=0).flatten()
    return (q * LAB_WEIGHTS / 100.0).astype(np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


@dataclass
class Resident:
    rid: int
    name: str
    signature: np.ndarray | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    visible_s: float = 0.0
    speed: DecayingStats = field(default_factory=lambda: DecayingStats(7 * 86400.0))
    depth: DecayingStats = field(default_factory=lambda: DecayingStats(7 * 86400.0))
    height: DecayingStats = field(default_factory=lambda: DecayingStats(7 * 86400.0))
    zone_time: dict[str, float] = field(default_factory=dict)
    state_time: dict[str, float] = field(default_factory=dict)
    max_speed_bls: float = 0.0
    sightings: int = 0
    last_pos: tuple[float, float] | None = None  # normalised by frame width, not persisted

    def favourite(self, table: dict[str, float]) -> str | None:
        return max(table, key=table.__getitem__) if table else None

    def summary(self) -> dict[str, Any]:
        total_states = sum(self.state_time.values()) or 1.0
        return {
            "id": self.rid,
            "name": self.name,
            "visible_min": round(self.visible_s / 60.0, 1),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "mean_speed_bls": round(self.speed.mean, 2),
            "max_speed_bls": round(self.max_speed_bls, 2),
            "mean_depth": round(self.depth.mean, 2) if self.depth.w > 0 else None,
            "mean_height": round(1.0 - self.height.mean, 2) if self.height.w > 0 else None,
            "favourite_zone": self.favourite(self.zone_time),
            "states": {k: round(v / total_states, 3) for k, v in self.state_time.items()},
            "sightings": self.sightings,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "rid": self.rid, "name": self.name,
            "signature": None if self.signature is None else self.signature.tolist(),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "visible_s": self.visible_s, "speed": self.speed.to_dict(),
            "depth": self.depth.to_dict(), "height": self.height.to_dict(),
            "zone_time": self.zone_time, "state_time": self.state_time,
            "max_speed_bls": self.max_speed_bls, "sightings": self.sightings,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Resident:
        r = cls(rid=d["rid"], name=d["name"])
        sig = d.get("signature")
        r.signature = None if sig is None else np.asarray(sig, dtype=np.float32)
        r.first_seen, r.last_seen, r.visible_s = d["first_seen"], d["last_seen"], d["visible_s"]
        r.speed = DecayingStats.from_dict(d["speed"])
        r.depth = DecayingStats.from_dict(d["depth"])
        r.height = DecayingStats.from_dict(d["height"])
        r.zone_time, r.state_time = d.get("zone_time", {}), d.get("state_time", {})
        r.max_speed_bls, r.sightings = d.get("max_speed_bls", 0.0), d.get("sightings", 0)
        return r


class ResidentRegistry:
    def __init__(self, reid: bool = True, max_distance: float = 0.25, warmup_frames: int = 8,
                 track_timeout_s: float = 3.0):
        self.reid = reid
        self.max_distance = max_distance
        self.warmup_frames = warmup_frames
        self.track_timeout_s = track_timeout_s
        self.residents: dict[int, Resident] = {}
        self._next_id = 1
        self._track_to_resident: dict[int, int] = {}
        self._track_last_seen: dict[int, float] = {}
        self._pending: dict[int, list[np.ndarray]] = {}
        self.new_residents: list[int] = []  # drained by the engine (events)
        self.reidentified: list[tuple[int, float]] = []  # (resident, gap seconds)

    def _create(self, t: float, signature: np.ndarray | None) -> Resident:
        rid = self._next_id
        self._next_id += 1
        r = Resident(rid=rid, name=f"Poisson #{rid}", signature=signature,
                     first_seen=t, last_seen=t)
        self.residents[rid] = r
        self.new_residents.append(rid)
        return r

    def assign(self, fish: list[TrackedFish], frame: np.ndarray | None, t: float) -> None:
        active = {
            self._track_to_resident[f.track_id] for f in fish
            if f.track_id in self._track_to_resident
        }
        for f in fish:
            self._track_last_seen[f.track_id] = t
            rid = self._track_to_resident.get(f.track_id)
            if rid is not None and rid in self.residents:
                f.resident_id = rid
                r = self.residents[rid]
                r.last_pos = self._norm_pos(f, frame)
                if self.reid and frame is not None and f.confidence >= 0.5 and r.sightings % 5 == 0:
                    d = descriptor(frame, f.bbox)
                    if d is not None:
                        r.signature = d if r.signature is None else 0.95 * r.signature + 0.05 * d
                r.sightings += 1
                continue
            if not self.reid or frame is None:
                r = self._create(t, None)
                self._bind(f, r, t, frame)
                continue
            d = descriptor(frame, f.bbox)
            buf = self._pending.setdefault(f.track_id, [])
            if d is not None:
                buf.append(d)
            if len(buf) < self.warmup_frames:
                continue  # identity not decided yet: f.resident_id stays None
            sig = np.median(buf, axis=0).astype(np.float32)
            del self._pending[f.track_id]
            best = self._match(sig, f, active, t, frame.shape[:2])
            if best is None:
                best = self._create(t, sig)
            else:
                self.reidentified.append((best.rid, t - best.last_seen))
            active.add(best.rid)
            self._bind(f, best, t, frame)
        self._expire(t)

    def _match(self, sig: np.ndarray, f: TrackedFish, active: set[int], t: float,
               shape: tuple[int, int]) -> Resident | None:
        """Pick the resident this new track most likely is, or None for a newcomer.

        Two kinds of evidence:
            * spatio-temporal: a resident lost < 3 s ago, close to where this track starts,
              is almost surely the same fish after a tracker id switch (occlusion). Accepted
              with a lenient appearance check.
            * appearance only (fish came back after a long absence): strict threshold plus
              a ratio test, so an ambiguous look-alike becomes a new resident rather than
              polluting someone else's statistics.
        """
        fh, fw = shape
        cx, cy = f.center
        pos = (cx / fw, cy / fw)
        scored: list[tuple[float, float, Resident]] = []
        for r in self.residents.values():
            if r.rid in active or r.signature is None or r.signature.shape != sig.shape:
                continue
            scored.append((distance(sig, r.signature), 0.0, r))
        if not scored:
            return None
        recent = []
        for app, _, r in scored:
            gap = t - r.last_seen
            if r.last_pos is not None and gap < 3.0:
                ds = float(np.hypot(pos[0] - r.last_pos[0], pos[1] - r.last_pos[1]))
                if ds < 0.12 + 0.1 * gap and app < 4.0 * self.max_distance:
                    recent.append((app * 0.5 + ds, r))
        if recent:
            return min(recent, key=lambda x: x[0])[1]
        scored.sort(key=lambda x: x[0])
        best_d, _, best = scored[0]
        if best_d > self.max_distance:
            return None
        if len(scored) > 1 and best_d > 0.8 * scored[1][0]:
            return None  # ambiguous
        return best

    def _bind(self, f: TrackedFish, r: Resident, t: float, frame: np.ndarray | None) -> None:
        self._track_to_resident[f.track_id] = r.rid
        f.resident_id = r.rid
        r.sightings += 1
        r.last_seen = t
        r.last_pos = self._norm_pos(f, frame)

    @staticmethod
    def _norm_pos(f: TrackedFish, frame: np.ndarray | None) -> tuple[float, float] | None:
        if frame is None:
            return None
        w = frame.shape[1]
        cx, cy = f.center
        return cx / w, cy / w

    def _expire(self, t: float) -> None:
        for tid, seen in list(self._track_last_seen.items()):
            if t - seen > self.track_timeout_s:
                self._track_last_seen.pop(tid, None)
                self._track_to_resident.pop(tid, None)
                self._pending.pop(tid, None)

    def prune(self, t: float, min_visible_s: float = 20.0, ghost_after_s: float = 3600.0) -> None:
        """Drop "ghost" residents: false detections seen a few seconds, long ago."""
        for rid, r in list(self.residents.items()):
            if r.visible_s < min_visible_s and t - r.last_seen > ghost_after_s:
                del self.residents[rid]

    def rename(self, rid: int, name: str) -> bool:
        r = self.residents.get(rid)
        if r is None:
            return False
        r.name = name.strip()[:40] or r.name
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"next_id": self._next_id,
                "residents": [r.to_dict() for r in self.residents.values()]}

    def load_dict(self, d: dict[str, Any]) -> None:
        self.residents = {r["rid"]: Resident.from_dict(r) for r in d.get("residents", [])}
        self._next_id = max([d.get("next_id", 1), *[rid + 1 for rid in self.residents]])
