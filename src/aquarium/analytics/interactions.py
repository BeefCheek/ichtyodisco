"""Fine-grained, low-latency interactions: the "game events" of the tank.

The slow `EventDetector` (2 Hz, minutes of context, cooldowns of minutes) produces the
journal and alerts. This module runs on every frame and emits short, musical triggers:

    school_leave   a fish breaks away from the shoal          (x y z, speed)
    school_return  ...and comes back                          (+ duration of the excursion)
    school_split   the shoal splits into two groups or more   (groups)
    school_merge   groups fuse back into one                   (groups)
    encounter      two fish meet (enter each other's space)   (distance)
    dash           a fish starts a sprint                      (speed)
    surface        a fish touches the surface                  (x)
    bottom         a fish touches the bottom                   (x)
    glass          a fish comes right up to the front glass    (x y)
    back_wall      a fish reaches the back wall                (x y)
    group_turn     the whole shoal turns together              (angle)
    appear         a fish enters the scene
    vanish         a fish leaves the scene / hides

Each comes with the fish id(s) and normalised position, so a score can pan, pitch or
filter the sample it fires. Thresholds are relative to the current shoal (median spacing),
so "leaving the school" means the same thing for a tight neon shoal and loose barbs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from aquarium.types import AquariumEvent

MESSAGES = {
    "school_leave": "{name} quitte le banc",
    "school_return": "{name} revient au banc ({duration:.1f} s)",
    "school_split": "Le banc se scinde en {groups} groupes",
    "school_merge": "Les groupes se rejoignent ({groups})",
    "encounter": "{name} croise {other}",
    "dash": "{name} sprinte",
    "surface": "{name} touche la surface",
    "bottom": "{name} touche le fond",
    "glass": "{name} vient à la vitre",
    "back_wall": "{name} file au fond de l'aquarium",
    "group_turn": "Virage collectif ({angle:.0f}°)",
    "appear": "{name} apparaît",
    "vanish": "{name} disparaît",
}


@dataclass
class FishFrame:
    fid: int
    name: str
    pos: np.ndarray  # tank units (x, y, z_scaled)
    xn: float  # 0..1 frame
    yn: float  # 0..1 frame (0 = surface)
    zn: float | None  # 0..1 tank depth (0 = glass)
    vel: np.ndarray
    speed_bls: float
    state: str


@dataclass
class _FishMemo:
    out_since: float | None = None
    left_at: float | None = None  # when school_leave was emitted
    zones: set[str] = field(default_factory=set)
    state: str = "cruise"
    last_seen: float = 0.0


def _hyst(inside: bool, value: float, enter: float, leave: float, below: bool = True) -> bool:
    """Schmitt trigger: returns the new 'inside' flag."""
    if below:
        return value < (leave if inside else enter)
    return value > (leave if inside else enter)


class InteractionDetector:
    def __init__(self, min_school: int = 3, leave_factor: float = 2.5, return_factor: float = 1.6,
                 min_out_s: float = 0.35, encounter_d: float = 0.045, cooldown_s: float = 1.5,
                 vanish_s: float = 0.8):
        self.min_school = min_school
        self.leave_factor = leave_factor
        self.return_factor = return_factor
        self.min_out_s = min_out_s
        self.encounter_d = encounter_d
        self.cooldown_s = cooldown_s
        self.vanish_s = vanish_s
        self._memo: dict[int, _FishMemo] = {}
        self._pairs: set[tuple[int, int]] = set()
        self._last: dict[tuple, float] = {}
        self._groups = 1
        self._groups_candidate = (1, 0.0)
        self._heading: list[tuple[float, float]] = []  # (t, angle) of the group, when polarised
        self.counts: dict[str, int] = {}

    # ------------------------------------------------------------------ helpers
    def _ok(self, key: tuple, t: float, cooldown: float | None = None) -> bool:
        cd = self.cooldown_s if cooldown is None else cooldown
        last = self._last.get(key)
        if last is not None and t - last < cd:
            return False
        self._last[key] = t
        return True

    def _event(self, kind: str, t: float, fish: list[FishFrame], **data: float) -> AquariumEvent:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        payload: dict = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}
        if fish:
            f = fish[0]
            payload.update(x=round(f.xn, 4), y=round(f.yn, 4),
                           z=None if f.zn is None else round(f.zn, 4),
                           speed=round(f.speed_bls, 3))
        names = {"name": fish[0].name if fish else "", "other": fish[1].name if len(fish) > 1 else ""}
        try:
            msg = MESSAGES[kind].format(**names, **{k: v for k, v in data.items() if v is not None})
        except (KeyError, ValueError):
            msg = kind
        return AquariumEvent(kind, t, 0.1, msg, fish=[f.fid for f in fish], data=payload)

    # ------------------------------------------------------------------ main
    def update(self, t: float, fish: list[FishFrame]) -> list[AquariumEvent]:
        out: list[AquariumEvent] = []
        ids = [f.fid for f in fish]
        by_id = {f.fid: f for f in fish}

        # appear / vanish -------------------------------------------------------------
        for f in fish:
            memo = self._memo.get(f.fid)
            if memo is None:
                self._memo[f.fid] = memo = _FishMemo(state=f.state)
                if self._ok(("appear", f.fid), t, 10.0) and len(self._memo) > 1:
                    out.append(self._event("appear", t, [f]))
            memo.last_seen = t
        for fid, memo in list(self._memo.items()):
            if fid not in by_id and t - memo.last_seen > self.vanish_s:
                if fish and self._ok(("vanish", fid), t, 10.0):
                    ghost = FishFrame(fid, f"#{fid}", np.zeros(3), 0.5, 0.5, None, np.zeros(3), 0.0, "rest")
                    out.append(self._event("vanish", t, [ghost]))
                del self._memo[fid]

        # per-fish zone contacts and dashes ------------------------------------------
        for f in fish:
            memo = self._memo[f.fid]
            checks = [("surface", f.yn, 0.07, 0.12, True), ("bottom", f.yn, 0.93, 0.88, False)]
            if f.zn is not None:
                checks += [("glass", f.zn, 0.08, 0.15, True), ("back_wall", f.zn, 0.92, 0.85, False)]
            for kind, value, enter, leave, below in checks:
                was = kind in memo.zones
                now_in = _hyst(was, value, enter, leave, below)
                if now_in and not was:
                    memo.zones.add(kind)
                    if self._ok((kind, f.fid), t, 3.0):
                        out.append(self._event(kind, t, [f]))
                elif was and not now_in:
                    memo.zones.discard(kind)
            if f.state == "burst" and memo.state != "burst" and self._ok(("dash", f.fid), t, 2.0):
                out.append(self._event("dash", t, [f]))
            memo.state = f.state

        n = len(fish)
        if n >= 2:
            pos = np.stack([f.pos for f in fish])
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            nnd = d.min(axis=1)
            spacing = float(np.median(nnd))

            # encounters ---------------------------------------------------------------
            close_now = set()
            for i in range(n):
                for j in range(i + 1, n):
                    key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                    was = key in self._pairs
                    # "meeting" is relative to the shoal's spacing: in a tight shoal everyone is
                    # close, only a real approach counts
                    enter = self.encounter_d if n < 4 else min(self.encounter_d, 0.5 * spacing)
                    if _hyst(was, float(d[i, j]), enter, enter * 1.8):
                        close_now.add(key)
                        if not was and self._ok(("encounter",) + key, t, 4.0):
                            out.append(self._event("encounter", t, [fish[i], fish[j]],
                                                   distance=float(d[i, j])))
            self._pairs = close_now

            # leaving / returning to the school ----------------------------------------
            if n >= self.min_school:
                for i, f in enumerate(fish):
                    others = np.delete(pos, i, axis=0)
                    centroid = others.mean(axis=0)
                    radius = max(float(np.median(np.linalg.norm(others - centroid, axis=1))),
                                 spacing, 0.02)
                    dist = float(np.linalg.norm(f.pos - centroid))
                    memo = self._memo[f.fid]
                    is_out = memo.out_since is not None
                    if not is_out and dist > self.leave_factor * radius:
                        memo.out_since = t
                    elif is_out and dist < self.return_factor * radius:
                        if memo.left_at is not None and self._ok(("return", f.fid), t, 0.5):
                            out.append(self._event("school_return", t, [f],
                                                   duration=t - (memo.out_since or memo.left_at),
                                                   distance=dist / radius))
                        memo.out_since, memo.left_at = None, None
                    if (memo.out_since is not None and memo.left_at is None
                            and t - memo.out_since >= self.min_out_s
                            and self._ok(("leave", f.fid), t, self.cooldown_s)):
                        memo.left_at = t
                        out.append(self._event("school_leave", t, [f], distance=dist / radius))

            # split / merge (single-linkage groups) ------------------------------------
            if n >= 4:
                link = max(2.5 * spacing, 0.06)
                groups = self._count_groups(d, link)
                cand, since = self._groups_candidate
                if groups != cand:
                    self._groups_candidate = (groups, t)
                elif groups != self._groups and t - since >= 0.6:
                    kind = "school_split" if groups > self._groups else "school_merge"
                    if self._ok((kind,), t, 3.0):
                        out.append(self._event(kind, t, [], groups=groups))
                    self._groups = groups

            # group turn -----------------------------------------------------------------
            v = np.stack([f.vel[:2] for f in fish])
            norms = np.linalg.norm(v, axis=1)
            moving = norms > 1e-3
            if moving.sum() >= 3:
                u = (v[moving] / norms[moving, None]).mean(axis=0)
                if np.linalg.norm(u) > 0.6:
                    ang = math.atan2(u[1], u[0])
                    self._heading = [(ht, a) for ht, a in self._heading if t - ht <= 0.8]
                    if self._heading:
                        a0 = self._heading[0][1]
                        delta = abs((ang - a0 + math.pi) % (2 * math.pi) - math.pi)
                        if delta > math.radians(100) and self._ok(("turn",), t, 3.0):
                            out.append(self._event("group_turn", t, [], angle=math.degrees(delta)))
                            self._heading = []
                    self._heading.append((t, ang))
                else:
                    self._heading = []
        return out

    @staticmethod
    def _count_groups(d: np.ndarray, link: float) -> int:
        n = d.shape[0]
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j in zip(*np.where(np.triu(d < link, k=1)), strict=True):
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
        sizes: dict[int, int] = {}
        for i in range(n):
            r = find(i)
            sizes[r] = sizes.get(r, 0) + 1
        return sum(1 for s in sizes.values() if s >= 2) or 1
