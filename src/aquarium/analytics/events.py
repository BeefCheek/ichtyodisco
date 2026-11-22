"""Rule-based behavioural event detection on top of the learnt baselines.

Each rule is "condition held for N seconds" + a cooldown, which makes events robust to
tracker noise and keeps the stream ticker readable. Thresholds are mostly expressed in
standard deviations from the *learnt* circadian baseline, so they adapt to each tank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aquarium.types import AquariumEvent


@dataclass
class FishView:
    fid: int
    name: str
    pos: np.ndarray  # tank units (3,)
    vel: np.ndarray  # (3,)
    speed_bls: float
    state: str
    y: float  # 0 surface .. 1 bottom


@dataclass
class EventContext:
    t: float
    fish: list[FishView]
    metrics: dict[str, float]
    zscores: dict[str, float]
    baseline_ready: bool
    rest_threshold: float
    burst_threshold: float
    pairs: list[tuple[int, int, float]] = field(default_factory=list)
    missing: list[tuple[int, str, float]] = field(default_factory=list)  # id, name, absent s
    new_residents: list[tuple[int, str]] = field(default_factory=list)
    reidentified: list[tuple[int, str, float]] = field(default_factory=list)
    records: list[tuple[int, str, float]] = field(default_factory=list)  # id, name, speed
    warmed_up: bool = True


def _sev(z: float, lo: float = 2.0, hi: float = 5.0) -> float:
    return float(np.clip((abs(z) - lo) / (hi - lo) * 0.7 + 0.3, 0.0, 1.0))


class EventDetector:
    def __init__(self, cooldown_s: float = 60.0, chase_distance: float = 0.08,
                 chase_min_s: float = 1.5, hiding_s: float = 45.0,
                 feeding_z: float = 1.2, feeding_tau_s: float = 4.0):
        self.cooldown_s = cooldown_s
        # Feeding uses an EMA of the activity z-score: per-snapshot activity is spiky (fish
        # dart and pause around the food), so a raw "z > 2 held 2 s" almost never holds.
        self.feeding_z = feeding_z
        self.feeding_tau_s = feeding_tau_s
        self._act_ema: float | None = None
        self._act_t: float | None = None
        self.chase_distance = chase_distance
        self.chase_min_s = chase_min_s
        self.hiding_s = hiding_s
        self._since: dict[Any, float] = {}
        self._last: dict[Any, float] = {}
        self.feeding = False
        self.feeding_started: float | None = None

    def _held(self, key: Any, cond: bool, t: float, seconds: float) -> bool:
        if not cond:
            self._since.pop(key, None)
            return False
        start = self._since.setdefault(key, t)
        return t - start >= seconds

    def _smooth_activity(self, act_z: float, t: float) -> float:
        if self._act_ema is None or self._act_t is None or t <= self._act_t:
            self._act_ema = act_z if self._act_ema is None else self._act_ema
        else:
            a = 1.0 - math.exp(-(t - self._act_t) / self.feeding_tau_s)
            self._act_ema += a * (act_z - self._act_ema)
        self._act_t = t
        return self._act_ema

    def _cool(self, key: Any, t: float, cooldown: float | None = None) -> bool:
        cd = self.cooldown_s if cooldown is None else cooldown
        last = self._last.get(key)
        if last is not None and t - last < cd:
            return False
        self._last[key] = t
        return True

    def update(self, c: EventContext) -> list[AquariumEvent]:
        out: list[AquariumEvent] = []
        t, m, z = c.t, c.metrics, c.zscores
        n = len(c.fish)
        act_z = z.get("activity", 0.0)
        act_ema = self._smooth_activity(act_z, t)

        # --- Feeding frenzy: sustained activity rise while the group rises to the surface.
        feeding_cond = n >= 1 and act_ema > self.feeding_z and (
            m.get("surface_ratio", 0) > 0.4 or m.get("mean_y", 1) < 0.4
        )
        if not self.feeding and self._held("feeding", feeding_cond, t, 2.0):
            self.feeding, self.feeding_started = True, t
            if self._cool("feeding", t, 600):
                out.append(AquariumEvent("feeding", t, _sev(act_ema, lo=1.0), "Festin ! Ruée vers "
                                         f"la surface (activité {act_ema:+.1f}σ)",
                                         data={"activity_z": act_ema}))
        elif self.feeding and act_ema < 1.0 and m.get("surface_ratio", 0) < 0.3:
            duration = t - (self.feeding_started or t)
            self.feeding, self.feeding_started = False, None
            out.append(AquariumEvent("feeding_end", t, 0.2,
                                     f"Fin du repas ({duration:.0f} s)",
                                     data={"duration_s": duration}))

        if c.baseline_ready and not self.feeding and n >= 1:
            if self._held("act_hi", act_z > 3.0, t, 20.0) and self._cool("act_hi", t, 900):
                out.append(AquariumEvent("agitation", t, _sev(act_z),
                                         f"Agitation inhabituelle pour cette heure "
                                         f"({act_z:+.1f}σ)", data={"activity_z": act_z}))
            if self._held("act_lo", act_z < -2.5, t, 90.0) and self._cool("act_lo", t, 1800):
                out.append(AquariumEvent("lethargy", t, _sev(act_z),
                                         f"Aquarium anormalement calme pour cette heure "
                                         f"({act_z:+.1f}σ)", data={"activity_z": act_z}))
            coh_z = z.get("cohesion", 0.0)
            if n >= 3 and self._held("tight", coh_z < -2.5, t, 8.0) and self._cool("tight", t, 600):
                out.append(AquariumEvent("tight_shoal", t, _sev(coh_z),
                                         f"Banc très resserré ({coh_z:+.1f}σ) — réaction "
                                         "d'alerte ?", data={"cohesion_z": coh_z}))
            pol = m.get("polarization", 0.0)
            if n >= 3 and self._held("polar", pol > 0.85, t, 4.0) and self._cool("polar", t, 600):
                out.append(AquariumEvent("schooling", t, 0.4,
                                         f"Nage synchronisée : {n} poissons dans la même "
                                         f"direction (polarisation {pol:.2f})",
                                         data={"polarization": pol}))

        if not self.feeding and n >= 2:
            crowd = m.get("surface_ratio", 0.0) > 0.6
            if self._held("surface", crowd, t, 120.0) and self._cool("surface", t, 1800):
                out.append(AquariumEvent("surface_crowding", t, 0.8,
                                         "Beaucoup de poissons en surface depuis 2 min — "
                                         "vérifier l'oxygénation / la température"))

        out.extend(self._chases(c))

        for fid, name, absent in c.missing:
            if self._held(("hide", fid), absent >= self.hiding_s, t, 0.0) and \
                    self._cool(("hide", fid), t, 1800):
                out.append(AquariumEvent("hiding", t, 0.35,
                                         f"{name} se cache depuis {absent:.0f} s", fish=[fid]))
        if c.warmed_up:
            for fid, name in c.new_residents:
                out.append(AquariumEvent("new_resident", t, 0.5,
                                         f"Nouveau résident repéré : {name}", fish=[fid]))
        for fid, name, gap in c.reidentified:
            if gap >= 300 and self._cool(("back", fid), t, 600):
                out.append(AquariumEvent("return", t, 0.3,
                                         f"Re-voilà {name} (absent {gap / 60:.0f} min)",
                                         fish=[fid], data={"gap_s": gap}))
        for fid, name, speed in c.records:
            if self._cool(("record", fid), t, 300):
                out.append(AquariumEvent("record", t, 0.45,
                                         f"Record de vitesse pour {name} : {speed:.1f} "
                                         "longueurs/s", fish=[fid], data={"speed_bls": speed}))
        return out

    def _chases(self, c: EventContext) -> list[AquariumEvent]:
        out: list[AquariumEvent] = []
        by_id = {f.fid: f for f in c.fish}
        live = set()
        fast = max(c.rest_threshold * 3, c.burst_threshold * 0.5)
        for a, b, dist in c.pairs:
            fa, fb = by_id.get(a), by_id.get(b)
            if fa is None or fb is None or dist > self.chase_distance:
                continue
            na, nb = np.linalg.norm(fa.vel), np.linalg.norm(fb.vel)
            if na < 1e-6 or nb < 1e-6:
                continue
            align = float(fa.vel @ fb.vel / (na * nb))
            cond = fa.speed_bls > fast and fb.speed_bls > fast and align > 0.6
            key = ("chase", min(a, b), max(a, b))
            live.add(key)
            if self._held(key, cond, c.t, self.chase_min_s) and self._cool(key, c.t, 120):
                # the chaser is the one whose velocity points at the other fish
                ahead_of_a = float(fa.vel @ (fb.pos - fa.pos))
                chaser, target = (fa, fb) if ahead_of_a > 0 else (fb, fa)
                out.append(AquariumEvent(
                    "chase", c.t, 0.55, f"{chaser.name} poursuit {target.name}",
                    fish=[chaser.fid, target.fid],
                    data={"distance": round(dist, 3), "alignment": round(align, 2)}))
        for key in [k for k in self._since if isinstance(k, tuple) and k[0] == "chase"]:
            if key not in live:
                self._since.pop(key, None)
        return out


def mood(metrics: dict[str, float], z: dict[str, float], feeding: bool) -> tuple[str, str]:
    """A single, stream-friendly word summarising the tank (label, emoji-free colour key)."""
    if metrics.get("fish_count", 0) == 0:
        return "désert", "grey"
    if feeding:
        return "festin", "gold"
    act, coh = z.get("activity", 0.0), z.get("cohesion", 0.0)
    if coh < -2.0 and act > 1.0:
        return "sur le qui-vive", "red"
    if act > 2.0 or metrics.get("burst_ratio", 0) > 0.3:
        return "excité", "orange"
    if metrics.get("rest_ratio", 0) > 0.6 or act < -1.5:
        return "endormi", "blue"
    if act > 0.7:
        return "joueur", "green"
    return "serein", "teal"


def describe_z(z: float) -> str:
    if not math.isfinite(z):
        return "?"
    if z > 2:
        return "très au-dessus de la normale"
    if z > 1:
        return "au-dessus de la normale"
    if z < -2:
        return "très en dessous de la normale"
    if z < -1:
        return "en dessous de la normale"
    return "dans la normale"
