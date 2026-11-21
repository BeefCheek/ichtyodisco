"""AnalyticsEngine: turns tracked fish into live stats, a learnt model of the tank and events."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from aquarium.analytics.behavior import STATE_LABELS_FR, STATES, BehaviorClassifier
from aquarium.analytics.events import EventContext, EventDetector, FishView, describe_z, mood
from aquarium.analytics.kinematics import FishTrack
from aquarium.analytics.residents import ResidentRegistry
from aquarium.analytics.social import group_metrics
from aquarium.analytics.stats import CircadianBaseline
from aquarium.analytics.zones import VERTICAL, OccupancyMap, zone_of
from aquarium.config import AnalyticsConfig
from aquarium.types import AquariumEvent, TrackedFish

BASELINE_METRICS = ("activity", "cohesion", "nnd", "polarization", "mean_y", "mean_z",
                    "fish_count", "burst_ratio", "rest_ratio")


class AnalyticsEngine:
    def __init__(self, cfg: AnalyticsConfig | None = None, tank_depth_ratio: float = 0.4):
        self.cfg = cfg or AnalyticsConfig()
        self.depth_ratio = tank_depth_ratio
        self.registry = ResidentRegistry(reid=self.cfg.reid, max_distance=self.cfg.reid_max_distance)
        self.classifier = BehaviorClassifier()
        self.baseline = CircadianBaseline(self.cfg.baseline_half_life_hours * 3600.0)
        self.occupancy = OccupancyMap(self.cfg.heatmap_size, self.cfg.heatmap_half_life_s)
        self.events = EventDetector(
            cooldown_s=self.cfg.event_cooldown_s,
            chase_distance=self.cfg.chase_distance,
            chase_min_s=self.cfg.chase_min_seconds,
            hiding_s=self.cfg.hiding_seconds,
        )
        self.tracks: dict[int, FishTrack] = {}
        self.states: dict[int, str] = {}
        self.zones: dict[int, dict[str, str]] = {}
        self.norm_pos: dict[int, tuple[float, float, float | None]] = {}
        self.recent_events: deque[AquariumEvent] = deque(maxlen=50)
        self.timeline: deque[dict[str, Any]] = deque(maxlen=3600)
        self.latest: dict[str, Any] = {}
        self.started_at = time.time()
        self.samples = 0
        self._last_t: float | None = None
        self._last_snapshot = 0.0
        self._pending_events: list[AquariumEvent] = []

    # ------------------------------------------------------------------ per frame
    def update(self, fish: list[TrackedFish], frame: np.ndarray | None,
               frame_shape: tuple[int, int], t: float | None = None,
               extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Feed one frame worth of tracked fish. Returns a snapshot when one is due."""
        t = time.time() if t is None else t
        dt = 0.0 if self._last_t is None else float(np.clip(t - self._last_t, 0.0, 0.5))
        self._last_t = t
        fh, fw = frame_shape
        self.registry.assign(fish, frame, t)

        points = []
        for f in fish:
            fid = f.resident_id
            if fid is None:
                continue
            cx, cy = f.center
            xn, yn = cx / fw, cy / fh
            body = max(f.size) / fw
            z_tank = None if f.z is None else f.z * self.depth_ratio
            track = self.tracks.get(fid)
            if track is None:
                track = self.tracks[fid] = FishTrack(fid, history_s=self.cfg.history_seconds)
            track.add(t, cx / fw, cy / fw, z_tank, body)
            k = track.kin
            state = self.classifier.classify(fid, k.speed_bls, k.tortuosity, t)
            self.states[fid] = state
            zone = zone_of(xn, yn, f.z)
            self.zones[fid] = zone
            self.norm_pos[fid] = (xn, yn, f.z)
            points.append((xn, yn, f.z))
            r = self.registry.residents.get(fid)
            if r is not None and dt > 0:
                r.visible_s += dt
                r.last_seen = t
                zkey = zone["vertical"] + ("-" + zone["depth"] if "depth" in zone else "")
                r.zone_time[zkey] = r.zone_time.get(zkey, 0.0) + dt
                r.state_time[state] = r.state_time.get(state, 0.0) + dt
        if points and dt > 0:
            self.occupancy.add(points, t, weight=dt)

        for fid in [i for i, tr in self.tracks.items() if t - tr.last_seen > self.cfg.history_seconds]:
            del self.tracks[fid]
            self.classifier.forget(fid)
            self.states.pop(fid, None)
            self.zones.pop(fid, None)
            self.norm_pos.pop(fid, None)

        if t - self._last_snapshot >= 1.0 / max(self.cfg.snapshot_rate_hz, 0.1):
            self._last_snapshot = t
            return self._snapshot(t, extra or {})
        return None

    # ------------------------------------------------------------------ snapshot
    def visible_ids(self, t: float, max_age: float = 0.5) -> list[int]:
        return [fid for fid, tr in self.tracks.items() if t - tr.last_seen <= max_age]

    def _snapshot(self, t: float, extra: dict[str, Any]) -> dict[str, Any]:
        ids = self.visible_ids(t)
        n = len(ids)
        pos = np.array([self.tracks[i].position for i in ids], dtype=np.float64).reshape(n, 3)
        vel = np.array([[self.tracks[i].kin.vx, self.tracks[i].kin.vy, self.tracks[i].kin.vz]
                        for i in ids], dtype=np.float64).reshape(n, 3)
        spd = np.array([self.tracks[i].kin.speed_bls for i in ids], dtype=np.float64)
        g = group_metrics(ids, pos, vel, spd, close_distance=self.cfg.chase_distance * 1.5)
        states = [self.states.get(i, "cruise") for i in ids]
        ys = [self.norm_pos[i][1] for i in ids]
        zs = [self.norm_pos[i][2] for i in ids if self.norm_pos[i][2] is not None]

        metrics = {
            "fish_count": float(n),
            "activity": g.activity,
            "cohesion": g.cohesion,
            "nnd": g.nnd,
            "polarization": g.polarization,
            "mean_y": float(np.mean(ys)) if ys else 0.5,
            "mean_z": float(np.mean(zs)) if zs else float("nan"),
            "burst_ratio": states.count("burst") / n if n else 0.0,
            "rest_ratio": states.count("rest") / n if n else 0.0,
            "surface_ratio": sum(1 for y in ys if y < 0.25) / n if n else 0.0,
        }
        zscores = {}
        for key in BASELINE_METRICS:
            v = metrics[key]
            needs_group = key in ("cohesion", "nnd", "polarization")
            if not np.isfinite(v) or (n == 0 and key != "fish_count") or (needs_group and n < 2):
                continue
            zscores[key] = self.baseline.zscore(key, v, t)
            self.baseline.update(key, v, t)
        self.samples += 1

        # Records must be checked before the resident's max is updated.
        rest_thr, burst_thr = self.classifier.thresholds()
        records = []
        fish_views = []
        for i, s in zip(ids, spd, strict=True):
            self.classifier.learn(float(s), t)
            r = self.registry.residents.get(i)
            name = r.name if r else f"#{i}"
            if r is not None:
                if s > burst_thr and s > r.max_speed_bls and r.visible_s > 60 and r.max_speed_bls > 0:
                    records.append((i, name, float(s)))
                r.max_speed_bls = max(r.max_speed_bls, float(s))
                r.speed.update(float(s), t)
                r.height.update(self.norm_pos[i][1], t)
                if self.norm_pos[i][2] is not None:
                    r.depth.update(self.norm_pos[i][2], t)
            tr = self.tracks[i]
            fish_views.append(FishView(i, name, np.asarray(tr.position), np.array(
                [tr.kin.vx, tr.kin.vy, tr.kin.vz]), float(s), self.states.get(i, "cruise"),
                self.norm_pos[i][1]))

        visible = set(ids)
        missing = [
            (r.rid, r.name, t - r.last_seen) for r in self.registry.residents.values()
            if r.rid not in visible and r.visible_s > 60 and 0 < t - r.last_seen < 600 and n > 0
        ]
        new_res = [(rid, self.registry.residents[rid].name) for rid in self.registry.new_residents
                   if rid in self.registry.residents]
        self.registry.new_residents.clear()
        reid = [(rid, self.registry.residents[rid].name, gap)
                for rid, gap in self.registry.reidentified if rid in self.registry.residents]
        self.registry.reidentified.clear()

        ctx = EventContext(
            t=t, fish=fish_views, metrics=metrics, zscores=zscores,
            baseline_ready=self.baseline.stats_for("activity", t) is not None,
            rest_threshold=rest_thr, burst_threshold=burst_thr, pairs=g.pairs,
            missing=missing, new_residents=new_res, reidentified=reid, records=records,
            warmed_up=(t - self.started_at) > 30,
        )
        new_events = self.events.update(ctx)
        self.recent_events.extend(new_events)
        self._pending_events.extend(new_events)
        mood_label, mood_color = mood(metrics, zscores, self.events.feeding)

        fish_out = []
        for i, s in zip(ids, spd, strict=True):
            tr = self.tracks[i]
            xn, yn, zn = self.norm_pos[i]
            r = self.registry.residents.get(i)
            fish_out.append({
                "id": i, "name": r.name if r else f"#{i}",
                "x": round(xn, 4), "y": round(yn, 4), "z": None if zn is None else round(zn, 4),
                "speed_bls": round(float(s), 3), "speed": round(tr.kin.speed, 4),
                "heading": round(float(np.degrees(np.arctan2(tr.kin.vy, tr.kin.vx))), 1),
                "tortuosity": round(tr.kin.tortuosity, 2),
                "state": self.states.get(i, "cruise"),
                "state_fr": STATE_LABELS_FR.get(self.states.get(i, "cruise"), ""),
                "state_for_s": round(self.classifier.time_in_state(i, t), 1),
                "zone": self.zones.get(i, {}),
            })

        coverage = self.baseline.coverage("activity")
        snap = {
            "t": t,
            "iso": datetime.fromtimestamp(t).isoformat(timespec="seconds"),
            "fish_count": n,
            "metrics": {k: (None if not np.isfinite(v) else round(float(v), 4))
                        for k, v in metrics.items()},
            "zscores": {k: round(float(v), 2) for k, v in zscores.items()},
            "states": {s: states.count(s) for s in STATES},
            "thresholds": {"rest_bls": round(rest_thr, 3), "burst_bls": round(burst_thr, 3)},
            "mood": {"label": mood_label, "color": mood_color},
            "feeding": self.events.feeding,
            "fish": fish_out,
            "group": {"centroid": [round(c, 4) for c in g.centroid]},
            "events": [e.to_dict() for e in new_events],
            "recent_events": [e.to_dict() for e in list(self.recent_events)[-12:]][::-1],
            "occupancy": {
                "vertical": dict(zip(VERTICAL, [round(v, 3) for v in self.occupancy.vertical_profile()], strict=True)),
                "depth": [round(v, 3) for v in self.occupancy.depth_profile()],
                "hotspot": self.occupancy.hotspot(),
            },
            "learning": {
                "hours_learnt": coverage,
                "samples": self.samples,
                "residents": len(self.registry.residents),
                "running_min": round((t - self.started_at) / 60.0, 1),
            },
            "insights": self._insights(metrics, zscores, n, t),
            **extra,
        }
        self.timeline.append({
            "t": round(t, 2), "fish_count": n,
            "activity": snap["metrics"]["activity"], "cohesion": snap["metrics"]["cohesion"],
            "polarization": snap["metrics"]["polarization"], "mean_z": snap["metrics"]["mean_z"],
            "mean_y": snap["metrics"]["mean_y"], "activity_z": snap["zscores"].get("activity"),
        })
        self.latest = snap
        return snap

    def live_fish(self, t: float) -> list[dict[str, Any]]:
        """Lightweight per-frame view of visible fish (for real-time consumers like music)."""
        out = []
        for i in self.visible_ids(t, max_age=0.3):
            xn, yn, zn = self.norm_pos[i]
            out.append({"id": i, "x": xn, "y": yn, "z": zn,
                        "speed_bls": self.tracks[i].kin.speed_bls,
                        "state": self.states.get(i, "cruise")})
        return out

    def drain_events(self) -> list[AquariumEvent]:
        out, self._pending_events = self._pending_events, []
        return out

    # ------------------------------------------------------------------ narration
    def _insights(self, m: dict[str, float], z: dict[str, float], n: int, t: float) -> list[str]:
        if n == 0:
            return ["Personne à l'écran… tout le monde se cache ?"]
        out = []
        hour = datetime.fromtimestamp(t).hour
        if "activity" in z and self.baseline.stats_for("activity", t) is not None:
            out.append(f"Activité {describe_z(z['activity'])} pour {hour}h ({z['activity']:+.1f}σ)")
        else:
            out.append("J'apprends encore le rythme de cet aquarium…")
        vert = self.occupancy.vertical_profile()
        if sum(vert) > 0:
            idx = int(np.argmax(vert))
            out.append(f"Zone préférée la dernière heure : {VERTICAL[idx]} ({vert[idx] * 100:.0f} %)")
        if np.isfinite(m.get("mean_z", float("nan"))):
            zmean = m["mean_z"]
            where = "près de la vitre" if zmean < 0.33 else ("au fond" if zmean > 0.66 else "à mi-profondeur")
            out.append(f"Les poissons nagent {where} (profondeur moy. {zmean:.2f})")
        if n >= 3 and m.get("polarization", 0) > 0.7:
            out.append(f"Banc polarisé : nage coordonnée ({m['polarization']:.2f})")
        hours = self.baseline.coverage("activity")
        out.append(f"Modèle appris : {hours}/24 h de la journée")
        return out

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "baseline": self.baseline.to_dict(),
            "classifier": self.classifier.to_dict(),
            "occupancy": self.occupancy.to_dict(),
            "registry": self.registry.to_dict(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if "baseline" in d:
            self.baseline = CircadianBaseline.from_dict(d["baseline"])
        if "classifier" in d:
            self.classifier.load_dict(d["classifier"])
        if "occupancy" in d:
            self.occupancy.load_dict(d["occupancy"])
        if "registry" in d:
            self.registry.load_dict(d["registry"])

    def residents_summary(self) -> list[dict[str, Any]]:
        return sorted((r.summary() for r in self.registry.residents.values()),
                      key=lambda s: -s["visible_min"])

    def hourly_profile(self, metric: str = "activity") -> list[dict[str, float]]:
        return self.baseline.hourly_profile(metric)
