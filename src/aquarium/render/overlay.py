"""Broadcast overlay: boxes, depth-coloured trails, top-view radar, depth inset and a HUD.

Performance: per-fish graphics are drawn with OpenCV every frame; the text-heavy HUD
(French accents need a real font, hence Pillow) is re-rendered only when a new analytics
snapshot arrives (~2 Hz) and alpha-blended on the panel rectangles in between.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aquarium.config import RenderConfig
from aquarium.types import DepthMap, TrackedFish

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    str(Path.home() / ".config/Ultralytics/Arial.ttf"),
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

MOOD_COLORS = {  # RGB for Pillow
    "grey": (150, 160, 170), "gold": (255, 200, 60), "red": (255, 90, 90),
    "orange": (255, 150, 60), "blue": (110, 160, 255), "green": (110, 220, 130),
    "teal": (80, 220, 210),
}
# Same categorical slots as the web dashboard (RGB): rest, cruise, explore, burst
STATE_COLORS = {"rest": (57, 135, 229), "cruise": (217, 89, 38), "explore": (25, 158, 112),
                "burst": (201, 133, 0)}
STATE_FR = {"rest": "repos", "cruise": "croisière", "explore": "explo", "burst": "sprint"}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in (BOLD_CANDIDATES if bold else []) + FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


_TURBO = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]


def depth_color(z: float | None) -> tuple[int, int, int]:
    """BGR colour for a depth: near the glass = warm, near the back wall = cold."""
    if z is None:
        return (235, 235, 235)
    idx = int(np.clip((1.0 - z) * 0.8 + 0.1, 0, 1) * 255)
    b, g, r = _TURBO[idx]
    return int(b), int(g), int(r)


class OverlayRenderer:
    def __init__(self, cfg: RenderConfig | None = None, size: tuple[int, int] = (1280, 720),
                 depth_ratio: float = 0.4):
        self.cfg = cfg or RenderConfig()
        self.w, self.h = size
        self.depth_ratio = depth_ratio
        s = self.h / 720.0
        self.s = s
        self.f_title = _font(int(26 * s), bold=True)
        self.f_big = _font(int(22 * s), bold=True)
        self.f_med = _font(int(16 * s))
        self.f_small = _font(int(13 * s))
        self.trails: dict[int, deque[tuple[float, int, int, float | None]]] = {}
        self._hud_bgr = np.zeros((self.h, self.w, 3), np.uint8)
        self._hud_a = np.zeros((self.h, self.w, 1), np.float32)
        self._rects: list[tuple[int, int, int, int]] = []
        self._hud_snapshot_t: float | None = None
        self._depth_inset: np.ndarray | None = None
        self._depth_t: float | None = None
        self._fps_hist: deque[float] = deque(maxlen=30)

    # ------------------------------------------------------------------ public
    def render(self, frame: np.ndarray, fish: list[TrackedFish], snapshot: dict[str, Any] | None,
               depth: DepthMap | None = None, stats: dict[str, Any] | None = None) -> np.ndarray:
        fh, fw = frame.shape[:2]
        out = frame if (fw, fh) == (self.w, self.h) else cv2.resize(
            frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        sx, sy = self.w / fw, self.h / fh
        now = time.monotonic()
        snap_fish = {f["id"]: f for f in (snapshot or {}).get("fish", [])}

        self._update_trails(fish, sx, sy, now)
        self._draw_trails(out, now)
        for f in fish:
            self._draw_fish(out, f, sx, sy, snap_fish.get(f.identity))

        if self.cfg.show_hud and snapshot is not None:
            if snapshot.get("t") != self._hud_snapshot_t:
                self._build_hud(snapshot, stats or {})
                self._hud_snapshot_t = snapshot.get("t")
            self._blend_hud(out)
        if self.cfg.show_top_view:
            self._draw_top_view(out, fish, fw, fh)
        if self.cfg.show_depth_inset and depth is not None:
            self._draw_depth_inset(out, depth)
        self._draw_clock(out)
        return out

    # ------------------------------------------------------------------ fish
    def _update_trails(self, fish: list[TrackedFish], sx: float, sy: float, now: float) -> None:
        horizon = self.cfg.trails_seconds
        for f in fish:
            cx, cy = f.center
            trail = self.trails.setdefault(f.identity, deque(maxlen=240))
            trail.append((now, int(cx * sx), int(cy * sy), f.z))
        for key in list(self.trails):
            trail = self.trails[key]
            while trail and now - trail[0][0] > horizon:
                trail.popleft()
            if not trail:
                del self.trails[key]

    def _draw_trails(self, out: np.ndarray, now: float) -> None:
        horizon = max(self.cfg.trails_seconds, 1e-3)
        for trail in self.trails.values():
            pts = list(trail)
            for (t0, x0, y0, _), (_, x1, y1, z1) in zip(pts, pts[1:], strict=False):
                age = (now - t0) / horizon
                thick = max(1, int((1 - age) * 4 * self.s * (1.3 - 0.6 * (z1 or 0.5))))
                col = depth_color(z1)
                fade = 1 - age
                col = tuple(int(c * fade) for c in col)
                cv2.line(out, (x0, y0), (x1, y1), col, thick, cv2.LINE_AA)

    def _draw_fish(self, out: np.ndarray, f: TrackedFish, sx: float, sy: float,
                   info: dict[str, Any] | None) -> None:
        x1, y1, x2, y2 = f.bbox
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        col = depth_color(f.z)
        L = max(6, int(min(p2[0] - p1[0], p2[1] - p1[1]) * 0.3))
        for (cx, cy), (dx, dy) in (((p1[0], p1[1]), (1, 1)), ((p2[0], p1[1]), (-1, 1)),
                                   ((p1[0], p2[1]), (1, -1)), ((p2[0], p2[1]), (-1, -1))):
            cv2.line(out, (cx, cy), (cx + dx * L, cy), col, 2, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * L), col, 2, cv2.LINE_AA)
        label = f"#{f.identity}" if f.resident_id is not None else "?"
        if info is not None:
            label += f" {info['speed_bls']:.1f}BL/s"  # depth is shown by colour + top view
        scale = 0.45 * self.s
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(th + 4, p1[1] - 6)
        cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 6, ty + 3), (20, 20, 20), -1)
        cv2.putText(out, label, (p1[0] + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1,
                    cv2.LINE_AA)
        if info is not None:
            state = info.get("state", "cruise")
            rgb = STATE_COLORS.get(state, (255, 255, 255))
            cv2.circle(out, (p2[0] - 5, p1[1] + 5), max(3, int(4 * self.s)), rgb[::-1], -1,
                       cv2.LINE_AA)

    # ------------------------------------------------------------------ insets
    def _panel(self, out: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
        roi = out[y:y + h, x:x + w]
        roi[:] = (roi * (1 - alpha) + np.array([25, 18, 10]) * alpha).astype(np.uint8)
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (120, 110, 90), 1, cv2.LINE_AA)

    def _draw_top_view(self, out: np.ndarray, fish: list[TrackedFish], fw: int, fh: int) -> None:
        w = int(250 * self.s)
        h = max(int(w * self.depth_ratio * 1.6), int(70 * self.s))
        m = int(14 * self.s)
        x0, y0 = self.w - w - m, m
        self._panel(out, x0, y0, w, h + int(18 * self.s))
        cv2.putText(out, "VUE DE DESSUS", (x0 + 8, y0 + int(14 * self.s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * self.s, (200, 200, 200), 1, cv2.LINE_AA)
        top = y0 + int(18 * self.s)
        cv2.putText(out, "fond", (x0 + w - int(36 * self.s), top + int(12 * self.s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * self.s, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(out, "vitre", (x0 + w - int(40 * self.s), top + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * self.s, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(out, (x0 + 4, top + h - 1), (x0 + w - 4, top + h - 1), (200, 180, 120), 2)
        for f in fish:
            if f.z is None:
                continue
            cx, _ = f.center
            px = x0 + 6 + int(cx / fw * (w - 12))
            py = top + 4 + int((1.0 - f.z) * (h - 10))  # front glass at the bottom
            col = depth_color(f.z)
            cv2.circle(out, (px, py), max(3, int(5 * self.s)), col, -1, cv2.LINE_AA)
            cv2.putText(out, str(f.identity), (px + 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * self.s, (230, 230, 230), 1, cv2.LINE_AA)

    def _draw_depth_inset(self, out: np.ndarray, depth: DepthMap) -> None:
        w = int(250 * self.s)
        h = int(w * self.h / self.w)
        m = int(14 * self.s)
        top_h = max(int(w * self.depth_ratio * 1.6), int(70 * self.s)) + int(18 * self.s)
        x0, y0 = self.w - w - m, m + top_h + int(8 * self.s)
        if self._depth_t != depth.timestamp or self._depth_inset is None:
            d = cv2.resize(depth.data, (w, h), interpolation=cv2.INTER_AREA)
            idx = np.clip((1.0 - d) * 0.8 + 0.1, 0, 1)
            self._depth_inset = cv2.applyColorMap((idx * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            self._depth_t = depth.timestamp
        if y0 + h + int(20 * self.s) > self.h:
            return
        out[y0:y0 + h, x0:x0 + w] = self._depth_inset
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), (120, 110, 90), 1)
        label = f"profondeur · {depth.model} · {1000 * depth.latency_s:.0f} ms"
        cv2.putText(out, label.replace("·", "-"), (x0 + 4, y0 + h + int(14 * self.s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * self.s, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_clock(self, out: np.ndarray) -> None:
        txt = datetime.now().strftime("%H:%M:%S")
        cv2.putText(out, txt, (int(18 * self.s), self.h - int(14 * self.s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * self.s, (220, 220, 220), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ HUD (Pillow, cached)
    def _build_hud(self, snap: dict[str, Any], stats: dict[str, Any]) -> None:
        img = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        s = self.s
        rects: list[tuple[int, int, int, int]] = []

        def panel(x: int, y: int, w: int, h: int) -> None:
            d.rounded_rectangle((x, y, x + w, y + h), radius=int(10 * s), fill=(10, 18, 28, 165),
                                outline=(90, 130, 160, 200), width=1)
            rects.append((x, y, w + 1, h + 1))

        # Title + mood ---------------------------------------------------------
        m = int(14 * s)
        pw, ph = int(360 * s), int(92 * s)
        panel(m, m, pw, ph)
        d.text((m + 14 * s, m + 8 * s), self.cfg.title, font=self.f_title, fill=(235, 245, 255, 255))
        mood = snap.get("mood", {})
        mc = MOOD_COLORS.get(mood.get("color", "teal"), (200, 200, 200))
        d.text((m + 14 * s, m + 44 * s), "humeur :", font=self.f_med, fill=(170, 190, 210, 255))
        d.text((m + 90 * s, m + 40 * s), str(mood.get("label", "?")).upper(), font=self.f_big,
               fill=(*mc, 255))
        n = snap.get("fish_count", 0)
        learn = snap.get("learning", {})
        d.text((m + 14 * s, m + 68 * s),
               f"{n} poisson{'s' if n > 1 else ''} à l'écran · {learn.get('residents', 0)} résidents"
               f" · modèle {learn.get('hours_learnt', 0)}/24 h",
               font=self.f_small, fill=(170, 190, 210, 255))

        # Metrics panel (bottom-right) -----------------------------------------
        metrics, zs = snap.get("metrics", {}), snap.get("zscores", {})
        pw, ph = int(330 * s), int(190 * s)
        x0, y0 = self.w - pw - m, self.h - ph - m
        panel(x0, y0, pw, ph)
        rows = [
            ("Activité", metrics.get("activity"), "BL/s", zs.get("activity")),
            ("Cohésion", metrics.get("cohesion"), "", zs.get("cohesion")),
            ("Polarisation", metrics.get("polarization"), "", zs.get("polarization")),
            ("Profondeur moy.", metrics.get("mean_z"), "", zs.get("mean_z")),
        ]
        y = y0 + 10 * s
        for name, val, unit, z in rows:
            d.text((x0 + 12 * s, y), name, font=self.f_med, fill=(190, 205, 220, 255))
            sval = "—" if val is None else f"{val:.2f} {unit}".strip()
            d.text((x0 + 150 * s, y), sval, font=self.f_med, fill=(240, 245, 250, 255))
            self._zgauge(d, x0 + 235 * s, y + 4 * s, 80 * s, 10 * s, z)
            y += 26 * s
        # state distribution bar
        states = snap.get("states", {})
        total = max(1, sum(states.values()))
        bx, bw, bh = x0 + 12 * s, pw - 24 * s, 14 * s
        y += 4 * s
        cx = bx
        for key in ("rest", "cruise", "explore", "burst"):
            frac = states.get(key, 0) / total
            if frac <= 0:
                continue
            seg = bw * frac
            d.rectangle((cx, y, cx + seg, y + bh), fill=(*STATE_COLORS[key], 230))
            cx += seg
        d.rectangle((bx, y, bx + bw, y + bh), outline=(90, 130, 160, 220))
        y += bh + 4 * s
        lx = bx
        for key in ("rest", "cruise", "explore", "burst"):
            d.ellipse((lx, y + 4 * s, lx + 8 * s, y + 12 * s), fill=(*STATE_COLORS[key], 255))
            label = f"{STATE_FR[key]} {states.get(key, 0)}"
            d.text((lx + 11 * s, y), label, font=self.f_small, fill=(190, 205, 220, 255))
            lx += d.textlength(label, font=self.f_small) + 24 * s
        y += 20 * s
        fps = stats.get("fps")
        perf = f"{fps:.0f} fps" if fps else ""
        if stats.get("depth_model"):
            perf += f" · profondeur {stats['depth_model']}"
        music = stats.get("music")
        if music and music.get("mode") == "score":
            nxt = f" → {music['next']}" if music.get("next") else ""
            perf = (f"♪ {music['section']}{nxt} · {len(music.get('layers', {}))} couches"
                    f" · {music['bpm']:.0f} bpm")
        elif music:
            perf = (f"♪ {music['key']} {music['mode']} · {music['chord']} ({music['roman']})"
                    f" · {music['bpm']:.0f} bpm")
        d.text((bx, y), perf, font=self.f_small, fill=(130, 150, 170, 255))

        # Events / insights ticker (bottom-left) -------------------------------
        events = snap.get("recent_events", [])[:3]
        insights = snap.get("insights", [])[:2]
        lines: list[tuple[str, tuple[int, int, int]]] = []
        for e in events:
            ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M")
            sev = e.get("severity", 0.3)
            col = (255, 200, 90) if sev > 0.6 else (150, 220, 255)
            lines.append((f"{ts}  {e['message']}", col))
        for text in insights:
            lines.append((f"› {text}", (190, 210, 225)))
        if lines:
            pw = int(min(self.w * 0.55, 620 * s))
            ph = int((16 + 22 * len(lines)) * s)
            x0, y0 = m, self.h - ph - int(36 * s)
            panel(x0, y0, pw, ph)
            y = y0 + 8 * s
            for text, col in lines:
                while d.textlength(text, font=self.f_med) > pw - 24 * s and len(text) > 4:
                    text = text[:-2]
                d.text((x0 + 12 * s, y), text, font=self.f_med, fill=(*col, 255))
                y += 22 * s

        arr = np.asarray(img)
        self._hud_bgr = arr[..., 2::-1].copy()
        self._hud_a = (arr[..., 3:4].astype(np.float32) / 255.0)
        self._rects = [(int(x), int(y), int(w), int(h)) for x, y, w, h in rects]

    def _zgauge(self, d: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float,
                z: float | None) -> None:
        """Horizontal gauge centred on the learnt normal; the band is ±1σ."""
        d.rectangle((x, y, x + w, y + h), fill=(30, 45, 60, 220))
        mid = x + w / 2
        d.rectangle((mid - w / 6, y, mid + w / 6, y + h), fill=(60, 90, 110, 230))
        d.line((mid, y - 2, mid, y + h + 2), fill=(160, 190, 210, 255))
        if z is None:
            return
        pos = mid + float(np.clip(z / 3.0, -1, 1)) * (w / 2)
        col = (255, 110, 90) if abs(z) > 2 else ((255, 200, 90) if abs(z) > 1 else (110, 230, 160))
        d.ellipse((pos - h / 2, y, pos + h / 2, y + h), fill=(*col, 255))

    def _blend_hud(self, out: np.ndarray) -> None:
        for x, y, w, h in self._rects:
            x2, y2 = min(self.w, x + w), min(self.h, y + h)
            x, y = max(0, x), max(0, y)
            if x2 <= x or y2 <= y:
                continue
            a = self._hud_a[y:y2, x:x2]
            roi = out[y:y2, x:x2]
            roi[:] = (roi * (1 - a) + self._hud_bgr[y:y2, x:x2] * a).astype(np.uint8)
