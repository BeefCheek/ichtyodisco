"""Zero-dependency live web dashboard (stdlib HTTP server, MJPEG video, Server-Sent Events).

Endpoints
    /                    full dashboard
    /overlay             transparent stats widget for an OBS "Browser source"
    /video.mjpg          live annotated video (MJPEG)
    /frame.jpg           latest annotated frame
    /api/live            Server-Sent Events, one analytics snapshot per message
    /api/snapshot        latest snapshot (JSON)
    /api/history         ?minutes=60  aggregated time series (SQLite, or in-memory fallback)
    /api/events          ?limit=50
    /api/residents       per-fish long-term profiles
    /api/profile         ?metric=activity  learnt 24 h circadian profile
    /api/daily           ?days=7  daily summaries
    /api/heatmap.png     ?view=front|top  occupancy heatmap
    POST /api/residents/<id>/name   {"name": "Nemo"}  rename a fish (chat can name them!)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from aquarium.storage.sqlite_store import SQLiteStore

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class LiveHub:
    """Thread-safe exchange point between the pipeline (producer) and HTTP clients."""

    def __init__(self, jpeg_quality: int = 75, max_fps: float = 15.0,
                 store: SQLiteStore | None = None):
        self.jpeg_quality = jpeg_quality
        self.min_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.store = store
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._last_encode = 0.0
        self._snapshot: dict[str, Any] | None = None
        self._snap_seq = 0
        self._extras: dict[str, Any] = {}
        self.video_clients = 0
        self.on_rename: Callable[[int, str], bool] | None = None

    # -- producer side ---------------------------------------------------------
    def push_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_encode < self.min_period:
            return
        if self.video_clients <= 0 and self._jpeg is not None and now - self._last_encode < 1.0:
            return  # nobody watching: refresh the still image once a second only
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        self._last_encode = now
        with self._cond:
            self._jpeg = buf.tobytes()
            self._jpeg_seq += 1
            self._cond.notify_all()

    def publish(self, snapshot: dict[str, Any], **extras: Any) -> None:
        with self._cond:
            self._snapshot = snapshot
            self._snap_seq += 1
            self._extras.update(extras)
            self._cond.notify_all()

    # -- consumer side ---------------------------------------------------------
    def wait_jpeg(self, last_seq: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._jpeg_seq > last_seq, timeout=timeout)
            return self._jpeg_seq, self._jpeg

    def wait_snapshot(self, last_seq: int, timeout: float = 15.0) -> tuple[int, dict | None]:
        with self._cond:
            got = self._cond.wait_for(lambda: self._snap_seq > last_seq, timeout=timeout)
            return self._snap_seq, (self._snapshot if got else None)

    def latest_snapshot(self) -> dict[str, Any] | None:
        with self._cond:
            return self._snapshot

    def latest_jpeg(self) -> bytes | None:
        with self._cond:
            return self._jpeg

    def extra(self, key: str, default: Any = None) -> Any:
        with self._cond:
            return self._extras.get(key, default)


def _sequential_lut() -> np.ndarray:
    """Single-hue blue ramp (BGR) from the dark chart surface to light blue: rare -> frequent."""
    stops = [(0.0, (0x1A, 0x1A, 0x19)), (0.35, (0x18, 0x4F, 0x95)), (0.7, (0x39, 0x87, 0xE5)),
             (1.0, (0xCD, 0xE2, 0xFB))]
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        v = i / 255
        for (a, ca), (b, cb) in zip(stops, stops[1:], strict=False):
            if a <= v <= b:
                f = (v - a) / (b - a)
                rgb = [ca[k] + (cb[k] - ca[k]) * f for k in range(3)]
                lut[i] = (rgb[2], rgb[1], rgb[0])
                break
    return lut


_LUT = _sequential_lut()


def heatmap_png(grid: np.ndarray, width: int = 480) -> bytes:
    g = np.asarray(grid, dtype=np.float64)
    if g.size == 0 or g.max() <= 0:
        g = np.zeros((9, 16))
    norm = np.sqrt(g / g.max()) if g.max() > 0 else g  # sqrt: show rarely-visited areas too
    img = _LUT[(norm * 255).astype(np.uint8)]
    h = max(1, int(width * g.shape[0] / g.shape[1]))
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def _make_handler(hub: LiveHub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ichtyodisco"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr logging
            log.debug("http: " + fmt, *args)

        # ---------------------------------------------------------- helpers
        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = 200) -> None:
            self._send(status, json.dumps(obj, default=_json_default).encode(), "application/json")

        # ---------------------------------------------------------- routes
        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            path = url.path.rstrip("/") or "/"
            try:
                if path == "/":
                    self._static("dashboard.html")
                elif path == "/overlay":
                    self._static("overlay.html")
                elif path == "/video.mjpg":
                    self._mjpeg()
                elif path == "/frame.jpg":
                    jpg = hub.latest_jpeg()
                    if jpg is None:
                        self._send(HTTPStatus.SERVICE_UNAVAILABLE, b"no frame yet", "text/plain")
                    else:
                        self._send(200, jpg, "image/jpeg")
                elif path == "/api/live":
                    self._sse()
                elif path == "/api/snapshot":
                    self._json(hub.latest_snapshot() or {})
                elif path == "/api/history":
                    self._history(float(q.get("minutes", 30)))
                elif path == "/api/events":
                    limit = int(q.get("limit", 50))
                    if hub.store is not None:
                        self._json(hub.store.events(limit=limit))
                    else:
                        snap = hub.latest_snapshot() or {}
                        self._json(snap.get("recent_events", [])[:limit])
                elif path == "/api/residents":
                    self._json(hub.extra("residents", []))
                elif path == "/api/profile":
                    profiles = hub.extra("profiles", {})
                    self._json(profiles.get(q.get("metric", "activity"), []))
                elif path == "/api/daily":
                    days = int(q.get("days", 7))
                    self._json(hub.store.daily_summary(days) if hub.store else [])
                elif path == "/api/heatmap.png":
                    view = q.get("view", "front")
                    grid = hub.extra(f"heatmap_{view}")
                    self._send(200, heatmap_png(grid if grid is not None else np.zeros((9, 16))),
                               "image/png")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:  # noqa: N802
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "residents"] and parts[3] == "name":
                try:
                    rid = int(parts[2])
                    length = min(int(self.headers.get("Content-Length", 0)), 4096)
                    body = json.loads(self.rfile.read(length) or b"{}")
                    name = str(body.get("name", ""))
                except (ValueError, json.JSONDecodeError):
                    self._json({"error": "bad request"}, 400)
                    return
                ok = bool(hub.on_rename and hub.on_rename(rid, name))
                self._json({"ok": ok}, 200 if ok else 404)
            else:
                self._send(404, b"not found", "text/plain")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _static(self, name: str) -> None:
            self._send(200, (STATIC / name).read_bytes(), "text/html; charset=utf-8")

        def _history(self, minutes: float) -> None:
            if hub.store is not None:
                self._json(hub.store.history(minutes))
            else:
                since = time.time() - minutes * 60
                self._json([p for p in hub.extra("timeline", []) if p["t"] >= since])

        def _mjpeg(self) -> None:
            boundary = "frame"
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            hub.video_clients += 1
            seq = 0
            try:
                while True:
                    seq, jpg = hub.wait_jpeg(seq)
                    if jpg is None:
                        continue
                    self.wfile.write(
                        f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError):
                pass
            finally:
                hub.video_clients -= 1
                self.close_connection = True

        def _sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            seq = 0
            snap = hub.latest_snapshot()
            try:
                if snap is not None:
                    self.wfile.write(f"data: {json.dumps(snap, default=_json_default)}\n\n".encode())
                    self.wfile.flush()
                while True:
                    seq, snap = hub.wait_snapshot(seq)
                    if snap is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        self.wfile.write(f"data: {json.dumps(snap, default=_json_default)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError):
                pass
            finally:
                self.close_connection = True

    return Handler


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not serialisable: {type(o)}")


class DashboardServer:
    def __init__(self, hub: LiveHub, host: str = "127.0.0.1", port: int = 8765):
        self.hub = hub
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(hub))
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
        return f"http://{shown}:{port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                        name="dashboard-http")
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
