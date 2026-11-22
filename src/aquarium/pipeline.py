"""Glue: capture -> tracking -> depth -> analytics -> overlay -> dashboard / RTMP / OSC / SQLite."""

from __future__ import annotations

import logging
import signal
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from aquarium.analytics.engine import AnalyticsEngine
from aquarium.capture.video_source import FrameSource
from aquarium.config import AppConfig
from aquarium.depth.async_depth import AsyncDepthWorker
from aquarium.depth.base import DepthEstimator
from aquarium.depth.sampling import FishDepthSmoother, TankDepthCalibration, sample_bbox_depth
from aquarium.render.overlay import OverlayRenderer
from aquarium.tracking.fish_tracker import Tracker
from aquarium.types import TrackedFish

log = logging.getLogger(__name__)
STATE_KEY = "engine_state"


def build_source(cfg: AppConfig) -> tuple[FrameSource, Tracker | None, DepthEstimator | None]:
    """Returns the frame source and, for the simulator, its ground-truth tracker/depth."""
    uri = cfg.source.uri
    if uri == "sim":
        from aquarium.sim import SimDepthEstimator, SimTracker, SimulatedAquarium

        sim = SimulatedAquarium(size=(cfg.source.capture_width, cfg.source.capture_height),
                                fps=cfg.source.target_fps)
        return sim, SimTracker(sim), SimDepthEstimator(sim)
    if uri.isdigit():
        from aquarium.capture.webcam_capture import WebcamCapture

        cam = WebcamCapture(
            webcam_index=int(uri),
            capture_resolution=(cfg.source.capture_width, cfg.source.capture_height),
            target_fps=cfg.source.target_fps,
        )
        return cam, None, None
    from aquarium.capture.video_source import VideoFileCapture

    return VideoFileCapture(uri, loop=cfg.source.loop), None, None


def build_tracker(cfg: AppConfig) -> Tracker:
    from aquarium.tracking.fish_tracker import FishTracker

    return FishTracker(cfg.tracking)


def build_depth(cfg: AppConfig) -> DepthEstimator:
    if cfg.depth.model.startswith("depth-pro-2"):
        from aquarium.depth.depthpro2_estimator import DepthPro2Estimator

        return DepthPro2Estimator(cfg.depth.model, device=cfg.depth.device,
                                  input_width=cfg.depth.input_width, window=cfg.depth.window)
    from aquarium.depth.hf_estimator import HFDepthEstimator

    return HFDepthEstimator(cfg.depth.model, device=cfg.depth.device,
                            input_width=cfg.depth.input_width)


class Pipeline:
    def __init__(self, cfg: AppConfig, source: FrameSource, tracker: Tracker,
                 depth_estimator: DepthEstimator | None = None, max_frames: int | None = None):
        self.cfg = cfg
        self.source = source
        self.tracker = tracker
        self.max_frames = max_frames
        self.engine = AnalyticsEngine(cfg.analytics, tank_depth_ratio=cfg.analytics.tank_depth_ratio)
        self.renderer = OverlayRenderer(cfg.render, size=(cfg.stream.width, cfg.stream.height),
                                        depth_ratio=cfg.analytics.tank_depth_ratio)
        self.depth_worker: AsyncDepthWorker | None = None
        if depth_estimator is not None and cfg.depth.enabled:
            calib = TankDepthCalibration(cfg.depth.near_percentile, cfg.depth.far_percentile,
                                         cfg.depth.calibration_smoothing)
            self.depth_worker = AsyncDepthWorker(depth_estimator, calib, cfg.depth.max_rate_hz)
        self.depth_smoother = FishDepthSmoother(alpha=cfg.depth.per_fish_smoothing)
        self.store = None
        self.hub = None
        self.server = None
        self.streamer = None
        self.osc = None
        self.music = None
        self._stop = False
        self._loop_times: deque[float] = deque(maxlen=60)
        self._timings: dict[str, float] = {}
        self._last_state_save = time.monotonic()
        self.frames = 0
        self.last_output: np.ndarray | None = None

    # ------------------------------------------------------------------ lifecycle
    def _start_outputs(self) -> None:
        cfg = self.cfg
        if cfg.storage.enabled:
            from aquarium.storage.sqlite_store import SQLiteStore

            self.store = SQLiteStore(cfg.storage.path)
            state = self.store.load_state(STATE_KEY)
            if state:
                self.engine.load_state_dict(state)
                log.info("restored learnt model: %d residents, %d/24 h learnt",
                         len(self.engine.registry.residents), self.engine.baseline.coverage("activity"))
        if cfg.dashboard.enabled:
            from aquarium.stream.dashboard import DashboardServer, LiveHub

            self.hub = LiveHub(cfg.dashboard.jpeg_quality, cfg.dashboard.max_fps, store=self.store)
            self.hub.on_rename = self.engine.registry.rename
            self.server = DashboardServer(self.hub, cfg.dashboard.host, cfg.dashboard.port)
            self.server.start()
            log.info("dashboard: %s  (OBS widget: %s/overlay)", self.server.url, self.server.url)
        if cfg.stream.enabled:
            from aquarium.stream.rtmp import FFmpegStreamer

            self.streamer = FFmpegStreamer(cfg.stream)
            self.streamer.start()
            log.info("streaming to %s", cfg.stream.url.split("/live")[0] + "/…")
        if cfg.osc.enabled:
            from aquarium.sound.osc_bridge import OscBridge

            self.osc = OscBridge(cfg.osc.host, cfg.osc.port, cfg.osc.rate_hz)
            log.info("OSC -> %s:%d", cfg.osc.host, cfg.osc.port)
        if cfg.music.enabled:
            from aquarium.music import Conductor, open_sinks

            sink = open_sinks(cfg.music.port, cfg.music.record_path)
            if sink.sinks:
                if cfg.music.mode == "score":
                    from aquarium.music.score import Score, ScorePlayer

                    if not cfg.music.score:
                        raise ValueError("music.mode = 'score' needs music.score = <path to .toml>")
                    score = Score.load(cfg.music.score)
                    self.music = ScorePlayer(score, sink, humanize=cfg.music.humanize,
                                             seed=cfg.music.seed, send_clock=cfg.music.send_clock)
                    log.info("music: score '%s' (%d sections, %d stingers)", score.name,
                             len(score.sections), len(score.stingers))
                else:
                    self.music = Conductor(cfg.music, sink)
                    log.info("music: generative, key %s, base %.0f bpm", cfg.music.key,
                             cfg.music.base_bpm)
                self.music.start()

    def stop(self, *_: Any) -> None:
        self._stop = True

    def run(self) -> None:
        self._start_outputs()
        self.source.start()
        if self.depth_worker:
            self.depth_worker.start()
        try:
            signal.signal(signal.SIGINT, self.stop)
            signal.signal(signal.SIGTERM, self.stop)
        except ValueError:
            pass  # not in main thread (tests)
        try:
            while not self._stop:
                frame = self.source.wait_for_new_frame(timeout_s=5.0)
                if frame is None:
                    if getattr(self.source, "finished", False):
                        break
                    log.warning("no frame from source for 5 s")
                    continue
                self.step(frame)
                if self.max_frames is not None and self.frames >= self.max_frames:
                    break
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.music:
            self.music.stop()
        self.source.stop()
        if self.depth_worker:
            self.depth_worker.stop()
        if self.streamer:
            self.streamer.stop()
        if self.store:
            self.store.save_state(STATE_KEY, self.engine.state_dict())
            self.store.flush()
            self.store.close()
        if self.server:
            self.server.stop()
        if self.cfg.render.preview_window:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------ per frame
    def step(self, frame: np.ndarray, t: float | None = None) -> np.ndarray:
        t = time.time() if t is None else t
        t0 = time.perf_counter()
        fish = self.tracker.update(frame)
        t1 = time.perf_counter()

        depth = None
        if self.depth_worker:
            self.depth_worker.submit(frame)
            depth = self.depth_worker.latest()
        self._attach_depth(fish, depth, frame.shape[:2], t)

        extra = {"perf": self.perf(), "depth": self._depth_info(depth)}
        if self.music:
            extra["music"] = self.music.status()
        snap = self.engine.update(fish, frame, frame.shape[:2], t=t, extra=extra)
        inter = self.engine.drain_interactions()
        if self.music:
            self.music.set_state(self.engine.live_fish(t), self.engine.latest)
            if inter:
                self.music.trigger([e.to_dict() for e in inter])
        if self.osc and inter:
            self.osc.send_events(inter)
        t2 = time.perf_counter()

        latest = self.engine.latest or None
        out = self.renderer.render(frame, fish, latest, depth,
                                   stats={"fps": self.perf().get("fps"),
                                          "depth_model": depth.model if depth else None,
                                          "music": self.music.status() if self.music else None})
        t3 = time.perf_counter()
        self.last_output = out

        if snap is not None:
            self._on_snapshot(snap)
        if self.hub:
            self.hub.push_frame(out)
        if self.streamer:
            self.streamer.push(out)
        if self.osc:
            self.osc.send_frame(fish, frame.shape[:2], latest)
        if self.cfg.render.preview_window:
            cv2.imshow("ichtyodisco", out)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                self.stop()

        self._timings = {"track_ms": (t1 - t0) * 1e3, "analytics_ms": (t2 - t1) * 1e3,
                         "render_ms": (t3 - t2) * 1e3}
        self._loop_times.append(time.monotonic())
        self.frames += 1
        return out

    def _attach_depth(self, fish: list[TrackedFish], depth: Any, shape: tuple[int, int],
                      t: float) -> None:
        if depth is None:
            return
        if t - depth.timestamp > 5.0:
            return  # stale map (model stalled): better no depth than wrong depth
        for f in fish:
            raw = sample_bbox_depth(depth.data, f.bbox, shape)
            # smooth by tracker id (resident ids are assigned later in the engine)
            f.z = self.depth_smoother.update(f.track_id, raw, t)

    def _depth_info(self, depth: Any) -> dict[str, Any]:
        if depth is None:
            return {"available": False}
        return {"available": True, "model": depth.model, "latency_ms": round(depth.latency_s * 1e3, 1),
                "age_s": round(time.time() - depth.timestamp, 2)}

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        events = self.engine.drain_events()
        for e in events:
            log.info("[event] %s", e.message)
        if self.store:
            self.store.write_snapshot(snap)
            self.store.write_events([e.to_dict() for e in events])
            if time.monotonic() - self._last_state_save > self.cfg.storage.baseline_save_every_s:
                self.engine.registry.prune(snap["t"])
                self.store.save_state(STATE_KEY, self.engine.state_dict())
                self._last_state_save = time.monotonic()
        if self.hub:
            self.hub.publish(
                snap,
                residents=self.engine.residents_summary(),
                profiles={m: self.engine.hourly_profile(m) for m in ("activity", "cohesion", "mean_z")},
                heatmap_front=self.engine.occupancy.front.copy(),
                heatmap_top=self.engine.occupancy.top.copy(),
                timeline=list(self.engine.timeline),
            )
        if self.osc:
            self.osc.send_snapshot(snap)
            self.osc.send_events(events)
        if self.music and events:
            self.music.trigger([e.to_dict() for e in events])

    def perf(self) -> dict[str, Any]:
        lt = self._loop_times
        fps = (len(lt) - 1) / (lt[-1] - lt[0]) if len(lt) >= 2 and lt[-1] > lt[0] else 0.0
        return {"fps": round(fps, 1), "source_fps": round(self.source.fps_actual, 1),
                **{k: round(v, 1) for k, v in self._timings.items()}}
