"""Command line entry point: ``ichtyodisco --help``."""

from __future__ import annotations

import argparse
import logging

from aquarium.config import AppConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ichtyodisco",
        description="Aquarium live: fish tracking, monocular depth, behaviour analytics, "
                    "web dashboard, RTMP livestream and OSC sound.",
    )
    p.add_argument("-c", "--config", help="TOML config file (see config/example.toml)")
    p.add_argument("-s", "--source", help="webcam index (0), video file / rtsp URL, or 'sim'")
    p.add_argument("--demo", action="store_true", help="synthetic aquarium, no camera or GPU needed")
    p.add_argument("--model", help="YOLO weights for detection (e.g. models/fish.pt)")
    p.add_argument("--depth", help="depth model: depth-pro | depth-anything-v2-small | ... | off")
    p.add_argument("--device", help="cuda:0 / mps / cpu for tracking and depth")
    p.add_argument("--stream", metavar="URL", help="rtmp://... (YouTube/Twitch) or output .mp4/.flv")
    p.add_argument("--port", type=int, help="dashboard port (default 8765)")
    p.add_argument("--host", help="dashboard bind address (default 127.0.0.1)")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--osc", metavar="HOST:PORT", help="send OSC to a synth, e.g. 127.0.0.1:57120")
    p.add_argument("--db", help="SQLite path (default outputs/ichtyodisco.sqlite)")
    p.add_argument("--preview", action="store_true", help="show an OpenCV window")
    p.add_argument("--frames", type=int, help="stop after N frames (benchmarks, tests)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def config_from_args(a: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load(a.config)
    if a.demo:
        cfg.source.uri = "sim"
        cfg.source.capture_width, cfg.source.capture_height = 1280, 720
    if a.source:
        cfg.source.uri = a.source
    if a.model:
        cfg.tracking.model = a.model
    if a.depth:
        if a.depth.lower() in ("off", "none", "0", "false"):
            cfg.depth.enabled = False
        else:
            cfg.depth.enabled, cfg.depth.model = True, a.depth
    if a.device:
        cfg.tracking.device = cfg.depth.device = a.device
    if a.stream:
        cfg.stream.enabled, cfg.stream.url = True, a.stream
    if a.port:
        cfg.dashboard.port = a.port
    if a.host:
        cfg.dashboard.host = a.host
    if a.no_dashboard:
        cfg.dashboard.enabled = False
    if a.osc:
        host, _, port = a.osc.rpartition(":")
        cfg.osc.enabled, cfg.osc.host, cfg.osc.port = True, host or "127.0.0.1", int(port)
    if a.db:
        cfg.storage.path = a.db
    if a.preview:
        cfg.render.preview_window = True
    return cfg


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("ichtyodisco")
    cfg = config_from_args(a)

    from aquarium.pipeline import Pipeline, build_depth, build_source, build_tracker

    source, sim_tracker, sim_depth = build_source(cfg)
    tracker = sim_tracker or build_tracker(cfg)
    depth = None
    if cfg.depth.enabled:
        if sim_depth is not None:
            depth = sim_depth
        else:
            try:
                depth = build_depth(cfg)
                log.info("depth model: %s on %s", cfg.depth.model, getattr(depth, "device", "?"))
            except ImportError:
                log.warning("depth disabled: install the extra with `uv sync --extra depth`")
    Pipeline(cfg, source, tracker, depth, max_frames=a.frames).run()


if __name__ == "__main__":
    main()
