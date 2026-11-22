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
    p.add_argument("--depth", help="depth-pro-2[-b|-l|-h] | depth-anything-v2-small | depth-pro | off")
    p.add_argument("--device", help="cuda:0 / mps / cpu for tracking and depth")
    p.add_argument("--stream", metavar="URL", help="rtmp://... (YouTube/Twitch) or output .mp4/.flv")
    p.add_argument("--audio-in", metavar="FMT:DEVICE",
                   help="audio of your synths into the stream, e.g. avfoundation::0, pulse:default, "
                        "'dshow:audio=Line 1'")
    p.add_argument("--port", type=int, help="dashboard port (default 8765)")
    p.add_argument("--host", help="dashboard bind address (default 127.0.0.1)")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--osc", metavar="HOST:PORT", help="send OSC to a synth, e.g. 127.0.0.1:57120")
    p.add_argument("--midi", nargs="?", const="ichtyodisco", metavar="PORT",
                   help="play the tank on your synths: virtual port 'ichtyodisco' by default, or "
                        "(part of) an existing port name, e.g. 'IAC', 'loopMIDI', 'Minilogue'")
    p.add_argument("--midi-record", metavar="FILE.mid", help="also record the music to a MIDI file")
    p.add_argument("--midi-list", action="store_true", help="list MIDI output ports and exit")
    p.add_argument("--key", help="musical key, e.g. D, F#, Bb (default D)")
    p.add_argument("--bpm", type=float, help="base tempo; the tank's activity pushes it +/-")
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
    if a.audio_in:
        cfg.stream.audio_input = a.audio_in
    if a.port:
        cfg.dashboard.port = a.port
    if a.host:
        cfg.dashboard.host = a.host
    if a.no_dashboard:
        cfg.dashboard.enabled = False
    if a.osc:
        host, _, port = a.osc.rpartition(":")
        cfg.osc.enabled, cfg.osc.host, cfg.osc.port = True, host or "127.0.0.1", int(port)
    if a.midi or a.midi_record:
        cfg.music.enabled = True
        cfg.music.port = a.midi
        cfg.music.record_path = a.midi_record or cfg.music.record_path
    if a.key:
        cfg.music.key = a.key
    if a.bpm:
        cfg.music.base_bpm = a.bpm
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
    if a.midi_list:
        from aquarium.music.midi_out import list_ports

        ports = list_ports()
        print("\n".join(ports) if ports else "no MIDI output (install: uv sync --extra midi)")
        return
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
            except ImportError as exc:
                hint = ("Depth Pro 2: uv pip install git+https://github.com/apple-aiml-research/ml-depth-pro-2"
                        if cfg.depth.model.startswith("depth-pro-2") else "uv sync --extra depth")
                log.warning("depth disabled (%s). Install: %s", exc, hint)
    Pipeline(cfg, source, tracker, depth, max_frames=a.frames).run()


if __name__ == "__main__":
    main()
