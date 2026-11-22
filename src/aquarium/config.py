"""Pipeline configuration: dataclasses with defaults, optionally overridden by a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class SourceConfig:
    # "0" / "1" = webcam index, a path / rtsp:// URL = video, "sim" = synthetic aquarium
    uri: str = "0"
    capture_width: int = 1920
    capture_height: int = 1080
    target_fps: float = 30.0
    loop: bool = True  # loop video files (handy for testing a stream overnight)


@dataclass
class TrackingConfig:
    model: str = "yolo11n.pt"  # fine-tuned fish weights recommended (models/fish.pt)
    tracker: str = "bytetrack.yaml"  # or "botsort.yaml" (re-id, slower)
    confidence: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    classes: list[int] | None = None  # None = all classes of the model
    device: str | None = None  # "cuda:0", "mps", "cpu" or None (auto)


@dataclass
class DepthConfig:
    enabled: bool = True
    # "depth-pro-2" (S) / "depth-pro-2-b" / "-l" / "-h": Apple Depth Pro 2 in live mode, steady
    #     depth, research licence (install: see README)
    # "depth-anything-v2-small" (Apache-2.0) / "-base" / "-large", "depth-pro" (v1), or any HF
    #     depth-estimation checkpoint (`uv sync --extra depth`)
    model: str = "depth-anything-v2-small"
    device: str | None = None
    max_rate_hz: float = 4.0  # depth is slow: run it asynchronously at a capped rate
    input_width: int = 518  # frame is downscaled before inference
    window: int = 4  # Depth Pro 2 only: past frames attended to (steadier but slower when larger)
    # Robust normalisation of the tank volume (percentiles of the depth map in the tank ROI)
    near_percentile: float = 3.0
    far_percentile: float = 97.0
    calibration_smoothing: float = 0.05  # EWMA factor for the near/far planes
    per_fish_smoothing: float = 0.35  # EWMA factor for each fish z (depth is noisy)


@dataclass
class AnalyticsConfig:
    history_seconds: float = 20.0
    snapshot_rate_hz: float = 2.0
    baseline_half_life_hours: float = 72.0  # how fast the learnt "normal" forgets the past
    chase_distance: float = 0.08  # normalised 3D distance
    chase_min_seconds: float = 1.5
    hiding_seconds: float = 45.0
    event_cooldown_s: float = 60.0
    heatmap_size: tuple[int, int] = (48, 27)
    heatmap_half_life_s: float = 3600.0
    reid: bool = True
    reid_max_distance: float = 0.25
    tank_depth_ratio: float = 0.4  # tank depth / tank width (e.g. 40 cm / 100 cm)


@dataclass
class StorageConfig:
    enabled: bool = True
    path: str = "outputs/ichtyodisco.sqlite"
    write_every_s: float = 1.0
    baseline_save_every_s: float = 60.0


@dataclass
class DashboardConfig:
    enabled: bool = True
    # 127.0.0.1 = only this machine. Use "0.0.0.0" to open it on the LAN (no auth!).
    host: str = "127.0.0.1"
    port: int = 8765
    jpeg_quality: int = 75
    max_fps: float = 15.0


@dataclass
class StreamConfig:
    enabled: bool = False
    # rtmp://a.rtmp.youtube.com/live2/<key>, rtmp://live.twitch.tv/app/<key>, or a file .mp4/.flv
    url: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate: str = "4500k"
    preset: str = "veryfast"
    encoder: str = "libx264"  # "h264_nvenc" / "h264_videotoolbox" when available
    silent_audio: bool = True  # YouTube refuses streams without an audio track
    # Put your synths in the stream: an ffmpeg audio input "<format>:<device>", e.g.
    #   macOS "avfoundation::0" · Linux "pulse:default" / "alsa:hw:1" · Windows "dshow:audio=Line 1"
    # (list devices: ffmpeg -f avfoundation -list_devices true -i "" / -f dshow -list_devices true -i dummy)
    audio_input: str | None = None


@dataclass
class OscConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 57120  # SuperCollider default
    rate_hz: float = 20.0


@dataclass
class MusicConfig:
    """Generative music sent as MIDI to your synths (see aquarium/music/conductor.py)."""

    enabled: bool = False
    # "generative": the conductor improvises from the tank's state.
    # "score": adaptive game-style score (sections / layers / stingers) read from `score`.
    mode: str = "generative"
    score: str | None = None  # path to a score .toml (see scores/demo/score.toml)
    port: str | None = "ichtyodisco"  # virtual port name, or part of an existing port's name
    record_path: str | None = None  # also record the session to a .mid file
    key: str = "D"
    base_bpm: float = 84.0
    pad_channel: int = 1
    bass_channel: int = 2
    lead_channel: int = 3  # event phrases
    fish_channels: list[int] = field(default_factory=lambda: [4, 5, 6, 7, 8, 9])
    drums_channel: int = 10
    drums: bool = True
    max_voices: int = 6
    fish_range: tuple[int, int] = (55, 88)  # MIDI notes: bottom of the tank -> surface
    send_clock: bool = False  # MIDI clock so arpeggiators / DAW follow the tank's tempo
    humanize: float = 0.08
    seed: int | None = None


@dataclass
class RenderConfig:
    trails_seconds: float = 4.0
    show_depth_inset: bool = True
    show_top_view: bool = True
    show_hud: bool = True
    title: str = "ICHTYODISCO"
    preview_window: bool = False  # cv2.imshow window (needs a display)


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    osc: OscConfig = field(default_factory=OscConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        cfg = cls()
        if path is not None:
            with open(path, "rb") as fh:
                _merge(cfg, tomllib.load(fh))
        return cfg


def _merge(target: Any, values: dict[str, Any]) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' in section {type(target).__name__}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(target, key, tuple(value))
        else:
            setattr(target, key, value)
