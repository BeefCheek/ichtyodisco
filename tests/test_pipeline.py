import shutil

import cv2
import pytest

from aquarium.cli import config_from_args, parse_args
from aquarium.pipeline import Pipeline, build_source


def test_cli_parsing():
    cfg = config_from_args(parse_args(["--demo", "--depth", "off", "--osc", "10.0.0.2:9000",
                                       "--stream", "rtmp://x/live/k"]))
    assert cfg.source.uri == "sim" and not cfg.depth.enabled
    assert (cfg.osc.host, cfg.osc.port) == ("10.0.0.2", 9000)
    assert cfg.stream.enabled


def test_full_pipeline_on_simulator(tmp_path):
    cfg = config_from_args(parse_args(["--demo", "--db", str(tmp_path / "db.sqlite"),
                                       "--no-dashboard"]))
    cfg.source.capture_width, cfg.source.capture_height = 640, 360
    cfg.stream.width, cfg.stream.height = 640, 360
    # Snapshots are paced on wall-clock time; on a fast machine 90 frames take < 0.5 s,
    # leaving only the frame-0 snapshot (before any resident exists). Snapshot every frame.
    cfg.analytics.snapshot_rate_hz = 1e6
    source, tracker, depth = build_source(cfg)
    source.realtime = False  # as fast as possible, but never skip a frame
    p = Pipeline(cfg, source, tracker, depth, max_frames=90)
    p.run()
    assert p.frames == 90
    assert p.last_output is not None and p.last_output.shape == (360, 640, 3)
    assert p.engine.latest["fish_count"] > 0
    # the learnt model was persisted and can be restored
    from aquarium.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(tmp_path / "db.sqlite")
    assert store.load_state("engine_state")["version"] == 1
    store.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_stream_to_file(tmp_path):
    out = tmp_path / "out.mp4"
    cfg = config_from_args(parse_args(["--demo", "--no-dashboard", "--stream", str(out),
                                       "--depth", "off"]))
    cfg.storage.enabled = False
    cfg.source.capture_width, cfg.source.capture_height = 640, 360
    cfg.stream.width, cfg.stream.height, cfg.stream.fps = 640, 360, 15
    source, tracker, _ = build_source(cfg)
    Pipeline(cfg, source, tracker, None, max_frames=60).run()
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
