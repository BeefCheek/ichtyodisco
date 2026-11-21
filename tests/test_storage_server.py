import json
import time
import urllib.request

import numpy as np

from aquarium.config import StreamConfig
from aquarium.storage.sqlite_store import SQLiteStore
from aquarium.stream.dashboard import DashboardServer, LiveHub
from aquarium.stream.rtmp import build_ffmpeg_cmd


def _snap(t, n=2):
    return {
        "t": t, "fish_count": n,
        "metrics": {"activity": 1.5, "cohesion": 0.1, "nnd": 0.05, "polarization": 0.5,
                    "mean_y": 0.4, "mean_z": 0.3, "burst_ratio": 0.0, "rest_ratio": 0.5,
                    "surface_ratio": 0.0},
        "zscores": {"activity": 0.4},
        "mood": {"label": "serein"},
        "fish": [{"id": i, "x": 0.5, "y": 0.5, "z": 0.2, "speed_bls": 1.0, "state": "cruise"}
                 for i in range(n)],
    }


def test_sqlite_store_roundtrip(tmp_path):
    store = SQLiteStore(tmp_path / "db.sqlite")
    now = time.time()
    for i in range(20):
        store.write_snapshot(_snap(now - 60 + i))
    store.write_events([{"timestamp": now, "kind": "chase", "severity": 0.5, "message": "x",
                         "fish": [1, 2], "data": {"distance": 0.03}}])
    store.save_state("engine_state", {"a": 1})
    store.flush()
    assert store.load_state("engine_state") == {"a": 1}
    hist = store.history(minutes=5, bucket_s=1)
    assert len(hist) == 20 and abs(hist[0]["activity"] - 1.5) < 1e-9
    ev = store.events()
    assert ev[0]["kind"] == "chase" and ev[0]["fish"] == [1, 2]
    assert store.daily_summary()[0]["events"] == 1
    store.close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_dashboard_endpoints():
    hub = LiveHub(max_fps=100)
    names = {}
    hub.on_rename = lambda rid, name: names.setdefault(rid, name) is not None
    server = DashboardServer(hub, "127.0.0.1", 0)
    server.start()
    try:
        hub.push_frame(np.zeros((90, 160, 3), np.uint8))
        hub.publish(_snap(time.time()), residents=[{"id": 1}],
                    profiles={"activity": [{"hour": 0}]},
                    heatmap_front=np.random.rand(9, 16), timeline=[{"t": time.time()}])
        base = server.url
        status, ctype, body = _get(base + "/")
        assert status == 200 and b"ICHTYODISCO" in body
        assert _get(base + "/overlay")[0] == 200
        assert _get(base + "/frame.jpg")[1] == "image/jpeg"
        assert json.loads(_get(base + "/api/snapshot")[2])["fish_count"] == 2
        assert json.loads(_get(base + "/api/residents")[2]) == [{"id": 1}]
        assert json.loads(_get(base + "/api/profile?metric=activity")[2]) == [{"hour": 0}]
        assert len(json.loads(_get(base + "/api/history?minutes=5")[2])) == 1
        assert _get(base + "/api/heatmap.png?view=front")[2][:4] == b"\x89PNG"
        req = urllib.request.Request(base + "/api/residents/3/name", data=b'{"name": "Nemo"}',
                                     method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read())["ok"] is True
        assert names == {3: "Nemo"}
        with urllib.request.urlopen(base + "/api/live", timeout=5) as r:
            line = r.readline()
            assert line.startswith(b"data: ")
    finally:
        server.stop()


def test_ffmpeg_command_for_youtube():
    cmd = build_ffmpeg_cmd(StreamConfig(enabled=True, url="rtmp://a.rtmp.youtube.com/live2/KEY"))
    assert cmd[-1].endswith("/KEY") and cmd[-2] == "flv"
    assert "anullsrc=channel_layout=stereo:sample_rate=44100" in cmd
    assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "60"


def test_ffmpeg_command_with_synth_audio_input():
    cmd = build_ffmpeg_cmd(StreamConfig(enabled=True, url="rtmp://x/live/KEY",
                                        audio_input="avfoundation::0"))
    i = cmd.index("avfoundation")
    assert cmd[i + 1:i + 3] == ["-i", ":0"]
    assert "anullsrc=channel_layout=stereo:sample_rate=44100" not in cmd
    assert "aresample=async=1000" in cmd
