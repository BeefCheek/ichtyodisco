"""End-to-end on the synthetic aquarium: tracks -> engine -> overlay, plus persistence."""

import json

import numpy as np

from aquarium.analytics.engine import AnalyticsEngine
from aquarium.render.overlay import OverlayRenderer
from aquarium.sim.aquarium import SimulatedAquarium
from aquarium.types import DepthMap


def _run(seconds=40, n_fish=5, id_switch_rate=0.0, seed=1):
    sim = SimulatedAquarium(n_fish=n_fish, size=(640, 360), id_switch_rate=id_switch_rate,
                            seed=seed, feeding_every_s=0)
    eng = AnalyticsEngine()
    t, snaps = 1_000_000.0, []
    frame = depth = tracks = None
    for _ in range(int(seconds * 30)):
        sim.step(1 / 30)
        t += 1 / 30
        frame, depth, tracks = sim.render()
        for tr in tracks:  # ground-truth depth, as the depth stage would provide
            cx, cy = tr.center
            tr.z = float(depth[int(cy / 4) % depth.shape[0], int(cx / 4) % depth.shape[1]])
        snap = eng.update(tracks, frame, frame.shape[:2], t=t)
        if snap:
            snaps.append(snap)
    return eng, snaps, frame, depth, tracks


def test_engine_produces_rich_snapshots():
    eng, snaps, *_ = _run()
    assert len(snaps) >= 70  # ~2 Hz
    last = snaps[-1]
    assert last["fish_count"] == 5
    assert set(last["states"]) == {"rest", "cruise", "explore", "burst"}
    assert last["metrics"]["activity"] > 0
    assert last["mood"]["label"]
    assert all(f["z"] is not None for f in last["fish"])
    assert "activity" in last["zscores"]
    json.dumps(last)  # must be JSON-serialisable for the dashboard / SQLite


def test_reid_keeps_resident_count_close_to_truth():
    eng, *_ = _run(seconds=60, n_fish=5, id_switch_rate=0.002, seed=4)
    # many tracker id switches happen, but residents should stay close to the 5 real fish
    assert len(eng.registry.residents) <= 10


def test_state_roundtrip_through_json():
    eng, *_ = _run(seconds=10)
    state = json.loads(json.dumps(eng.state_dict()))
    fresh = AnalyticsEngine()
    fresh.load_state_dict(state)
    assert len(fresh.registry.residents) == len(eng.registry.residents)
    assert fresh.baseline.metrics.keys() == eng.baseline.metrics.keys()
    assert np.allclose(fresh.occupancy.front, eng.occupancy.front, atol=1e-3)


def test_overlay_renders_at_output_size():
    eng, snaps, frame, depth, tracks = _run(seconds=5)
    r = OverlayRenderer(size=(1280, 720))
    dm = DepthMap(depth, 1.0, frame.shape[:2], model="sim", latency_s=0.05)
    out = r.render(frame, tracks, snaps[-1], dm, stats={"fps": 30})
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
