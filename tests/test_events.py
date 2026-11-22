import numpy as np

from aquarium.analytics.events import EventContext, EventDetector, FishView, mood


def _fish(fid, pos, vel, speed, y=0.5):
    return FishView(fid, f"Poisson #{fid}", np.array(pos, float), np.array(vel, float), speed,
                    "burst", y)


def _ctx(t, **kw):
    base = dict(t=t, fish=[], metrics={}, zscores={}, baseline_ready=True, rest_threshold=0.3,
                burst_threshold=4.0)
    base.update(kw)
    return EventContext(**base)


def test_feeding_frenzy_start_and_end():
    det = EventDetector()
    fish = [_fish(i, [0.1 * i, 0.05, 0.1], [0.1, 0, 0], 5.0, y=0.05) for i in range(4)]
    kinds = []
    for k in range(10):
        evs = det.update(_ctx(k * 0.5, fish=fish, metrics={"surface_ratio": 0.9, "mean_y": 0.1},
                              zscores={"activity": 3.5}))
        kinds += [e.kind for e in evs]
    assert kinds.count("feeding") == 1 and det.feeding
    evs = det.update(_ctx(10.0, fish=fish, metrics={"surface_ratio": 0.0}, zscores={"activity": 0}))
    assert [e.kind for e in evs] == ["feeding_end"]


def test_chase_detects_chaser():
    det = EventDetector(chase_distance=0.1, chase_min_s=1.0)
    out = []
    for k in range(40):
        t = k / 10
        a = _fish(1, [0.30 + 0.2 * t, 0.5, 0.1], [0.2, 0, 0], 5.0)  # behind, same heading
        b = _fish(2, [0.35 + 0.2 * t, 0.5, 0.1], [0.2, 0, 0], 5.0)
        out += det.update(_ctx(t, fish=[a, b], pairs=[(1, 2, 0.05)]))
    chases = [e for e in out if e.kind == "chase"]
    assert len(chases) == 1
    assert chases[0].fish == [1, 2] and "Poisson #1 poursuit Poisson #2" in chases[0].message


def test_cooldown_prevents_spam():
    det = EventDetector()
    fish = [_fish(i, [0.1 * i, 0.5, 0.5], [0, 0, 0], 0.1) for i in range(3)]
    n = 0
    for k in range(200):
        n += len(det.update(_ctx(k * 1.0, fish=fish, zscores={"cohesion": -4.0})))
    assert n == 1


def test_mood_labels():
    assert mood({"fish_count": 0}, {}, False)[0] == "désert"
    assert mood({"fish_count": 3}, {}, True)[0] == "festin"
    assert mood({"fish_count": 3}, {"activity": 2.5}, False)[0] == "excité"
    assert mood({"fish_count": 3, "rest_ratio": 0.8}, {}, False)[0] == "endormi"
    assert mood({"fish_count": 3}, {}, False)[0] == "serein"


def test_feeding_detected_despite_spiky_activity():
    # Real (and simulated) meals: fish dart and pause, so the activity z-score oscillates.
    det = EventDetector()
    fish = [_fish(i, [0.1 * i, 0.05, 0.1], [0.1, 0, 0], 5.0, y=0.05) for i in range(4)]
    kinds = []
    for k in range(20):
        z = 3.0 if k % 2 == 0 else 0.0
        evs = det.update(_ctx(k * 0.5, fish=fish, metrics={"surface_ratio": 1.0, "mean_y": 0.12},
                              zscores={"activity": z}))
        kinds += [e.kind for e in evs]
    assert kinds.count("feeding") == 1


def test_no_feeding_from_surface_alone():
    det = EventDetector()
    fish = [_fish(i, [0.1 * i, 0.05, 0.1], [0.1, 0, 0], 1.0, y=0.05) for i in range(4)]
    kinds = []
    for k in range(40):
        evs = det.update(_ctx(k * 0.5, fish=fish, metrics={"surface_ratio": 1.0, "mean_y": 0.12},
                              zscores={"activity": 0.3}))
        kinds += [e.kind for e in evs]
    assert "feeding" not in kinds
