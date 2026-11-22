import numpy as np

from aquarium.analytics.interactions import FishFrame, InteractionDetector


def _frame(fid, x, y, z=0.5, vx=0.1, vy=0.0, speed=1.0, state="cruise"):
    return FishFrame(fid, f"Poisson #{fid}", np.array([x, y, z * 0.4]), x, y, z,
                     np.array([vx, vy, 0.0]), speed, state)


def _school(t, extra=None):
    """5 fish in a tight group around (0.5, 0.5); `extra` overrides fish 1's position."""
    fish = [_frame(i, 0.5 + 0.02 * (i % 3), 0.5 + 0.02 * (i // 3)) for i in range(1, 6)]
    if extra is not None:
        fish[0] = _frame(1, *extra)
    return fish


def test_fish_leaves_and_returns_to_school():
    det = InteractionDetector()
    t, kinds, ret = 0.0, [], None
    for _ in range(10):  # all together
        kinds += [e.kind for e in det.update(t, _school(t))]
        t += 0.1
    for _ in range(12):  # fish 1 darts away for 1.2 s
        kinds += [e.kind for e in det.update(t, _school(t, extra=(0.9, 0.5)))]
        t += 0.1
    for _ in range(5):  # and comes back
        evs = det.update(t, _school(t))
        kinds += [e.kind for e in evs]
        ret = ret or next((e for e in evs if e.kind == "school_return"), None)
        t += 0.1
    assert kinds.count("school_leave") == 1
    assert ret is not None and ret.fish == [1]
    assert 1.0 <= ret.data["duration"] <= 1.5
    assert ret.data["x"] > 0.4 and "revient au banc" in ret.message


def test_short_wobble_is_not_an_excursion():
    det = InteractionDetector(min_out_s=0.35)
    t, kinds = 0.0, []
    for k in range(20):
        extra = (0.9, 0.5) if k in (10, 11) else None  # 0.2 s outside only
        kinds += [e.kind for e in det.update(t, _school(t, extra=extra))]
        t += 0.1
    assert "school_leave" not in kinds and "school_return" not in kinds


def test_zone_contacts_and_dash():
    det = InteractionDetector()
    seq = [_frame(1, 0.5, 0.5), _frame(1, 0.5, 0.05, z=0.05), _frame(1, 0.5, 0.05, z=0.05),
           _frame(1, 0.5, 0.5, state="burst", speed=6.0)]
    kinds = []
    for i, f in enumerate(seq):
        kinds += [e.kind for e in det.update(i * 0.1, [f, _frame(2, 0.2, 0.5)])]
    assert kinds.count("surface") == 1 and kinds.count("glass") == 1
    assert "dash" in kinds


def test_encounter_and_split_merge():
    det = InteractionDetector()
    kinds = []
    t = 0.0
    # two fish approach each other: encounter
    for d in (0.3, 0.2, 0.1, 0.02):
        kinds += [e.kind for e in det.update(t, [_frame(1, 0.5 - d, 0.5), _frame(2, 0.5 + d, 0.5)])]
        t += 0.2
    assert "encounter" in kinds
    det2 = InteractionDetector()
    kinds2 = []
    for k in range(30):
        spread = 0.0 if k < 10 or k >= 20 else 0.3
        fish = [_frame(10 + i, 0.4 + 0.01 * i, 0.5) for i in range(3)]
        fish += [_frame(20 + i, 0.45 + spread + 0.01 * i, 0.5) for i in range(3)]
        kinds2 += [e.kind for e in det2.update(k * 0.1, fish)]
    assert kinds2.count("school_split") == 1 and kinds2.count("school_merge") == 1
