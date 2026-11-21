import math

import numpy as np

from aquarium.analytics.behavior import BehaviorClassifier
from aquarium.analytics.kinematics import FishTrack
from aquarium.analytics.social import group_metrics


def test_constant_velocity_speed_and_body_lengths():
    tr = FishTrack(1)
    for i in range(60):
        t = i / 30
        tr.add(t, 0.1 + 0.2 * t, 0.3, 0.1, body_len=0.05)
    assert math.isclose(tr.kin.speed, 0.2, rel_tol=0.02)
    assert math.isclose(tr.kin.speed_bls, 4.0, rel_tol=0.02)  # 0.2 / 0.05
    assert tr.kin.tortuosity < 1.05


def test_depth_contributes_to_speed():
    tr = FishTrack(1)
    for i in range(30):
        t = i / 30
        tr.add(t, 0.5, 0.5, 0.1 * t, body_len=0.05)
    assert math.isclose(tr.kin.vz, 0.1, rel_tol=0.05)


def test_unknown_depth_is_ignored_in_velocity():
    tr = FishTrack(1)
    for i in range(30):
        tr.add(i / 30, 0.5, 0.5, None, body_len=0.05)
    assert tr.kin.speed < 1e-6


def test_history_is_bounded():
    tr = FishTrack(1, history_s=2.0)
    for i in range(300):
        tr.add(i / 30, 0.5, 0.5, None, 0.05)
    assert tr.t[-1] - tr.t[0] <= 2.0 + 1e-9


def test_behavior_classifier_adapts_and_has_hysteresis():
    bc = BehaviorClassifier(dwell_s=0.5)
    rng = np.random.default_rng(0)
    for i in range(2000):
        bc.learn(float(rng.lognormal(0, 0.6)), float(i))
    rest, burst = bc.thresholds()
    assert rest < 1.0 < burst
    t = 0.0
    assert bc.classify(1, 1.0, 1.0, t) == "cruise"
    assert bc.classify(1, 0.01, 1.0, t + 0.1) == "cruise"  # not long enough yet
    assert bc.classify(1, 0.01, 1.0, t + 0.7) == "rest"
    assert bc.classify(1, burst * 2, 1.0, t + 0.75) == "burst"  # bursts are immediate


def test_group_metrics():
    ids = [1, 2, 3]
    pos = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=float)
    vel = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    g = group_metrics(ids, pos, vel, np.array([1.0, 2.0, 3.0]), close_distance=0.12)
    assert g.count == 3
    assert math.isclose(g.polarization, 1.0)
    assert math.isclose(g.activity, 2.0)
    assert math.isclose(g.nnd, 0.1)
    assert {(a, b) for a, b, _ in g.pairs} == {(1, 2), (1, 3)}
    assert group_metrics([], np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)).count == 0


def test_gap_resets_segment_instead_of_fake_sprint():
    tr = FishTrack(1)
    for i in range(30):
        tr.add(i / 30, 0.2, 0.5, None, body_len=0.05)
    # lost for 1 s, then found 0.3 tank widths away: must not register as a sprint
    for i in range(5):
        tr.add(2.0 + i / 30, 0.5, 0.5, None, body_len=0.05)
    assert tr.kin.speed_bls < 0.5
