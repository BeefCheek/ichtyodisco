import math

from aquarium.analytics.stats import CircadianBaseline, DecayingHistogram, DecayingStats


def test_decaying_stats_mean_std():
    s = DecayingStats(half_life=1e9)
    for i, x in enumerate([1.0, 2.0, 3.0, 4.0]):
        s.update(x, float(i))
    assert math.isclose(s.mean, 2.5, rel_tol=1e-6)
    assert math.isclose(s.std, math.sqrt(1.25), rel_tol=1e-6)


def test_decaying_stats_forgets_old_data():
    s = DecayingStats(half_life=10.0)
    for t in range(100):
        s.update(0.0, float(t))
    for t in range(100, 200):
        s.update(10.0, float(t))
    assert s.mean > 9.9  # old regime has been forgotten


def test_decaying_stats_roundtrip():
    s = DecayingStats(60.0)
    s.update(3.0, 1.0)
    s.update(5.0, 2.0)
    r = DecayingStats.from_dict(s.to_dict())
    assert math.isclose(r.mean, s.mean) and math.isclose(r.std, s.std)


def test_histogram_quantiles():
    h = DecayingHistogram(0.0, 100.0, bins=100, half_life=1e9)
    for i in range(1000):
        h.update(i % 100 + 0.5, float(i))
    assert abs(h.quantile(0.5) - 50) < 2
    assert abs(h.quantile(0.9) - 90) < 2
    assert DecayingHistogram(0, 1).quantile(0.5, default=-1) == -1


def test_circadian_baseline_is_hour_specific():
    from datetime import datetime

    b = CircadianBaseline(half_life_s=1e9, min_weight=10)
    day = datetime(2026, 11, 20)
    t_night = day.replace(hour=3).timestamp()
    t_dinner = day.replace(hour=18).timestamp()
    for i in range(50):
        b.update("activity", 0.2 + 0.01 * (i % 3), t_night + i)
        b.update("activity", 3.0 + 0.1 * (i % 3), t_dinner + i)
    # 3 BL/s is normal at dinner time but very abnormal at night
    assert abs(b.zscore("activity", 3.0, t_dinner + 100)) < 2
    assert b.zscore("activity", 3.0, t_night + 100) > 10
    assert b.coverage("activity") == 2
    restored = CircadianBaseline.from_dict(b.to_dict())
    assert restored.coverage("activity") == 2
