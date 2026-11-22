from pathlib import Path

import mido
import pytest

from aquarium.music.clips import load_clip
from aquarium.music.expr import Expr, ExprError
from aquarium.music.score import Score, ScorePlayer

DEMO = Path(__file__).resolve().parents[1] / "scores" / "demo" / "score.toml"


class ListSink:
    def __init__(self):
        self.msgs = []
        self.closed = False

    def send(self, msg, t):
        self.msgs.append((t, msg))

    def close(self):
        self.closed = True


# ---------------------------------------------------------------- expressions
def test_expressions_are_safe_and_forgiving():
    assert Expr("clip(fish_count / 4, 0.2, 1)")({"fish_count": 2}) == 0.5
    assert Expr("1 if mood == 'festin' else 0")({"mood": "festin"}) == 1
    assert Expr("unknown_metric + 1")({}) == 1  # unknown names are 0
    assert Expr("1 / zero").num({"zero": 0}, default=7.0) == 0.0  # errors -> 0, never raise
    for bad in ("__import__('os')", "().__class__", "open('x')", "[x for x in y]", "lambda: 1"):
        with pytest.raises(ExprError):
            Expr(bad)


# ---------------------------------------------------------------- clips
def _write_mid(path, notes, bars=1, tpb=480):
    tr = mido.MidiTrack()
    ev = []
    for start, dur, note, ch in notes:
        ev += [(int(start * tpb), mido.Message("note_on", channel=ch, note=note, velocity=100)),
               (int((start + dur) * tpb), mido.Message("note_off", channel=ch, note=note))]
    ev.sort(key=lambda e: e[0])
    last = 0
    for tick, m in ev:
        tr.append(m.copy(time=tick - last))
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=bars * 4 * tpb - last))
    f = mido.MidiFile(ticks_per_beat=tpb)
    f.tracks.append(tr)
    f.save(path)


def test_clip_loading_and_looping(tmp_path):
    p = tmp_path / "c.mid"
    _write_mid(p, [(0, 1, 60, 0), (2.5, 0.5, 64, 1)], bars=1)
    c = load_clip(p)
    assert c.length == 4 and [n.note for n in c.notes] == [60, 64]
    assert [(s, n.note) for s, n in c.window(4, 8)] == [(4.0, 60), (6.5, 64)]  # second loop
    assert load_clip(p, channel=5).notes[1].channel == 4


# ---------------------------------------------------------------- player
SCORE = """
[score]
bpm = 120
beats_per_bar = 4
default_section = "calm"

[signal.heat]
on = ["school_return"]
add = 1
half_life_s = 10

[section.calm]
layers = [ { clip = "pad.mid", name = "pad" }, { clip = "bass.mid", name = "bass", gain = "clip(fish_count / 4)" } ]

[section.party]
when = "feeding"
priority = 5
min_bars = 1
layers = [ { clip = "pad.mid", name = "pad" }, { clip = "drums.mid", name = "drums" } ]

[[overlay]]
name = "heat"
clip = "heat.mid"
gain = "clip(heat)"
fade_beats = 0

[[transition]]
from = "calm"
to = "party"
at = "bar"
bridge = { clip = "riser.mid" }

[[stinger]]
on = "school_return"
when = "duration < 3"
note = 76
channel = 10
quantize = "beat"
pan = "x"
cooldown_s = 0.5
"""


@pytest.fixture
def score(tmp_path):
    _write_mid(tmp_path / "pad.mid", [(0, 3.9, 50, 0)])
    _write_mid(tmp_path / "bass.mid", [(0, 0.5, 38, 1), (2, 0.5, 38, 1)])
    _write_mid(tmp_path / "drums.mid", [(b, 0.1, 36, 9) for b in range(4)])
    _write_mid(tmp_path / "heat.mid", [(0.5, 0.2, 90, 4)])
    _write_mid(tmp_path / "riser.mid", [(k, 0.5, 70 + k, 3) for k in range(4)])
    (tmp_path / "score.toml").write_text(SCORE)
    return Score.load(tmp_path / "score.toml")


def _run(p, t0, seconds, dt=0.01):
    t = t0
    while t < t0 + seconds:
        p.advance(t)
        t += dt
    return t


def _ons(sink, channel=None, after=None):
    return [(t, m) for t, m in sink.msgs if m.type == "note_on" and m.velocity > 0
            and (channel is None or m.channel == channel) and (after is None or t >= after)]


def test_layers_follow_metrics(score):
    sink = ListSink()
    p = ScorePlayer(score, sink, humanize=0)
    p.set_state([{}] * 0, {"fish_count": 0})
    t = _run(p, 0.0, 4.0)  # 2 bars at 120 bpm with no fish: bass muted
    assert _ons(sink, 0) and not _ons(sink, 1)
    p.set_state([{"id": i} for i in range(4)], {"fish_count": 4})
    _run(p, t, 4.0)
    assert _ons(sink, 1, after=t)  # bass came in


def test_section_switch_waits_for_bar_and_plays_bridge(score):
    sink = ListSink()
    p = ScorePlayer(score, sink, humanize=0)
    p.set_state([{"id": 1}], {"fish_count": 1})
    t = _run(p, 0.0, 4.5)  # past bar 2 (bar = 2 s)
    p.set_state([{"id": 1}], {"fish_count": 1, "feeding": True})
    _run(p, t, 6.0)
    riser = _ons(sink, 3)
    assert len(riser) == 4
    assert abs(riser[0][0] - 6.0) < 0.02  # bridge starts on the next bar line, not immediately
    drums = _ons(sink, 9)
    assert drums and abs(drums[0][0] - 8.0) < 0.02  # new section after the 1-bar bridge
    assert p.section == "party" and p.history[-1][1] == "party"
    # the shared pad never stopped: one note per bar on channel 0 through the transition
    pad_starts = [round(t) for t, _ in _ons(sink, 0)]
    assert pad_starts == list(range(0, 11, 2))


def test_stinger_is_quantised_panned_and_cooled_down(score):
    sink = ListSink()
    p = ScorePlayer(score, sink, humanize=0)
    p.set_state([{"id": 1}], {"fish_count": 1})
    t = _run(p, 0.0, 1.13)
    ev = {"kind": "school_return", "fish": [3], "data": {"duration": 1.2, "x": 1.0, "y": 0.5}}
    p.trigger([ev, ev])  # duplicate within the cooldown
    p.trigger([{"kind": "school_return", "fish": [4], "data": {"duration": 9.0}}])  # filtered by `when`
    _run(p, t, 1.0)
    hits = _ons(sink, 9)
    assert len(hits) == 1
    assert abs(hits[0][0] - 1.5) < 0.02  # next beat at 120 bpm
    pans = [m for _, m in sink.msgs if m.type == "control_change" and m.control == 10]
    assert pans and pans[-1].value == 127
    assert p.stingers_fired == 1


def test_signals_rise_with_events_and_decay(score):
    sink = ListSink()
    p = ScorePlayer(score, sink, humanize=0)
    p.set_state([{"id": 1}], {"fish_count": 1})
    t = _run(p, 0.0, 0.5)
    p.trigger([{"kind": "school_return", "fish": [i], "data": {"duration": 9}} for i in range(3)])
    t = _run(p, t, 0.1)
    assert abs(p._signal("heat", t) - 3.0) < 0.1
    t = _run(p, t, 3.0)
    assert _ons(sink, 4)  # the overlay driven by the signal played
    assert p._signal("heat", t + 10.0) < 1.6  # halves every 10 s


def test_no_stuck_notes_after_stop(score):
    sink = ListSink()
    p = ScorePlayer(score, sink, humanize=0)
    p.set_state([{"id": 1}], {"fish_count": 4, "feeding": True})
    _run(p, 0.0, 5.0)
    p.stop()
    sounding = set()
    for _, m in sink.msgs:
        if m.type == "note_on" and m.velocity > 0:
            sounding.add((m.channel, m.note))
        elif m.type in ("note_on", "note_off"):
            sounding.discard((m.channel, m.note))
    assert not sounding and sink.closed


def test_demo_score_loads_and_plays():
    score = Score.load(DEMO)
    assert {"calm", "explore", "schooling", "feeding", "night", "empty"} <= set(score.sections)
    sink = ListSink()
    p = ScorePlayer(score, sink, humanize=0)
    fish = [{"id": i, "x": 0.5, "y": 0.5} for i in range(5)]
    p.set_state(fish, {"fish_count": 5, "metrics": {"polarization": 0.5}, "zscores": {"activity": 0}})
    t = _run(p, 0.0, 6.0)
    p.trigger([{"kind": "school_return", "fish": [2], "data": {"duration": 1.0, "x": 0.2, "speed": 3}}])
    _run(p, t, 2.0)
    assert any(m.note == 76 and m.channel == 9 for _, m in _ons(sink))  # the spike
    assert p.status()["section"] in score.sections
