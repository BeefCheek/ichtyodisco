import mido

from aquarium.config import MusicConfig
from aquarium.music.conductor import Conductor
from aquarium.music.midi_out import MidiFileRecorder, MultiSink
from aquarium.music.theory import (
    MODES,
    chord_name,
    chord_pcs,
    note_number,
    roman,
    scale_notes,
    voice_chord,
)


class ListSink:
    def __init__(self):
        self.msgs: list[tuple[float, mido.Message]] = []
        self.closed = False

    def send(self, msg, t):
        self.msgs.append((t, msg))

    def close(self):
        self.closed = True


def test_theory_basics():
    assert note_number("D") == 2 and note_number("Bb") == 10 and note_number("f#") == 6
    assert chord_name(2, "ionian", 0) == "D" and chord_name(2, "aeolian", 0) == "Dm"
    assert chord_name(2, "lydian", 1) == "E"  # the characteristic major II of lydian
    assert roman("lydian", 1) == "II" and roman("dorian", 0) == "i"
    assert all((n - 2) % 12 in MODES["dorian"] for n in scale_notes(2, "dorian", 50, 80))
    assert chord_pcs(0, "ionian", 4, 4) == [7, 11, 2, 5]  # G7
    v = voice_chord([0, 4, 7], 60)
    assert sorted(n % 12 for n in v) == [0, 4, 7] and all(54 <= n < 66 for n in v)
    assert max(voice_chord([0, 4, 7, 11], 60, spread=1.0)) > 66  # open voicing


def _snap(mood="serein", act=0.0, feeding=False, n=3):
    return {"mood": {"label": mood}, "zscores": {"activity": act}, "feeding": feeding,
            "metrics": {"cohesion": 0.08, "polarization": 0.4, "mean_z": 0.3},
            "thresholds": {"burst_bls": 4.0}, "fish_count": n}


def _fish(n=3, state="cruise", speed=2.0):
    return [{"id": i + 1, "x": 0.2 * i, "y": 0.3 + 0.1 * i, "z": 0.2 * i, "speed_bls": speed,
             "state": state} for i in range(n)]


def _run(c: Conductor, seconds: float, t0: float = 0.0, dt: float = 0.01):
    t = t0
    while t < t0 + seconds:
        c.advance(t)
        t += dt
    return t


def test_conductor_plays_in_key_on_the_right_channels():
    sink = ListSink()
    c = Conductor(MusicConfig(enabled=True, key="D", seed=1), sink)
    c.set_state(_fish(), _snap())
    _run(c, 12.0)
    ons = [m for _, m in sink.msgs if m.type == "note_on" and m.velocity > 0]
    assert len(ons) > 30
    melodic = [m for m in ons if m.channel != 9]  # channel 10 = drums
    lydian = MODES["lydian"]
    assert all((m.note - 2) % 12 in lydian for m in melodic), "every pitched note is in D lydian"
    chans = {m.channel for m in ons}
    assert {0, 1} <= chans  # pad + bass
    assert chans & {3, 4, 5}  # fish voices
    st = c.status()
    assert st["key"] == "D" and st["mode"] == "lydien"


def test_mood_changes_mode_after_a_stable_bar():
    sink = ListSink()
    c = Conductor(MusicConfig(enabled=True, seed=2), sink)
    c.set_state(_fish(), _snap("serein"))
    t = _run(c, 6.0)
    assert c.mode == "lydian"
    c.set_state(_fish(), _snap("sur le qui-vive", act=1.5))
    _run(c, 10.0, t0=t)
    assert c.mode == "phrygian"


def test_activity_drives_tempo():
    calm = Conductor(MusicConfig(enabled=True, seed=3), ListSink())
    calm.set_state(_fish(), _snap(act=-2.0))
    _run(calm, 15.0)
    busy = Conductor(MusicConfig(enabled=True, seed=3), ListSink())
    busy.set_state(_fish(), _snap(act=2.5, feeding=True))
    _run(busy, 15.0)
    assert busy.bpm > calm.bpm * 1.2


def test_resting_fish_play_less_than_sprinting_fish():
    def count(state, speed):
        sink = ListSink()
        c = Conductor(MusicConfig(enabled=True, seed=4, drums=False), sink)
        c.set_state(_fish(3, state, speed), _snap())
        _run(c, 10.0)
        return sum(1 for _, m in sink.msgs if m.type == "note_on" and m.velocity > 0
                   and m.channel in (3, 4, 5))
    assert count("burst", 6.0) > 3 * max(1, count("rest", 0.1))


def test_event_phrases_and_no_stuck_notes():
    sink = ListSink()
    c = Conductor(MusicConfig(enabled=True, seed=5), sink)
    c.set_state(_fish(), _snap())
    t = _run(c, 2.0)
    c.trigger([{"kind": "feeding"}, {"kind": "chase", "fish": [1, 2]}])
    _run(c, 3.0, t0=t)
    lead = [m for _, m in sink.msgs if m.type == "note_on" and m.velocity > 0 and m.channel == 2]
    assert len(lead) >= 8  # the rising feeding arpeggio
    assert any(m.type == "note_on" and m.note == 49 and m.channel == 9 for _, m in sink.msgs)
    c.stop()  # panic: everything released, sink closed
    assert sink.closed
    sounding = set()
    for _, m in sink.msgs:
        if m.type == "note_on" and m.velocity > 0:
            sounding.add((m.channel, m.note))
        elif m.type in ("note_off", "note_on"):
            sounding.discard((m.channel, m.note))
    assert not sounding


def test_empty_tank_is_a_quiet_drone():
    sink = ListSink()
    c = Conductor(MusicConfig(enabled=True, seed=6), sink)
    c.set_state([], {"mood": {"label": "désert"}})
    _run(c, 6.0)
    ons = [m for _, m in sink.msgs if m.type == "note_on" and m.velocity > 0]
    assert ons and all(m.channel == 0 and m.velocity <= 40 for m in ons)


def test_midi_file_recording(tmp_path):
    path = tmp_path / "session.mid"
    rec = MidiFileRecorder(path)
    c = Conductor(MusicConfig(enabled=True, seed=7), MultiSink([rec]))
    c.set_state(_fish(), _snap())
    _run(c, 5.0, t0=100.0)
    c.panic(105.0)
    rec.close()
    mid = mido.MidiFile(path)
    assert 4.5 < mid.length < 6.0  # real-time duration preserved
    assert sum(1 for m in mid if m.type == "note_on") > 10
