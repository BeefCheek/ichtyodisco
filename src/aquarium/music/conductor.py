"""The conductor: turns what the fish are doing into music, sent as MIDI to real synths.

Mapping (all of it follows the analytics, so the music evolves with the tank):

    tank mood         -> mode + chord progression (lydian when serene, phrygian when on alert...)
    activity vs norm  -> tempo, bass rhythm, drum density
    shoal cohesion    -> pad voicing (tight shoal = close voicing, scattered = open)
    polarisation      -> mod wheel (CC1) on the pad
    each fish         -> its own MIDI channel / voice:
        height (y)        pitch register (surface = high), snapped to chord / scale tones
        swim state        rhythm: resting = rare long notes, cruising = 8ths, sprint = 16th bursts
        speed             velocity
        left-right (x)    pan (CC10)
        depth (z)         brightness CC74 (near the glass = bright) + reverb send CC91 (far = wet)
    events            -> musical phrases (feeding = rising arpeggio + crash, chase = call and
                         response between the two fishes' voices, newcomer = bell...)

Channels (1-16, configurable): pad 1, bass 2, lead/events 3, fish 4-9, drums 10 (GM map).
In a DAW, put one instrument per channel; on hardware, set each synth's MIDI channel.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import random
import threading
import time
from collections import deque
from typing import Any

import mido

from aquarium.config import MusicConfig
from aquarium.music.midi_out import MidiSink
from aquarium.music.theory import (
    MODE_FR,
    MOOD_MUSIC,
    chord_name,
    chord_pcs,
    nearest_in,
    note_number,
    roman,
    scale_notes,
    voice_chord,
)

log = logging.getLogger(__name__)

KICK, RIM, HAT, OPEN_HAT, CRASH, SHAKER = 36, 37, 42, 46, 49, 82


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Conductor:
    def __init__(self, cfg: MusicConfig, sink: MidiSink):
        self.cfg = cfg
        self.sink = sink
        self.root = note_number(cfg.key)
        self.rng = random.Random(cfg.seed)
        self._lock = threading.Lock()
        # latest view of the tank (written by the pipeline thread)
        self._fish: list[dict[str, Any]] = []
        self._snap: dict[str, Any] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=32)
        # musical state (only touched by the conductor thread)
        self.mood = "désert"
        self.mode, self.progression, self.bars_per_chord = MOOD_MUSIC[self.mood]
        self._pending_mood: str | None = None
        self._pending_bar = 0
        self.bpm = float(cfg.base_bpm)
        self.step = 0
        self.next_step_t: float | None = None
        self.chord_idx = 0
        self.degree = 0
        self.pad_notes: list[int] = []
        self._timeline: list[tuple[float, int, mido.Message]] = []
        self._seq = itertools.count()
        self._active: set[tuple[int, int]] = set()
        self._fish_channel: dict[int, int] = {}
        self._last_note: dict[int, int] = {}
        self._cc_cache: dict[tuple[int, int], int] = {}
        self.notes_played = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ inputs (any thread)
    def set_state(self, fish: list[dict[str, Any]], snapshot: dict[str, Any] | None) -> None:
        with self._lock:
            self._fish = fish
            if snapshot:
                self._snap = snapshot

    def trigger(self, events: list[dict[str, Any]]) -> None:
        with self._lock:
            self._events.extend(events)

    # ------------------------------------------------------------------ clock
    @property
    def step_s(self) -> float:
        return 60.0 / self.bpm / 4.0  # one 16th note

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="conductor")
        self._thread.start()

    def _run(self) -> None:
        if self.cfg.send_clock:
            self._emit(mido.Message("start"), time.perf_counter())
        while not self._stop.is_set():
            now = time.perf_counter()
            self.advance(now)
            nxt = self.next_step_t if self.next_step_t is not None else now + 0.01
            if self._timeline:
                nxt = min(nxt, self._timeline[0][0])
            time.sleep(_clip(nxt - time.perf_counter(), 0.0005, 0.02))

    def advance(self, now: float) -> None:
        """Play everything due up to ``now`` (called by the thread, or directly in tests)."""
        if self.next_step_t is None:
            self.next_step_t = now
        while self.next_step_t <= now:
            t = self.next_step_t
            self._flush(t)
            self._do_step(t)
            self.next_step_t = t + self.step_s
        self._flush(now)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.panic(time.perf_counter())
        self.sink.close()

    def panic(self, t: float) -> None:
        """Release every sounding note (no stuck notes on the synths when we quit)."""
        self._timeline.clear()
        for ch, note in list(self._active):
            self._emit(mido.Message("note_off", channel=ch, note=note, velocity=0), t)
        self._active.clear()
        for ch in self._channels():
            self._emit(mido.Message("control_change", channel=ch, control=123, value=0), t)
        if self.cfg.send_clock:
            self._emit(mido.Message("stop"), t)

    # ------------------------------------------------------------------ plumbing
    def _ch(self, one_based: int) -> int:
        return int(_clip(one_based, 1, 16)) - 1

    def _channels(self) -> set[int]:
        c = self.cfg
        return {self._ch(x) for x in [c.pad_channel, c.bass_channel, c.lead_channel,
                                      c.drums_channel, *c.fish_channels]}

    def _emit(self, msg: mido.Message, t: float) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            self._active.add((msg.channel, msg.note))
            self.notes_played += 1
        elif msg.type in ("note_off", "note_on"):
            self._active.discard((msg.channel, msg.note))
        self.sink.send(msg, t)

    def _at(self, t: float, msg: mido.Message) -> None:
        heapq.heappush(self._timeline, (t, next(self._seq), msg))

    def _flush(self, until: float) -> None:
        while self._timeline and self._timeline[0][0] <= until:
            t, _, msg = heapq.heappop(self._timeline)
            if msg.type == "note_on" and (msg.channel, msg.note) in self._active:
                self._emit(mido.Message("note_off", channel=msg.channel, note=msg.note), t)
            self._emit(msg, t)

    def _play(self, t: float, ch: int, note: int, vel: float, dur: float) -> None:
        note = int(_clip(note, 0, 127))
        vel = int(_clip(vel * (1 + self.rng.uniform(-self.cfg.humanize, self.cfg.humanize)), 1, 127))
        self._at(t, mido.Message("note_on", channel=ch, note=note, velocity=vel))
        self._at(t + max(dur, 0.02), mido.Message("note_off", channel=ch, note=note, velocity=0))

    def _cc(self, t: float, ch: int, control: int, value: float, min_delta: int = 3) -> None:
        v = int(_clip(value, 0, 127))
        key = (ch, control)
        if abs(self._cc_cache.get(key, -100) - v) >= min_delta:
            self._cc_cache[key] = v
            self._emit(mido.Message("control_change", channel=ch, control=control, value=v), t)

    # ------------------------------------------------------------------ the music
    def _do_step(self, t: float) -> None:
        with self._lock:
            fish = list(self._fish)
            snap = self._snap
            events = list(self._events)
            self._events.clear()
        pos = self.step % 16
        m = snap.get("metrics", {}) or {}
        z = snap.get("zscores", {}) or {}
        ctx = {
            "fish": fish, "pos": pos, "count": len(fish),
            "act_z": float(z.get("activity", 0.0) or 0.0),
            "cohesion": m.get("cohesion"), "polarization": m.get("polarization") or 0.0,
            "mean_z": m.get("mean_z"), "feeding": bool(snap.get("feeding")),
            "mood": (snap.get("mood") or {}).get("label", "désert") if fish else "désert",
            "burst": float((snap.get("thresholds") or {}).get("burst_bls", 4.0) or 4.0),
        }
        if pos == 0:
            self._bar(t, ctx)
        if self.cfg.send_clock:
            for k in range(6):  # 24 ppqn
                self._at(t + k * self.step_s / 6, mido.Message("clock"))
        self._bass(t, ctx)
        if self.cfg.drums:
            self._drums(t, ctx)
        self._fish_notes(t, ctx)
        for e in events:
            self._phrase(t, e)
        self.step += 1

    def _bar(self, t: float, c: dict[str, Any]) -> None:
        bar = self.step // 16
        # mood -> mode, only after it has been stable for a full bar (no flapping)
        if c["mood"] != self.mood:
            if self.step == 0:  # first bar: take the current mood right away
                self._pending_mood, self._pending_bar = c["mood"], bar - 1
            if self._pending_mood != c["mood"]:
                self._pending_mood, self._pending_bar = c["mood"], bar
            elif bar - self._pending_bar >= 1:
                self.mood = c["mood"]
                self.mode, self.progression, self.bars_per_chord = MOOD_MUSIC.get(
                    self.mood, MOOD_MUSIC["serein"])
                self.chord_idx = 0
                self._pending_mood = None
                log.info("music: %s -> %s %s", self.mood, self.cfg.key, MODE_FR[self.mode])
        else:
            self._pending_mood = None
        # tempo follows activity relative to the learnt normal
        target = self.cfg.base_bpm * _clip(1 + 0.08 * c["act_z"], 0.75, 1.3)
        if c["feeding"]:
            target *= 1.12
        if self.mood == "endormi":
            target *= 0.85
        self.bpm += (target - self.bpm) * 0.5
        # harmony
        pad = self._ch(self.cfg.pad_channel)
        if bar % self.bars_per_chord == 0:
            self.degree = self.progression[self.chord_idx % len(self.progression)]
            self.chord_idx += 1
            for n in self.pad_notes:
                self._emit(mido.Message("note_off", channel=pad, note=n, velocity=0), t)
            if c["count"] == 0:
                notes = [36 + self.root, 43 + self.root]  # bare drone: the tank is empty
                vel = 32
            else:
                spread = _clip((c["cohesion"] or 0.1) / 0.2, 0, 1)
                notes = voice_chord(chord_pcs(self.root, self.mode, self.degree, 4), 62, spread)
                vel = 42 + 38 * min(1.0, c["count"] / 8) + (18 if c["feeding"] else 0)
            self.pad_notes = notes
            for n in notes:
                self._emit(mido.Message("note_on", channel=pad, note=n,
                                        velocity=int(_clip(vel, 1, 127))), t)
        self._cc(t, pad, 1, c["polarization"] * 127)
        if c["mean_z"] is not None:
            self._cc(t, pad, 74, (1 - c["mean_z"]) * 127)
        self._cc(t, pad, 11, 70 + 20 * _clip(c["act_z"], -2, 2))

    def _bass(self, t: float, c: dict[str, Any]) -> None:
        if c["count"] == 0:
            return
        pos, act = c["pos"], c["act_z"]
        if c["feeding"] or act > 1.0 or self.mood == "excité":
            hits = (0, 3, 6, 8, 11, 14)
        elif act > -0.5 and self.mood != "endormi":
            hits = (0, 8, 10)
        else:
            hits = (0,)
        if pos not in hits:
            return
        nxt = next((h for h in hits if h > pos), 16)
        pcs = chord_pcs(self.root, self.mode, self.degree, 3)
        pc = pcs[2] if pos in (8, 10) and self.rng.random() < 0.35 else pcs[0]
        note = 36 + (pc - 36) % 12
        self._play(t, self._ch(self.cfg.bass_channel), note, 78 + (14 if pos == 0 else 0),
                   (nxt - pos) * self.step_s * 0.9)

    def _drums(self, t: float, c: dict[str, Any]) -> None:
        if c["count"] == 0:
            return
        ch, pos = self._ch(self.cfg.drums_channel), c["pos"]
        d = 1.0 if c["feeding"] else _clip((c["act_z"] + 1) / 3, 0, 1)
        r = self.rng.random
        if (pos == 0 and d > 0.25) or (pos == 8 and d > 0.6) or (pos == 10 and d > 0.85):
            self._play(t, ch, KICK, 100, self.step_s)
        if pos in (4, 12) and d > 0.5:
            self._play(t, ch, RIM, 70, self.step_s)
        if pos % 2 == 0 and r() < d * 0.9:
            self._play(t, ch, HAT if pos != 14 or d < 0.8 else OPEN_HAT, 55 + 30 * d, self.step_s)
        elif pos % 2 == 1 and r() < d * 0.35:
            self._play(t, ch, SHAKER, 45 + 20 * d, self.step_s)

    def _channel_for(self, fid: int) -> int:
        ch = self._fish_channel.get(fid)
        if ch is None:
            chans = self.cfg.fish_channels or [self.cfg.lead_channel]
            used = set(self._fish_channel.values())
            free = [self._ch(x) for x in chans if self._ch(x) not in used]
            ch = free[0] if free else self._ch(chans[fid % len(chans)])
            self._fish_channel[fid] = ch
        return ch

    def _fish_notes(self, t: float, c: dict[str, Any]) -> None:
        pos = c["pos"]
        weight = 1.0 if pos % 4 == 0 else (0.55 if pos % 2 == 0 else 0.3)
        lo, hi = self.cfg.fish_range
        chord = set(chord_pcs(self.root, self.mode, self.degree, 4))
        scale = scale_notes(self.root, self.mode, lo, hi)
        tones = [n for n in scale if n % 12 in chord]
        voices = sorted(c["fish"], key=lambda f: -f.get("speed_bls", 0))[: self.cfg.max_voices]
        for f in voices:
            state, sp = f.get("state", "cruise"), float(f.get("speed_bls", 0.0))
            if state == "rest":
                p, steps = (0.12 if pos % 8 == 0 else 0.0), 8
            elif state == "cruise":
                p, steps = _clip(0.08 + 0.1 * sp, 0, 0.6) * weight, 2
            elif state == "explore":
                p, steps = _clip(0.12 + 0.12 * sp, 0, 0.7) * weight, 1
            else:  # burst
                p, steps = 0.85, 1
            if self.rng.random() >= p:
                continue
            fid = int(f["id"])
            ch = self._channel_for(fid)
            target = lo + (1.0 - float(f.get("y", 0.5))) * (hi - lo)
            last = self._last_note.get(fid)
            if last is not None:
                target = 0.6 * target + 0.4 * last  # melodic continuity
            cands = tones if (pos % 4 == 0 or not scale) and tones else scale
            note = nearest_in(target, cands)
            if note == last and self.rng.random() < 0.5:
                i = cands.index(note)
                note = cands[max(0, min(len(cands) - 1, i + self.rng.choice((-1, 1))))]
            self._last_note[fid] = note
            zf = f.get("z")
            near = 0.5 if zf is None else 1.0 - float(zf)
            vel = 38 + 55 * min(1.0, sp / max(c["burst"], 0.5)) + 22 * near
            self._cc(t, ch, 10, float(f.get("x", 0.5)) * 127)
            if zf is not None:
                self._cc(t, ch, 74, near * 127)
                self._cc(t, ch, 91, float(zf) * 127)
            self._play(t, ch, note, vel, steps * self.step_s * 0.95)

    def _phrase(self, t: float, e: dict[str, Any]) -> None:
        kind = e.get("kind", "")
        lead, s = self._ch(self.cfg.lead_channel), self.step_s
        pcs = chord_pcs(self.root, self.mode, self.degree, 4)
        scale = scale_notes(self.root, self.mode, 48, 96)
        arp = [n for n in scale if n % 12 in pcs]
        if kind == "feeding":
            if self.cfg.drums:
                self._play(t, self._ch(self.cfg.drums_channel), CRASH, 110, 4 * s)
            run = [n for n in arp if n >= 60][:12]
            for i, n in enumerate(run):
                self._play(t + i * s, lead, n, 70 + 3 * i, s * 0.9)
        elif kind == "feeding_end":
            for i, n in enumerate(sorted([n for n in arp if 60 <= n <= 84], reverse=True)[:4]):
                self._play(t + 2 * i * s, lead, n, 60 - 6 * i, 2 * s)
        elif kind == "chase":
            fish = e.get("fish") or []
            ch_a = self._channel_for(fish[0]) if len(fish) > 0 else lead
            ch_b = self._channel_for(fish[1]) if len(fish) > 1 else lead
            hi = [n for n in scale if 72 <= n <= 91]
            for i, n in enumerate(sorted(hi, reverse=True)[:4]):  # the chaser calls...
                self._play(t + i * s, ch_a, n, 95, s * 0.8)
            for i, n in enumerate(hi[:4]):  # ...the target answers
                self._play(t + (5 + i) * s, ch_b, n, 85, s * 0.8)
        elif kind in ("new_resident", "return"):
            top = [n for n in arp if 79 <= n <= 96][:2]
            for n in top:
                self._play(t, lead, n, 92, 8 * s)
        elif kind == "record":
            for i, n in enumerate([n for n in scale if n >= 67][:8]):
                self._play(t + i * s / 2, lead, n, 80 + 4 * i, s * 0.45)
        elif kind == "hiding":
            self._play(t, lead, 48 + (pcs[0] - 48) % 12, 40, 16 * s)
        elif kind in ("agitation", "tight_shoal", "surface_crowding", "lethargy"):
            base = 60 + (pcs[0] - 60) % 12
            for i in range(6):  # minor-second tremolo: something is off
                self._play(t + i * s, lead, base + (i % 2), 70, s * 0.8)
        elif kind == "schooling":
            target = nearest_in(74, arp) if arp else 74
            for ch in list(self._fish_channel.values())[: self.cfg.max_voices]:
                self._play(t, ch, target, 80, 4 * s)

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        return {
            "key": self.cfg.key,
            "mode": MODE_FR.get(self.mode, self.mode),
            "bpm": round(self.bpm, 1),
            "chord": chord_name(self.root, self.mode, self.degree),
            "roman": roman(self.mode, self.degree),
            "bar": self.step // 16,
            "notes": self.notes_played,
        }
