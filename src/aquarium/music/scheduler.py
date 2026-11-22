"""Shared MIDI plumbing for the music engines: a time-ordered event heap, note bookkeeping
(so nothing is ever left hanging on a synth), CC de-duplication and an all-notes-off panic."""

from __future__ import annotations

import heapq
import itertools
import random

import mido

from aquarium.music.midi_out import MidiSink


def clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class MidiScheduler:
    def __init__(self, sink: MidiSink, humanize: float = 0.0, seed: int | None = None):
        self.sink = sink
        self.humanize = humanize
        self.rng = random.Random(seed)
        self._timeline: list[tuple[float, int, mido.Message]] = []
        self._seq = itertools.count()
        self._active: set[tuple[int, int]] = set()
        self._cc_cache: dict[tuple[int, int], int] = {}
        self.notes_played = 0

    def _emit(self, msg: mido.Message, t: float) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            self._active.add((msg.channel, msg.note))
            self.notes_played += 1
        elif msg.type in ("note_off", "note_on"):
            self._active.discard((msg.channel, msg.note))
        self.sink.send(msg, t)

    def _at(self, t: float, msg: mido.Message) -> None:
        heapq.heappush(self._timeline, (t, next(self._seq), msg))

    def _next_event_t(self) -> float | None:
        return self._timeline[0][0] if self._timeline else None

    def _flush(self, until: float) -> None:
        while self._timeline and self._timeline[0][0] <= until:
            t, _, msg = heapq.heappop(self._timeline)
            if msg.type == "note_on" and (msg.channel, msg.note) in self._active:
                # retrigger: close the previous instance first so the synth doesn't drop it
                self._emit(mido.Message("note_off", channel=msg.channel, note=msg.note), t)
            self._emit(msg, t)

    def _play(self, t: float, ch: int, note: int, vel: float, dur: float) -> None:
        note = int(clip(note, 0, 127))
        if self.humanize:
            vel *= 1 + self.rng.uniform(-self.humanize, self.humanize)
        vel = int(clip(vel, 1, 127))
        self._at(t, mido.Message("note_on", channel=ch, note=note, velocity=vel))
        self._at(t + max(dur, 0.02), mido.Message("note_off", channel=ch, note=note, velocity=0))

    def _cc(self, t: float, ch: int, control: int, value: float, min_delta: int = 3) -> None:
        v = int(clip(value, 0, 127))
        key = (ch, control)
        if abs(self._cc_cache.get(key, -100) - v) >= min_delta:
            self._cc_cache[key] = v
            self._emit(mido.Message("control_change", channel=ch, control=control, value=v), t)

    def _panic(self, t: float, channels: set[int]) -> None:
        """Release every sounding note and send all-notes-off on the channels we used."""
        self._timeline.clear()
        for ch, note in list(self._active):
            self._emit(mido.Message("note_off", channel=ch, note=note, velocity=0), t)
        self._active.clear()
        for ch in sorted(channels):
            self._emit(mido.Message("control_change", channel=ch, control=123, value=0), t)
