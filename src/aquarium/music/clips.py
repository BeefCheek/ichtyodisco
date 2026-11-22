"""MIDI clips (loops, bridges, stingers) loaded from .mid files composed in a DAW."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mido


@dataclass(frozen=True, slots=True)
class Note:
    start: float  # beats from the clip start
    dur: float  # beats
    channel: int  # 0-15
    note: int
    velocity: int


@dataclass
class Clip:
    name: str
    notes: list[Note]
    length: float  # beats (a whole number of bars for loops)

    def window(self, a: float, b: float) -> list[tuple[float, Note]]:
        """Notes starting in [a, b) of an *infinitely looped* clip, with their absolute start."""
        if self.length <= 0 or not self.notes:
            return []
        out = []
        k = math.floor(a / self.length)
        while k * self.length < b:
            base = k * self.length
            for n in self.notes:
                s = base + n.start
                if a <= s < b:
                    out.append((s, n))
            k += 1
        return out

    def once(self, a: float, b: float, origin: float) -> list[tuple[float, Note]]:
        """Notes starting in [a, b) for a single (non-looped) play started at ``origin``."""
        return [(origin + n.start, n) for n in self.notes if a <= origin + n.start < b]


def load_clip(path: str | Path, beats_per_bar: int = 4, bars: float | None = None,
              channel: int | None = None) -> Clip:
    """Read a Standard MIDI File into beat-based notes.

    ``channel`` (1-16) forces every note onto one channel. The length is ``bars`` if given,
    else the end of the file rounded up to whole bars.
    """
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    pending: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    notes: list[Note] = []
    tick = 0
    end_tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        tick += msg.time
        end_tick = max(end_tick, tick)
        if msg.type == "note_on" and msg.velocity > 0:
            pending[(msg.channel, msg.note)].append((tick, msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            stack = pending.get((msg.channel, msg.note))
            if stack:
                start, vel = stack.pop(0)
                ch = msg.channel if channel is None else channel - 1
                notes.append(Note(start / tpb, max(tick - start, 1) / tpb, ch, msg.note, vel))
    for (ch, note), stack in pending.items():  # unterminated notes: close at file end
        for start, vel in stack:
            c = ch if channel is None else channel - 1
            notes.append(Note(start / tpb, max(end_tick - start, tpb // 4) / tpb, c, note, vel))
    notes.sort(key=lambda n: (n.start, n.channel, n.note))
    if bars is not None:
        length = float(bars) * beats_per_bar
    else:
        last = max([end_tick / tpb] + [n.start + n.dur for n in notes] + [1e-9])
        length = max(1, math.ceil(last / beats_per_bar - 1e-6)) * beats_per_bar
    return Clip(Path(path).stem, notes, length)


def single_note(note: int, channel: int, velocity: int = 100, beats: float = 0.25) -> Clip:
    """A one-shot that just fires a note: ideal to trigger a sample in a sampler / drum rack."""
    return Clip(f"note{note}", [Note(0.0, beats, channel - 1, note, velocity)], beats)
