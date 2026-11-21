"""Where the notes go: live MIDI ports (synths, DAW) and/or a .mid recording."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Protocol

import mido

log = logging.getLogger(__name__)


class MidiSink(Protocol):
    def send(self, msg: mido.Message, t: float) -> None: ...
    def close(self) -> None: ...


class PortSink:
    """A real MIDI output: a virtual port named e.g. "ichtyodisco" (macOS / Linux), or an
    existing port (USB synth, IAC Driver on macOS, loopMIDI on Windows) matched by name."""

    def __init__(self, name: str = "ichtyodisco", virtual: bool | None = None):
        names = mido.get_output_names()
        match = next((n for n in names if name.lower() in n.lower()), None)
        if match is not None and not virtual:
            self.port = mido.open_output(match)
            log.info("MIDI -> port '%s'", match)
        else:
            self.port = mido.open_output(name, virtual=True)
            log.info("MIDI -> virtual port '%s' (select it as input in your synth / DAW)", name)
        self._lock = threading.Lock()

    def send(self, msg: mido.Message, t: float) -> None:
        with self._lock:
            self.port.send(msg)

    def close(self) -> None:
        self.port.close()


class MidiFileRecorder:
    """Records everything to a Standard MIDI File (real time preserved), e.g. to rework a
    session in a DAW or render it later."""

    def __init__(self, path: str | Path, ticks_per_beat: int = 480):
        self.path = Path(path)
        self.tpb = ticks_per_beat
        self.events: list[tuple[float, mido.Message]] = []
        self._lock = threading.Lock()

    def send(self, msg: mido.Message, t: float) -> None:
        with self._lock:
            self.events.append((t, msg))

    def close(self) -> None:
        with self._lock:
            events = sorted(self.events, key=lambda e: e[0])
        mid = mido.MidiFile(type=0, ticks_per_beat=self.tpb)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        tempo = 500_000  # 120 bpm: 1 s = 2 beats = 2 * tpb ticks; real timing is kept as-is
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        track.append(mido.MetaMessage("track_name", name="ichtyodisco", time=0))
        t0 = events[0][0] if events else 0.0
        last = 0
        for t, msg in events:
            if msg.type == "clock":
                continue
            tick = int(round((t - t0) * 2 * self.tpb))
            track.append(msg.copy(time=max(0, tick - last)))
            last = max(last, tick)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mid.save(self.path)
        log.info("MIDI recording saved: %s (%d messages)", self.path, len(events))


class MultiSink:
    def __init__(self, sinks: list[MidiSink]):
        self.sinks = sinks

    def send(self, msg: mido.Message, t: float) -> None:
        for s in self.sinks:
            s.send(msg, t)

    def close(self) -> None:
        for s in self.sinks:
            try:
                s.close()
            except Exception:  # one failing sink must not lose the recording
                log.exception("closing MIDI sink failed")


def open_sinks(port: str | None, record_path: str | None) -> MultiSink:
    sinks: list[MidiSink] = []
    if port:
        try:
            sinks.append(PortSink(port))
        except (ImportError, OSError) as exc:  # no rtmidi backend / no MIDI subsystem
            log.warning("live MIDI unavailable (%s). Install `uv sync --extra midi`.", exc)
    if record_path:
        sinks.append(MidiFileRecorder(record_path))
    return MultiSink(sinks)


def list_ports() -> list[str]:
    try:
        return mido.get_output_names()
    except ImportError:
        return []
