"""Adaptive score: the "video game" music mode.

You compose MIDI clips in your DAW; a TOML score says how the tank plays them:

* **sections** (horizontal re-sequencing): the tank's state picks the current section
  (``when`` expressions + priority). Switches wait for a musical boundary (beat / bar /
  phrase) and can go through a **bridge** clip specific to the transition.
* **layers** (vertical remixing): each section is a stack of looping clips whose gain follows
  live metrics. **Overlays** are layers that live outside sections, and layers shared by
  several sections keep playing across a switch: that is how sections intersect.
* **stingers**: interactions (a fish leaves the school and comes back, two fish meet, a dash,
  the surface is touched...) fire one-shot clips or single notes (a sample in a sampler),
  quantised to the grid, panned / transposed / scaled from the event itself.
* **signals**: game-style intensity meters fed by events and decaying over time
  (``excursions``, ``chase_heat``...), usable in any expression.

All loops are aligned on one global bar grid, so a layer shared between two sections never
jumps back to its start when the section changes.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mido

from aquarium.music.clips import Clip, load_clip, single_note
from aquarium.music.expr import Expr
from aquarium.music.midi_out import MidiSink
from aquarium.music.scheduler import MidiScheduler, clip

log = logging.getLogger(__name__)

STEPS_PER_BEAT = 4  # 16th-note grid


@dataclass
class LayerSpec:
    key: str
    clip: Clip
    gain: Expr | None = None
    fade_beats: float = 2.0


@dataclass
class SectionSpec:
    name: str
    layers: list[LayerSpec]
    when: Expr | None = None
    priority: int = 0
    min_bars: int = 2
    bpm: Expr | None = None
    on_enter: Clip | None = None


@dataclass
class TransitionSpec:
    src: str
    dst: str
    at: str = "bar"  # now | beat | bar | phrase
    bridge: Clip | None = None


@dataclass
class StingerSpec:
    on: set[str]
    clip: Clip
    when: Expr | None = None
    quantize: str = "16th"  # now | 16th | beat | bar
    gain: Expr | None = None
    transpose: Expr | None = None
    pan: Expr | None = None
    cooldown_s: float = 0.0
    per_fish: bool = True
    name: str = ""


@dataclass
class SignalSpec:
    name: str
    on: set[str]
    add: Expr
    half_life_s: float = 15.0
    max: float = math.inf
    when: Expr | None = None


@dataclass
class Score:
    name: str
    bpm: Expr
    beats_per_bar: int
    phrase_bars: int
    default_section: str
    sections: dict[str, SectionSpec]
    overlays: list[LayerSpec] = field(default_factory=list)
    transitions: list[TransitionSpec] = field(default_factory=list)
    stingers: list[StingerSpec] = field(default_factory=list)
    signals: dict[str, SignalSpec] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> Score:
        path = Path(path)
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        base = path.parent
        meta = data.get("score", {})
        bpb = int(meta.get("beats_per_bar", 4))
        cache: dict[tuple, Clip] = {}

        def clip_of(spec: dict[str, Any], where: str) -> Clip:
            if "clip" in spec:
                key = (spec["clip"], spec.get("bars"), spec.get("channel"))
                if key not in cache:
                    cache[key] = load_clip(base / spec["clip"], bpb, spec.get("bars"),
                                           spec.get("channel"))
                return cache[key]
            if "note" in spec:
                return single_note(int(spec["note"]), int(spec.get("channel", 10)),
                                   int(spec.get("velocity", 100)), float(spec.get("beats", 0.25)))
            raise ValueError(f"{where}: needs 'clip = \"file.mid\"' or 'note = <0-127>'")

        def opt(spec: dict[str, Any], key: str) -> Expr | None:
            return Expr(spec[key]) if key in spec else None

        def layer(spec: dict[str, Any], where: str) -> LayerSpec:
            c = clip_of(spec, where)
            key = spec.get("name") or f"{spec.get('clip', c.name)}@{spec.get('channel', '')}"
            return LayerSpec(key, c, opt(spec, "gain"), float(spec.get("fade_beats", 2.0)))

        sections = {}
        for name, s in data.get("section", {}).items():
            sections[name] = SectionSpec(
                name=name,
                layers=[layer(ls, f"section.{name}") for ls in s.get("layers", [])],
                when=opt(s, "when"), priority=int(s.get("priority", 0)),
                min_bars=int(s.get("min_bars", 2)), bpm=opt(s, "bpm"),
                on_enter=clip_of(s["on_enter"], f"section.{name}.on_enter") if "on_enter" in s else None,
            )
        if not sections:
            raise ValueError(f"{path}: a score needs at least one [section.<name>]")
        default = meta.get("default_section") or next(iter(sections))
        if default not in sections:
            raise ValueError(f"{path}: default_section '{default}' is not a section")
        transitions = [
            TransitionSpec(t.get("from", "*"), t.get("to", "*"), t.get("at", "bar"),
                           clip_of(t["bridge"], "transition") if "bridge" in t else None)
            for t in data.get("transition", [])
        ]
        for t in transitions:
            if t.at not in ("now", "beat", "bar", "phrase"):
                raise ValueError(f"transition {t.src}->{t.dst}: 'at' must be now/beat/bar/phrase")
        stingers = []
        for i, s in enumerate(data.get("stinger", [])):
            on = s["on"]
            stingers.append(StingerSpec(
                on={on} if isinstance(on, str) else set(on), clip=clip_of(s, f"stinger[{i}]"),
                when=opt(s, "when"), quantize=s.get("quantize", "16th"), gain=opt(s, "gain"),
                transpose=opt(s, "transpose"), pan=opt(s, "pan"),
                cooldown_s=float(s.get("cooldown_s", 0.0)), per_fish=bool(s.get("per_fish", True)),
                name=s.get("name", f"{'/'.join(sorted({on} if isinstance(on, str) else on))}#{i}"),
            ))
        signals = {
            name: SignalSpec(name, {s["on"]} if isinstance(s["on"], str) else set(s["on"]),
                             Expr(s.get("add", 1.0)), float(s.get("half_life_s", 15.0)),
                             float(s.get("max", math.inf)), opt(s, "when"))
            for name, s in data.get("signal", {}).items()
        }
        return cls(
            name=meta.get("name", path.stem), bpm=Expr(meta.get("bpm", 90)), beats_per_bar=bpb,
            phrase_bars=int(meta.get("phrase_bars", 4)), default_section=default,
            sections=sections, overlays=[layer(o, "overlay") for o in data.get("overlay", [])],
            transitions=transitions, stingers=stingers, signals=signals,
        )


class ScorePlayer(MidiScheduler):
    def __init__(self, score: Score, sink: MidiSink, humanize: float = 0.03,
                 seed: int | None = None, send_clock: bool = False):
        super().__init__(sink, humanize=humanize, seed=seed)
        self.score = score
        self.send_clock = send_clock
        self._lock = threading.Lock()
        self._fish: list[dict[str, Any]] = []
        self._snap: dict[str, Any] = {}
        self._queue: list[dict[str, Any]] = []
        self.step = 0
        self.next_step_t: float | None = None
        self.bpm = float(score.bpm.num({}, 90.0)) or 90.0
        self.section = score.default_section
        self.section_start_step = 0
        self.pending: tuple[str, int] | None = None  # (target, switch step)
        self.bridge: tuple[Clip, float, str] | None = None  # clip, start beat, target section
        self._enter: tuple[Clip, float] | None = (
            (score.sections[self.section].on_enter, 0.0)
            if score.sections[self.section].on_enter else None)
        self.gains: dict[str, float] = {}
        self._layers: dict[str, LayerSpec] = {}
        self.signals: dict[str, tuple[float, float]] = {}
        self._stinger_last: dict[tuple, float] = {}
        self.stingers_fired = 0
        self.last_stinger = ""
        self.history: list[tuple[int, str]] = [(0, self.section)]  # (bar, section) changes
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ inputs
    def set_state(self, fish: list[dict[str, Any]], snapshot: dict[str, Any] | None) -> None:
        with self._lock:
            self._fish = fish
            if snapshot:
                self._snap = snapshot

    def trigger(self, events: list[dict[str, Any]]) -> None:
        with self._lock:
            self._queue.extend(events)

    # ------------------------------------------------------------------ clock
    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def step_s(self) -> float:
        return self.beat_s / STEPS_PER_BEAT

    @property
    def steps_per_bar(self) -> int:
        return STEPS_PER_BEAT * self.score.beats_per_bar

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="score-player")
        self._thread.start()

    def _run(self) -> None:
        if self.send_clock:
            self._emit(mido.Message("start"), time.perf_counter())
        while not self._stop.is_set():
            now = time.perf_counter()
            self.advance(now)
            nxt = self.next_step_t if self.next_step_t is not None else now + 0.01
            ev = self._next_event_t()
            if ev is not None:
                nxt = min(nxt, ev)
            time.sleep(clip(nxt - time.perf_counter(), 0.0005, 0.01))

    def advance(self, now: float) -> None:
        if self.next_step_t is None:
            self.next_step_t = now
        self._handle_events(now)
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
        self._panic(t, {ch for ch, _ in self._active} | self._used_channels())
        if self.send_clock:
            self._emit(mido.Message("stop"), t)

    def _used_channels(self) -> set[int]:
        s = self.score
        clips = [ly.clip for sec in s.sections.values() for ly in sec.layers]
        clips += [ly.clip for ly in s.overlays] + [st.clip for st in s.stingers]
        return {n.channel for c in clips for n in c.notes}

    # ------------------------------------------------------------------ environment
    def _signal(self, name: str, t: float) -> float:
        v, t0 = self.signals.get(name, (0.0, t))
        spec = self.score.signals.get(name)
        if spec is None or spec.half_life_s <= 0:
            return v
        return v * 0.5 ** (max(t - t0, 0.0) / spec.half_life_s)

    def env(self, t: float) -> dict[str, Any]:
        with self._lock:
            snap, fish = self._snap, self._fish
        m = snap.get("metrics") or {}
        e: dict[str, Any] = {k: (0.0 if v is None else v) for k, v in m.items()}
        e.update({f"{k}_z": v for k, v in (snap.get("zscores") or {}).items()})
        e["fish_count"] = len(fish) if fish else snap.get("fish_count", 0)
        e["feeding"] = bool(snap.get("feeding"))
        e["mood"] = (snap.get("mood") or {}).get("label", "")
        for k, v in (snap.get("states") or {}).items():
            e[f"n_{k}"] = v
        hour = datetime.now().hour
        e["hour"], e["night"] = hour, hour >= 22 or hour < 7
        e["section"] = self.section
        e["section_bars"] = (self.step - self.section_start_step) // self.steps_per_bar
        e["bar"] = self.step // self.steps_per_bar
        for name in self.score.signals:
            e[name] = self._signal(name, t)
        return e

    # ------------------------------------------------------------------ events & stingers
    def _handle_events(self, now: float) -> None:
        with self._lock:
            events, self._queue = self._queue, []
        if not events:
            return
        base = self.env(now)
        for ev in events:
            kind = ev.get("kind", "")
            env = {**base, **(ev.get("data") or {}), "kind": kind, "n_fish": len(ev.get("fish") or [])}
            for name, sig in self.score.signals.items():
                if kind in sig.on and (sig.when is None or sig.when(env)):
                    v = min(self._signal(name, now) + sig.add.num(env, 1.0), sig.max)
                    self.signals[name] = (v, now)
                    base[name] = v
            for st in self.score.stingers:
                if kind not in st.on or (st.when is not None and not st.when(env)):
                    continue
                fish = (ev.get("fish") or [None])[0] if st.per_fish else None
                key = (st.name, fish)
                last = self._stinger_last.get(key)
                if last is not None and now - last < st.cooldown_s:
                    continue
                self._stinger_last[key] = now
                self._fire(st, env, now)

    def _quantized_time(self, q: str, now: float) -> float:
        if q == "now" or self.next_step_t is None:
            return now
        per = {"16th": 1, "beat": STEPS_PER_BEAT, "bar": self.steps_per_bar}.get(q, 1)
        target = -(-self.step // per) * per  # next multiple of `per`, at or after the next step
        return self.next_step_t + (target - self.step) * self.step_s

    def _fire(self, st: StingerSpec, env: dict[str, Any], now: float) -> None:
        t0 = self._quantized_time(st.quantize, now)
        gain = clip(st.gain.num(env, 1.0), 0.0, 1.5) if st.gain else 1.0
        if gain <= 0.01:
            return
        shift = int(round(st.transpose.num(env))) if st.transpose else 0
        if st.pan is not None:
            pan = clip(st.pan.num(env, 0.5), 0.0, 1.0) * 127
            for ch in {n.channel for n in st.clip.notes}:
                self._cc(max(t0 - 0.002, now), ch, 10, pan, min_delta=1)
        for n in st.clip.notes:
            self._play(t0 + n.start * self.beat_s, n.channel, n.note + shift, n.velocity * gain,
                       n.dur * self.beat_s)
        self.stingers_fired += 1
        self.last_stinger = st.name

    # ------------------------------------------------------------------ sections
    def _select(self, env: dict[str, Any]) -> str:
        best: SectionSpec | None = None
        for sec in self.score.sections.values():
            if sec.when is None or not sec.when(env):
                continue
            if best is None or sec.priority > best.priority or (
                    sec.priority == best.priority and sec.name == self.section):
                best = sec
        return best.name if best else self.score.default_section

    def _transition(self, src: str, dst: str) -> TransitionSpec:
        for a, b in ((src, dst), ("*", dst), (src, "*"), ("*", "*")):
            for tr in self.score.transitions:
                if tr.src == a and tr.dst == b:
                    return tr
        return TransitionSpec(src, dst)

    def _boundary(self, at: str) -> int:
        if at == "now":
            return self.step
        per = {"beat": STEPS_PER_BEAT, "bar": self.steps_per_bar,
               "phrase": self.steps_per_bar * self.score.phrase_bars}[at]
        return (self.step // per + 1) * per

    def _enter_section(self, name: str) -> None:
        if name != self.section:
            self.section = name
            self.section_start_step = self.step
            self.history.append((self.step // self.steps_per_bar, name))
            sec = self.score.sections[name]
            self._enter = (sec.on_enter, self.step / STEPS_PER_BEAT) if sec.on_enter else None
            log.info("score: section -> %s (bar %d)", name, self.step // self.steps_per_bar)

    def _do_step(self, t: float) -> None:
        on_beat = self.step % STEPS_PER_BEAT == 0
        if on_beat:
            env = self.env(t)
            # 1) apply a due switch / end of bridge
            if self.bridge is not None:
                bclip, bstart, target = self.bridge
                if self.step / STEPS_PER_BEAT >= bstart + bclip.length:
                    self.bridge = None
                    self._enter_section(target)
            if self.pending is not None and self.step >= self.pending[1]:
                target = self.pending[0]
                self.pending = None
                tr = self._transition(self.section, target)
                if tr.bridge is not None:
                    self.bridge = (tr.bridge, self.step / STEPS_PER_BEAT, target)
                    log.info("score: bridge %s -> %s", self.section, target)
                else:
                    self._enter_section(target)
            # 2) tempo, at bar starts
            if self.step % self.steps_per_bar == 0:
                sec = self.score.sections[self.section]
                bpm = (sec.bpm or self.score.bpm).num(env, self.bpm)
                self.bpm = clip(bpm, 30.0, 220.0)
            # 3) choose where to go next
            if self.bridge is None:
                target = self._select(env)
                cur = self.score.sections[self.section]
                if target == self.section:
                    self.pending = None
                else:
                    bars_in = (self.step - self.section_start_step) // self.steps_per_bar
                    tgt = self.score.sections[target]
                    if bars_in >= cur.min_bars or tgt.priority > cur.priority:
                        at = self._transition(self.section, target).at
                        when = self._boundary(at)
                        if self.pending is None or self.pending[0] != target:
                            self.pending = (target, when if self.pending is None else self.pending[1])
                        if when == self.step:  # "now": switch on this very step
                            self.pending = None
                            self._enter_section(target)
            # 4) layer gains
            self._update_gains(env)
        self._schedule_notes(t)
        if self.send_clock:
            for k in range(6):  # 24 ppqn = 6 per 16th
                self._at(t + k * self.step_s / 6, mido.Message("clock"))
        self.step += 1

    def _active_layers(self) -> list[LayerSpec]:
        cur = self.score.sections[self.section].layers
        if self.bridge is not None:  # only what both sections share keeps playing over a bridge
            nxt_keys = {ly.key for ly in self.score.sections[self.bridge[2]].layers}
            cur = [ly for ly in cur if ly.key in nxt_keys]
        return cur + self.score.overlays

    def _update_gains(self, env: dict[str, Any]) -> None:
        active = {ly.key: ly for ly in self._active_layers()}
        self._layers.update(active)
        for key, ly in self._layers.items():
            target = clip(ly.gain.num(env, 1.0), 0.0, 1.0) if (key in active and ly.gain) else (
                1.0 if key in active else 0.0)
            g = self.gains.get(key, 0.0)
            if key not in self.gains and key in active:
                g = target  # first appearance at a boundary: start at its level
            alpha = 1.0 if ly.fade_beats <= 0 else 1.0 - math.exp(-1.0 / ly.fade_beats)
            self.gains[key] = g + (target - g) * alpha

    def _schedule_notes(self, t: float) -> None:
        a = self.step / STEPS_PER_BEAT
        b = a + 1.0 / STEPS_PER_BEAT
        for key, g in self.gains.items():
            if g < 0.04:
                continue
            vel_scale = math.sqrt(g)
            for start, n in self._layers[key].clip.window(a, b):
                self._play(t + (start - a) * self.beat_s, n.channel, n.note,
                           n.velocity * vel_scale, n.dur * self.beat_s * 0.98)
        for once in (self.bridge, self._enter):
            if once is None:
                continue
            c, origin = once[0], once[1]
            for start, n in c.once(a, b, origin):
                self._play(t + (start - a) * self.beat_s, n.channel, n.note, n.velocity,
                           n.dur * self.beat_s * 0.98)

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        return {
            "mode": "score",
            "score": self.score.name,
            "section": self.section,
            "next": self.pending[0] if self.pending else (self.bridge[2] if self.bridge else None),
            "bridge": self.bridge is not None,
            "layers": {k.split("@")[0].split("/")[-1].removesuffix(".mid"): round(g, 2)
                       for k, g in self.gains.items() if g >= 0.04},
            "bpm": round(self.bpm, 1),
            "bar": self.step // self.steps_per_bar,
            "notes": self.notes_played,
            "stingers": self.stingers_fired,
            "last_stinger": self.last_stinger,
            "signals": {k: round(self._signal(k, time.perf_counter()), 2) for k in self.score.signals},
        }
