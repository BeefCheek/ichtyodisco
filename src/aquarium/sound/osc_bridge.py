"""Send the aquarium to a synth (SuperCollider, Max/MSP, Pure Data, TouchDesigner...) over OSC.

Address space (all floats normalised 0..1 unless stated):
    /aq/fish        id x y z speed_bls state_index      one message per visible fish
    /aq/group       count activity cohesion polarization mean_z activity_z
    /aq/mood        label
    /aq/event       kind severity message                on every detected event
    /aq/frame       count                                 end-of-frame marker
State index: 0 rest, 1 cruise, 2 explore, 3 burst.

Ideas: z -> reverb / low-pass (far fish sound "underwater"), y -> pitch, speed -> amplitude,
activity_z -> tension of the harmony, events -> one-shot samples.
"""

from __future__ import annotations

import time
from typing import Any

from aquarium.analytics.behavior import STATES
from aquarium.types import AquariumEvent, TrackedFish


class OscBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 57120, rate_hz: float = 20.0):
        from pythonosc.udp_client import SimpleUDPClient

        self.client = SimpleUDPClient(host, port)
        self.min_period = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._last = 0.0

    def send_frame(self, fish: list[TrackedFish], frame_shape: tuple[int, int],
                   snapshot: dict[str, Any] | None) -> None:
        now = time.monotonic()
        if now - self._last < self.min_period:
            return
        self._last = now
        fh, fw = frame_shape
        info = {f["id"]: f for f in (snapshot or {}).get("fish", [])}
        for f in fish:
            cx, cy = f.center
            meta = info.get(f.identity, {})
            state = meta.get("state", "cruise")
            self.client.send_message("/aq/fish", [
                int(f.identity), cx / fw, cy / fh, -1.0 if f.z is None else float(f.z),
                float(meta.get("speed_bls", 0.0)), STATES.index(state) if state in STATES else 1,
            ])
        self.client.send_message("/aq/frame", [len(fish)])

    def send_snapshot(self, snap: dict[str, Any]) -> None:
        m, z = snap.get("metrics", {}), snap.get("zscores", {})
        self.client.send_message("/aq/group", [
            int(snap.get("fish_count", 0)), float(m.get("activity") or 0.0),
            float(m.get("cohesion") or 0.0), float(m.get("polarization") or 0.0),
            float(m.get("mean_z") if m.get("mean_z") is not None else -1.0),
            float(z.get("activity", 0.0)),
        ])
        self.client.send_message("/aq/mood", [snap.get("mood", {}).get("label", "")])

    def send_events(self, events: list[AquariumEvent]) -> None:
        for e in events:
            self.client.send_message("/aq/event", [e.kind, float(e.severity), e.message])
