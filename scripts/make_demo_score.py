"""Generate the placeholder MIDI clips of scores/demo (D dorian, 4/4).

    uv run scripts/make_demo_score.py

They are deliberately simple: replace them with your own clips (same file names, or edit
scores/demo/score.toml). Channels follow General MIDI so a quick preview with any GM
soundfont makes sense: 1 pad, 2 bass, 3 bells/arps, 4 lead, 5 ostinato, 6 drone, 10 drums.
"""

from __future__ import annotations

from pathlib import Path

import mido

OUT = Path(__file__).resolve().parents[1] / "scores" / "demo" / "clips"
TPB = 480
D, E, F, G, A, B, C = 62, 64, 65, 67, 69, 71, 72  # D dorian around middle C


def write(name: str, notes: list[tuple[float, float, int, int, int]], bars: float) -> None:
    """notes: (start_beat, dur_beats, note, velocity, channel 1-16)."""
    events = []
    for start, dur, note, vel, ch in notes:
        events.append((round(start * TPB), 1, mido.Message("note_on", channel=ch - 1, note=note, velocity=vel)))
        events.append((round((start + dur) * TPB), 0, mido.Message("note_off", channel=ch - 1, note=note, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    last = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - last))
        last = tick
    end = round(bars * 4 * TPB)
    track.append(mido.MetaMessage("end_of_track", time=max(0, end - last)))
    mid = mido.MidiFile(ticks_per_beat=TPB)
    mid.tracks.append(track)
    OUT.mkdir(parents=True, exist_ok=True)
    mid.save(OUT / f"{name}.mid")


def chord(start: float, dur: float, notes: list[int], vel: int, ch: int) -> list:
    return [(start, dur, n, vel, ch) for n in notes]


def main() -> None:
    prog = [[D - 12, F, A, C], [G - 12, B - 12, D, F], [A - 12, C, E, G], [C - 12, E, G, B]]
    bass_roots = [D - 24, G - 24, A - 24, C - 24]

    # --- loops (4 bars) ---------------------------------------------------------------
    write("pad", [n for i, c in enumerate(prog) for n in chord(i * 4, 3.9, c, 58, 1)], 4)
    write("night_pad", [n for i, c in enumerate([prog[0], prog[2]]) for n in chord(i * 8, 7.8, c, 46, 1)], 4)
    write("bass", [(i * 4 + b, 0.9 if b else 1.8, r + (7 if b == 2.5 else 0), 92 if b == 0 else 72, 2)
                   for i, r in enumerate(bass_roots) for b in (0, 2.5, 3)], 4)
    write("bass_busy", [(i * 4 + k * 0.5, 0.4, r + (12 if k % 4 == 3 else 0), 100 if k % 2 == 0 else 78, 2)
                        for i, r in enumerate(bass_roots) for k in range(8)], 4)
    arp_pattern = [0, 1, 2, 3, 2, 1, 2, 3]
    write("arps", [(i * 4 + k * 0.5, 0.45, c[arp_pattern[k]] + 12, 64 + (12 if k == 0 else 0), 3)
                   for i, c in enumerate(prog) for k in range(8)], 4)
    write("ostinato", [(i * 4 + k * 0.25, 0.2, (c[0] if k % 2 == 0 else c[2]) + 12, 70 if k % 4 == 0 else 52, 5)
                       for i, c in enumerate(prog) for k in range(16)], 4)
    write("drums_light", [(b * 4 + k, 0.1, 36, 90, 10) for b in range(4) for k in (0, 2.5)]
          + [(b * 4 + k * 0.5, 0.1, 42, 50 if k % 2 else 64, 10) for b in range(4) for k in range(8)], 4)
    write("drums_groove", [(b * 4 + k, 0.1, 36, 105, 10) for b in range(4) for k in (0, 1.5, 2.5)]
          + [(b * 4 + k, 0.1, 38, 96, 10) for b in range(4) for k in (1, 3)]
          + [(b * 4 + k * 0.25, 0.1, 42, 70 if k % 2 == 0 else 48, 10) for b in range(4) for k in range(16)]
          + [(b * 4 + 3.75, 0.1, 46, 70, 10) for b in range(4)], 4)
    write("feast_lead", [(0, 1, A + 12, 96, 4), (1, 0.5, C + 12, 90, 4), (1.5, 0.5, D + 12, 94, 4),
                         (2, 2, E + 12, 100, 4), (4, 1, D + 12, 90, 4), (5, 1, C + 12, 86, 4),
                         (6, 2, A + 12, 92, 4), (8, 0.5, G + 12, 88, 4), (8.5, 0.5, A + 12, 90, 4),
                         (9, 1, C + 12, 96, 4), (10, 2, D + 12, 100, 4), (12, 4, D + 12, 92, 4)], 4)
    write("chase_pulse", [(b * 4 + k * 0.25, 0.12, (D - 24) if k % 4 else (D - 12), 96 if k % 4 == 0 else 70, 5)
                          for b in range(4) for k in range(16)], 4)
    write("shimmer", [(b * 4 + k * 0.75, 0.3, n, 50, 3)
                      for b in range(4) for k, n in enumerate([D + 24, A + 12, E + 24, C + 24, F + 24])
                      if k * 0.75 < 4], 4)
    write("drone", [(0, 15.9, D - 24, 70, 6), (0, 15.9, A - 24, 60, 6)], 4)

    # --- bridges / section entries (1 bar) -----------------------------------------------
    write("riser", [(k * 0.25, 0.22, n, 60 + 4 * k, 4)
                    for k, n in enumerate([D, E, F, G, A, B, C, D + 12, E + 12, F + 12, G + 12, A + 12,
                                           B + 12, C + 12, D + 24, E + 24])]
          + [(3.5, 0.5, 49, 110, 10)], 1)
    write("feast_out", [(k * 0.5, 0.45, n, 90 - 8 * k, 4)
                        for k, n in enumerate([A + 12, G + 12, F + 12, E + 12, D + 12, C + 12, A, D])], 1)
    write("night_enter", [(0, 3.5, D + 24, 60, 3), (1, 2.5, A + 12, 50, 3)], 1)

    # --- stingers -------------------------------------------------------------------------
    write("welcome_back", [(0, 0.25, A + 12, 80, 3), (0.25, 0.25, D + 24, 90, 3), (0.5, 1.5, E + 24, 70, 3)], 1)
    write("encounter", [(0, 0.5, D + 12, 85, 3), (0.25, 0.75, A + 12, 75, 3)], 1)
    write("dash_roll", [(k * 0.125, 0.1, 42, 50 + 8 * k, 10) for k in range(8)], 1)
    bubbles = [D + 24, F + 24, A + 24, C + 36, E + 36]
    write("bubbles", [(k * 0.125, 0.1, n, 70 - 5 * k, 3) for k, n in enumerate(bubbles)], 1)
    write("split_stab", chord(0, 1.0, [D, F, A + 12, C + 12], 100, 4), 1)
    write("merge_resolve", chord(0, 2.0, [D, A, D + 12, F + 12], 90, 4), 1)
    print(f"clips written to {OUT}")


if __name__ == "__main__":
    main()
