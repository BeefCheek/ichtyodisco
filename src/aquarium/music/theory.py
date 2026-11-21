"""Just enough music theory: keys, modes, diatonic chords, quantisation."""

from __future__ import annotations

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

MODES: dict[str, list[int]] = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
}
MODE_FR = {"ionian": "majeur", "dorian": "dorien", "phrygian": "phrygien", "lydian": "lydien",
           "mixolydian": "mixolydien", "aeolian": "mineur"}

# How the tank's mood colours the music: (mode, chord progression as scale degrees, bars per chord)
MOOD_MUSIC: dict[str, tuple[str, list[int], int]] = {
    "serein": ("lydian", [0, 1, 0, 4], 2),  # I  II  I  V  — the bright, floating #4
    "joueur": ("mixolydian", [0, 6, 3, 0], 1),  # I  bVII  IV  I
    "excité": ("dorian", [0, 3, 0, 6], 1),  # i  IV  i  bVII
    "endormi": ("aeolian", [0, 5, 2, 6], 4),  # i  bVI  bIII  bVII, very slow
    "sur le qui-vive": ("phrygian", [0, 1, 0, 6], 1),  # i  bII — tension
    "festin": ("ionian", [0, 3, 4, 0], 1),  # I  IV  V  I
    "désert": ("aeolian", [0], 4),  # drone
}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


def note_number(name: str) -> int:
    """Pitch class of a key name ("D", "f#", "Bb")."""
    n = name.strip()
    n = n[0].upper() + n[1:]
    n = _FLATS.get(n, n)
    if n not in NOTE_NAMES:
        raise ValueError(f"unknown key: {name}")
    return NOTE_NAMES.index(n)


def scale_notes(root_pc: int, mode: str, lo: int = 0, hi: int = 127) -> list[int]:
    steps = MODES[mode]
    return [n for n in range(lo, hi + 1) if (n - root_pc) % 12 in steps]


def chord_pcs(root_pc: int, mode: str, degree: int, size: int = 4) -> list[int]:
    """Diatonic chord built by stacking thirds inside the mode (pitch classes, root first)."""
    steps = MODES[mode]
    return [(root_pc + steps[(degree + 2 * k) % len(steps)]) % 12 for k in range(size)]


def chord_name(root_pc: int, mode: str, degree: int) -> str:
    pcs = chord_pcs(root_pc, mode, degree, 3)
    third = (pcs[1] - pcs[0]) % 12
    fifth = (pcs[2] - pcs[0]) % 12
    quality = {(4, 7): "", (3, 7): "m", (3, 6): "dim", (4, 8): "aug"}.get((third, fifth), "sus")
    return NOTE_NAMES[pcs[0]] + quality


def roman(mode: str, degree: int) -> str:
    third = (MODES[mode][(degree + 2) % 7] - MODES[mode][degree]) % 12
    r = ROMAN[degree]
    return r if third == 4 else r.lower()


def nearest_in(target: float, candidates: list[int]) -> int:
    return min(candidates, key=lambda n: (abs(n - target), n))


def voice_chord(pcs: list[int], center: int, spread: float = 0.0) -> list[int]:
    """Place pitch classes around ``center``; spread 0 = close voicing, 1 = open (drop-2-ish)."""
    notes = []
    for i, pc in enumerate(pcs):
        n = center - 6 + ((pc - (center - 6)) % 12)
        if spread > 0.5 and i % 2 == 1:
            n += 12
        notes.append(n)
    return sorted(set(notes))
