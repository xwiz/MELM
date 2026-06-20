"""Pure-Python MIDI generation engine for MELM Assistant OS.

No external dependencies — uses only Python stdlib (struct, bytes/bytearray).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class MusicDescription:
    """Describes a piece of music to generate."""

    genre: str = "classical"
    tempo_bpm: int = 120
    key: str = "C"
    mode: str = "major"
    mood: str = "calm"
    length_bars: int = 16
    chord_complexity: str = "simple"


# ---------------------------------------------------------------------------
# Internal low-level MIDI file builder
# ---------------------------------------------------------------------------

class _MidiWriter:
    """Low-level MIDI file builder (format 1, variable-length quantitites)."""

    def __init__(self) -> None:
        self._tracks: list[bytes] = []

    def add_track(self, track_bytes: bytes) -> None:
        self._tracks.append(track_bytes)

    def note_on(self, channel: int, note: int, velocity: int,
                delta_time: int) -> bytes:
        return (self._encode_varint(delta_time)
                + bytes([0x90 | channel, note, velocity]))

    def note_off(self, channel: int, note: int, velocity: int,
                 delta_time: int) -> bytes:
        return (self._encode_varint(delta_time)
                + bytes([0x80 | channel, note, velocity]))

    def tempo_event(self, microseconds_per_beat: int) -> bytes:
        packed = struct.pack(">I", microseconds_per_beat)
        return b'\x00\xFF\x51\x03' + packed[1:]

    def end_of_track(self) -> bytes:
        return b'\x00\xFF\x2F\x00'

    def build(self) -> bytes:
        header = struct.pack(">4sIHHH", b"MThd", 6, 1, len(self._tracks), 480)
        result = bytearray(header)
        for track_bytes in self._tracks:
            chunk = struct.pack(">4sI", b"MTrk", len(track_bytes)) + track_bytes
            result.extend(chunk)
        return bytes(result)

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        if value < 0:
            value = 0
        if value < 128:
            return bytes([value])
        buf = bytearray()
        buf.append(value & 0x7F)
        value >>= 7
        while value > 0:
            buf.append(0x80 | (value & 0x7F))
            value >>= 7
        buf.reverse()
        return bytes(buf)


# ---------------------------------------------------------------------------
# Stateless music theory utilities
# ---------------------------------------------------------------------------

class MusicTheoryEngine:
    """Stateless music theory functions (scales, chords, progressions)."""

    _SCALE_INTERVALS: ClassVar[dict[str, tuple[int, ...]]] = {
        "major": (0, 2, 4, 5, 7, 9, 11),
        "minor": (0, 2, 3, 5, 7, 8, 10),
    }

    _CHORD_INTERVALS: ClassVar[dict[str, tuple[int, ...]]] = {
        "major": (0, 4, 7),
        "minor": (0, 3, 7),
        "dim": (0, 3, 6),
        "aug": (0, 4, 8),
        "seventh": (0, 4, 7, 10),
        "minor_seventh": (0, 3, 7, 10),
        "major_seventh": (0, 4, 7, 11),
    }

    _PROGRESSIONS: ClassVar[dict[str, list[tuple[int, str]]]] = {
        "simple": [(0, "major"), (3, "major"), (4, "major"), (0, "major")],
        "classical": [
            (0, "major"),
            (5, "minor"),
            (3, "major"),
            (4, "major"),
            (0, "major"),
        ],
        "jazz": [(1, "minor"), (4, "major"), (0, "major")],
        "modal": [(0, "minor"), (3, "minor"), (0, "minor")],
    }

    _PREFERRED_QUALITY: ClassVar[dict[str, str]] = {
        "waltz": "major",
        "jazz": "seventh",
    }

    _KEY_SEMITONES: ClassVar[dict[str, int]] = {
        "C": 0, "G": 7, "D": 2, "A": 9, "E": 4, "F": 5, "Bb": 10,
    }

    def scale_notes(self, key: str, mode: str, octave: int = 4) -> list[int]:
        intervals = self._SCALE_INTERVALS.get(mode, self._SCALE_INTERVALS["major"])
        root = self._KEY_SEMITONES.get(key, 0) + (octave + 1) * 12
        return [root + i for i in intervals]

    def chord_notes(self, root: int, quality: str = "major") -> list[int]:
        intervals = self._CHORD_INTERVALS.get(quality, self._CHORD_INTERVALS["major"])
        return [root + i for i in intervals]

    def progression(self, key: str, mode: str,
                    style: str = "simple") -> list[tuple[int, str]]:
        _ = key, mode
        return list(self._PROGRESSIONS.get(style, self._PROGRESSIONS["simple"]))

    def chord_quality_for_style(self, style: str) -> str:
        return self._PREFERRED_QUALITY.get(style, "major")


# ---------------------------------------------------------------------------
# Genre-to-progression mapping and melody patterns
# ---------------------------------------------------------------------------

_GENRE_TO_PROGRESSION: dict[str, str] = {
    "classical": "classical",
    "waltz": "simple",
    "lullaby": "simple",
    "jazz": "jazz",
    "ambient": "modal",
}

_MELODY_PATTERNS: dict[str, tuple[int, ...]] = {
    "intro": (0, 2, 4, 3, 1, 0, 2, 4, 5, 4, 3, 2, 1, 0, 2, 3),
    "main": (
        0, 2, 4, 5, 6, 5, 4, 3,
        2, 1, 0, 1, 2, 3, 4, 3,
        4, 5, 6, 5, 4, 3, 2, 1,
        0, 2, 4, 3, 2, 1, 0, 0,
    ),
    "outro": (4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 0, 1, 0, 0, 0),
}

_TICKS_PER_BEAT = 480
_BEATS_PER_BAR = 4
_MELODY_VELOCITY = 80
_CHORD_VELOCITY = 60


class _ProgressionResolver:
    """Resolves a MusicDescription into per-bar chord note sets."""

    @staticmethod
    def resolve(description: MusicDescription) -> list[list[int]]:
        engine = MusicTheoryEngine()
        scale = engine.scale_notes(description.key, description.mode)
        prog_style = _GENRE_TO_PROGRESSION.get(description.genre, "simple")
        progression = engine.progression(description.key, description.mode,
                                         prog_style)

        num_chords = len(progression)
        bars = description.length_bars
        if num_chords == 0:
            return [[60, 64, 67]] * bars

        bars_per_chord = bars // num_chords
        remainder = bars % num_chords

        result: list[list[int]] = []
        for i, (degree, quality) in enumerate(progression):
            root = scale[degree % len(scale)]
            chord = engine.chord_notes(root - 12, quality)
            count = bars_per_chord + (1 if i < remainder else 0)
            for _ in range(count):
                result.append(chord)

        return result


class MidiRenderer:
    """Renders a MusicDescription into a playable MIDI file (bytes)."""

    @staticmethod
    def render(description: MusicDescription) -> bytes:
        engine = MusicTheoryEngine()
        writer = _MidiWriter()

        tempo_us = 60_000_000 // description.tempo_bpm
        scale = engine.scale_notes(description.key, description.mode)
        bars = description.length_bars

        intro_bars = bars // 4
        main_bars = bars // 2
        outro_bars = bars - intro_bars - main_bars

        melody_notes: list[int] = []
        sections: list[tuple[str, int]] = [
            ("intro", intro_bars),
            ("main", main_bars),
            ("outro", outro_bars),
        ]
        for name, num_bars in sections:
            pattern = _MELODY_PATTERNS[name]
            total_notes = num_bars * _BEATS_PER_BAR
            for i in range(total_notes):
                idx = pattern[i % len(pattern)]
                safe_idx = idx % len(scale)
                melody_notes.append(scale[safe_idx])

        # -- Track 0: tempo + melody ---------------------------------------
        track0 = bytearray()
        track0 += writer.tempo_event(tempo_us)
        for note in melody_notes:
            track0 += writer.note_on(0, note, _MELODY_VELOCITY, 0)
            track0 += writer.note_off(0, note, _MELODY_VELOCITY,
                                      _TICKS_PER_BEAT)
        track0 += writer.end_of_track()
        writer.add_track(bytes(track0))

        # -- Track 1: chords (block chords on beat 1 of each bar) ----------
        chord_sets = _ProgressionResolver.resolve(description)
        track1 = bytearray()
        bar_ticks = _TICKS_PER_BEAT * _BEATS_PER_BAR
        for chord in chord_sets:
            for note in chord:
                track1 += writer.note_on(1, note, _CHORD_VELOCITY, 0)
            for i, note in enumerate(chord):
                delta = bar_ticks if i == 0 else 0
                track1 += writer.note_off(1, note, _CHORD_VELOCITY, delta)
        track1 += writer.end_of_track()
        writer.add_track(bytes(track1))

        return writer.build()
