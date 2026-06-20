"""Typed-value extraction (slice 5).

Sidecar extraction over raw text — does NOT mutate the frozen UOL. Emits typed
numbers and metalinguistic char-count targets the solvers consume. Deterministic
(regex + a fixed number-word table); no ML.
"""

from __future__ import annotations

import re

# Explicit number words only. Articles ("a"/"an") are deliberately excluded so
# "a shop" is not read as the quantity 1.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

_TOKEN_RE = re.compile(r"\b([A-Za-z]+|\d+(?:\.\d+)?)\b")
_DIGIT_RE = re.compile(r"^\d+(?:\.\d+)?$")


def extract_numbers(text: str) -> list[tuple[float, str, int]]:
    """Ordered numeric quantities as ``(value, surface, start_index)``.

    Recognises digits and the explicit number words above, preserving order.
    """
    out: list[tuple[float, str, int]] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(1)
        if _DIGIT_RE.match(tok):
            out.append((float(tok), tok, m.start()))
        else:
            val = _NUMBER_WORDS.get(tok.lower())
            if val is not None:
                out.append((float(val), tok, m.start()))
    return out


# "how many r's in strawberry", "how many letter a in banana",
# "how many letters e are there in the word excellence"
_CHAR_COUNT_RE = re.compile(
    r"how\s+many\s+(?:letter[s]?\s+|character[s]?\s+)?"
    r"['\"]?([A-Za-z])['\"]?(?:'s|s)?\s+(?:are\s+|is\s+)?(?:there\s+)?in\s+"
    r"(?:the\s+word\s+)?['\"]?([A-Za-z]+)['\"]?",
    re.IGNORECASE,
)


def extract_char_count_target(text: str) -> dict | None:
    """Return ``{"char","word"}`` for a metalinguistic count question, else None."""
    m = _CHAR_COUNT_RE.search(text)
    if not m:
        return None
    return {"char": m.group(1).lower(), "word": m.group(2).lower()}


_UNIT_TO_KM = {
    "m": 0.001, "meter": 0.001, "meters": 0.001, "metre": 0.001, "metres": 0.001,
    "km": 1.0, "kilometer": 1.0, "kilometre": 1.0, "kilometers": 1.0, "kilometres": 1.0,
    "mile": 1.609344, "miles": 1.609344, "mi": 1.609344,
    "ft": 0.0003048, "feet": 0.0003048, "foot": 0.0003048,
}
_DISTANCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(meters?|metres?|kilometers?|kilometres?|miles?|feet|foot|km|mi|ft|m)\b",
    re.IGNORECASE,
)


def extract_distance(text: str) -> dict | None:
    """Return ``{"value","unit","value_km"}`` for a distance measure, else None."""
    m = _DISTANCE_RE.search(text)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return {"value": value, "unit": unit, "value_km": value * _UNIT_TO_KM[unit]}
