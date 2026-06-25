"""Small absolute-date parser for deterministic temporal reasoning.

This module is deliberately narrow and stdlib-only. It handles common English
absolute dates until the language adapters expose a fuller temporal normalizer.
"""

from __future__ import annotations

import calendar
import re
from datetime import date


_MONTH_INDEX = {
    name.lower(): idx
    for idx, name in enumerate(calendar.month_name)
    if name
}
_MONTH_INDEX.update({
    name.lower(): idx
    for idx, name in enumerate(calendar.month_abbr)
    if name
})

_MONTH_RE = "|".join(sorted((re.escape(m) for m in _MONTH_INDEX), key=len, reverse=True))
_ORDINAL_SUFFIX_RE = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)

_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_RE})\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{_MONTH_RE})\.?(?:,)?\s+"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)


def parse_absolute_date(text: str) -> str | None:
    """Return an ISO date (YYYY-MM-DD) when *text* contains an absolute date."""

    normalized = _ORDINAL_SUFFIX_RE.sub("", text or "")
    for pattern in (_MONTH_DAY_YEAR_RE, _DAY_MONTH_YEAR_RE):
        match = pattern.search(normalized)
        if match is None:
            continue
        try:
            month = _MONTH_INDEX[match.group("month").lower().rstrip(".")]
            day = int(match.group("day"))
            year = int(match.group("year"))
            return date(year, month, day).isoformat()
        except (KeyError, ValueError):
            return None
    return None


def format_iso_date(iso_date: str) -> str:
    """Format an ISO date for display with a stable English month name."""

    parsed = date.fromisoformat(iso_date)
    return f"{calendar.month_name[parsed.month]} {parsed.day}, {parsed.year}"
