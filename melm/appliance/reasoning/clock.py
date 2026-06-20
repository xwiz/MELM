"""Clock/calendar primitives for temporal reasoning (slice 7).

Single source of "now" (injectable for tests via monkeypatching ``now``).
Stdlib-only; no timezone database dependency — uses local wall-clock time, which
matches a user asking "what time is it?" on their own device.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def now() -> datetime:
    return datetime.now()


def format_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def format_date(dt: datetime) -> str:
    return dt.strftime("%A, %B %d, %Y")


def weekday_name(dt: datetime) -> str:
    return dt.strftime("%A")


def shift_days(dt: datetime, n: int) -> datetime:
    return dt + timedelta(days=int(n))
