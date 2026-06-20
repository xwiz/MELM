"""Geospatial primitives for reasoning (slice 9). Stdlib-only."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def next_weekday(now: datetime, target_idx: int, hour: int = 8) -> datetime:
    """Next datetime whose weekday == target_idx (Mon=0..Sun=6), at *hour*:00.

    If today is the target weekday, returns today at *hour*:00.
    """
    days_ahead = (target_idx - now.weekday()) % 7
    target = now + timedelta(days=days_ahead)
    return target.replace(hour=hour, minute=0, second=0, microsecond=0)
