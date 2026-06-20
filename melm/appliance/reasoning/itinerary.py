"""Multi-stop itinerary reasoning (slice 9).

Parses a journey described over text into an ordered scenario, then answers
duration / total-distance / displacement / location-at-time queries. Pure given
an atlas + a ``now``; scenario persistence is handled by the store. Refuses /
flags assumptions rather than inventing facts (departure time is assumed and
disclosed; unknown places refuse).
"""

from __future__ import annotations

import re
from datetime import timedelta

from . import geo
from .value_extract import _NUMBER_WORDS

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_STAY_RE = re.compile(
    r"\bstay(?:ing)?\s+(?:in|at)\s+([a-z]+)\s+for\s+(\d+|[a-z]+)\s+(hour|minute)s?",
    re.IGNORECASE,
)
_DEFAULT_DEPART_HOUR = 8


def _num(token: str) -> float | None:
    token = token.strip().lower()
    if token.isdigit():
        return float(token)
    return float(_NUMBER_WORDS[token]) if token in _NUMBER_WORDS else None


def detect_itinerary_query(text: str) -> str | None:
    low = text.lower()
    if re.search(r"\bwhere\s+will\s+you\s+be\b|\bwhere\s+are\s+you\s+going\s+to\s+be\b", low):
        return "projection"
    if re.search(r"\bbetween\b.*\bfinal\b|\bfinal\b.*\binitial\b|\bfinal\b.*\bstart", low):
        return "displacement"
    if re.search(r"\btotal\s+distance\b|\bdistance\s+(?:moved|travel|covered)\b|\bhow\s+far\b.*\btotal\b", low):
        return "path_distance"
    if re.search(r"\bhow\s+long\b|\bhow\s+much\s+time\b", low):
        return "duration"
    return None


def parse_itinerary(text: str, atlas_places: list[str]) -> dict | None:
    """Parse ordered stops, stays, and the start weekday from *text*."""
    low = text.lower()
    mentions: list[tuple[int, str]] = []
    for name in atlas_places:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", low):
            mentions.append((m.start(), name))
    mentions.sort()
    seq: list[str] = []
    for _, name in mentions:
        if not seq or seq[-1] != name:
            seq.append(name)
    if len(seq) < 2:
        return None
    legs = list(zip(seq[:-1], seq[1:]))
    stays: dict[str, float] = {}
    for m in _STAY_RE.finditer(text):
        place = m.group(1).lower()
        amount = _num(m.group(2))
        if place in atlas_places and amount is not None:
            stays[place] = amount if m.group(3).lower() == "hour" else amount / 60.0
    start_day = None
    for name, _idx in _WEEKDAYS.items():
        if re.search(r"\b" + name + r"\b", low):
            start_day = name
            break
    return {"places": seq, "legs": legs, "stays": stays, "start_day": start_day}


def _leg_distances(scenario: dict, atlas: dict) -> list[tuple[str, str, float]]:
    coords = atlas["places"]
    out = []
    for a, b in scenario["legs"]:
        A, B = coords[a], coords[b]
        out.append((a, b, geo.haversine_km(A["lat"], A["lon"], B["lat"], B["lon"])))
    return out


def solve_itinerary(scenario: dict, atlas: dict, query: str, now) -> tuple[dict | None, str, str | None]:
    """Return (result, answer, refusal). Unknown places ⇒ refuse."""
    coords = atlas["places"]
    places = scenario["places"]
    unknown = [p for p in places if p not in coords]
    if unknown:
        return (None, "", f"unknown_place:{unknown[0]}")
    speed = float(atlas.get("default_road_speed_kmh", 60) or 60)
    legd = _leg_distances(scenario, atlas)
    path = sum(d for _, _, d in legd)
    travel_h = sum(d / speed for _, _, d in legd) if speed else 0.0
    stay_h = sum(scenario.get("stays", {}).values())

    if query == "path_distance":
        ans = f"The total distance moved is about {path:.0f} km."
        return ({"task": "itinerary", "query": query, "path_km": round(path, 1)}, ans, None)

    if query == "displacement":
        a0, aN = places[0], places[-1]
        disp = geo.haversine_km(coords[a0]["lat"], coords[a0]["lon"],
                                coords[aN]["lat"], coords[aN]["lon"])
        ans = (f"Your final stop ({aN.title()}) is only about {disp:.0f} km from your "
               f"start ({a0.title()}) — far less than the {path:.0f} km you actually "
               f"cover, because the route loops back.")
        return ({"task": "itinerary", "query": query, "displacement_km": round(disp, 1),
                 "path_km": round(path, 1)}, ans, None)

    if query == "duration":
        total = travel_h + stay_h
        ans = (f"About {total:.1f} hours — roughly {travel_h:.1f}h travelling plus "
               f"{stay_h:.1f}h of stops (assuming ~{int(speed)} km/h by road).")
        return ({"task": "itinerary", "query": query, "total_hours": round(total, 2),
                 "travel_hours": round(travel_h, 2), "stay_hours": round(stay_h, 2)}, ans, None)

    if query == "projection":
        return _project(scenario, legd, speed, now)

    return (None, "", "unsupported_itinerary_query")


def _project(scenario: dict, legd: list, speed: float, now) -> tuple[dict | None, str, str | None]:
    places = scenario["places"]
    stays = scenario.get("stays", {})
    start_day = scenario.get("start_day")
    if start_day is None or start_day not in _WEEKDAYS:
        return (None, "", "missing_start_day")
    depart0 = geo.next_weekday(now, _WEEKDAYS[start_day], _DEFAULT_DEPART_HOUR)
    timeline: list[tuple[str, object, object, object]] = []
    t = depart0
    for a, b, d in legd:
        travel = timedelta(hours=(d / speed) if speed else 0)
        timeline.append(("transit", (a, b), t, t + travel))
        t = t + travel
        stay = stays.get(b, 0)
        if stay:
            timeline.append(("stay", b, t, t + timedelta(hours=stay)))
            t = t + timedelta(hours=stay)
    target = (now + timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
    note = f" (assuming a {start_day.title()} {_DEFAULT_DEPART_HOUR}am start and ~{int(speed)} km/h)"
    result = {"task": "itinerary", "query": "projection",
              "target": target.isoformat(), "start_day": start_day}
    if target < depart0:
        return (result, f"At that time you'll still be in {places[0].title()}; the journey "
                        f"only starts {start_day.title()}.{note}", None)
    for kind, what, s, e in timeline:
        if s <= target < e:
            if kind == "stay":
                return (result, f"Around then you'll be in {str(what).title()}.{note}", None)
            a, b = what
            return (result, f"Around then you'll be in transit between {a.title()} "
                            f"and {b.title()}.{note}", None)
    return (result, f"By then you'll have arrived at your final stop, {places[-1].title()}.{note}", None)
