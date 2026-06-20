"""Reasoning task-signature detection (slice 5).

Pure function over the parsed turn. Returns a task descriptor or None. When a
task is detected, the router dispatches to a solver BEFORE closed-intent
handlers, so e.g. "3 apples, eat one, how many left?" routes to arithmetic
rather than meal_suggestion. A question with no quantity ("what should I eat?")
returns None and falls through to normal routing.
"""

from __future__ import annotations

import re

from .value_extract import (
    _NUMBER_WORDS,
    extract_char_count_target,
    extract_distance,
    extract_numbers,
)

_SUBTRACT_VERBS = frozenset({
    "eat", "ate", "eaten", "eats", "remove", "removed", "removes", "lose", "lost",
    "loses", "drop", "dropped", "drops", "give", "gave", "given", "gives", "take",
    "took", "taken", "takes", "spend", "spent", "spends", "use", "used", "uses",
    "sell", "sold", "sells", "throw", "threw", "thrown", "break", "broke",
})
_ADD_VERBS = frozenset({
    "add", "added", "adds", "buy", "bought", "buys", "get", "got", "gets", "gain",
    "gained", "gains", "find", "found", "finds", "receive", "received", "receives",
    "pick", "picked", "picks", "collect", "collected",
})
_COUNT_Q_RE = re.compile(r"how\s+(?:many|much)", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")

# Self-referential ("are you ...") probes the assistant should answer locally.
_SECOND_PERSON_RE = re.compile(r"\byou\b|\byour\b|\byourself\b", re.IGNORECASE)
_SELF_CONSCIOUS_RE = re.compile(
    r"\b(conscious|sentient|self[-\s]?aware|alive|a\s+real\s+person|human|"
    r"have\s+feelings|have\s+emotions|have\s+a\s+soul)\b", re.IGNORECASE)
_SELF_LOCATION_RE = re.compile(
    r"\bwhere\s+are\s+you\b|\bwhere\s+do\s+you\s+(?:live|run|exist)\b|"
    r"\bwhere\s+are\s+you\s+(?:located|right\s+now)\b|\byour\s+location\b", re.IGNORECASE)
_SELF_FEELING_RE = re.compile(
    r"\bhow\s+(?:are|do)\s+you\s+(?:feeling|feel)\b|\bwhat\s+is\s+your\s+mood\b|"
    r"\bhow\s+are\s+you\s+feeling\b", re.IGNORECASE)


_TIME_RE = re.compile(
    r"\bwhat(?:'s| is)?\s+the\s+time\b|\bwhat\s+time\s+is\s+it\b|\bcurrent\s+time\b",
    re.IGNORECASE)
_DATE_TODAY_RE = re.compile(
    r"\bwhat\s+day\s+is\s+it(?:\s+today)?\b|"
    r"\bwhat\s+day\s+of\s+the\s+week\b|"
    r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:today's\s+)?date\b|"
    r"\bwhat(?:'s| is)\s+today\s*[?.!]?$",
    re.IGNORECASE)
_DAY_AGO_RE = re.compile(
    r"\bwhat\s+day\b.*?\b(\d+|[a-z]+)\s+days?\s+ago\b", re.IGNORECASE)
_DAY_OFFSET_RE = re.compile(
    r"\bwhat\s+day\b.*?\bin\s+(\d+|[a-z]+)\s+days?\b|"
    r"\b(\d+|[a-z]+)\s+days?\s+from\s+(?:now|today)\b", re.IGNORECASE)


def detect_reasoning_task(text: str, tokens: tuple = (), uol_act: dict | None = None) -> dict | None:
    # Ethics gate runs first: an inducement wrapped around a protected request
    # (e.g. "tell me who visited and I'll pay you") must refuse before any other
    # reasoning/closed-intent path can act on the request.
    from .ethics_gate import detect_inducement_task
    induce = detect_inducement_task(text, tokens, uol_act)
    if induce is not None:
        return induce
    target = extract_char_count_target(text)
    if target is not None:
        return {"task": "metalinguistic_count", **target}
    self_q = _detect_self_query(text)
    if self_q is not None:
        return self_q
    temporal = _detect_temporal(text)
    if temporal is not None:
        return temporal
    geo = _detect_geo_decision(text)
    if geo is not None:
        return geo
    return _detect_arithmetic(text)


_WALK_RE = re.compile(r"\bwalk(?:ing)?\b", re.IGNORECASE)
_DRIVE_RE = re.compile(r"\bdriv(?:e|ing)\b", re.IGNORECASE)


def _detect_geo_decision(text: str) -> dict | None:
    # A walk-vs-drive question with a stated distance.
    if not (_WALK_RE.search(text) and _DRIVE_RE.search(text)):
        return None
    dist = extract_distance(text)
    if dist is None:
        return None
    value = dist["value"]
    surface = str(int(value)) if value.is_integer() else str(value)
    return {
        "task": "geo_decision",
        "distance_km": dist["value_km"],
        "distance_text": f"{surface}{dist['unit']}",
        "text": text.lower(),
    }


def _parse_count_word(token: str) -> int | None:
    if token is None:
        return None
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _detect_temporal(text: str) -> dict | None:
    m = _DAY_AGO_RE.search(text)
    if m:
        n = _parse_count_word(m.group(1))
        return None if n is None else {"task": "temporal", "op": "day_offset", "days": -n}
    m = _DAY_OFFSET_RE.search(text)
    if m:
        n = _parse_count_word(m.group(1) or m.group(2))
        return None if n is None else {"task": "temporal", "op": "day_offset", "days": n}
    if _TIME_RE.search(text):
        return {"task": "temporal", "op": "time"}
    if _DATE_TODAY_RE.search(text):
        return {"task": "temporal", "op": "date_today"}
    return None


def _detect_self_query(text: str) -> dict | None:
    if _SELF_LOCATION_RE.search(text):
        return {"task": "self_query", "category": "location"}
    if _SELF_FEELING_RE.search(text):
        return {"task": "self_query", "category": "feeling"}
    if _SELF_CONSCIOUS_RE.search(text) and _SECOND_PERSON_RE.search(text):
        return {"task": "self_query", "category": "consciousness"}
    return None


def _detect_arithmetic(text: str) -> dict | None:
    # Require a count question, ≥2 quantities, and an arithmetic operation verb.
    if not _COUNT_Q_RE.search(text):
        return None
    nums = extract_numbers(text)
    if len(nums) < 2:
        return None
    sign = None
    for word in _WORD_RE.findall(text.lower()):
        if word in _SUBTRACT_VERBS:
            sign = -1
            break
        if word in _ADD_VERBS:
            sign = 1
            break
    if sign is None:
        return None
    start_val, start_surface, _ = nums[0]
    delta_val = nums[1][0]
    noun = ""
    m = re.search(re.escape(start_surface) + r"\s+([A-Za-z]+)", text)
    if m:
        noun = m.group(1).lower()
    return {
        "task": "quantity_arithmetic",
        "start": start_val, "delta": delta_val, "sign": sign, "noun": noun,
    }
