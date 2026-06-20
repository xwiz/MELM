"""Deterministic solvers (slice 5).

Each returns ``(result_dict | None, answer_text, refusal_reason | None)``. The
answer is copy-slot rendered from the computed result (faithful — no free
generation). Missing/invalid inputs refuse rather than invent.
"""

from __future__ import annotations


def solve(task: dict) -> tuple[dict | None, str, str | None]:
    kind = task.get("task")
    if kind == "metalinguistic_count":
        return _metalinguistic_count(task)
    if kind == "quantity_arithmetic":
        return _quantity_arithmetic(task)
    if kind == "temporal":
        return _temporal(task)
    if kind == "geo_decision":
        return _geo_decision(task)
    if kind == "ethics_gate":
        return _ethics_gate(task)
    return (None, "", "unsupported_task")


def _ethics_gate(task: dict) -> tuple[dict | None, str, str | None]:
    from .ethics_gate import render_ethics_refusal
    answer = render_ethics_refusal(task)
    result = {
        "task": "ethics_gate",
        "inducement_type": task.get("inducement_type"),
        "credibility_score": task.get("credibility_score"),
        "privacy_nonnegotiable": task.get("privacy_nonnegotiable"),
    }
    return (result, answer, task.get("refusal_reason", "privacy_nonnegotiable"))


def _geo_decision(task: dict) -> tuple[dict | None, str, str | None]:
    from melm.contracts import load_geo_decision
    cfg = load_geo_decision()
    threshold = float(cfg.get("walk_threshold_km", 1.0))
    text = str(task.get("text", ""))
    dist_km = task.get("distance_km")
    dist_text = str(task.get("distance_text", "the stated distance"))
    if dist_km is None:
        return (None, "", "missing_distance")
    # Entity-purpose detection: a car wash needs the car present, which overrides
    # the pure distance heuristic.
    purpose = None
    place = "there"
    for phrase, tag in cfg.get("place_purposes", {}).items():
        if phrase in text:
            purpose, place = tag, phrase
            break
    overrides = cfg.get("purpose_overrides", {})
    if purpose and overrides.get(purpose, {}).get("requires_vehicle"):
        note = overrides[purpose].get("note", "it needs your vehicle")
        answer = f"It's only {dist_text} away, so by distance I'd say walk — but {note}, so drive."
        decision = "drive"
    elif dist_km <= threshold:
        answer = f"At {dist_text}, that's close — walk."
        decision = "walk"
    else:
        answer = f"At {dist_text}, that's a drive."
        decision = "drive"
    result = {"task": "geo_decision", "decision": decision,
              "distance_km": dist_km, "place": place, "purpose": purpose}
    return (result, answer, None)


def _temporal(task: dict) -> tuple[dict | None, str, str | None]:
    from . import clock
    n = clock.now()
    op = task.get("op")
    if op == "time":
        t = clock.format_time(n)
        return ({"task": "temporal", "op": "time", "value": t}, f"It is {t}.", None)
    if op == "date_today":
        d = clock.format_date(n)
        return ({"task": "temporal", "op": "date_today", "value": d}, f"Today is {d}.", None)
    if op == "day_offset":
        days = task.get("days")
        if days is None:
            return (None, "", "missing_offset")
        target = clock.shift_days(n, days)
        weekday = clock.weekday_name(target)
        date = clock.format_date(target)
        mag = abs(int(days))
        unit = "day" if mag == 1 else "days"
        if days < 0:
            answer = f"{mag} {unit} ago it was {weekday} ({date})."
        else:
            answer = f"In {mag} {unit} it will be {weekday} ({date})."
        result = {"task": "temporal", "op": "day_offset", "days": days,
                  "weekday": weekday, "date": date}
        return (result, answer, None)
    return (None, "", "unsupported_temporal_op")


def _metalinguistic_count(task: dict) -> tuple[dict | None, str, str | None]:
    char = str(task.get("char", ""))
    word = str(task.get("word", ""))
    if not char or not word:
        return (None, "", "missing_count_operands")
    count = word.count(char)
    verb = "is" if count == 1 else "are"
    plural = "" if count == 1 else "s"
    answer = f'There {verb} {count} "{char}"{plural} in "{word}".'
    return ({"task": "metalinguistic_count", "char": char, "word": word, "count": count}, answer, None)


def _quantity_arithmetic(task: dict) -> tuple[dict | None, str, str | None]:
    start = task.get("start")
    delta = task.get("delta")
    sign = task.get("sign")
    noun = str(task.get("noun", ""))
    if start is None or delta is None or sign is None:
        return (None, "", "missing_arithmetic_operands")
    value = float(start) + float(sign) * float(delta)
    if value < 0:
        return (None, "", "negative_quantity")
    value_out = int(value) if float(value).is_integer() else round(value, 4)
    answer = f"{value_out} {noun}.".strip() if noun else f"{value_out}."
    return ({"task": "quantity_arithmetic", "value": value_out, "noun": noun}, answer, None)
