"""Deterministic solvers (slice 5).

Each returns ``(result_dict | None, answer_text, refusal_reason | None)``. The
answer is copy-slot rendered from the computed result (faithful — no free
generation). Missing/invalid inputs refuse rather than invent.
"""

from __future__ import annotations

from typing import Any


def _load_entity_causal_rules_for_merge(store: Any) -> dict[str, list[dict[str, Any]]]:
    """Return approved causal_rule entities grouped by cause_lemma.

    Format: {cause: [{effect_state, effect_domain, confidence}, ...]}.
    """
    if store is None:
        return {}
    try:
        rules = store.query_causal_rules(review_status="approved", scope=None)
    except Exception:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rule in rules:
        cause = str(rule.get("cause_lemma", "")).lower()
        effect = str(rule.get("effect_state", "")).lower()
        domain = str(rule.get("effect_domain", "general") or "general").lower()
        confidence = float(rule.get("confidence", 0.5))
        if not cause or not effect:
            continue
        grouped.setdefault(cause, []).append({
            "state": effect,
            "domain": domain,
            "confidence": confidence,
            "provenance": "entity:causal_rule",
        })
    return grouped


def solve(task: dict, *, store: Any | None = None) -> tuple[dict | None, str, str | None]:
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
    if kind == "causal_explanation":
        return _causal_explanation(task, store=store)
    if kind == "causal_prediction":
        return _causal_prediction(task, store=store)
    if kind == "causal_contrast":
        return _causal_contrast(task, store=store)
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
    note = ""
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
              "distance_km": dist_km, "distance_text": dist_text,
              "place": place, "purpose": purpose, "note": note}
    return (result, answer, None)


def _temporal(task: dict) -> tuple[dict | None, str, str | None]:
    from . import clock
    n = clock.now()
    op = task.get("op")
    if op == "time":
        t = clock.format_time(n)
        return ({"task": "temporal", "op": "time", "value": t, "display": t}, f"It is {t}.", None)
    if op == "date_today":
        d = clock.format_date(n)
        return ({"task": "temporal", "op": "date_today", "value": d, "display": d}, f"Today is {d}.", None)
    if op == "absolute_date":
        from datetime import date
        from .dates import format_iso_date
        iso_date = str(task.get("date", ""))
        try:
            target = date.fromisoformat(iso_date)
        except ValueError:
            return (None, "", "invalid_date")
        weekday = target.strftime("%A")
        display = format_iso_date(iso_date)
        today = n.date()
        if target < today:
            verb = "was"
        elif target > today:
            verb = "will be"
        else:
            verb = "is"
        relation = "past" if target < today else "future" if target > today else "today"
        answer = f"{display} {verb} a {weekday}."
        result = {
            "task": "temporal",
            "op": "absolute_date",
            "date": iso_date,
            "weekday": weekday,
            "display": display,
            "verb": verb,
            "relation": relation,
        }
        return (result, answer, None)
    if op == "day_offset":
        days = task.get("days")
        if days is None:
            return (None, "", "missing_offset")
        target = clock.shift_days(n, days)
        weekday = clock.weekday_name(target)
        date = clock.format_date(target)
        mag = abs(int(days))
        unit = "day" if mag == 1 else "days"
        direction = "ago" if days < 0 else "from_now"
        if days < 0:
            answer = f"{mag} {unit} ago it was {weekday} ({date})."
        else:
            answer = f"In {mag} {unit} it will be {weekday} ({date})."
        result = {"task": "temporal", "op": "day_offset", "days": days,
                  "weekday": weekday, "date": date,
                  "magnitude": mag, "unit": unit, "direction": direction}
        return (result, answer, None)
    return (None, "", "unsupported_temporal_op")


def _metalinguistic_count(task: dict) -> tuple[dict | None, str, str | None]:
    char = str(task.get("char", ""))
    word = str(task.get("word", ""))
    if not char or not word:
        return (None, "", "missing_count_operands")
    count = word.count(char)
    plural = "" if count == 1 else "s"
    answer = f'There {"is" if count == 1 else "are"} {count} "{char}"{plural} in "{word}".'
    result = {"task": "metalinguistic_count", "char": char, "word": word,
              "count": count, "count_word": str(count), "plural": plural}
    return (result, answer, None)


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
    result = {"task": "quantity_arithmetic", "value": value_out,
              "noun": noun, "value_str": str(value_out)}
    return (result, answer, None)


def _causal_explanation(task: dict, *, store: Any | None = None) -> tuple[dict | None, str, str | None]:
    effect = str(task.get("effect", "")).lower()
    if not effect:
        return None, "", "missing_effect"
    theme = str(task.get("theme", ""))

    from .causal_frames import explain_effect_state, load_causal_frame_index, normalize_causal_surface

    # Normalize compound effects via surface aliases
    normalized = normalize_causal_surface(effect)
    index = load_causal_frame_index()

    # Resolve via surface alias: "gun shot" -> resolution info
    surface_resolution = None
    alias_entry = index.by_surface_alias.get(effect)
    if alias_entry is None:
        alias_entry = index.by_surface_alias.get(normalized)
    if alias_entry is not None:
        surface_resolution = {
            "surface": effect,
            "canonical": alias_entry.get("canonical", effect),
            "sense_type": alias_entry.get("sense_type", ""),
        }

    # If the surface alias has caused_by data, use it directly.
    # Otherwise follow the canonical chain to find caused_by.
    caused_by_data = None
    if alias_entry:
        caused_by_data = alias_entry.get("caused_by")
        if not caused_by_data:
            # Follow canonical chain (start seen with effect only, not normalized)
            follow = alias_entry.get("canonical")
            seen = {effect}
            while follow and isinstance(follow, str) and follow not in seen:
                seen.add(follow)
                canonical_entry = index.by_surface_alias.get(follow)
                if canonical_entry:
                    caused_by_data = canonical_entry.get("caused_by")
                    if caused_by_data:
                        if surface_resolution is None:
                            surface_resolution = {
                                "surface": effect,
                                "canonical": follow,
                                "sense_type": canonical_entry.get("sense_type", ""),
                            }
                        break
                    follow = canonical_entry.get("canonical")
                else:
                    break
    if caused_by_data:
        result = {
            "task": "causal_explanation",
            "effect": normalized,
            "theme": theme,
            "state_definition": {"state": normalized, "definition": ""},
            "candidate_causes": list(caused_by_data),
            "selected_cause": caused_by_data[0].get("predicate_id") if caused_by_data else None,
            "surface_resolution": surface_resolution,
        }
        answer = f"A {normalized} is typically caused by {caused_by_data[0]['predicate_id']}"
        if caused_by_data[0].get("instrument"):
            answer += f" using a {caused_by_data[0]['instrument']}"
        answer += "."
        return result, answer, None

    # Get base explanation
    result = explain_effect_state(normalized, theme=theme)

    # Attach surface resolution if applicable
    if surface_resolution:
        result["surface_resolution"] = surface_resolution

    # Merge entity rules: add entity-sourced causes for the same effect
    entity_rules = _load_entity_causal_rules_for_merge(store)
    for cause_lemma, entries in entity_rules.items():
        for entry in entries:
            if entry["state"] == normalized:
                # Check if this cause_lemma already exists in candidates
                existing = [c for c in result.get("candidate_causes", []) if c.get("predicate_id") == cause_lemma]
                if existing:
                    # Update confidence if higher
                    if entry["confidence"] > existing[0].get("confidence", 0):
                        existing[0]["confidence"] = entry["confidence"]
                        existing[0]["provenance"] = "contract:causal_frames.v1+entity:causal_rule"
                else:
                    result.setdefault("candidate_causes", []).append({
                        "predicate_id": cause_lemma,
                        "cause_kind": "unknown",
                        "confidence": entry["confidence"],
                        "provenance": "entity:causal_rule",
                    })
        # Re-sort candidates by confidence
        if result.get("candidate_causes"):
            result["candidate_causes"] = sorted(
                result["candidate_causes"], key=lambda c: -c["confidence"]
            )
            result["selected_cause"] = result["candidate_causes"][0]["predicate_id"]

    # Generate answer text from structured result
    if result.get("selected_cause"):
        state_def = result.get("state_definition", {}).get("definition", "")
        cause_kind_hint = ""
        for c in result.get("candidate_causes", []):
            if c["predicate_id"] == result["selected_cause"]:
                cause_kind_hint = c.get("cause_kind", "")
                break
        prefix = f"{theme} being " if theme else ""
        kind_tag = f" ({cause_kind_hint})" if cause_kind_hint else ""
        answer = f"{prefix}{effect} means {state_def}. A likely cause is {result['selected_cause']}{kind_tag}."
        return result, answer, None

    return (
        result,
        f"I don't have a causal explanation for {effect} in my local knowledge.",
        "no_cause_found",
    )


def _causal_prediction(task: dict, *, store: Any | None = None) -> tuple[dict | None, str, str | None]:
    cause = str(task.get("cause", "")).lower()
    if not cause:
        return None, "", "missing_cause"
    actor = str(task.get("actor", ""))
    patient = str(task.get("patient", ""))

    from .causal_frames import predict_effects, load_causal_frame_index

    index = load_causal_frame_index()

    # Get base prediction
    result = predict_effects(cause, actor=actor, patient=patient)

    # Merge entity rules for the same cause predicate
    entity_rules = _load_entity_causal_rules_for_merge(store)
    for entry in entity_rules.get(cause, []):
        existing = [e for e in result.get("effects", []) if e.get("state") == entry["state"]]
        if existing:
            # Update confidence if higher
            if entry["confidence"] > existing[0].get("confidence", 0):
                existing[0]["confidence"] = entry["confidence"]
        else:
            result.setdefault("effects", []).append({
                "state": entry["state"],
                "domain": entry["domain"],
                "target_role": "patient",
                "confidence": entry["confidence"],
            })
    # Sort effects by confidence
    if result.get("effects"):
        result["effects"] = sorted(result["effects"], key=lambda e: -e["confidence"])

    # Generate answer text from structured result
    result["patient"] = patient
    if result.get("effects"):
        state_labels = []
        for e in result["effects"]:
            label = e["state"]
            def_text = e.get("state_definition_text", "")
            if def_text:
                label = f"{label} ({def_text})"
            if "precondition_state" in e:
                label += f" if {e['precondition_state']}"
            state_labels.append(label)
        effect_text = ", ".join(state_labels[:5])
        cause_phrase = cause
        if actor:
            cause_phrase = actor + " " + cause_phrase
        if patient:
            cause_phrase += f" someone or something ({patient})" if patient != "person" else " someone"
        answer = f"If {cause_phrase} happens, likely effects include {effect_text}."
        return result, answer, None

    return (
        result,
        f"I don't have enough information to predict what {cause} would lead to.",
        "no_effect_found",
    )


def _causal_contrast(task: dict, *, store: Any | None = None) -> tuple[dict | None, str, str | None]:
    cause_a = str(task.get("cause_a", "")).lower()
    cause_b = str(task.get("cause_b", "")).lower()
    if not cause_a or not cause_b:
        return None, "", "missing_contrast_causes"

    from .causal_frames import predict_effects

    result_a = predict_effects(cause_a)
    result_b = predict_effects(cause_b)

    # Merge approved entity rules into the contrast, mirroring _causal_prediction.
    entity_rules = _load_entity_causal_rules_for_merge(store)
    for cause, result in ((cause_a, result_a), (cause_b, result_b)):
        for entry in entity_rules.get(cause, []):
            existing = [e for e in result.get("effects", []) if e.get("state") == entry["state"]]
            if existing:
                if entry["confidence"] > existing[0].get("confidence", 0):
                    existing[0]["confidence"] = entry["confidence"]
            else:
                result.setdefault("effects", []).append({
                    "state": entry["state"],
                    "domain": entry["domain"],
                    "target_role": "patient",
                    "confidence": entry["confidence"],
                })
        if result.get("effects"):
            result["effects"] = sorted(result["effects"], key=lambda e: -e["confidence"])

    effects_a = [e["state"] for e in result_a.get("effects", []) if "state" in e]
    effects_b = [e["state"] for e in result_b.get("effects", []) if "state" in e]

    if not effects_a and not effects_b:
        return (
            {"task": "causal_contrast", "cause_a": cause_a, "cause_b": cause_b},
            f"I do not have enough local causal knowledge to compare {cause_a} vs {cause_b}.",
            "no_contrast_data",
        )

    result = {
        "task": "causal_contrast",
        "cause_a": cause_a,
        "cause_b": cause_b,
        "effects_a": effects_a[:3],
        "effects_b": effects_b[:3],
    }

    a_text = ", ".join(effects_a[:3]) if effects_a else "nothing (unknown)"
    b_text = ", ".join(effects_b[:3]) if effects_b else "nothing (unknown)"
    answer = f"If {cause_a} happens, likely effects include {a_text}. In contrast, if {cause_b} happens, likely effects include {b_text}."
    return result, answer, None

