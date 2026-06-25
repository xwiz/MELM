"""Generic reasoning-result NLG renderer (V0.4).

Renders structured reasoning results into text via answer_templates.v1.json
reasoning_templates. Falls back to raw answer text when no template matches.
"""

from __future__ import annotations

from typing import Any


def render_reasoning_result(
    result: dict[str, Any] | None,
    templates: dict[str, Any],
) -> str | None:
    """Render a structured reasoning result using contract templates.

    ``templates`` is the reasoning_templates dict from answer_templates.v1.json
    (not the full payload). Returns rendered text, or None if no template matches
    and caller should fall back to the decision.answer or result answer text.
    """
    if not result or not isinstance(result, dict):
        return None
    task = str(result.get("task", ""))
    if not task:
        return None

    if not templates or not isinstance(templates, dict):
        return None

    task_templates = templates.get(task)
    if not task_templates or not isinstance(task_templates, dict):
        return None

    # Select the appropriate template based on result content
    if task == "causal_explanation":
        return _render_causal_explanation(result, task_templates)
    elif task == "causal_prediction":
        return _render_causal_prediction(result, task_templates)
    elif task == "causal_contrast":
        return _render_causal_contrast(result, task_templates)
    elif task == "geo_decision":
        return _render_geo_decision(result, task_templates)
    elif task == "temporal":
        return _render_temporal(result, task_templates)
    elif task == "metalinguistic_count":
        return _render_metalinguistic_count(result, task_templates)
    elif task == "quantity_arithmetic":
        return _render_quantity_arithmetic(result, task_templates)

    return None


def _render_causal_explanation(result: dict[str, Any], templates: dict[str, Any]) -> str | None:
    selected = result.get("selected_cause")
    theme = result.get("theme", "")
    effect = result.get("effect", "")
    state_def = result.get("state_definition", {}).get("definition", "")
    candidates = result.get("candidate_causes", [])

    if not selected:
        unknown = templates.get("unknown", "")
        return unknown.format(effect=effect) if unknown else None

    # Determine if single or multi cause
    has_multi = len(candidates) > 1
    if has_multi:
        template = templates.get("multi_cause", "")
        if template:
            theme_prefix = f"{theme} being " if theme else ""
            cause_labels = []
            for c in candidates[:5]:
                label = c.get("predicate_id", "")
                kind = c.get("cause_kind", "")
                if kind and kind != "unknown":
                    label += f" ({kind})"
                cause_labels.append(label)
            return template.format(
                theme_prefix=theme_prefix,
                effect=effect,
                state_definition=state_def,
                candidate_causes=", ".join(cause_labels),
                selected_cause=selected,
            )

    # Single cause
    template = templates.get("single_cause", "")
    if template:
        theme_prefix = f"{theme} being " if theme else ""
        return template.format(
            theme_prefix=theme_prefix,
            effect=effect,
            state_definition=state_def,
            selected_cause=selected,
        )

    return None


def _render_causal_prediction(result: dict[str, Any], templates: dict[str, Any]) -> str | None:
    cause = result.get("cause", "")
    actor = result.get("actor", "")
    effects = result.get("effects", [])

    if not effects:
        unknown = templates.get("unknown", "")
        return unknown.format(cause=cause) if unknown else None

    template = templates.get("effects", "")
    if not template:
        return None

    cause_phrase = cause
    if actor:
        cause_phrase = f"{actor} {cause_phrase}"

    effect_labels = []
    definitions: list[str] = []
    for e in effects:
        label = e.get("state", "")
        def_text = e.get("state_definition_text", "")
        if def_text:
            definitions.append(f"{label}: {def_text}")
        precond = e.get("precondition_state")
        if precond:
            label += f" if the surface is {precond}"
        effect_labels.append(label)

    return template.format(
        cause_phrase=cause_phrase,
        cause=cause,
        effects=", ".join(effect_labels[:5]),
        selected_definitions="; ".join(definitions[:3]),
    )


def _render_causal_contrast(result: dict[str, Any], templates: dict[str, Any]) -> str | None:
    cause_a = result.get("cause_a", "")
    cause_b = result.get("cause_b", "")
    effects_a = result.get("effects_a", [])
    effects_b = result.get("effects_b", [])

    if not effects_a and not effects_b:
        unknown = templates.get("unknown", "")
        return unknown.format(cause_a=cause_a, cause_b=cause_b) if unknown else None

    template = templates.get("contrast", "")
    if not template:
        return None

    a_text = ", ".join(effects_a[:3])
    b_text = ", ".join(effects_b[:3])
    return template.format(
        cause_a=cause_a,
        cause_b=cause_b,
        effects_a=a_text,
        effects_b=b_text,
    )


def _render_geo_decision(result: dict[str, Any], templates: dict[str, Any]) -> str | None:
    decision = result.get("decision", "")
    dist_text = result.get("distance_text", "")
    purpose = result.get("purpose")
    note = result.get("note", "")

    if purpose and note:
        template = templates.get("drive_with_purpose", "")
        if template:
            return template.format(distance_text=dist_text, note=note)

    template = templates.get(decision)
    if template:
        return template.format(distance_text=dist_text)

    return None


def _render_temporal(result: dict[str, Any], templates: dict[str, Any]) -> str | None:
    op = result.get("op", "")
    if op == "time":
        template = templates.get("time", "")
        if template:
            return template.format(display=result.get("display", ""))
    elif op == "date_today":
        template = templates.get("date_today", "")
        if template:
            return template.format(display=result.get("display", ""))
    elif op == "absolute_date":
        template = templates.get("absolute_date", "")
        if template:
            return template.format(
                display=result.get("display", ""),
                verb=result.get("verb", ""),
                weekday=result.get("weekday", ""),
            )
    elif op == "day_offset":
        direction = result.get("direction", "from_now")
        if direction == "ago":
            template = templates.get("day_offset_ago", "")
        else:
            template = templates.get("day_offset_from_now", "")
        if template:
            return template.format(
                magnitude=result.get("magnitude", ""),
                unit=result.get("unit", ""),
                weekday=result.get("weekday", ""),
                date=result.get("date", ""),
            )
    return None


def _render_metalinguistic_count(result: dict[str, Any], templates: dict[str, Any]) -> str | None:
    template = templates.get("count", "")
    if template:
        return template.format(
            count_word=result.get("count_word", ""),
            char=result.get("char", ""),
            plural=result.get("plural", ""),
            word=result.get("word", ""),
        )
    return None


def _render_quantity_arithmetic(result: dict[str, Any], templates: dict[str, Any]) -> str | None:
    template = templates.get("result", "")
    if template:
        noun = result.get("noun", "")
        value_str = result.get("value_str", "")
        formatted = template.format(value_str=value_str, noun=noun).strip()
        return formatted
    return None
