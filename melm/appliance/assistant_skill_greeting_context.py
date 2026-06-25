"""Greeting context injector for the Local Assistant OS.

Surfaces deferred tasks and epistemic states at session start.
"""

from __future__ import annotations

from typing import Any

_DEFERRED_TASK_TEMPLATES_CACHE: dict[str, str] | None = None


def _load_deferred_task_templates_cached() -> dict[str, str] | None:
    global _DEFERRED_TASK_TEMPLATES_CACHE
    if _DEFERRED_TASK_TEMPLATES_CACHE is None:
        try:
            from ..contracts.validation import load_deferred_task_templates
            _DEFERRED_TASK_TEMPLATES_CACHE = load_deferred_task_templates()
        except Exception:
            _DEFERRED_TASK_TEMPLATES_CACHE = {}
    return _DEFERRED_TASK_TEMPLATES_CACHE or None


def _render_completed_task(topic: str, summary: str) -> str:
    templates = _load_deferred_task_templates_cached()
    if templates:
        template = templates.get("task_completed", "")
        if template:
            if summary:
                return template.format(topic=topic, summary=summary)
            return template.format(topic=topic, summary="").rstrip()
    if summary:
        return f"I was looking into {topic} — {summary}"
    return f"I finished looking into {topic}."


def build_greeting_context(
    store: Any,
    current_session_id: str,
    profile: Any,
) -> str | None:
    """Build a natural-language context string for greeting injection.
    
    Checks for completed deferred tasks from previous sessions and
    open epistemic states.
    """
    context_parts = []
    
    try:
        entities = store.find_entities(kind="deferred_task")
        for ent in entities:
            eid = ent.entity_id
            status_slot = store.get_entity_slot(eid, "status")
            owner_slot = store.get_entity_slot(eid, "owner_session_id")
            if status_slot and status_slot.value_json:
                status = _json_load(status_slot.value_json)
                owner = _json_load(owner_slot.value_json) if owner_slot and owner_slot.value_json else ""
                if status == "completed" and owner and owner != current_session_id:
                    topic_slot = store.get_entity_slot(eid, "topic")
                    topic = _json_load(topic_slot.value_json) if topic_slot and topic_slot.value_json else "something"
                    result_slot = store.get_entity_slot(eid, "result_summary")
                    summary = _json_load(result_slot.value_json) if result_slot and result_slot.value_json else ""
                    context_parts.append(_render_completed_task(topic, summary))
    except Exception:
        pass
    
    try:
        from .assistant_skill_epistemic import surface_open_states
        states_text = surface_open_states(store)
        if states_text:
            context_parts.append(states_text)
    except Exception:
        pass
    
    if context_parts:
        return " ".join(context_parts)
    return None


def _json_load(val: str) -> str:
    try:
        import json
        return str(json.loads(val))
    except Exception:
        return str(val)


def inject_greeting(decision: Any, context_text: str) -> Any:
    """Prepend greeting context to a decision's answer."""
    from dataclasses import replace
    existing = getattr(decision, "answer", "")
    if existing:
        new_answer = f"{context_text} {existing}"
    else:
        new_answer = context_text
    return replace(decision, answer=new_answer)
