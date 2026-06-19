"""Deferred task management skill for the Local Assistant OS.

Creates, queries, and manages deferred_task entities in the entity store.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

DEFERRED_TASK_CLASS = "deferred_task"


def queue_deferred_task(
    store: Any,
    topic: str,
    action: str,
    scheduled_at: str | None = None,
    **kwargs: Any,
) -> str | None:
    """Create a deferred_task entity and return its entity_id."""
    if store is None:
        return None
    try:
        entity_id = f"dt_{uuid.uuid4().hex[:12]}"
        store.add_entity(
            entity_id=entity_id,
            kind=DEFERRED_TASK_CLASS,
            label=f"deferred: {action}",
            semantic_class_id="deferred_task",
            canonical_lemma=topic[:80],
        )
        now = datetime.now(timezone.utc).isoformat()
        store.set_entity_slot(entity_id, "topic", topic)
        store.set_entity_slot(entity_id, "action", action)
        store.set_entity_slot(entity_id, "status", "queued")
        store.set_entity_slot(entity_id, "scheduled_at", scheduled_at or now)
        if "due_at" in kwargs:
            store.set_entity_slot(entity_id, "due_at", kwargs["due_at"])
        if "priority" in kwargs:
            store.set_entity_slot(entity_id, "priority", str(kwargs["priority"]))
        if "engagement_prompt" in kwargs:
            store.set_entity_slot(entity_id, "engagement_prompt", kwargs["engagement_prompt"])
        if "owner_session_id" in kwargs:
            store.set_entity_slot(entity_id, "owner_session_id", kwargs["owner_session_id"])
        return entity_id
    except Exception:
        return None


def find_due_tasks(
    store: Any,
    before: str | None = None,
) -> list[dict[str, Any]]:
    """Find deferred_task entities with status='queued' and scheduled_at <= before."""
    if store is None:
        return []
    try:
        entities = store.find_entities(kind=DEFERRED_TASK_CLASS)
        due = []
        cutoff = before or datetime.now(timezone.utc).isoformat()
        for ent in entities:
            eid = ent.entity_id
            status = store.get_entity_slot(eid, "status")
            if status is not None and status.value_json and json.loads(status.value_json) == "queued":
                scheduled = store.get_entity_slot(eid, "scheduled_at")
                if scheduled is not None and scheduled.value_json:
                    sched_val = json.loads(scheduled.value_json)
                    if sched_val <= cutoff:
                        slots = {}
                        for slot_name in ["topic", "action", "status", "engagement_prompt"]:
                            slot = store.get_entity_slot(eid, slot_name)
                            if slot is not None and slot.value_json:
                                slots[slot_name] = json.loads(slot.value_json)
                        slots["entity_id"] = eid
                        due.append(slots)
        return due
    except Exception:
        return []


def surface_task_context(store: Any, task_entity: dict[str, Any]) -> str:
    """Render a deferred task as a natural-language context string."""
    topic = task_entity.get("topic", "something")
    action = task_entity.get("action", "research")
    if action == "auto_research":
        return f"I was looking into {topic}."
    elif action == "novelty_review":
        return f"I came across something interesting about {topic}."
    return f"I have a pending task about {topic}."


def complete_deferred_task(
    store: Any,
    entity_id: str,
    result_summary: str = "",
    result_entity_id: str = "",
) -> None:
    """Mark a deferred task as completed with results."""
    if store is None:
        return
    try:
        store.set_entity_slot(entity_id, "status", "completed")
        if result_summary:
            store.set_entity_slot(entity_id, "result_summary", result_summary)
        if result_entity_id:
            store.set_entity_slot(entity_id, "result_entity_id", result_entity_id)
    except Exception:
        pass


def cancel_deferred_task(store: Any, entity_id: str) -> None:
    """Cancel a deferred task."""
    if store is None:
        return
    try:
        store.set_entity_slot(entity_id, "status", "cancelled")
    except Exception:
        pass
