"""Epistemic state tracking for the Local Assistant OS.

Tracks confusion, curiosity, expectation, and surprise states.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import json

_EPISTEMIC_STATES = None


def _load_epistemic_config():
    global _EPISTEMIC_STATES
    if _EPISTEMIC_STATES is None:
        try:
            from melm.contracts import load_epistemic_states
            _EPISTEMIC_STATES = load_epistemic_states()
        except Exception:
            _EPISTEMIC_STATES = {}
    return _EPISTEMIC_STATES


def record_epistemic_state(
    store: Any,
    state_type: str,
    topic: str,
    valence: float = 0.0,
    source_event_id: str = "",
) -> str | None:
    """Record an epistemic state entity. Returns entity_id."""
    if store is None:
        return None
    import uuid
    try:
        entity_id = f"es_{uuid.uuid4().hex[:12]}"
        store.add_entity(
            entity_id=entity_id,
            kind="epistemic_state",
            label=f"epistemic: {state_type}",
            semantic_class_id="epistemic_state",
            canonical_lemma=topic[:80],
        )
        store.set_entity_slot(entity_id, "state_type", state_type)
        store.set_entity_slot(entity_id, "topic", topic)
        store.set_entity_slot(entity_id, "valence", str(valence))
        if source_event_id:
            store.set_entity_slot(entity_id, "source_event_id", source_event_id)
        return entity_id
    except Exception:
        return None


def load_open_epistemic_states(
    store: Any,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load unresolved epistemic states (no resolved_at set)."""
    if store is None:
        return []
    config = _load_epistemic_config()
    max_open = config.get("max_open_states", 10)
    
    try:
        entities = store.find_entities(kind="epistemic_state")
        results = []
        for ent in entities:
            eid = ent.entity_id
            resolved = store.get_entity_slot(eid, "resolved_at")
            if resolved is not None and resolved.value_json:
                continue
            state_type_slot = store.get_entity_slot(eid, "state_type")
            topic_slot = store.get_entity_slot(eid, "topic")
            if state_type_slot and state_type_slot.value_json:
                results.append({
                    "entity_id": eid,
                    "state_type": json.loads(state_type_slot.value_json),
                    "topic": json.loads(topic_slot.value_json) if topic_slot and topic_slot.value_json else "",
                })
            if len(results) >= max_open:
                break
        return results
    except Exception:
        return []


def surface_open_states(store: Any) -> str | None:
    """Render open epistemic states as a natural-language string."""
    states = load_open_epistemic_states(store)
    if not states:
        return None
    curious = [s for s in states if s.get("state_type") == "curiosity"]
    confused = [s for s in states if s.get("state_type") == "confusion"]
    parts = []
    if curious:
        topics = [s.get("topic", "something") for s in curious[:3]]
        parts.append(f"I was curious about {' and '.join(topics)}")
    if confused:
        topics = [s.get("topic", "something") for s in confused[:3]]
        parts.append(f"I was unsure about {' and '.join(topics)}")
    return " — ".join(parts) if parts else None
