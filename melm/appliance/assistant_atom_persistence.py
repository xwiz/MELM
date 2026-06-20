"""UOL atom persistence — stores utterance meaning (T1) per turn."""

import uuid
from typing import Any


def record_uol_parse(
    store: Any,
    event_id: str,
    uol_act: dict | None,
) -> str | None:
    """Write a ``uol_parse`` entity for a single turn's UOL atoms.

    Parameters
    ----------
    store: The active entity store.
    event_id: The event_id of the turn.
    uol_act: The serialized UOL act dict, or *None*.

    Returns
    -------
    str | None — entity_id or *None* when no atoms exist.
    """
    if store is None:
        return None
    if not isinstance(uol_act, dict):
        return None
    content = uol_act.get("content")
    if not content or not isinstance(content, (list, tuple)):
        return None
    first = content[0]
    if not isinstance(first, dict):
        return None
    entity_id = f"uol_{uuid.uuid4().hex[:12]}"
    utterance = first.get("predicate", {}).get("lemma", "")[:80]
    store.add_entity(
        entity_id=entity_id,
        kind="uol_parse",
        label=f"uol: {utterance}",
        semantic_class_id="uol_parse",
        canonical_lemma=utterance,
    )
    store.set_entity_slot(entity_id, "uol_json", uol_act, provenance="atom_persistence")
    store.set_entity_slot(entity_id, "event_id", event_id, provenance="atom_persistence")
    return entity_id
