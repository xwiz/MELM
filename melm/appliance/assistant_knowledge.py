"""Knowledge typing — classify claims by UOL structure.

This module provides deterministic classification of user utterances
into knowledge types (static_fact, negated_fact, opinion, literary_device)
based on UOL atoms. Conservative by design: ambiguous input returns None,
falling through to existing behavior.
"""

from typing import Any

_KNOWLEDGE_TYPES_CACHE: dict | None = None
_WORLD_RELATIONS_CACHE: dict | None = None

KnowledgeType = str  # static_fact | negated_fact | opinion | literary_device | None


def classify_knowledge(uol_act: dict | None, text: str) -> KnowledgeType | None:
    """Classify a claim by its UOL structure.
    
    Returns None for non-claims, ambiguous input, or unrecognized patterns —
    the caller falls through to existing behavior.
    """
    if uol_act is None:
        return None
    act_type = uol_act.get("act", "")
    if act_type != "claim":
        return None
    content = uol_act.get("content", [])
    if not content:
        return None
    atom = content[0] if isinstance(content, (list, tuple)) else {}
    if not isinstance(atom, dict):
        return None
    context = atom.get("context", {}) or {}
    polarity = str(context.get("polarity", "positive")).lower()
    is_negated = polarity == "negative" or bool(context.get("negation_scope"))
    pred = atom.get("predicate", {}) or {}
    pred_id = str(pred.get("id", "")).lower()
    roles = atom.get("roles", [])
    subjects = [
        str(r.get("value", "")).lower()
        for r in roles
        if isinstance(r, dict) and r.get("role") in ("agent", "subject")
    ]
    objects = [
        str(r.get("value", "")).lower()
        for r in roles
        if isinstance(r, dict) and r.get("role") in ("theme", "patient", "object")
    ]
    subject = subjects[0] if subjects else ""
    obj = objects[0] if objects else ""

    # Personal subjects — not a world fact (goes to profile path)
    if subject in ("i", "we", "my", "our"):
        return None

    # Literary device detection (conservative — require marker)
    lower_text = text.lower().strip().rstrip("?.")
    _ensure_caches()
    literary_stems = (
        _KNOWLEDGE_TYPES_CACHE.get("type_markers", {}).get("literary_stems", [])
        if _KNOWLEDGE_TYPES_CACHE else []
    )
    for stem in literary_stems:
        if lower_text.startswith(stem):
            return "literary_device"

    # Need a clean subject + predicate + object for factual claims
    if not subject or not pred_id or not obj:
        return None

    # Negated claim
    if is_negated:
        return "negated_fact"

    # Opinion detection
    opinion_markers = (
        _KNOWLEDGE_TYPES_CACHE.get("type_markers", {}).get("opinion_markers", [])
        if _KNOWLEDGE_TYPES_CACHE else []
    )
    if any(marker in lower_text.split() for marker in opinion_markers):
        return "opinion"

    # Static fact (copular or known relation predicate)
    if pred_id in ("be", "is", "are", "was", "were"):
        return "static_fact"
    relations = _WORLD_RELATIONS_CACHE if _WORLD_RELATIONS_CACHE else {}
    if pred_id in relations.get("predicate_to_relation", {}):
        return "static_fact"

    return None


def _ensure_caches() -> None:
    global _KNOWLEDGE_TYPES_CACHE, _WORLD_RELATIONS_CACHE
    if _KNOWLEDGE_TYPES_CACHE is None:
        from ..contracts.validation import load_knowledge_types
        _KNOWLEDGE_TYPES_CACHE = load_knowledge_types()
    if _WORLD_RELATIONS_CACHE is None:
        from ..contracts.validation import load_world_relations
        _WORLD_RELATIONS_CACHE = load_world_relations()


def extract_proposition(uol_act: dict) -> dict | None:
    """Extract (subject, relation, object) from UOL atoms.

    Returns None when the atom structure doesn't support extraction.
    """
    content = uol_act.get("content", [])
    if not content:
        return None
    atom = content[0] if isinstance(content, (list, tuple)) else {}
    if not isinstance(atom, dict):
        return None
    pred = atom.get("predicate", {}) or {}
    pred_id = str(pred.get("id", "")).lower()
    roles = atom.get("roles", [])
    subjects = [
        str(r.get("value", "")).lower()
        for r in roles
        if isinstance(r, dict) and r.get("role") in ("agent", "subject")
    ]
    objects = [
        str(r.get("value", "")).lower()
        for r in roles
        if isinstance(r, dict) and r.get("role") in ("theme", "patient", "object")
    ]
    if not subjects or not objects:
        return None
    _ensure_caches()
    relations_data = _WORLD_RELATIONS_CACHE.get("predicate_to_relation", {}) if _WORLD_RELATIONS_CACHE else {}
    rel_entry = relations_data.get(pred_id, {})
    rel_id = rel_entry.get("relation_id", pred_id)
    rel_conf = rel_entry.get("confidence", 0.5)
    return {
        "subject": subjects[0],
        "relation": rel_id,
        "object": objects[0],
        "confidence": rel_conf,
    }
