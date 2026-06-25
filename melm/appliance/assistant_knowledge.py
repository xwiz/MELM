"""Knowledge typing — classify claims by UOL structure.

This module provides deterministic classification of user utterances
into knowledge types (static_fact, negated_fact, opinion, literary_device)
based on UOL atoms. Conservative by design: ambiguous input returns None,
falling through to existing behavior.
"""

import re
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
    if act_type not in ("claim", "statement"):
        return None
    atom = _first_atom(uol_act)
    if atom is None:
        return None
    context = atom.get("context", {}) or {}
    polarity = str(context.get("polarity", "positive")).lower()
    is_negated = (
        polarity == "negative"
        or bool(context.get("negation_scope"))
        or bool(re.search(r"\b(?:not|never|no|n't)\b", text.lower()))
    )
    pred_id, subject, obj = _atom_predicate_subject_object(atom, uol_act)

    # Personal subjects — not a world fact (goes to profile path)
    if subject in ("i", "we", "my", "our", "you", "your", "user", "this", "that", "it"):
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
    if pred_id in ("be", "is", "are", "was", "were") or _looks_copular_fact(text, subject, obj):
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


def _first_atom(uol_act: dict) -> dict | None:
    content = uol_act.get("content", [])
    if isinstance(content, (list, tuple)) and content and isinstance(content[0], dict):
        return content[0]
    if isinstance(uol_act.get("predicate"), dict):
        return uol_act
    return None


def _role_values(atom: dict, names: tuple[str, ...]) -> list[str]:
    roles = atom.get("roles", [])
    return [
        str(r.get("value", "")).lower()
        for r in roles
        if isinstance(r, dict) and r.get("role") in names and str(r.get("value", "")).strip()
    ]


def _atom_predicate_subject_object(atom: dict, uol_act: dict) -> tuple[str, str, str]:
    pred = atom.get("predicate", {}) or {}
    pred_id = str(pred.get("id", "")).lower()
    subjects = _role_values(atom, ("agent", "subject"))
    objects = _role_values(atom, ("theme", "patient", "object"))

    # Current production UOL for copular statements can project the grammatical
    # subject as the atom predicate plus a role named "predicate".
    if not subjects:
        subjects = _role_values(atom, ("predicate",))
    if not objects:
        top_obj = uol_act.get("object")
        if isinstance(top_obj, str) and top_obj.strip():
            objects = [top_obj.lower()]
    if not subjects:
        top_subject = uol_act.get("subject")
        if isinstance(top_subject, str) and top_subject.strip():
            subjects = [top_subject.lower()]
    # For state atoms (copular "X is Y"), the complement/stative object is
    # stored in modifiers rather than roles. Extract it as a fallback.
    if not objects and atom.get("kind") == "state":
        modifiers = atom.get("modifiers", [])
        if isinstance(modifiers, (list, tuple)):
            for m in modifiers:
                if isinstance(m, dict):
                    lemma = m.get("lemma", "")
                    if lemma and lemma.strip():
                        objects.append(lemma.lower())

    top_action = uol_act.get("action")
    if pred_id == "" and isinstance(top_action, str):
        pred_id = top_action.lower()
    return pred_id, subjects[0] if subjects else "", " ".join(objects)


def _looks_copular_fact(text: str, subject: str, obj: str) -> bool:
    if not subject or not obj:
        return False
    escaped_subject = re.escape(subject)
    return bool(re.search(rf"\b{escaped_subject}\b\s+(?:is|are|was|were|be)\b", text.lower()))


def extract_proposition(uol_act: dict, text: str = "") -> dict | None:
    """Extract (subject, relation, object) from UOL atoms.

    ``text`` is the original utterance, used to detect copular patterns
    when the UOL atomizer incorrectly uses the subject as predicate
    (e.g. "Abuja is the capital" → pred_id="abuja" instead of "be").
    Returns None when the atom structure doesn't support extraction.
    """
    atom = _first_atom(uol_act)
    if atom is None:
        return None
    pred_id, subject, obj = _atom_predicate_subject_object(atom, uol_act)
    if not subject or not obj:
        return None
    _ensure_caches()
    relations_data = _WORLD_RELATIONS_CACHE.get("predicate_to_relation", {}) if _WORLD_RELATIONS_CACHE else {}
    rel_entry = relations_data.get(pred_id, {})
    rel_id = rel_entry.get("relation_id", pred_id)
    rel_conf = rel_entry.get("confidence", 0.5)
    _COPULAR = {"be", "is", "are", "was", "were", "am"}
    if rel_id == pred_id and pred_id not in relations_data:
        # The UOL atomizer can use the subject as the predicate for
        # copular statements (e.g. "Abuja is the capital").
        # Check if the original text has a copular pattern.
        is_copular = (
            pred_id in _COPULAR
            or (text and bool(re.search(rf"\b{re.escape(subject)}\b\s+(?:is|are|was|were|be|am)\b", text.lower())))
        )
        if is_copular:
            rel_id = "is_a"
            rel_conf = 0.8
        else:
            rel_id = pred_id
            rel_conf = 0.7
    return {
        "subject": subject,
        "relation": rel_id,
        "object": obj,
        "confidence": rel_conf,
    }
