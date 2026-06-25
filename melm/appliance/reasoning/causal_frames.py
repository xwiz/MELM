"""Cached causal frame index (V0.4 atomic causality).

Builds in-memory indexes from causal_frames.v1.json with fallback to
causal_effects.v1.json seed data. Single module-level cache per import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import re


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalEffect:
    predicate_id: str
    state: str
    domain: str
    target_role: str
    relation: str
    cause_kind: str
    confidence: float
    provenance: str


@dataclass(frozen=True)
class CausalFrameIndex:
    by_predicate: Mapping[str, Sequence[CausalEffect]]
    by_effect_state: Mapping[str, Sequence[CausalEffect]]
    by_active_entity: Mapping[str, Sequence[str]]
    by_surface_alias: Mapping[str, dict[str, Any]]
    state_definitions: Mapping[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_INDEX_CACHE: CausalFrameIndex | None = None


def _clear_cache() -> None:
    global _INDEX_CACHE
    _INDEX_CACHE = None


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


def _load_and_validate_frames() -> dict[str, Any]:
    from melm.contracts import load_causal_frames
    return load_causal_frames()


def _load_causal_effects_fallback() -> dict[str, Any]:
    try:
        from melm.contracts import load_causal_effects
        return load_causal_effects()
    except Exception:
        return {"rules": {}}


def load_causal_frame_index() -> CausalFrameIndex:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    frames_data = _load_and_validate_frames()
    predicate_frames: dict[str, Any] = frames_data.get("predicate_frames", {})
    state_defs: dict[str, Any] = frames_data.get("state_definitions", {})
    entity_affordances: dict[str, Any] = frames_data.get("active_entity_affordances", {})
    surface_aliases: dict[str, Any] = frames_data.get("surface_aliases", {})

    # Build by_predicate index from causal_frames.v1.json
    by_predicate: dict[str, list[CausalEffect]] = {}
    for pid, frame in predicate_frames.items():
        cause_kind = frame.get("default_cause_kind", "unknown")
        effects_list = frame.get("effects", [])
        effects: list[CausalEffect] = []
        for eff in effects_list:
            effects.append(CausalEffect(
                predicate_id=pid,
                state=str(eff.get("state", "")),
                domain=str(eff.get("domain", "")),
                target_role=str(eff.get("target_role", "")),
                relation=str(eff.get("relation", "")),
                cause_kind=cause_kind,
                confidence=float(eff.get("confidence", 0.5)),
                provenance="contract:causal_frames.v1",
            ))
        if effects:
            by_predicate[pid] = effects

    # Fallback: import predicates from causal_effects.v1.json that are absent
    fallback = _load_causal_effects_fallback()
    for verb, rule in fallback.get("rules", {}).items():
        if verb in by_predicate:
            continue
        confidence = float(rule.get("confidence", 0.5))
        effects = rule.get("effects", {})
        effect_list: list[CausalEffect] = []
        for domain, states in effects.items():
            for state in states:
                effect_list.append(CausalEffect(
                    predicate_id=verb,
                    state=str(state).lower(),
                    domain=str(domain).lower(),
                    target_role="patient",
                    relation="causes",
                    cause_kind="unknown",
                    confidence=confidence,
                    provenance="seed:causal_effects.v1",
                ))
        if effect_list:
            by_predicate[verb] = effect_list

    # Build by_effect_state reverse index
    by_effect_state: dict[str, list[CausalEffect]] = {}
    for effects in by_predicate.values():
        for eff in effects:
            by_effect_state.setdefault(eff.state, []).append(eff)

    # Build by_active_entity index
    by_active_entity: dict[str, list[str]] = {}
    for entity_id, entry in entity_affordances.items():
        bindings = entry.get("role_bindings", [])
        pred_ids = [b["predicate_id"] for b in bindings if isinstance(b.get("predicate_id"), str)]
        for surface in entry.get("surface_forms", []):
            by_active_entity[surface.lower()] = pred_ids

    # Build by_surface_alias index (lowercased)
    by_surface_alias: dict[str, dict[str, Any]] = {}
    for alias, entry in surface_aliases.items():
        if alias.lower() not in by_surface_alias:
            by_surface_alias[alias.lower()] = entry
        # Also index the canonical form (only if not already set by a primary entry)
        canonical = entry.get("canonical")
        if canonical and isinstance(canonical, str) and canonical.lower() != alias.lower():
            if canonical.lower() not in by_surface_alias:
                by_surface_alias[canonical.lower()] = entry

    _INDEX_CACHE = CausalFrameIndex(
        by_predicate=by_predicate,
        by_effect_state=by_effect_state,
        by_active_entity=by_active_entity,
        by_surface_alias=by_surface_alias,
        state_definitions=state_defs,
    )
    return _INDEX_CACHE


# ---------------------------------------------------------------------------
# Surface normalization
# ---------------------------------------------------------------------------


def normalize_causal_surface(text: str, index: CausalFrameIndex | None = None) -> str:
    """Normalize surface text through surface_aliases index.

    Returns canonical alias or the original text lowercased.
    """
    if index is None:
        index = load_causal_frame_index()
    key = text.strip().lower()
    entry = index.by_surface_alias.get(key)
    if entry is not None:
        canonical = entry.get("canonical")
        if isinstance(canonical, str) and canonical:
            return canonical
    # Strip articles for matching
    stripped = _ARTICLE_RE.sub("", key).strip()
    if stripped != key:
        entry = index.by_surface_alias.get(stripped)
        if entry is not None:
            canonical = entry.get("canonical")
            if isinstance(canonical, str) and canonical:
                return canonical
    return key


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


# ---------------------------------------------------------------------------
# State definition lookup
# ---------------------------------------------------------------------------


def explain_effect_state(
    effect_state: str,
    *,
    theme: str = "",
    store: Any | None = None,
    index: CausalFrameIndex | None = None,
) -> dict[str, Any]:
    """Return state definition + candidate causes for a given effect state.

    Structured result shape:
      {task, effect, theme, state_definition, candidate_causes, selected_cause}
    """
    if index is None:
        index = load_causal_frame_index()

    state_key = effect_state.strip().lower()
    state_def = index.state_definitions.get(state_key, {})
    definition_text = str(state_def.get("definition", "")) if state_def else ""
    definition = {"state": state_key, "definition": definition_text} if definition_text else {"state": state_key}

    # Find candidate causes from by_effect_state
    candidates = index.by_effect_state.get(state_key, [])
    cause_list: list[dict[str, Any]] = []
    for cand in candidates:
        entry = {
            "predicate_id": cand.predicate_id,
            "cause_kind": cand.cause_kind,
            "confidence": cand.confidence,
            "provenance": cand.provenance,
        }
        if cand.target_role:
            entry["target_role"] = cand.target_role
        cause_list.append(entry)

    # Deduplicate by predicate_id (keep highest confidence)
    seen: dict[str, dict[str, Any]] = {}
    for entry in cause_list:
        pid = entry["predicate_id"]
        if pid not in seen or entry["confidence"] > seen[pid]["confidence"]:
            seen[pid] = entry
    deduped = sorted(seen.values(), key=lambda e: -e["confidence"])

    selected = deduped[0]["predicate_id"] if deduped else None

    return {
        "task": "causal_explanation",
        "effect": state_key,
        "theme": theme,
        "state_definition": definition,
        "candidate_causes": deduped,
        "selected_cause": selected,
    }


# ---------------------------------------------------------------------------
# Effect prediction
# ---------------------------------------------------------------------------


def predict_effects(
    cause: str,
    *,
    actor: str = "",
    patient: str = "",
    store: Any | None = None,
    index: CausalFrameIndex | None = None,
) -> dict[str, Any]:
    """Return predicted effects for a given cause predicate.

    Structured result shape:
      {task, cause, actor, patient, effects}
    Each effect includes state_definition from the index when available.
    """
    if index is None:
        index = load_causal_frame_index()

    cause_key = cause.strip().lower()
    effects = index.by_predicate.get(cause_key, [])

    effect_list: list[dict[str, Any]] = []
    for eff in effects:
        entry: dict[str, Any] = {
            "state": eff.state,
            "domain": eff.domain,
            "target_role": eff.target_role,
            "confidence": eff.confidence,
        }
        # Look up the original frame for precondition_state
        frames_data = _load_and_validate_frames()
        pred_frame = frames_data.get("predicate_frames", {}).get(cause_key, {})
        for orig_eff in pred_frame.get("effects", []):
            if orig_eff.get("state") == eff.state and "precondition_state" in orig_eff:
                entry["precondition_state"] = orig_eff["precondition_state"]
                break
        # Include state definition for richer NLG rendering
        state_def = index.state_definitions.get(eff.state, {})
        if state_def:
            entry["state_definition_text"] = state_def.get("definition", "")
            entry["state_aliases"] = state_def.get("aliases", [])
        effect_list.append(entry)

    return {
        "task": "causal_prediction",
        "cause": cause_key,
        "actor": actor,
        "patient": patient,
        "effects": effect_list,
    }


# ---------------------------------------------------------------------------
# Active entity resolution
# ---------------------------------------------------------------------------


def resolve_active_entity(
    entity_surface: str,
    index: CausalFrameIndex | None = None,
) -> Sequence[str]:
    """Resolve entity surface text to predicate IDs via active_entity_affordances."""
    if index is None:
        index = load_causal_frame_index()
    key = entity_surface.strip().lower()
    return index.by_active_entity.get(key, [])
