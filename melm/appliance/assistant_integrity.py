"""Response-integrity scoring for the local assistant improvement loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .local_assistant_router import AssistantDebugParse, AssistantDecision


UNDERSTANDING_RESEARCH_FLOOR = 0.72
RESPONSE_INTEGRITY_FLOOR = 0.75


@dataclass(frozen=True)
class ResponseIntegrityAssessment:
    """Explain whether a turn was understood, grounded, and worth researching."""

    understanding_score: float
    response_integrity_score: float
    overall_score: float
    band: str
    research_recommended: bool
    candidate_kinds: tuple[str, ...]
    research_topics: tuple[str, ...]
    components: dict[str, float]
    flags: tuple[str, ...]
    schema: str = "melm.response_integrity.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "understanding_score": self.understanding_score,
            "response_integrity_score": self.response_integrity_score,
            "overall_score": self.overall_score,
            "band": self.band,
            "research_recommended": self.research_recommended,
            "candidate_kinds": list(self.candidate_kinds),
            "research_topics": list(self.research_topics),
            "components": dict(self.components),
            "flags": list(self.flags),
            "promotion_policy": "quarantine_evaluate_then_promote",
        }


def assess_response_integrity(
    decision: AssistantDecision,
    debug_parse: AssistantDebugParse | dict[str, Any],
    *,
    synthesis: dict[str, Any] | None = None,
    membrane: dict[str, Any] | None = None,
) -> ResponseIntegrityAssessment:
    """Score language understanding separately from response integrity.

    A correct abstention may have high response integrity while still exposing
    a language gap. This distinction keeps research pressure from rewarding
    fabricated local answers.
    """

    debug = debug_parse.to_dict() if isinstance(debug_parse, AssistantDebugParse) else dict(debug_parse)
    nlp = dict(debug.get("nlp", {}))
    uol = dict(debug.get("uol", {}))
    frame = dict(debug.get("chat_frame", {}))
    synthesis_payload = dict(synthesis or {})
    membrane_payload = dict(membrane or {})

    token_count = max(1, int(nlp.get("token_count", len(debug.get("tokens", []))) or 0))
    raw_unknown_tokens = tuple(
        dict.fromkeys(str(item).lower() for item in nlp.get("unknown_tokens", []) if str(item).strip())
    )
    semantic_unknown_tokens = tuple(
        dict.fromkeys(
            str(item).lower()
            for item in nlp.get("semantic_unknown_tokens", [])
            if str(item).strip()
        )
    )
    unknown_tokens = _researchable_unknown_tokens(
        tuple(dict.fromkeys((*raw_unknown_tokens, *semantic_unknown_tokens))),
        uol,
    )
    unknown_count = min(token_count, len(unknown_tokens))
    lexical_coverage = max(0.0, 1.0 - unknown_count / token_count)
    parse_score = _bounded(uol.get("parse_score", 0.0))
    composition_strength = _composition_strength(nlp, frame)
    intent_resolution = 0.0 if decision.intent == "unknown" else 1.0
    route_alignment = 1.0 if (
        str(frame.get("intent", "")) == decision.intent
        and str(frame.get("route", "")) == decision.route
    ) else 0.0
    understanding_score = _rounded(
        0.35 * lexical_coverage
        + 0.25 * parse_score
        + 0.2 * composition_strength
        + 0.1 * intent_resolution
        + 0.1 * route_alignment
    )

    route_confidence = _bounded(decision.confidence)
    grounding_strength = _grounding_strength(decision, synthesis_payload)
    route_discipline = _route_discipline(decision, synthesis_payload)
    boundary_integrity = _boundary_integrity(decision, membrane_payload, synthesis_payload)
    response_integrity_score = _rounded(
        0.3 * route_confidence
        + 0.3 * grounding_strength
        + 0.2 * route_discipline
        + 0.2 * boundary_integrity
    )
    overall_score = _rounded(0.6 * understanding_score + 0.4 * response_integrity_score)

    flags: list[str] = []
    candidate_kinds: list[str] = []
    research_topics: list[str] = list(unknown_tokens)
    if unknown_tokens:
        flags.append("unknown_tokens")
        candidate_kinds.append("language_gap")
    if composition_strength < 0.7:
        flags.append("weak_uol_composition")
        candidate_kinds.append("language_gap")
    if decision.intent == "unknown":
        flags.append("unknown_intent")
        candidate_kinds.append("language_gap")
    if route_confidence < 0.65:
        flags.append("low_route_confidence")
        candidate_kinds.append("routing_review")
    if response_integrity_score < RESPONSE_INTEGRITY_FLOOR:
        flags.append("low_response_integrity")
        candidate_kinds.append("response_quality")
    warnings = tuple(str(item) for item in synthesis_payload.get("quality", {}).get("warnings", []))
    if warnings:
        flags.extend(f"synthesis:{warning}" for warning in warnings)
        candidate_kinds.append("response_quality")
    if decision.route == "cloud_handoff":
        flags.append("cloud_capability_gap")
        candidate_kinds.append("capability_gap")
    elif decision.route == "clarify" and decision.reason == "unknown_intent":
        flags.append("clarification_language_gap")
        candidate_kinds.append("language_gap")

    if not research_topics and "language_gap" in candidate_kinds:
        object_value = str(uol.get("object", "") or "").strip()
        action_value = str(uol.get("action", "") or "").strip()
        research_topics.extend(item for item in (object_value, action_value) if item)

    candidate_kinds = list(dict.fromkeys(candidate_kinds))
    research_topics = list(dict.fromkeys(research_topics))
    research_recommended = bool(
        understanding_score < UNDERSTANDING_RESEARCH_FLOOR
        or response_integrity_score < RESPONSE_INTEGRITY_FLOOR
        or "language_gap" in candidate_kinds
    )
    band = _integrity_band(overall_score)
    if research_recommended and "language_gap" in candidate_kinds and band == "reliable":
        band = "review"
    return ResponseIntegrityAssessment(
        understanding_score=understanding_score,
        response_integrity_score=response_integrity_score,
        overall_score=overall_score,
        band=band,
        research_recommended=research_recommended,
        candidate_kinds=tuple(candidate_kinds),
        research_topics=tuple(research_topics),
        components={
            "lexical_coverage": _rounded(lexical_coverage),
            "raw_unknown_token_ratio": _rounded(len(raw_unknown_tokens) / token_count),
            "parse_score": _rounded(parse_score),
            "composition_strength": _rounded(composition_strength),
            "intent_resolution": _rounded(intent_resolution),
            "route_alignment": _rounded(route_alignment),
            "route_confidence": _rounded(route_confidence),
            "grounding_strength": _rounded(grounding_strength),
            "route_discipline": _rounded(route_discipline),
            "boundary_integrity": _rounded(boundary_integrity),
        },
        flags=tuple(dict.fromkeys(flags)),
    )


def _composition_strength(nlp: dict[str, Any], frame: dict[str, Any]) -> float:
    primary = dict(nlp.get("primary_domain_evidence", {}))
    source = str(primary.get("source", ""))
    source_score = {
        "token_role_relation": 1.0,
        "slot_role_relation": 1.0,
        "weighted_functional_relation": 0.92,
        "no_local_composition": 0.15,
    }.get(source, 0.35)
    registry_score = 1.0 if str(frame.get("frame_registry", "")).strip() else 0.0
    routing_basis = tuple(str(item) for item in frame.get("primary_routing_basis", []))
    basis_score = 1.0 if routing_basis else 0.0
    return _bounded(0.5 * source_score + 0.3 * registry_score + 0.2 * basis_score)


def _researchable_unknown_tokens(
    unknown_tokens: tuple[str, ...],
    uol: dict[str, Any],
) -> tuple[str, ...]:
    resolved_slots = {
        token
        for key in ("subject", "action", "source", "target")
        for token in str(uol.get(key, "") or "").lower().split()
    }
    functional = {
        "am",
        "an",
        "as",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "does",
        "had",
        "has",
        "have",
        "may",
        "might",
        "of",
        "on",
        "or",
        "please",
        "shall",
        "was",
        "were",
        "will",
        "with",
        "would",
    }
    return tuple(
        token
        for token in unknown_tokens
        if token not in resolved_slots and token not in functional
    )


def _grounding_strength(decision: AssistantDecision, synthesis: dict[str, Any]) -> float:
    quality = dict(synthesis.get("quality", {}))
    if quality:
        return _bounded(quality.get("score", 0.0))
    if decision.route in {"cloud_handoff", "external_fetch", "clarify", "reject"}:
        return 1.0
    if decision.evidence_keys:
        return 0.8
    return 0.25


def _route_discipline(decision: AssistantDecision, synthesis: dict[str, Any]) -> float:
    quality = dict(synthesis.get("quality", {}))
    if "route_discipline" in quality:
        return _bounded(quality.get("route_discipline", 0.0))
    if decision.route in {"cloud_handoff", "external_fetch", "clarify", "reject"}:
        return 1.0
    return 0.8 if decision.evidence_keys else 0.4


def _boundary_integrity(
    decision: AssistantDecision,
    membrane: dict[str, Any],
    synthesis: dict[str, Any],
) -> float:
    quality = dict(synthesis.get("quality", {}))
    privacy = _bounded(quality.get("local_privacy_discipline", 1.0))
    if not membrane:
        return privacy
    allowed = bool(membrane.get("allowed", False))
    boundary = str(membrane.get("boundary_crossed", ""))
    if decision.route == "reject" and not allowed:
        return 1.0
    if not allowed:
        return 0.0
    if boundary == "cloud" and membrane.get("personal_facts_included"):
        return 0.0
    return privacy


def _integrity_band(score: float) -> str:
    if score >= 0.8:
        return "reliable"
    if score >= 0.6:
        return "review"
    return "low"


def _bounded(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _rounded(value: float) -> float:
    return round(_bounded(value), 3)
