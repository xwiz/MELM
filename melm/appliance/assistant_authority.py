"""M4 authority infrastructure: evidence packets, answer plans, and verification."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .local_assistant_router import AssistantDecision


@dataclass(frozen=True)
class AuthorityEvidenceItem:
    key: str
    kind: str
    value: str
    source: str
    license: str
    local_only: bool


@dataclass(frozen=True)
class AuthorityEvidencePacket:
    packet_id: str
    items: tuple[AuthorityEvidenceItem, ...]
    admitted_count: int
    blocked_keys: tuple[str, ...]
    boundary: str
    membrane_decision_id: str = ""


@dataclass(frozen=True)
class AnswerPlan:
    plan_id: str
    route: str
    mode: str
    requires: tuple[str, ...]
    forbids: tuple[str, ...]
    evidence_packet_id: str


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    schema_valid: bool
    packet_bound: bool
    answer_nonempty: bool
    failure_codes: tuple[str, ...]
    constraint_retention: float


@dataclass(frozen=True)
class DecoderResult:
    answer: str
    decoder: str
    tokens_generated: int


@dataclass(frozen=True)
class AuthorityInfo:
    evidence_packet: AuthorityEvidencePacket
    answer_plan: AnswerPlan
    verification: VerificationResult


_MODE_MAP: dict[str, str] = {
    "story": "narrative",
    "weather": "factual",
    "health_advice": "factual",
    "meal_suggestion": "factual",
    "common_sense_safety": "factual",
}

_FORBIDS_MAP: dict[str, tuple[str, ...]] = {
    "health_advice": ("diagnosis",),
    "meal_suggestion": (),
    "weather": (),
    "story": (),
}

_REQUIRES_MAP: dict[str, tuple[str, ...]] = {
    "weather": ("weather",),
    "health_advice": ("health_goal",),
    "meal_suggestion": ("food_inventory",),
}


def _packet_id(evidence_keys: tuple[str, ...], boundary: str) -> str:
    raw = ",".join(sorted(evidence_keys)) + "|" + boundary
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def build_evidence_packet(
    evidence_keys: tuple[str, ...],
    items: tuple[AuthorityEvidenceItem, ...],
    boundary: str,
    membrane_decision_id: str = "",
) -> AuthorityEvidencePacket:
    item_map = {item.key: item for item in items}
    admitted: list[AuthorityEvidenceItem] = []
    blocked: list[str] = []
    for key in evidence_keys:
        if key in item_map:
            admitted.append(item_map[key])
        else:
            blocked.append(key)
    return AuthorityEvidencePacket(
        packet_id=_packet_id(evidence_keys, boundary),
        items=tuple(admitted),
        admitted_count=len(admitted),
        blocked_keys=tuple(blocked),
        boundary=boundary,
        membrane_decision_id=membrane_decision_id,
    )


def build_answer_plan(
    decision: AssistantDecision,
    packet: AuthorityEvidencePacket,
) -> AnswerPlan:
    raw = decision.intent + "|" + decision.route + "|" + "|".join(decision.evidence_keys)
    plan_id = hashlib.md5(raw.encode()).hexdigest()[:16]
    if decision.route == "reject":
        mode = "refusal"
    else:
        mode = _MODE_MAP.get(decision.intent, "factual")
    requires = _REQUIRES_MAP.get(decision.intent, (decision.intent,))
    forbids = _FORBIDS_MAP.get(decision.intent, ())
    return AnswerPlan(
        plan_id=plan_id,
        route=decision.route,
        mode=mode,
        requires=requires,
        forbids=forbids,
        evidence_packet_id=packet.packet_id,
    )


_VALID_MODES = frozenset({"factual", "narrative", "refusal"})


def verify_answer(
    plan: AnswerPlan,
    answer: str,
    packet: AuthorityEvidencePacket,
) -> VerificationResult:
    failure_codes: list[str] = []

    # Schema validation
    schema_valid = bool(plan.plan_id) and plan.mode in _VALID_MODES
    if not schema_valid:
        failure_codes.append("schema_invalid")

    # Packet binding
    packet_bound = plan.evidence_packet_id == packet.packet_id
    if not packet_bound:
        failure_codes.append("packet_unbound")

    # Non-empty answer
    answer_nonempty = bool(answer and answer.strip())
    if not answer_nonempty:
        failure_codes.append("empty_answer")

    # Constraint check — forbids (honor negation: "not a diagnosis" or "no diagnosis")
    constraint_retention = 1.0
    answer_lower = answer.lower() if answer else ""
    for forbidden in plan.forbids:
        f_lower = forbidden.lower()
        if f_lower in answer_lower:
            negated = re.search(
                rf'\b(?:not\s+a(?:n)?|no)\s+{re.escape(f_lower)}\b',
                answer_lower,
            )
            if not negated:
                if "constraint_violation" not in failure_codes:
                    failure_codes.append("constraint_violation")
                constraint_retention = 0.0
                break

    passed = not failure_codes
    return VerificationResult(
        passed=passed,
        schema_valid=schema_valid,
        packet_bound=packet_bound,
        answer_nonempty=answer_nonempty,
        failure_codes=tuple(failure_codes),
        constraint_retention=constraint_retention,
    )
