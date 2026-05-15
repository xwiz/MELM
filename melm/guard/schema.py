"""Typed records for MELM Guard procedural working memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


GuardStatus = Literal["allow", "deny", "warn", "abstain"]
RuleEffect = Literal["allow", "deny", "warn", "require_approval"]


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str
    predicate: str
    value: Any
    time_index: int
    source_event_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return fact_key(self.subject, self.predicate)


@dataclass(frozen=True)
class Condition:
    subject: str
    predicate: str
    operator: str
    value: Any = None
    value_source: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    conditions: tuple[Condition, ...]
    effect: RuleEffect
    severity: str = "hard"
    provenance: str = ""


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    action_type: str
    parameters: dict[str, Any]
    source_query: str
    proposed_by: str = "fixture_agent"
    malformed: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardDecision:
    status: GuardStatus
    triggered_rule_ids: tuple[str, ...]
    required_facts: tuple[str, ...]
    missing_facts: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    explanation: str


def fact_key(subject: str, predicate: str) -> str:
    return f"{subject}:{predicate}"
