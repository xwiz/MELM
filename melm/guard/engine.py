"""Deterministic rule engine for MELM Guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import (
    ActionProposal,
    Condition,
    Fact,
    GuardDecision,
    GuardStatus,
    Rule,
    fact_key,
)


@dataclass(frozen=True)
class ConditionResult:
    matched: bool
    required_fact: str | None = None
    missing_fact: str | None = None
    evidence_event_ids: tuple[str, ...] = ()


class WorkingMemory:
    """Small append-only fact store with latest-fact lookup."""

    def __init__(self, facts: list[Fact] | tuple[Fact, ...] = ()) -> None:
        self._facts: list[Fact] = []
        for fact in facts:
            self.add(fact)

    def add(self, fact: Fact) -> None:
        self._facts.append(fact)
        self._facts.sort(key=lambda item: (item.time_index, item.fact_id))

    def facts(self) -> tuple[Fact, ...]:
        return tuple(self._facts)

    def latest_fact(
        self,
        subject: str,
        predicate: str,
        *,
        at_or_before: int | None = None,
    ) -> Fact | None:
        candidates = [
            fact
            for fact in self._facts
            if fact.subject == subject
            and fact.predicate == predicate
            and (at_or_before is None or fact.time_index <= at_or_before)
        ]
        return candidates[-1] if candidates else None

    def derive(
        self,
        *,
        fact_id: str,
        subject: str,
        predicate: str,
        value: Any,
        time_index: int,
        source_event_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Fact:
        fact = Fact(
            fact_id=fact_id,
            subject=subject,
            predicate=predicate,
            value=value,
            time_index=time_index,
            source_event_id=source_event_id,
            metadata=metadata or {"derived": "true"},
        )
        self.add(fact)
        return fact


class RuleEngine:
    """Evaluate production-style rules against working memory and an action."""

    STATUS_PRECEDENCE: tuple[GuardStatus, ...] = ("deny", "abstain", "warn", "allow")

    def __init__(self, rules: list[Rule] | tuple[Rule, ...]) -> None:
        self.rules = tuple(rules)

    def decide(
        self,
        memory: WorkingMemory,
        proposal: ActionProposal,
        *,
        current_time: int,
    ) -> GuardDecision:
        triggered: list[Rule] = []
        required_facts: list[str] = []
        missing_facts: list[str] = []
        evidence_event_ids: list[str] = []

        for rule in self.rules:
            condition_results = [
                evaluate_condition(memory, proposal, condition, current_time=current_time)
                for condition in rule.conditions
            ]
            required_facts.extend(
                result.required_fact for result in condition_results if result.required_fact
            )
            if not all(result.matched for result in condition_results):
                continue

            triggered.append(rule)
            missing_facts.extend(
                result.missing_fact for result in condition_results if result.missing_fact
            )
            for result in condition_results:
                evidence_event_ids.extend(result.evidence_event_ids)

        status = _decision_status(triggered)
        explanation = _explanation(status, triggered, proposal)
        return GuardDecision(
            status=status,
            triggered_rule_ids=tuple(rule.rule_id for rule in triggered),
            required_facts=tuple(dict.fromkeys(required_facts)),
            missing_facts=tuple(dict.fromkeys(missing_facts)),
            evidence_event_ids=tuple(dict.fromkeys(evidence_event_ids)),
            explanation=explanation,
        )


def evaluate_condition(
    memory: WorkingMemory,
    proposal: ActionProposal,
    condition: Condition,
    *,
    current_time: int,
) -> ConditionResult:
    subject = _format_template(condition.subject, proposal.parameters)
    predicate = _format_template(condition.predicate, proposal.parameters)
    required = fact_key(subject, predicate)
    fact = _lookup_condition_fact(memory, proposal, subject, predicate, current_time=current_time)
    evidence = (fact.source_event_id,) if fact and fact.source_event_id else ()

    if condition.operator == "missing":
        return ConditionResult(
            matched=fact is None,
            required_fact=required,
            missing_fact=required if fact is None else None,
            evidence_event_ids=(),
        )
    if condition.operator == "exists":
        return ConditionResult(
            matched=fact is not None,
            required_fact=required,
            missing_fact=required if fact is None else None,
            evidence_event_ids=evidence,
        )
    if fact is None:
        return ConditionResult(False, required_fact=required, missing_fact=required)

    expected = _resolve_value(memory, proposal, condition, current_time=current_time)
    matched = _compare(fact.value, condition.operator, expected, current_time - fact.time_index)
    return ConditionResult(
        matched=matched,
        required_fact=required,
        evidence_event_ids=evidence if matched else (),
    )


def _lookup_condition_fact(
    memory: WorkingMemory,
    proposal: ActionProposal,
    subject: str,
    predicate: str,
    *,
    current_time: int,
) -> Fact | None:
    if subject == "action":
        if predicate not in proposal.parameters:
            return None
        return Fact(
            fact_id=f"action:{proposal.action_id}:{predicate}",
            subject="action",
            predicate=predicate,
            value=proposal.parameters[predicate],
            time_index=current_time,
            source_event_id=None,
        )
    return memory.latest_fact(subject, predicate, at_or_before=current_time)


def _resolve_value(
    memory: WorkingMemory,
    proposal: ActionProposal,
    condition: Condition,
    *,
    current_time: int,
) -> Any:
    if not condition.value_source:
        return condition.value
    if condition.value_source.startswith("param:"):
        return proposal.parameters.get(condition.value_source.removeprefix("param:"))
    if condition.value_source.startswith("fact:"):
        ref = condition.value_source.removeprefix("fact:")
        subject, predicate = ref.rsplit(":", 1)
        subject = _format_template(subject, proposal.parameters)
        predicate = _format_template(predicate, proposal.parameters)
        fact = memory.latest_fact(subject, predicate, at_or_before=current_time)
        return fact.value if fact else None
    raise ValueError(f"Unsupported value_source: {condition.value_source!r}")


def _compare(left: Any, operator: str, right: Any, age: int) -> bool:
    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    if operator == "gt":
        return _to_float(left) > _to_float(right)
    if operator == "gte":
        return _to_float(left) >= _to_float(right)
    if operator == "lt":
        return _to_float(left) < _to_float(right)
    if operator == "lte":
        return _to_float(left) <= _to_float(right)
    if operator == "fresh_within":
        return age <= int(right)
    if operator == "stale_after":
        return age > int(right)
    raise ValueError(f"Unsupported condition operator: {operator!r}")


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_template(template: str, parameters: dict[str, Any]) -> str:
    try:
        return template.format(**parameters)
    except KeyError:
        return template


def _decision_status(triggered: list[Rule]) -> GuardStatus:
    effects = {rule.effect for rule in triggered}
    if "deny" in effects:
        return "deny"
    if "require_approval" in effects:
        return "abstain"
    if "warn" in effects:
        return "warn"
    return "allow"


def _explanation(status: GuardStatus, triggered: list[Rule], proposal: ActionProposal) -> str:
    if not triggered:
        return f"{proposal.action_type} allowed: no blocking rule matched."
    descriptions = "; ".join(f"{rule.rule_id}: {rule.description}" for rule in triggered)
    if status == "abstain":
        return f"{proposal.action_type} needs more approval/evidence: {descriptions}"
    return f"{proposal.action_type} {status}: {descriptions}"
