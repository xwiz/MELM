"""Benchmark evaluation helpers for MELM Guard."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from .engine import RuleEngine, WorkingMemory
from .schema import ActionProposal, GuardDecision, GuardStatus, Rule


class GuardCaseLike(Protocol):
    proposal: ActionProposal
    expected_status: GuardStatus
    category: str


@dataclass(frozen=True)
class GuardPrediction:
    action_id: str
    category: str
    expected_status: GuardStatus
    schema_only_status: GuardStatus
    prompt_only_status: GuardStatus
    melm_status: GuardStatus
    melm_decision: GuardDecision


@dataclass(frozen=True)
class GuardBenchmarkReport:
    cases: int
    melm_accuracy: float
    schema_only_accuracy: float
    prompt_only_accuracy: float
    melm_false_allow_rate: float
    schema_only_false_allow_rate: float
    prompt_only_false_allow_rate: float
    false_allow_reduction_vs_schema: float
    valid_action_allow_rate: float
    traceability: float
    gate_passed: bool
    predictions: list[GuardPrediction]
    by_category: dict[str, "GuardBenchmarkReport"] | None = None


REQUIRED_PARAMS_BY_ACTION = {
    "approve_refund": ("order_id", "customer_id", "amount"),
    "deny_refund": ("order_id", "customer_id", "reason"),
    "request_manager_approval": ("order_id", "customer_id", "amount"),
}


def evaluate_guard_benchmark(
    facts,
    rules: list[Rule] | tuple[Rule, ...],
    cases: list[GuardCaseLike],
    *,
    current_time: int,
) -> GuardBenchmarkReport:
    memory = WorkingMemory(facts)
    engine = RuleEngine(rules)
    predictions = [
        GuardPrediction(
            action_id=case.proposal.action_id,
            category=case.category,
            expected_status=case.expected_status,
            schema_only_status=schema_only_status(case.proposal),
            prompt_only_status=prompt_only_status(case.proposal),
            melm_status=(
                decision := engine.decide(
                    memory,
                    case.proposal,
                    current_time=getattr(case, "current_time", None) or current_time,
                )
            ).status,
            melm_decision=decision,
        )
        for case in cases
    ]
    report = _summarize(predictions)
    buckets: dict[str, list[GuardPrediction]] = defaultdict(list)
    for prediction in predictions:
        buckets[prediction.category].append(prediction)
    return GuardBenchmarkReport(
        cases=report.cases,
        melm_accuracy=report.melm_accuracy,
        schema_only_accuracy=report.schema_only_accuracy,
        prompt_only_accuracy=report.prompt_only_accuracy,
        melm_false_allow_rate=report.melm_false_allow_rate,
        schema_only_false_allow_rate=report.schema_only_false_allow_rate,
        prompt_only_false_allow_rate=report.prompt_only_false_allow_rate,
        false_allow_reduction_vs_schema=report.false_allow_reduction_vs_schema,
        valid_action_allow_rate=report.valid_action_allow_rate,
        traceability=report.traceability,
        gate_passed=report.gate_passed,
        predictions=predictions,
        by_category={category: _summarize(bucket) for category, bucket in sorted(buckets.items())},
    )


def schema_only_status(proposal: ActionProposal) -> GuardStatus:
    required = REQUIRED_PARAMS_BY_ACTION.get(proposal.action_type)
    if proposal.malformed or required is None:
        return "deny"
    if all(param in proposal.parameters for param in required):
        return "allow"
    return "deny"


def prompt_only_status(proposal: ActionProposal) -> GuardStatus:
    return "deny" if proposal.malformed else "allow"


def _summarize(predictions: list[GuardPrediction]) -> GuardBenchmarkReport:
    invalid = [prediction for prediction in predictions if prediction.expected_status != "allow"]
    valid = [prediction for prediction in predictions if prediction.expected_status == "allow"]

    melm_false_allow = _false_allow_rate(predictions, "melm_status")
    schema_false_allow = _false_allow_rate(predictions, "schema_only_status")
    prompt_false_allow = _false_allow_rate(predictions, "prompt_only_status")
    valid_allow = (
        sum(1 for prediction in valid if prediction.melm_status == "allow") / len(valid)
        if valid
        else 0.0
    )
    traceable = [
        prediction
        for prediction in predictions
        if prediction.melm_status != "allow"
    ]
    traceability = (
        sum(
            1
            for prediction in traceable
            if prediction.melm_decision.triggered_rule_ids or prediction.melm_decision.missing_facts
        )
        / len(traceable)
        if traceable
        else 1.0
    )
    reduction = (
        (schema_false_allow - melm_false_allow) / schema_false_allow
        if schema_false_allow
        else 1.0
    )
    gate_passed = (
        reduction >= 0.50
        and valid_allow >= 0.90
        and traceability == 1.0
        and melm_false_allow < schema_false_allow
    )
    return GuardBenchmarkReport(
        cases=len(predictions),
        melm_accuracy=_accuracy(predictions, "melm_status"),
        schema_only_accuracy=_accuracy(predictions, "schema_only_status"),
        prompt_only_accuracy=_accuracy(predictions, "prompt_only_status"),
        melm_false_allow_rate=melm_false_allow,
        schema_only_false_allow_rate=schema_false_allow,
        prompt_only_false_allow_rate=prompt_false_allow,
        false_allow_reduction_vs_schema=reduction,
        valid_action_allow_rate=valid_allow,
        traceability=traceability,
        gate_passed=gate_passed,
        predictions=predictions,
        by_category=None,
    )


def _accuracy(predictions: list[GuardPrediction], field_name: str) -> float:
    if not predictions:
        return 0.0
    return sum(
        1
        for prediction in predictions
        if getattr(prediction, field_name) == prediction.expected_status
    ) / len(predictions)


def _false_allow_rate(predictions: list[GuardPrediction], field_name: str) -> float:
    invalid = [prediction for prediction in predictions if prediction.expected_status != "allow"]
    if not invalid:
        return 0.0
    return sum(1 for prediction in invalid if getattr(prediction, field_name) == "allow") / len(invalid)
