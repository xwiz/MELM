"""Validation-suite summaries for Phase 1 reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    severity: str
    passed: bool
    metric: float | None
    threshold: float | None
    detail: str


@dataclass(frozen=True)
class ValidationSuiteReport:
    overall_passed: bool
    hard_failures: int
    checks: list[ValidationCheck]


def evaluate_validation_suite(payload: dict[str, Any]) -> ValidationSuiteReport:
    """Evaluate a Phase 1 report payload against current validation gates."""

    checks = [
        _memory_check(payload),
        _abstention_check(payload),
        _tokenizer_decision_check(payload),
        _tokenizer_stability_check(payload),
        _grounding_check(payload),
        _state_resolution_check(payload),
    ]
    if payload.get("authored_dialogue"):
        checks.extend(_dialogue_checks(payload))
    if payload.get("sample_transcript"):
        checks.extend(_sample_transcript_checks(payload))

    hard_failures = sum(
        1 for check in checks if check.severity == "hard" and not check.passed
    )
    return ValidationSuiteReport(
        overall_passed=hard_failures == 0,
        hard_failures=hard_failures,
        checks=checks,
    )


def _memory_check(payload: dict[str, Any]) -> ValidationCheck:
    gate = payload["memory_gate"]
    return ValidationCheck(
        name="synthetic_event_memory_gain",
        severity="hard",
        passed=bool(gate["passed"]),
        metric=float(gate["metric"]),
        threshold=float(gate["threshold"]),
        detail=gate["recommendation"],
    )


def _abstention_check(payload: dict[str, Any]) -> ValidationCheck:
    gate = payload["memory_abstention"]["gate"]
    selected = payload["memory_abstention"]["calibrated"]["selected"]
    report = selected["evaluation_report"]
    return ValidationCheck(
        name="held_out_abstention_calibration",
        severity="hard",
        passed=bool(gate["passed"]),
        metric=float(gate["metric"]),
        threshold=float(gate["threshold"]),
        detail=(
            f"{selected['confidence_method']} at threshold {selected['threshold']:.2f}; "
            f"positive recall {report['positive_recall']:.2%}, "
            f"negative abstention {report['negative_abstention']:.2%}"
        ),
    )


def _tokenizer_decision_check(payload: dict[str, Any]) -> ValidationCheck:
    decision = payload["tokenizer_decision"]
    internally_consistent = (
        (decision["gate_passed"] and decision["decision"] == "primary_candidate")
        or (
            not decision["gate_passed"]
            and decision["decision"] in {"auxiliary_only", "reject_for_now"}
        )
    )
    return ValidationCheck(
        name="tokenizer_decision_consistency",
        severity="hard",
        passed=internally_consistent,
        metric=float(decision["lm_nll_gain"]),
        threshold=0.0,
        detail=(
            f"decision={decision['decision']}; "
            f"LM gain {decision['lm_nll_gain']:.3f}; "
            f"boundary gain {decision['boundary_f1_gain']:.2%}"
        ),
    )


def _tokenizer_stability_check(payload: dict[str, Any]) -> ValidationCheck:
    stability = payload.get("tokenizer_stability")
    if not stability:
        return ValidationCheck(
            name="tokenizer_stability",
            severity="advisory",
            passed=False,
            metric=None,
            threshold=None,
            detail="No cross-fold tokenizer stability report is present.",
        )
    return ValidationCheck(
        name="tokenizer_stability",
        severity="advisory",
        passed=bool(stability["stable_primary_candidate"]),
        metric=float(stability["average_lm_nll_gain"]),
        threshold=0.0,
        detail=(
            f"morphology win rate {stability['morph_win_rate']:.2%}; "
            f"best baseline {stability['best_baseline_tokenizer']}"
        ),
    )


def _grounding_check(payload: dict[str, Any]) -> ValidationCheck:
    grounding = payload["grounding"]
    return ValidationCheck(
        name="state_grounding_seed_accuracy",
        severity="hard",
        passed=float(grounding["accuracy"]) >= 0.95,
        metric=float(grounding["accuracy"]),
        threshold=0.95,
        detail=f"{grounding['cases']} seed cases",
    )


def _state_resolution_check(payload: dict[str, Any]) -> ValidationCheck:
    report = payload.get("state_resolution")
    if not report:
        return ValidationCheck(
            name="state_resolution",
            severity="hard",
            passed=False,
            metric=None,
            threshold=0.95,
            detail="No state-resolution report is present.",
        )
    passed = float(report["accuracy"]) >= 0.95 and float(report["false_positive_rate"]) <= 0.05
    return ValidationCheck(
        name="state_resolution",
        severity="hard",
        passed=passed,
        metric=float(report["accuracy"]),
        threshold=0.95,
        detail=(
            f"{report['cases']} object-location cases; "
            f"false positive rate {report['false_positive_rate']:.2%}"
        ),
    )


def _dialogue_checks(payload: dict[str, Any]) -> list[ValidationCheck]:
    dialogue = payload["authored_dialogue"]
    memory_gate = dialogue["memory_gate"]
    abstention_gate = dialogue["abstention_gate"]
    abstention = dialogue["abstention_report"]
    return [
        ValidationCheck(
            name="authored_dialogue_memory_gain",
            severity="hard",
            passed=bool(memory_gate["passed"]),
            metric=float(memory_gate["metric"]),
            threshold=float(memory_gate["threshold"]),
            detail=memory_gate["recommendation"],
        ),
        ValidationCheck(
            name="authored_dialogue_abstention",
            severity="hard",
            passed=bool(abstention_gate["passed"]),
            metric=float(abstention_gate["metric"]),
            threshold=float(abstention_gate["threshold"]),
            detail=(
                f"positive recall {abstention['positive_recall']:.2%}; "
                f"negative abstention {abstention['negative_abstention']:.2%}"
            ),
        ),
    ]


def _sample_transcript_checks(payload: dict[str, Any]) -> list[ValidationCheck]:
    sample = payload["sample_transcript"]
    memory_gate = sample["memory_gate"]
    abstention_gate = sample["abstention_gate"]
    abstention = sample["abstention_report"]
    checks = [
        ValidationCheck(
            name="sample_transcript_memory_gain",
            severity="hard",
            passed=bool(memory_gate["passed"]),
            metric=float(memory_gate["metric"]),
            threshold=float(memory_gate["threshold"]),
            detail=memory_gate["recommendation"],
        ),
        ValidationCheck(
            name="sample_transcript_abstention",
            severity="hard",
            passed=bool(abstention_gate["passed"]),
            metric=float(abstention_gate["metric"]),
            threshold=float(abstention_gate["threshold"]),
            detail=(
                f"positive recall {abstention['positive_recall']:.2%}; "
                f"negative abstention {abstention['negative_abstention']:.2%}"
            ),
        ),
    ]
    state = sample.get("state_resolution_report")
    if state:
        checks.append(
            ValidationCheck(
                name="sample_transcript_state_resolution",
                severity="hard",
                passed=(
                    float(state["accuracy"]) >= 0.95
                    and float(state["false_positive_rate"]) <= 0.05
                ),
                metric=float(state["accuracy"]),
                threshold=0.95,
                detail=(
                    f"{sample['state_cases']} annotated state cases; "
                    f"false positive rate {state['false_positive_rate']:.2%}"
                ),
            )
        )
    return checks
