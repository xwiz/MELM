"""Gate for moving from memory components to a persistent-dialogue demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryIntegrationGate:
    candidate: str
    decision: str
    state_assisted_accuracy: float
    state_assisted_lift: float
    state_answer_rate: float
    synthetic_memory_gain: float
    authored_dialogue_memory_gain: float
    sample_transcript_memory_gain: float
    authored_dialogue_abstention_metric: float
    sample_transcript_abstention_metric: float
    recommendation: str


def decide_memory_integration_gate(
    validation_suite_payload: dict,
    state_assisted_payload: dict,
    *,
    candidate: str = "tiered_morph_unigram",
    minimum_state_assisted_accuracy: float = 0.95,
    minimum_state_answer_rate: float = 0.80,
    minimum_memory_gain: float = 0.15,
    minimum_abstention_metric: float = 1.0,
) -> MemoryIntegrationGate:
    """Decide whether event/state memory is ready for a dialogue demo."""

    checks = {
        check["name"]: check
        for check in validation_suite_payload["checks"]
    }
    state_run = _state_run(state_assisted_payload, candidate)
    state_report = state_run["report"]
    synthetic_gain = float(checks["synthetic_event_memory_gain"]["metric"])
    authored_gain = float(checks["authored_dialogue_memory_gain"]["metric"])
    transcript_gain = float(checks["sample_transcript_memory_gain"]["metric"])
    authored_abstention = float(checks["authored_dialogue_abstention"]["metric"])
    transcript_abstention = float(checks["sample_transcript_abstention"]["metric"])
    state_accuracy = float(state_report["accuracy"])
    state_lift = float(state_report["accuracy_lift"])
    state_answer_rate = float(state_report["state_answer_rate"])

    passed = (
        bool(validation_suite_payload["overall_passed"])
        and state_accuracy >= minimum_state_assisted_accuracy
        and state_answer_rate >= minimum_state_answer_rate
        and synthetic_gain >= minimum_memory_gain
        and authored_gain >= minimum_memory_gain
        and transcript_gain >= minimum_memory_gain
        and authored_abstention >= minimum_abstention_metric
        and transcript_abstention >= minimum_abstention_metric
    )
    if passed:
        decision = "advance_to_persistent_dialogue_demo"
        recommendation = (
            "Build the persistent child-level dialogue demo with tiered morphology-Unigram, explicit event/state memory, and evidence-gated responses."
        )
    else:
        decision = "hold_for_memory_reliability"
        recommendation = (
            "Do not build the demo yet; improve event recall, state coverage, or abstention before integration."
        )

    return MemoryIntegrationGate(
        candidate=candidate,
        decision=decision,
        state_assisted_accuracy=state_accuracy,
        state_assisted_lift=state_lift,
        state_answer_rate=state_answer_rate,
        synthetic_memory_gain=synthetic_gain,
        authored_dialogue_memory_gain=authored_gain,
        sample_transcript_memory_gain=transcript_gain,
        authored_dialogue_abstention_metric=authored_abstention,
        sample_transcript_abstention_metric=transcript_abstention,
        recommendation=recommendation,
    )


def _state_run(payload: dict, candidate: str) -> dict:
    for run in payload["runs"]:
        if run["tokenizer"] == candidate:
            return run
    raise ValueError(f"Missing state-assisted run for {candidate!r}")
