"""Decision helper for saved tiny-LM checkpoint validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointTokenizerDecision:
    candidate_tokenizer: str
    best_baseline_tokenizer: str
    candidate_bits_per_byte: float
    best_baseline_bits_per_byte: float
    relative_bits_per_byte_gain: float
    candidate_minimal_pair_accuracy: float
    best_baseline_minimal_pair_accuracy: float
    minimal_pair_accuracy_delta: float
    decision: str
    recommendation: str


def decide_checkpoint_tokenizer_validation(
    artifact_runs: list[dict],
    minimal_pair_runs: list[dict],
    *,
    candidate_tokenizer: str = "capped_morpheme",
    baseline_prefixes: tuple[str, ...] = ("hf_",),
    minimum_relative_bits_per_byte_gain: float = 0.01,
    minimum_accuracy_delta: float = 0.0,
) -> CheckpointTokenizerDecision:
    """Combine checkpoint loss and minimal-pair smoke results into one decision."""

    candidate_artifact = _find_artifact_run(artifact_runs, candidate_tokenizer)
    baseline_artifacts = [
        run
        for run in artifact_runs
        if any(_artifact_tokenizer(run).startswith(prefix) for prefix in baseline_prefixes)
    ]
    if not baseline_artifacts:
        raise ValueError("Checkpoint decision requires at least one tokenizer baseline")

    best_baseline_artifact = min(
        baseline_artifacts,
        key=lambda run: _artifact_bits_per_byte(run),
    )
    candidate_bpb = _artifact_bits_per_byte(candidate_artifact)
    baseline_bpb = _artifact_bits_per_byte(best_baseline_artifact)
    relative_bpb_gain = (baseline_bpb - candidate_bpb) / baseline_bpb if baseline_bpb else 0.0

    candidate_pair = _find_minimal_pair_run(minimal_pair_runs, candidate_tokenizer)
    baseline_pairs = [
        run
        for run in minimal_pair_runs
        if any(str(run["tokenizer"]).startswith(prefix) for prefix in baseline_prefixes)
    ]
    if not baseline_pairs:
        raise ValueError("Checkpoint decision requires at least one minimal-pair baseline")

    best_baseline_pair = max(
        baseline_pairs,
        key=lambda run: float(run["report"]["accuracy"]),
    )
    candidate_accuracy = float(candidate_pair["report"]["accuracy"])
    baseline_accuracy = float(best_baseline_pair["report"]["accuracy"])
    accuracy_delta = candidate_accuracy - baseline_accuracy

    if relative_bpb_gain < minimum_relative_bits_per_byte_gain:
        decision = "do_not_promote"
        recommendation = "Keep morphology as auxiliary supervision unless stronger loss and task evidence appears."
    elif accuracy_delta >= minimum_accuracy_delta:
        decision = "promote_to_small_model_training"
        recommendation = "Proceed to longer BabyLM-style training and downstream evaluation."
    else:
        decision = "hold_for_quality_evidence"
        recommendation = (
            "Capped morphology has a loss/compression signal, but downstream smoke results lag baselines; "
            "run stronger BabyLM-style evaluations before primary-tokenizer claims."
        )

    return CheckpointTokenizerDecision(
        candidate_tokenizer=candidate_tokenizer,
        best_baseline_tokenizer=_artifact_tokenizer(best_baseline_artifact),
        candidate_bits_per_byte=candidate_bpb,
        best_baseline_bits_per_byte=baseline_bpb,
        relative_bits_per_byte_gain=relative_bpb_gain,
        candidate_minimal_pair_accuracy=candidate_accuracy,
        best_baseline_minimal_pair_accuracy=baseline_accuracy,
        minimal_pair_accuracy_delta=accuracy_delta,
        decision=decision,
        recommendation=recommendation,
    )


def _find_artifact_run(runs: list[dict], tokenizer: str) -> dict:
    for run in runs:
        if _artifact_tokenizer(run) == tokenizer:
            return run
    raise ValueError(f"Missing artifact run for tokenizer {tokenizer!r}")


def _find_minimal_pair_run(runs: list[dict], tokenizer: str) -> dict:
    for run in runs:
        if run.get("tokenizer") == tokenizer:
            return run
    raise ValueError(f"Missing minimal-pair run for tokenizer {tokenizer!r}")


def _artifact_tokenizer(run: dict) -> str:
    if "evaluation" in run:
        return str(run["evaluation"]["tokenizer"])
    return str(run["tokenizer"])


def _artifact_bits_per_byte(run: dict) -> float:
    if "evaluation" in run:
        return float(run["evaluation"]["validation_bits_per_byte"])
    return float(run["validation_bits_per_byte"])
