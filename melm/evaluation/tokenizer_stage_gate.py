"""Stage-gate decision for tokenizer candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerStageGate:
    candidate: str
    decision: str
    latest_step: int
    candidate_latest_bits_per_byte: float
    best_baseline: str
    best_baseline_latest_bits_per_byte: float
    relative_bits_per_byte_gain: float
    blimp_wins: int
    blimp_reports: int
    entity_best_baseline: str
    candidate_entity_accuracy: float
    best_baseline_entity_accuracy: float
    entity_accuracy_delta: float
    proxy_decision: str | None
    proxy_best_baseline: str | None
    proxy_relative_bits_per_byte_gain: float | None
    proxy_supports_scale: bool | None
    recommendation: str


def decide_tokenizer_stage_gate(
    progression_payload: dict,
    blimp_payloads: list[dict],
    entity_payload: dict,
    *,
    proxy_decision_payload: dict | None = None,
    candidate: str = "tiered_morph_unigram",
    baselines: tuple[str, ...] = ("hf_bpe", "hf_unigram"),
    minimum_relative_bpb_gain: float = 0.01,
    minimum_blimp_wins: int = 1,
    max_entity_regression: float = 0.03,
    minimum_proxy_relative_bpb_gain: float = 0.01,
) -> TokenizerStageGate:
    """Decide whether a tokenizer candidate advances to small-model ablation."""

    progressions = {
        item["tokenizer"]: item
        for item in progression_payload["progressions"]
    }
    candidate_progression = progressions[candidate]
    candidate_latest_point = _latest_point(candidate_progression)
    baseline_latest_points = {
        baseline: _latest_point(progressions[baseline])
        for baseline in baselines
        if baseline in progressions
    }
    if not baseline_latest_points:
        raise ValueError("At least one baseline progression is required")

    best_baseline, best_baseline_point = min(
        baseline_latest_points.items(),
        key=lambda item: float(item[1]["mean_bits_per_byte"]),
    )
    candidate_bpb = float(candidate_latest_point["mean_bits_per_byte"])
    baseline_bpb = float(best_baseline_point["mean_bits_per_byte"])
    relative_bpb_gain = (baseline_bpb - candidate_bpb) / baseline_bpb if baseline_bpb else 0.0

    blimp_wins = sum(
        1 for payload in blimp_payloads if _candidate_wins(payload, candidate, baselines)
    )

    entity_runs = {
        run["tokenizer"]: float(run["report"]["accuracy"])
        for run in entity_payload["runs"]
    }
    candidate_entity_accuracy = entity_runs[candidate]
    entity_baselines = {
        baseline: accuracy
        for baseline, accuracy in entity_runs.items()
        if baseline in baselines
    }
    entity_best_baseline, entity_best_accuracy = max(
        entity_baselines.items(),
        key=lambda item: item[1],
    )
    entity_delta = candidate_entity_accuracy - entity_best_accuracy
    proxy = _proxy_summary(
        proxy_decision_payload,
        candidate=candidate,
        minimum_relative_bpb_gain=minimum_proxy_relative_bpb_gain,
    )

    passed = (
        relative_bpb_gain >= minimum_relative_bpb_gain
        and blimp_wins >= minimum_blimp_wins
        and entity_delta >= -max_entity_regression
    )
    if passed:
        if proxy_decision_payload is None:
            decision = "advance_to_small_model_ablation"
            recommendation = (
                "Run the first larger proxy model before scheduling longer BabyLM-style neural ablations."
            )
        elif proxy["supports_scale"]:
            decision = "advance_to_scaled_neural_ablation"
            recommendation = (
                "Schedule longer matched BabyLM-style neural ablations for tiered morphology-Unigram, HF BPE, HF Unigram, and capped morphology."
            )
        else:
            decision = "hold_for_proxy_signal"
            recommendation = (
                "The tokenizer gate passed, but the proxy model does not yet justify a longer neural ablation."
            )
    elif relative_bpb_gain < minimum_relative_bpb_gain:
        decision = "hold_for_loss_signal"
        recommendation = "Do not scale until the candidate beats BPE/Unigram on stable validation bits-per-byte."
    elif blimp_wins < minimum_blimp_wins:
        decision = "hold_for_language_quality"
        recommendation = "Do not scale until the candidate wins at least one official fast-BLiMP scoring view."
    else:
        decision = "hold_for_entity_tracking"
        recommendation = "Improve or compensate for entity/state tracking before scaling the tokenizer."

    return TokenizerStageGate(
        candidate=candidate,
        decision=decision,
        latest_step=int(candidate_latest_point["steps"]),
        candidate_latest_bits_per_byte=candidate_bpb,
        best_baseline=best_baseline,
        best_baseline_latest_bits_per_byte=baseline_bpb,
        relative_bits_per_byte_gain=relative_bpb_gain,
        blimp_wins=blimp_wins,
        blimp_reports=len(blimp_payloads),
        entity_best_baseline=entity_best_baseline,
        candidate_entity_accuracy=candidate_entity_accuracy,
        best_baseline_entity_accuracy=entity_best_accuracy,
        entity_accuracy_delta=entity_delta,
        proxy_decision=proxy["decision"],
        proxy_best_baseline=proxy["best_baseline"],
        proxy_relative_bits_per_byte_gain=proxy["relative_bits_per_byte_gain"],
        proxy_supports_scale=proxy["supports_scale"],
        recommendation=recommendation,
    )


def _latest_point(progression: dict) -> dict:
    return max(progression["points"], key=lambda point: int(point["steps"]))


def _candidate_wins(payload: dict, candidate: str, baselines: tuple[str, ...]) -> bool:
    accuracies = {
        run["tokenizer"]: float(run["report"]["accuracy"])
        for run in payload["runs"]
    }
    candidate_accuracy = accuracies[candidate]
    best_baseline_accuracy = max(
        accuracy for tokenizer, accuracy in accuracies.items() if tokenizer in baselines
    )
    return candidate_accuracy > best_baseline_accuracy


def _proxy_summary(
    payload: dict | None,
    *,
    candidate: str,
    minimum_relative_bpb_gain: float,
) -> dict:
    if payload is None:
        return {
            "decision": None,
            "best_baseline": None,
            "relative_bits_per_byte_gain": None,
            "supports_scale": None,
        }

    decision = payload.get("decision", payload)
    if isinstance(decision, str):
        decision = {"decision": decision}
    decision_name = str(decision.get("decision", ""))
    proxy_candidate = str(decision.get("candidate_tokenizer", ""))
    relative_gain = float(decision.get("relative_bits_per_byte_gain", 0.0))
    supports_scale = (
        proxy_candidate == candidate
        and decision_name == "promote_to_scaled_neural_ablation"
        and relative_gain >= minimum_relative_bpb_gain
    )
    return {
        "decision": decision_name,
        "best_baseline": decision.get("best_baseline_tokenizer"),
        "relative_bits_per_byte_gain": relative_gain,
        "supports_scale": supports_scale,
    }
