"""Decision gate for the checkpointed small-model tokenizer stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmallModelStageGate:
    candidate: str
    decision: str
    candidate_bits_per_byte: float
    best_hf_baseline: str
    best_hf_baseline_bits_per_byte: float
    relative_bits_per_byte_gain: float
    compression_control: str
    compression_control_bits_per_byte: float
    blimp_wins: int
    blimp_reports: int
    entity_best_hf_baseline: str
    candidate_entity_accuracy: float
    best_hf_entity_accuracy: float
    entity_accuracy_delta: float
    symbolic_entity_accuracy: float | None
    recommendation: str


def decide_small_model_stage_gate(
    multiseed_payload: dict,
    blimp_payloads: list[dict],
    entity_payload: dict,
    *,
    symbolic_entity_payload: dict | None = None,
    candidate: str = "tiered_morph_unigram",
    hf_baselines: tuple[str, ...] = ("hf_bpe", "hf_unigram"),
    minimum_relative_bpb_gain: float = 0.01,
    minimum_blimp_wins: int = 2,
    minimum_entity_delta: float = 0.0,
) -> SmallModelStageGate:
    """Decide whether the checkpointed small-model stage supports integration."""

    summaries = {
        item["tokenizer"]: item
        for item in multiseed_payload["summaries"]
    }
    candidate_summary = summaries[candidate]
    hf_summaries = {
        tokenizer: summary
        for tokenizer, summary in summaries.items()
        if tokenizer in hf_baselines
    }
    if not hf_summaries:
        raise ValueError("Small-model gate requires at least one HF baseline")
    best_hf, best_hf_summary = min(
        hf_summaries.items(),
        key=lambda item: float(item[1]["mean_bits_per_byte"]),
    )
    candidate_bpb = float(candidate_summary["mean_bits_per_byte"])
    best_hf_bpb = float(best_hf_summary["mean_bits_per_byte"])
    relative_gain = (best_hf_bpb - candidate_bpb) / best_hf_bpb if best_hf_bpb else 0.0

    compression_control, compression_summary = min(
        summaries.items(),
        key=lambda item: float(item[1]["mean_bits_per_byte"]),
    )

    blimp_wins = sum(
        1 for payload in blimp_payloads if _candidate_wins(payload, candidate, hf_baselines)
    )
    entity_runs = _accuracies_by_tokenizer(entity_payload)
    candidate_entity = entity_runs[candidate]
    entity_hf = {
        tokenizer: accuracy
        for tokenizer, accuracy in entity_runs.items()
        if tokenizer in hf_baselines
    }
    entity_best_hf, entity_best_accuracy = max(
        entity_hf.items(),
        key=lambda item: item[1],
    )
    entity_delta = candidate_entity - entity_best_accuracy
    symbolic_accuracy = _symbolic_accuracy(symbolic_entity_payload)

    if (
        relative_gain >= minimum_relative_bpb_gain
        and blimp_wins >= minimum_blimp_wins
        and entity_delta >= minimum_entity_delta
    ):
        decision = "advance_to_event_memory_integration"
        recommendation = (
            "Keep tiered morphology-Unigram as the primary MELM tokenizer candidate, keep capped morphology as a compression control, and integrate explicit event/state memory next."
        )
    elif relative_gain < minimum_relative_bpb_gain:
        decision = "hold_for_loss_signal"
        recommendation = "Do not integrate yet; repeat or revise until the candidate beats HF baselines on multiseed bits/byte."
    elif blimp_wins < minimum_blimp_wins:
        decision = "hold_for_language_quality"
        recommendation = "Do not integrate yet; improve fast-BLiMP quality before adding memory complexity."
    else:
        decision = "hold_for_entity_tracking"
        recommendation = "Do not integrate yet; fix entity/state tracking before the next model stage."

    return SmallModelStageGate(
        candidate=candidate,
        decision=decision,
        candidate_bits_per_byte=candidate_bpb,
        best_hf_baseline=best_hf,
        best_hf_baseline_bits_per_byte=best_hf_bpb,
        relative_bits_per_byte_gain=relative_gain,
        compression_control=compression_control,
        compression_control_bits_per_byte=float(compression_summary["mean_bits_per_byte"]),
        blimp_wins=blimp_wins,
        blimp_reports=len(blimp_payloads),
        entity_best_hf_baseline=entity_best_hf,
        candidate_entity_accuracy=candidate_entity,
        best_hf_entity_accuracy=entity_best_accuracy,
        entity_accuracy_delta=entity_delta,
        symbolic_entity_accuracy=symbolic_accuracy,
        recommendation=recommendation,
    )


def _candidate_wins(payload: dict, candidate: str, baselines: tuple[str, ...]) -> bool:
    accuracies = _accuracies_by_tokenizer(payload)
    candidate_accuracy = accuracies[candidate]
    best_baseline_accuracy = max(
        accuracy for tokenizer, accuracy in accuracies.items() if tokenizer in baselines
    )
    return candidate_accuracy > best_baseline_accuracy


def _accuracies_by_tokenizer(payload: dict) -> dict[str, float]:
    return {
        run["tokenizer"]: float(run["report"]["accuracy"])
        for run in payload["runs"]
    }


def _symbolic_accuracy(payload: dict | None) -> float | None:
    if payload is None:
        return None
    report = payload.get("report", payload)
    return float(report["accuracy"])
