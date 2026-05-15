"""Gate helpers matching the validation plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    recommendation: str


def memory_gate(event_memory_recall: float, rag_recall: float, threshold: float = 0.15) -> GateResult:
    gain = event_memory_recall - rag_recall
    return GateResult(
        name="event_memory_vs_rag",
        passed=gain >= threshold,
        metric=gain,
        threshold=threshold,
        recommendation=(
            "Proceed with structured event memory"
            if gain >= threshold
            else "Simplify to RAG plus temporal/entity metadata"
        ),
    )


def abstention_gate(
    negative_abstention: float,
    threshold: float = 0.80,
    *,
    positive_recall: float | None = None,
    positive_recall_threshold: float = 0.75,
) -> GateResult:
    positive_passed = (
        True
        if positive_recall is None
        else positive_recall >= positive_recall_threshold
    )
    negative_passed = negative_abstention >= threshold
    passed = negative_passed and positive_passed
    metric = (
        negative_abstention
        if positive_recall is None
        else min(
            negative_abstention / threshold if threshold else 0.0,
            positive_recall / positive_recall_threshold if positive_recall_threshold else 0.0,
        )
    )
    return GateResult(
        name="event_memory_abstention",
        passed=passed,
        metric=metric,
        threshold=1.0 if positive_recall is not None else threshold,
        recommendation=(
            "Use calibrated confidence threshold for evidence-set admission"
            if passed
            else "Improve the recall/rejection tradeoff with better evidence sufficiency checks and no-answer cases"
        ),
    )


def morphology_gate(best_morph_score: float, best_baseline_score: float) -> GateResult:
    gain = best_morph_score - best_baseline_score
    return GateResult(
        name="morphology_vs_tokenizer_baseline",
        passed=gain > 0.0,
        metric=gain,
        threshold=0.0,
        recommendation=(
            "Use morphology-aware tokenizer or objective"
            if gain > 0.0
            else "Keep morphology as auxiliary supervision only"
        ),
    )
