"""Evaluation helpers for event memory vs RAG."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Protocol

from .store import EventMemory, RetrievalConfig


class RecallCase(Protocol):
    query: str
    expected_event_id: str
    category: str


@dataclass(frozen=True)
class MemoryComparison:
    cases: int
    rag_recall_at_k: float
    event_memory_recall_at_k: float
    absolute_gain: float
    rag_mrr_at_k: float = 0.0
    event_memory_mrr_at_k: float = 0.0
    mrr_gain: float = 0.0
    by_category: dict[str, "MemoryComparison"] | None = None

    @property
    def passes_gate(self) -> bool:
        return self.absolute_gain >= 0.15


def evaluate_memory(
    memory: EventMemory,
    cases: list[tuple[str, str]] | list[RecallCase],
    k: int = 3,
) -> MemoryComparison:
    """Evaluate recall@k for cases.

    Accepts either `(query, expected_event_id)` tuples or objects with
    `query`, `expected_event_id`, and `category` attributes.
    """

    if not cases:
        return MemoryComparison(0, 0.0, 0.0, 0.0)

    normalized = [_normalize_case(case) for case in cases]
    overall = _evaluate_normalized(memory, normalized, k)

    buckets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for case in normalized:
        buckets[case[2]].append(case)

    by_category = {
        category: _evaluate_normalized(memory, bucket, k)
        for category, bucket in sorted(buckets.items())
    }
    return MemoryComparison(
        cases=overall.cases,
        rag_recall_at_k=overall.rag_recall_at_k,
        event_memory_recall_at_k=overall.event_memory_recall_at_k,
        absolute_gain=overall.absolute_gain,
        rag_mrr_at_k=overall.rag_mrr_at_k,
        event_memory_mrr_at_k=overall.event_memory_mrr_at_k,
        mrr_gain=overall.mrr_gain,
        by_category=by_category,
    )


def evaluate_memory_variants(
    memory: EventMemory,
    cases: list[tuple[str, str]] | list[RecallCase],
    k: int = 3,
) -> dict[str, MemoryComparison]:
    """Evaluate retrieval-component ablations.

    Each variant is compared against vector-only RAG as the baseline fields in
    `MemoryComparison`, so gains are directly comparable.
    """

    normalized = [_normalize_case(case) for case in cases]
    rag_results = _evaluate_with_retriever(
        memory,
        normalized,
        k,
        lambda query: memory.retrieve_rag(query, k=k),
    )

    configs = [
        RetrievalConfig(name="entity_only", use_action=False, use_salience=False, use_explicit_temporal_refs=False, expand_temporal_neighbors=False, expand_causal_links=False),
        RetrievalConfig(name="entity_action", use_salience=False, use_explicit_temporal_refs=False, expand_temporal_neighbors=False, expand_causal_links=False),
        RetrievalConfig(name="entity_action_temporal", use_salience=False, expand_causal_links=False),
        RetrievalConfig(name="entity_action_causal", use_salience=False, expand_temporal_neighbors=False),
        RetrievalConfig.full(),
    ]

    output: dict[str, MemoryComparison] = {}
    for config in configs:
        variant = _evaluate_with_retriever(
            memory,
            normalized,
            k,
            lambda query, config=config: memory.retrieve_structured(query, k=k, config=config),
        )
        output[config.name] = MemoryComparison(
            cases=variant.cases,
            rag_recall_at_k=rag_results.event_memory_recall_at_k,
            event_memory_recall_at_k=variant.event_memory_recall_at_k,
            absolute_gain=variant.event_memory_recall_at_k - rag_results.event_memory_recall_at_k,
            rag_mrr_at_k=rag_results.event_memory_mrr_at_k,
            event_memory_mrr_at_k=variant.event_memory_mrr_at_k,
            mrr_gain=variant.event_memory_mrr_at_k - rag_results.event_memory_mrr_at_k,
        )
    return output


def _evaluate_normalized(
    memory: EventMemory,
    cases: list[tuple[str, str, str]],
    k: int,
) -> MemoryComparison:
    rag_hits = 0
    event_hits = 0
    rag_rr = 0.0
    event_rr = 0.0

    for query, expected_event_id, _category in cases:
        rag_results = memory.retrieve_rag(query, k=k)
        event_results = memory.retrieve_event_memory(query, k=k)
        rag_ids = {result.event.event_id for result in rag_results}
        event_ids = {result.event.event_id for result in event_results}
        rag_hits += int(expected_event_id in rag_ids)
        event_hits += int(expected_event_id in event_ids)
        rag_rr += _reciprocal_rank(rag_results, expected_event_id)
        event_rr += _reciprocal_rank(event_results, expected_event_id)

    rag_recall = rag_hits / len(cases)
    event_recall = event_hits / len(cases)
    rag_mrr = rag_rr / len(cases)
    event_mrr = event_rr / len(cases)
    return MemoryComparison(
        cases=len(cases),
        rag_recall_at_k=rag_recall,
        event_memory_recall_at_k=event_recall,
        absolute_gain=event_recall - rag_recall,
        rag_mrr_at_k=rag_mrr,
        event_memory_mrr_at_k=event_mrr,
        mrr_gain=event_mrr - rag_mrr,
    )


def _evaluate_with_retriever(memory: EventMemory, cases: list[tuple[str, str, str]], k: int, retriever) -> MemoryComparison:
    hits = 0
    reciprocal_rank = 0.0
    for query, expected_event_id, _category in cases:
        results = retriever(query)
        ids = {result.event.event_id for result in results}
        hits += int(expected_event_id in ids)
        reciprocal_rank += _reciprocal_rank(results, expected_event_id)

    recall = hits / len(cases) if cases else 0.0
    mrr = reciprocal_rank / len(cases) if cases else 0.0
    return MemoryComparison(
        cases=len(cases),
        rag_recall_at_k=0.0,
        event_memory_recall_at_k=recall,
        absolute_gain=0.0,
        rag_mrr_at_k=0.0,
        event_memory_mrr_at_k=mrr,
        mrr_gain=0.0,
    )


def _normalize_case(case: tuple[str, str] | RecallCase) -> tuple[str, str, str]:
    if isinstance(case, tuple):
        return (case[0], case[1], "overall")
    return (case.query, case.expected_event_id, case.category)


def _reciprocal_rank(results: list, expected_event_id: str) -> float:
    for index, result in enumerate(results, start=1):
        if result.event.event_id == expected_event_id:
            return 1.0 / index
    return 0.0
