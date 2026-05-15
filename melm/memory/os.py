"""Support-oriented Memory OS layer over MELM event memory."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Protocol

from melm.guard import Fact

from .schema import Event
from .store import EventMemory, RetrievalConfig, RetrievalResult


ORDER_RE = re.compile(r"\bo\d+\b", re.IGNORECASE)
PREDICATE_QUERY_TERMS = {
    "status": ("status", "state", "stand"),
    "refund_status": ("refund", "refunded"),
    "fraud_flag": ("fraud", "flag", "risk"),
    "manager_approval": ("approval", "approved", "manager"),
    "identity_verified": ("identity", "verified", "verification"),
}


class SupportMemoryCaseLike(Protocol):
    query: str
    kind: str
    category: str
    order_id: str
    expected_event_id: str | None
    predicate: str | None
    expected_value: str | None


@dataclass(frozen=True)
class StateFactResult:
    order_id: str
    predicate: str
    value: str
    event_id: str
    time_index: int


@dataclass(frozen=True)
class MemoryOSPrediction:
    query: str
    category: str
    kind: str
    expected: str | None
    vector_correct: bool
    temporal_entity_correct: bool
    memory_os_correct: bool
    memory_os_answered: bool
    vector_event_ids: tuple[str, ...]
    temporal_entity_event_ids: tuple[str, ...]
    memory_os_event_id: str | None


@dataclass(frozen=True)
class MemoryOSReport:
    cases: int
    vector_accuracy: float
    temporal_entity_accuracy: float
    memory_os_accuracy: float
    memory_os_gain_vs_vector: float
    positive_recall: float
    negative_abstention: float
    gate_passed: bool
    predictions: list[MemoryOSPrediction]
    by_category: dict[str, "MemoryOSReport"] | None = None


class SupportMemoryOS:
    """Indexed support/refund memory with current-state projection."""

    def __init__(self, events: list[Event] | tuple[Event, ...]) -> None:
        self.event_memory = EventMemory(list(events))
        self._events = tuple(events)
        self._events_by_order: dict[str, list[Event]] = defaultdict(list)
        self._events_by_customer: dict[str, list[Event]] = defaultdict(list)
        self._facts_by_key: dict[tuple[str, str], list[Fact]] = defaultdict(list)
        for event in self._events:
            order_id = event.metadata.get("order_id", "").lower()
            customer_id = event.metadata.get("customer_id", "").lower()
            if order_id:
                self._events_by_order[order_id].append(event)
            if customer_id:
                self._events_by_customer[customer_id].append(event)
            fact = _fact_from_event(event)
            if fact is not None:
                self._facts_by_key[(fact.subject, fact.predicate)].append(fact)

        for bucket in self._events_by_order.values():
            bucket.sort(key=lambda item: (item.time_index, item.event_id))
        for bucket in self._events_by_customer.values():
            bucket.sort(key=lambda item: (item.time_index, item.event_id))
        for bucket in self._facts_by_key.values():
            bucket.sort(key=lambda item: (item.time_index, item.fact_id))

    def latest_fact(self, subject: str, predicate: str) -> Fact | None:
        facts = self._facts_by_key.get((subject, predicate), [])
        return facts[-1] if facts else None

    def resolve_order_state(self, order_id: str, predicate: str) -> StateFactResult | None:
        fact = self.latest_fact(f"order:{order_id.lower()}", predicate)
        if fact is None or fact.source_event_id is None:
            return None
        return StateFactResult(
            order_id=order_id.lower(),
            predicate=predicate,
            value=str(fact.value),
            event_id=fact.source_event_id,
            time_index=fact.time_index,
        )

    def retrieve_temporal_entity_rag(self, query: str, *, k: int = 2) -> list[RetrievalResult]:
        base = self.event_memory.retrieve_structured(
            query,
            k=k,
            config=RetrievalConfig(
                name="temporal_entity_rag",
                use_entities=True,
                use_action=True,
                use_salience=True,
                use_explicit_temporal_refs=True,
                expand_temporal_neighbors=True,
                expand_causal_links=False,
            ),
        )
        by_id = {result.event.event_id: result for result in base}
        query_lower = query.lower()
        for order_id in _query_order_ids(query):
            indexed_events = list(self._events_by_order.get(order_id, [])[-2:])
            for event in self._events_by_order.get(order_id, []):
                predicate = event.metadata.get("fact_predicate", "").lower()
                if _predicate_matches_query(predicate, query_lower):
                    indexed_events.append(event)
            for event in indexed_events:
                predicate = event.metadata.get("fact_predicate", "").lower()
                predicate_bonus = 0.75 if _predicate_matches_query(predicate, query_lower) else 0.0
                candidate = RetrievalResult(
                    event=event,
                    score=2.0 + predicate_bonus + event.time_index * 0.0001,
                    reason="order_index",
                )
                if event.event_id not in by_id or candidate.score > by_id[event.event_id].score:
                    by_id[event.event_id] = candidate
        results = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
        return results[:k]


def evaluate_memory_os(
    os_memory: SupportMemoryOS,
    cases: list[SupportMemoryCaseLike],
    *,
    k: int = 2,
) -> MemoryOSReport:
    predictions = [_evaluate_case(os_memory, case, k=k) for case in cases]
    report = _summarize(predictions)
    buckets: dict[str, list[MemoryOSPrediction]] = defaultdict(list)
    for prediction in predictions:
        buckets[prediction.category].append(prediction)
    return MemoryOSReport(
        cases=report.cases,
        vector_accuracy=report.vector_accuracy,
        temporal_entity_accuracy=report.temporal_entity_accuracy,
        memory_os_accuracy=report.memory_os_accuracy,
        memory_os_gain_vs_vector=report.memory_os_gain_vs_vector,
        positive_recall=report.positive_recall,
        negative_abstention=report.negative_abstention,
        gate_passed=report.gate_passed,
        predictions=predictions,
        by_category={category: _summarize(bucket) for category, bucket in sorted(buckets.items())},
    )


def _evaluate_case(os_memory: SupportMemoryOS, case: SupportMemoryCaseLike, *, k: int) -> MemoryOSPrediction:
    vector_results = os_memory.event_memory.retrieve_rag(case.query, k=k)
    temporal_results = os_memory.retrieve_temporal_entity_rag(case.query, k=k)
    vector_ids = tuple(result.event.event_id for result in vector_results)
    temporal_ids = tuple(result.event.event_id for result in temporal_results)

    if case.kind == "event":
        expected = case.expected_event_id
        memory_results = os_memory.event_memory.retrieve_event_memory(case.query, k=k)
        memory_ids = {result.event.event_id for result in memory_results}
        memory_event_id = case.expected_event_id if case.expected_event_id in memory_ids else None
        return MemoryOSPrediction(
            query=case.query,
            category=case.category,
            kind=case.kind,
            expected=expected,
            vector_correct=bool(expected and expected in vector_ids),
            temporal_entity_correct=bool(expected and expected in temporal_ids),
            memory_os_correct=bool(expected and expected in memory_ids),
            memory_os_answered=bool(memory_ids),
            vector_event_ids=vector_ids,
            temporal_entity_event_ids=temporal_ids,
            memory_os_event_id=memory_event_id,
        )

    state = (
        os_memory.resolve_order_state(case.order_id, case.predicate or "")
        if case.predicate
        else None
    )
    expected_event_id = state.event_id if state and case.expected_value is not None else None
    memory_answered = state is not None
    memory_correct = (
        state is None
        if case.expected_value is None
        else state is not None and _norm(state.value) == _norm(case.expected_value)
    )
    return MemoryOSPrediction(
        query=case.query,
        category=case.category,
        kind=case.kind,
        expected=case.expected_value,
        vector_correct=bool(expected_event_id and expected_event_id in vector_ids),
        temporal_entity_correct=bool(expected_event_id and expected_event_id in temporal_ids),
        memory_os_correct=memory_correct,
        memory_os_answered=memory_answered,
        vector_event_ids=vector_ids,
        temporal_entity_event_ids=temporal_ids,
        memory_os_event_id=state.event_id if state else None,
    )


def _summarize(predictions: list[MemoryOSPrediction]) -> MemoryOSReport:
    positives = [prediction for prediction in predictions if prediction.expected is not None]
    negatives = [prediction for prediction in predictions if prediction.expected is None]
    vector_accuracy = _accuracy(predictions, "vector_correct")
    temporal_accuracy = _accuracy(predictions, "temporal_entity_correct")
    memory_accuracy = _accuracy(predictions, "memory_os_correct")
    positive_recall = (
        sum(1 for prediction in positives if prediction.memory_os_correct) / len(positives)
        if positives
        else 0.0
    )
    negative_abstention = (
        sum(1 for prediction in negatives if not prediction.memory_os_answered) / len(negatives)
        if negatives
        else 1.0
    )
    gain = memory_accuracy - vector_accuracy
    return MemoryOSReport(
        cases=len(predictions),
        vector_accuracy=vector_accuracy,
        temporal_entity_accuracy=temporal_accuracy,
        memory_os_accuracy=memory_accuracy,
        memory_os_gain_vs_vector=gain,
        positive_recall=positive_recall,
        negative_abstention=negative_abstention,
        gate_passed=gain >= 0.15 and positive_recall >= 0.75 and negative_abstention >= 0.85,
        predictions=predictions,
        by_category=None,
    )


def _accuracy(predictions: list[MemoryOSPrediction], field_name: str) -> float:
    if not predictions:
        return 0.0
    return sum(1 for prediction in predictions if getattr(prediction, field_name)) / len(predictions)


def _query_order_ids(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0).lower() for match in ORDER_RE.finditer(query)))


def _predicate_matches_query(predicate: str, query_lower: str) -> bool:
    if not predicate:
        return False
    terms = PREDICATE_QUERY_TERMS.get(predicate, tuple(predicate.split("_")))
    return any(term in query_lower for term in terms)


def _fact_from_event(event: Event) -> Fact | None:
    subject = event.metadata.get("fact_subject")
    predicate = event.metadata.get("fact_predicate")
    if not subject or not predicate:
        return None
    return Fact(
        fact_id=f"memory_fact_{event.event_id}",
        subject=subject,
        predicate=predicate,
        value=event.metadata.get("fact_value", ""),
        time_index=event.time_index,
        source_event_id=event.event_id,
        metadata={"memory_os": "true"},
    )


def _norm(value: str | None) -> str | None:
    return None if value is None else " ".join(str(value).lower().strip().split())
