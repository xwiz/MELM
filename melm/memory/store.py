"""In-memory RAG and event-memory retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections import Counter

from .schema import Event


TOKEN_RE = re.compile(r"[a-z0-9']+")
TOKEN_ALIASES = {
    "left": ("put", "placed"),
    "leave": ("put", "placed"),
    "noticed": ("saw",),
    "notice": ("saw",),
    "went": ("moved", "slid", "carried"),
}


@dataclass(frozen=True)
class RetrievalResult:
    event: Event
    score: float
    reason: str


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    use_entities: bool = True
    use_action: bool = True
    use_salience: bool = True
    use_explicit_temporal_refs: bool = True
    expand_temporal_neighbors: bool = True
    expand_causal_links: bool = True
    use_access_history: bool = False
    access_weight: float = 0.05
    recency_weight: float = 0.05

    @classmethod
    def vector_only(cls) -> "RetrievalConfig":
        return cls(
            name="vector_only",
            use_entities=False,
            use_action=False,
            use_salience=False,
            use_explicit_temporal_refs=False,
            expand_temporal_neighbors=False,
            expand_causal_links=False,
        )

    @classmethod
    def full(cls) -> "RetrievalConfig":
        return cls(name="event_memory")

    @classmethod
    def with_access_history(cls) -> "RetrievalConfig":
        return cls(name="event_memory_access_history", use_access_history=True)


class EventMemory:
    """Small deterministic event store.

    `retrieve_rag` is intentionally plain vector-style retrieval over bag-of-word
    cosine similarity. `retrieve_event_memory` adds entity and temporal structure
    so we can measure whether the structured memory hypothesis helps.
    """

    def __init__(self, events: list[Event] | None = None) -> None:
        self._events: dict[str, Event] = {}
        self._vectors: dict[str, Counter[str]] = {}
        self._access_counts: Counter[str] = Counter()
        self._last_access_step: dict[str, int] = {}
        self._clock = 0
        for event in events or []:
            self.add(event)

    def add(self, event: Event) -> None:
        self._events[event.event_id] = event
        self._vectors[event.event_id] = _vectorize(event.searchable_text())

    def get(self, event_id: str) -> Event | None:
        return self._events.get(event_id)

    def events(self) -> tuple[Event, ...]:
        return tuple(self._events.values())

    def retrieve_rag(self, query: str, k: int = 3) -> list[RetrievalResult]:
        return self.retrieve_structured(query, k=k, config=RetrievalConfig.vector_only())

    def retrieve_event_memory(self, query: str, k: int = 3) -> list[RetrievalResult]:
        return self.retrieve_structured(query, k=k, config=RetrievalConfig.full())

    def retrieve_structured(
        self,
        query: str,
        k: int = 3,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        config = config or RetrievalConfig.full()
        query_vec = _vectorize(query)
        query_terms = set(query_vec)
        query_lower = query.lower()
        wants_after = (
            "right after" in query_lower
            or "what happened after" in query_lower
            or "what came after" in query_lower
            or "next event" in query_lower
            or ("after" in query_terms and ("next" in query_terms or "came" in query_terms))
        )
        wants_before = (
            "right before" in query_lower
            or "just before" in query_lower
            or "what happened before" in query_lower
            or "what came before" in query_lower
            or "previous event" in query_lower
            or ("before" in query_terms and ("happened" in query_terms or "just" in query_terms))
        )
        wants_cause = (
            "why" in query_terms
            or "explain" in query_terms
            or "because" in query_terms
            or ("before" in query_terms and "moved" in query_terms)
        )
        scored: list[RetrievalResult] = []

        for event_id, event in self._events.items():
            vector_score = _cosine(query_vec, self._vectors[event_id])
            entity_overlap = len(event.entities & query_terms)
            entity_score = min(entity_overlap * 0.25, 0.75) if config.use_entities else 0.0
            action_score = (
                0.25
                if config.use_action and event.action_or_state.lower() in query_terms
                else 0.0
            )
            salience_score = (
                min(max(event.salience, 0.0), 2.0) * 0.05
                if config.use_salience
                else 0.0
            )
            access_score = (
                self._access_history_score(event_id, config)
                if config.use_access_history
                else 0.0
            )

            temporal_score = 0.0
            if config.use_explicit_temporal_refs:
                if event.previous_event_id and event.previous_event_id in query_terms:
                    temporal_score += 0.2
                if event.next_event_id and event.next_event_id in query_terms:
                    temporal_score += 0.2

            score = (
                vector_score
                + entity_score
                + action_score
                + salience_score
                + temporal_score
                + access_score
            )
            reason_parts = ["vector"]
            if config.use_entities and entity_overlap:
                reason_parts.append("entity")
            if temporal_score:
                reason_parts.append("temporal")
            if config.expand_causal_links and event.causal_links:
                reason_parts.append("causal")
            if access_score:
                reason_parts.append("access")

            scored.append(RetrievalResult(event=event, score=score, reason="+".join(reason_parts)))

        scored.sort(key=lambda item: item.score, reverse=True)
        expanded = scored[:k]
        if config.expand_temporal_neighbors and (wants_after or wants_before):
            expanded = self._with_temporal_neighbors(expanded, wants_after=wants_after, wants_before=wants_before)
        if config.expand_causal_links and wants_cause:
            expanded = self._with_causal_links(expanded)
        expanded.sort(key=lambda item: item.score, reverse=True)
        selected = expanded[:k]
        self.mark_access(result.event.event_id for result in selected)
        return selected

    def mark_access(self, event_ids) -> None:
        for event_id in event_ids:
            if event_id not in self._events:
                continue
            self._clock += 1
            self._access_counts[event_id] += 1
            self._last_access_step[event_id] = self._clock

    def access_count(self, event_id: str) -> int:
        return int(self._access_counts[event_id])

    def _with_temporal_neighbors(
        self,
        results: list[RetrievalResult],
        *,
        wants_after: bool = False,
        wants_before: bool = False,
    ) -> list[RetrievalResult]:
        by_id = {result.event.event_id: result for result in results}
        source_results = results[:1] if wants_after or wants_before else list(results)
        for result in source_results:
            neighbor_ids: list[tuple[str | None, float]] = []
            if wants_before or not wants_after:
                neighbor_ids.append((result.event.previous_event_id, 1.08 if wants_before else 0.82))
            if wants_after or not wants_before:
                neighbor_ids.append((result.event.next_event_id, 1.08 if wants_after else 0.82))

            for neighbor_id, multiplier in neighbor_ids:
                if not neighbor_id or neighbor_id in by_id:
                    continue
                neighbor = self._events.get(neighbor_id)
                if neighbor is None:
                    continue
                by_id[neighbor_id] = RetrievalResult(
                    event=neighbor,
                    score=result.score * multiplier,
                    reason=f"temporal_neighbor:{result.event.event_id}",
                )
        return list(by_id.values())

    def _with_causal_links(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        by_id = {result.event.event_id: result for result in results}
        for result in list(results):
            for cause_id in result.event.causal_links:
                if cause_id in by_id:
                    continue
                cause = self._events.get(cause_id)
                if cause is None:
                    continue
                by_id[cause_id] = RetrievalResult(
                    event=cause,
                    score=result.score * 1.10,
                    reason=f"causal_link:{result.event.event_id}",
                )
        return list(by_id.values())

    def _access_history_score(self, event_id: str, config: RetrievalConfig) -> float:
        count = self._access_counts[event_id]
        frequency = math.log1p(count) * config.access_weight
        last_step = self._last_access_step.get(event_id)
        if last_step is None or self._clock == 0:
            return frequency
        age = max(0, self._clock - last_step)
        recency = (1.0 / (1.0 + age)) * config.recency_weight
        return frequency + recency


def _vectorize(text: str) -> Counter[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        token = _normalize_token(raw_token)
        tokens.append(token)
        tokens.extend(TOKEN_ALIASES.get(token, ()))
    return Counter(tokens)


def _normalize_token(token: str) -> str:
    if token.endswith("'s") and len(token) > 3:
        return token[:-2]
    return token


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
