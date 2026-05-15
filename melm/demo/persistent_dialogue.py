"""Persistent dialogue demo backed by explicit event memory."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from melm.memory import Event, EventMemory, decide_evidence


TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "at",
    "been",
    "before",
    "did",
    "earlier",
    "event",
    "first",
    "for",
    "had",
    "happened",
    "he",
    "her",
    "his",
    "how",
    "in",
    "it",
    "later",
    "of",
    "on",
    "right",
    "she",
    "that",
    "the",
    "there",
    "to",
    "was",
    "what",
    "where",
    "who",
    "why",
}
STATE_CHANGE_ACTIONS = {
    "carried",
    "moved",
    "placed",
    "put",
    "rebuilt",
    "tucked",
}


class EvidenceCaseLike(Protocol):
    query: str
    expected_event_id: str | None
    category: str


@dataclass(frozen=True)
class DialogueDemoResponse:
    query: str
    status: str
    answer: str
    evidence_event_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class DialogueDemoReport:
    cases: int
    positives: int
    negatives: int
    accuracy: float
    positive_recall: float
    negative_abstention: float
    answered: int
    answer_rate: float
    responses: list[DialogueDemoResponse]


@dataclass(frozen=True)
class _ResolvedEvidence:
    event_ids: tuple[str, ...]
    confidence: float


class PersistentDialogueDemo:
    """Small evidence-gated dialogue surface over MELM event memory."""

    def __init__(
        self,
        events: list[Event] | tuple[Event, ...] = (),
        *,
        threshold: float = 1.25,
        k: int = 2,
        confidence_method: str = "score_with_evidence_veto",
    ) -> None:
        self.memory = EventMemory(list(events))
        self.threshold = threshold
        self.k = k
        self.confidence_method = confidence_method

    def remember(self, event: Event) -> None:
        self.memory.add(event)

    def ask(self, query: str) -> DialogueDemoResponse:
        decision = decide_evidence(
            self.memory,
            query,
            k=self.k,
            threshold=self.threshold,
            retriever="event_memory",
            confidence_method=self.confidence_method,
            selection_mode="evidence_set",
        )
        if not decision.answered:
            resolved = self._resolve_structured_evidence(query, base_confidence=decision.confidence)
            if resolved is not None:
                return self._response_from_event_ids(
                    query,
                    resolved.event_ids,
                    confidence=resolved.confidence,
                )
            return DialogueDemoResponse(
                query=query,
                status="abstained",
                answer="I do not have enough evidence for that yet.",
                evidence_event_ids=(),
                confidence=decision.confidence,
            )

        return self._response_from_event_ids(
            query,
            decision.candidate_event_ids,
            confidence=decision.confidence,
        )

    def _resolve_structured_evidence(
        self,
        query: str,
        *,
        base_confidence: float,
    ) -> _ResolvedEvidence | None:
        resolved = self._resolve_causal_source(query, base_confidence=base_confidence)
        if resolved is not None:
            return resolved
        resolved = self._resolve_state_change(query, base_confidence=base_confidence)
        if resolved is not None:
            return resolved
        return self._resolve_first_observation(query, base_confidence=base_confidence)

    def _resolve_causal_source(
        self,
        query: str,
        *,
        base_confidence: float,
    ) -> _ResolvedEvidence | None:
        lower = query.lower()
        if not (("why" in lower or "explain" in lower or "because" in lower) and "earlier" in lower):
            return None

        query_terms = _content_terms(query)
        query_actors = _query_actors(self.memory.events(), query_terms)
        candidates: list[tuple[float, tuple[str, ...]]] = []
        for event in self.memory.events():
            if not event.causal_links:
                continue
            if query_actors and not any(actor in query_actors for actor in event.actors):
                continue
            object_score = _object_overlap_score(event, query_terms)
            if object_score == 0:
                continue

            linked_ids: list[str] = []
            for cause_id in event.causal_links:
                cause = self.memory.get(cause_id)
                if cause is None or cause.time_index >= event.time_index:
                    continue
                if _object_overlap_score(cause, query_terms) == 0 and object_score == 0:
                    continue
                linked_ids.append(cause.event_id)
            if not linked_ids:
                continue

            actor_bonus = 2.0 if query_actors else 0.0
            score = object_score + actor_bonus + (event.time_index * 0.001)
            event_ids = tuple(dict.fromkeys([*linked_ids, event.event_id]))
            candidates.append((score, event_ids))

        if not candidates:
            return None
        _score, event_ids = max(candidates, key=lambda item: item[0])
        return _ResolvedEvidence(
            event_ids=event_ids,
            confidence=max(self.threshold, base_confidence),
        )

    def _resolve_state_change(
        self,
        query: str,
        *,
        base_confidence: float,
    ) -> _ResolvedEvidence | None:
        lower = query.lower()
        if "what changed" not in lower and "changed where" not in lower:
            return None

        query_terms = _content_terms(query)
        query_actors = _query_actors(self.memory.events(), query_terms)
        candidates: list[tuple[float, str]] = []
        for event in self.memory.events():
            action = event.action_or_state.lower()
            if action not in STATE_CHANGE_ACTIONS:
                continue
            if query_actors and not any(actor in query_actors for actor in event.actors):
                continue

            object_score = _object_overlap_score(event, query_terms)
            if object_score < 2:
                continue
            action_bonus = 1.0 if action in {"carried", "moved", "placed"} else 0.5
            score = object_score + action_bonus + (event.time_index * 0.001)
            candidates.append((score, event.event_id))

        if not candidates:
            return None
        _score, event_id = max(candidates, key=lambda item: item[0])
        return _ResolvedEvidence(
            event_ids=(event_id,),
            confidence=max(self.threshold, base_confidence),
        )

    def _resolve_first_observation(
        self,
        query: str,
        *,
        base_confidence: float,
    ) -> _ResolvedEvidence | None:
        lower = query.lower()
        if "first" not in lower and "at first" not in lower:
            return None

        query_terms = _content_terms(query)
        query_actors = _query_actors(self.memory.events(), query_terms)
        candidates: list[tuple[int, int, str]] = []
        for event in self.memory.events():
            actor_score = 1 if any(actor in query_actors for actor in event.actors) else 0
            object_score = _object_overlap_score(event, query_terms)
            if object_score < 2:
                continue
            if query_actors and actor_score == 0:
                continue
            score = object_score + actor_score
            candidates.append((score, -event.time_index, event.event_id))

        if not candidates:
            return None
        _score, _time_order, event_id = max(candidates, key=lambda item: item)
        return _ResolvedEvidence(
            event_ids=(event_id,),
            confidence=max(self.threshold, base_confidence),
        )

    def _response_from_event_ids(
        self,
        query: str,
        event_ids: tuple[str, ...],
        *,
        confidence: float,
    ) -> DialogueDemoResponse:
        events = [self.memory.get(event_id) for event_id in event_ids]
        evidence_spans = [event.source_span for event in events if event is not None]
        answer = "I remember: " + " ".join(evidence_spans)
        return DialogueDemoResponse(
            query=query,
            status="answered",
            answer=answer,
            evidence_event_ids=event_ids,
            confidence=confidence,
        )


def _content_terms(text: str) -> set[str]:
    return {
        _stem(term)
        for term in TOKEN_RE.findall(text.lower())
        if term not in STOPWORDS and len(term) > 1
    }


def _query_actors(events: tuple[Event, ...], query_terms: set[str]) -> set[str]:
    actors = {actor for event in events for actor in event.actors}
    return {
        actor
        for actor in actors
        if _phrase_terms(actor) <= query_terms
    }


def _object_overlap_score(event: Event, query_terms: set[str]) -> int:
    best_score = 0
    for object_phrase in event.objects:
        object_terms = _phrase_terms(object_phrase)
        if not object_terms:
            continue
        overlap = len(object_terms & query_terms)
        if object_terms <= query_terms:
            overlap += 2
        best_score = max(best_score, overlap)
    return best_score


def _phrase_terms(phrase: str) -> set[str]:
    return {_stem(term) for term in TOKEN_RE.findall(phrase.lower()) if len(term) > 1}


def _stem(term: str) -> str:
    if term.endswith("'s") and len(term) > 3:
        term = term[:-2]
    for suffix in ("ing", "ed", "s"):
        if len(term) > len(suffix) + 2 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def evaluate_dialogue_demo(
    demo: PersistentDialogueDemo,
    cases: list[EvidenceCaseLike],
) -> DialogueDemoReport:
    """Evaluate answer/admit behavior for a persistent dialogue demo."""

    responses = [demo.ask(case.query) for case in cases]
    positives = sum(1 for case in cases if case.expected_event_id is not None)
    negatives = len(cases) - positives
    true_positives = 0
    true_negatives = 0
    answered = 0
    for case, response in zip(cases, responses):
        answered += int(response.status == "answered")
        if case.expected_event_id is None:
            true_negatives += int(response.status == "abstained")
        elif case.expected_event_id in response.evidence_event_ids:
            true_positives += 1

    accuracy = (true_positives + true_negatives) / len(cases) if cases else 0.0
    positive_recall = true_positives / positives if positives else 0.0
    negative_abstention = true_negatives / negatives if negatives else 0.0
    return DialogueDemoReport(
        cases=len(cases),
        positives=positives,
        negatives=negatives,
        accuracy=accuracy,
        positive_recall=positive_recall,
        negative_abstention=negative_abstention,
        answered=answered,
        answer_rate=answered / len(cases) if cases else 0.0,
        responses=responses,
    )
