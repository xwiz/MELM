"""Abstention and evidence calibration probes for event memory."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import re
from typing import Literal, Protocol

from .store import EventMemory, RetrievalResult


class EvidenceCaseLike(Protocol):
    query: str
    expected_event_id: str | None
    category: str


RetrieverName = Literal["rag", "event_memory"]
ConfidenceMethod = Literal["top_score", "score_margin", "evidence", "score_with_evidence_veto"]
SelectionMode = Literal["evidence_set", "top_event"]
EVIDENCE_VETO_THRESHOLD = 0.60
EVIDENCE_SUPPORT_BONUS = 0.15


CONTENT_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "before",
    "came",
    "child",
    "children",
    "could",
    "after",
    "did",
    "do",
    "down",
    "event",
    "first",
    "had",
    "happen",
    "happened",
    "how",
    "in",
    "it",
    "knew",
    "later",
    "next",
    "of",
    "on",
    "or",
    "remember",
    "right",
    "say",
    "still",
    "that",
    "the",
    "there",
    "time",
    "to",
    "been",
    "was",
    "went",
    "were",
    "what",
    "where",
    "which",
    "who",
    "why",
    "you",
}
CANONICAL_TERMS = {
    "left": "put",
    "leave": "put",
    "noticed": "saw",
    "notice": "saw",
}


@dataclass(frozen=True)
class AbstentionReport:
    cases: int
    positives: int
    negatives: int
    threshold: float
    accuracy: float
    precision: float
    positive_recall: float
    negative_abstention: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    negative_false_positives: int
    false_negatives: int
    true_negatives: int
    answered: int
    answer_rate: float
    retriever: str = "event_memory"
    confidence_method: str = "top_score"
    selection_mode: str = "evidence_set"
    by_category: dict[str, "AbstentionReport"] | None = None


@dataclass(frozen=True)
class CalibratedAbstentionRun:
    retriever: str
    confidence_method: str
    selection_mode: str
    threshold: float
    target_negative_abstention: float
    calibration_cases: int
    evaluation_cases: int
    calibration_report: AbstentionReport
    evaluation_report: AbstentionReport


@dataclass(frozen=True)
class EvidenceDecision:
    query: str
    threshold: float
    confidence: float
    answered: bool
    candidate_event_ids: tuple[str, ...]
    retriever: str
    confidence_method: str
    selection_mode: str


def evaluate_abstention(
    memory: EventMemory,
    cases: list[EvidenceCaseLike],
    *,
    k: int = 2,
    threshold: float = 1.0,
    retriever: RetrieverName = "event_memory",
    confidence_method: ConfidenceMethod = "top_score",
    selection_mode: SelectionMode = "evidence_set",
) -> AbstentionReport:
    """Evaluate whether retrieval scores can support answer/abstain decisions."""

    normalized = [
        (case.query, case.expected_event_id, case.category)
        for case in cases
    ]
    overall = _evaluate_normalized(
        memory,
        normalized,
        k=k,
        threshold=threshold,
        retriever=retriever,
        confidence_method=confidence_method,
        selection_mode=selection_mode,
    )

    buckets: dict[str, list[tuple[str, str | None, str]]] = defaultdict(list)
    for case in normalized:
        buckets[case[2]].append(case)

    return AbstentionReport(
        cases=overall.cases,
        positives=overall.positives,
        negatives=overall.negatives,
        threshold=overall.threshold,
        accuracy=overall.accuracy,
        precision=overall.precision,
        positive_recall=overall.positive_recall,
        negative_abstention=overall.negative_abstention,
        false_positive_rate=overall.false_positive_rate,
        true_positives=overall.true_positives,
        false_positives=overall.false_positives,
        negative_false_positives=overall.negative_false_positives,
        false_negatives=overall.false_negatives,
        true_negatives=overall.true_negatives,
        answered=overall.answered,
        answer_rate=overall.answer_rate,
        retriever=retriever,
        confidence_method=confidence_method,
        selection_mode=selection_mode,
        by_category={
            category: _evaluate_normalized(
                memory,
                bucket,
                k=k,
                threshold=threshold,
                retriever=retriever,
                confidence_method=confidence_method,
                selection_mode=selection_mode,
            )
            for category, bucket in sorted(buckets.items())
        },
    )


def decide_evidence(
    memory: EventMemory,
    query: str,
    *,
    k: int = 2,
    threshold: float = 1.0,
    retriever: RetrieverName = "event_memory",
    confidence_method: ConfidenceMethod = "top_score",
    selection_mode: SelectionMode = "evidence_set",
) -> EvidenceDecision:
    """Return the answer/admit decision for one query."""

    results = _retrieve(memory, query, k=k, retriever=retriever)
    confidence = _confidence(memory, query, results, confidence_method) if results else 0.0
    if confidence < threshold:
        candidate_ids: tuple[str, ...] = ()
    elif selection_mode == "top_event":
        candidate_ids = (results[0].event.event_id,)
    else:
        candidate_ids = tuple(dict.fromkeys(result.event.event_id for result in results))

    return EvidenceDecision(
        query=query,
        threshold=threshold,
        confidence=confidence,
        answered=bool(candidate_ids),
        candidate_event_ids=candidate_ids,
        retriever=retriever,
        confidence_method=confidence_method,
        selection_mode=selection_mode,
    )


def sweep_abstention_thresholds(
    memory: EventMemory,
    cases: list[EvidenceCaseLike],
    *,
    thresholds: list[float] | None = None,
    k: int = 2,
    retriever: RetrieverName = "event_memory",
    confidence_method: ConfidenceMethod = "top_score",
    selection_mode: SelectionMode = "evidence_set",
) -> list[AbstentionReport]:
    """Evaluate a fixed threshold grid for evidence calibration."""

    thresholds = thresholds or default_thresholds(confidence_method)
    return [
        evaluate_abstention(
            memory,
            cases,
            k=k,
            threshold=threshold,
            retriever=retriever,
            confidence_method=confidence_method,
            selection_mode=selection_mode,
        )
        for threshold in thresholds
    ]


def best_abstention_report(reports: list[AbstentionReport]) -> AbstentionReport:
    """Select the best report by accuracy, then positive recall, then precision."""

    if not reports:
        raise ValueError("best_abstention_report requires at least one report")
    return max(
        reports,
        key=lambda report: (
            report.accuracy,
            report.positive_recall,
            report.precision,
            -report.false_positive_rate,
        ),
    )


def calibrate_abstention_threshold(
    memory: EventMemory,
    calibration_cases: list[EvidenceCaseLike],
    evaluation_cases: list[EvidenceCaseLike],
    *,
    thresholds: list[float] | None = None,
    k: int = 2,
    retriever: RetrieverName = "event_memory",
    confidence_method: ConfidenceMethod = "evidence",
    selection_mode: SelectionMode = "evidence_set",
    target_negative_abstention: float = 0.80,
) -> CalibratedAbstentionRun:
    """Select a threshold on calibration cases and report held-out performance."""

    calibration_reports = sweep_abstention_thresholds(
        memory,
        calibration_cases,
        thresholds=thresholds,
        k=k,
        retriever=retriever,
        confidence_method=confidence_method,
        selection_mode=selection_mode,
    )
    candidates = [
        report
        for report in calibration_reports
        if report.negative_abstention >= target_negative_abstention
    ]
    if candidates:
        selected = max(
            candidates,
            key=lambda report: (
                report.positive_recall,
                report.accuracy,
                report.precision,
                -report.threshold,
            ),
        )
    else:
        selected = best_abstention_report(calibration_reports)

    evaluation_report = evaluate_abstention(
        memory,
        evaluation_cases,
        k=k,
        threshold=selected.threshold,
        retriever=retriever,
        confidence_method=confidence_method,
        selection_mode=selection_mode,
    )
    return CalibratedAbstentionRun(
        retriever=retriever,
        confidence_method=confidence_method,
        selection_mode=selection_mode,
        threshold=selected.threshold,
        target_negative_abstention=target_negative_abstention,
        calibration_cases=len(calibration_cases),
        evaluation_cases=len(evaluation_cases),
        calibration_report=selected,
        evaluation_report=evaluation_report,
    )


def split_evidence_cases(
    cases: list[EvidenceCaseLike],
    *,
    calibration_fraction: float = 0.40,
) -> tuple[list[EvidenceCaseLike], list[EvidenceCaseLike]]:
    """Split evidence cases by story ID when available."""

    if not cases:
        return [], []

    story_ids = [getattr(case, "story_id", "") for case in cases]
    if all(story_ids):
        ordered_story_ids = sorted(set(story_ids), key=_stable_split_key)
        cutoff = max(1, min(len(ordered_story_ids) - 1, round(len(ordered_story_ids) * calibration_fraction)))
        calibration_ids = set(ordered_story_ids[:cutoff])
        return (
            [case for case in cases if getattr(case, "story_id", "") in calibration_ids],
            [case for case in cases if getattr(case, "story_id", "") not in calibration_ids],
        )

    cutoff = max(1, min(len(cases) - 1, round(len(cases) * calibration_fraction)))
    return cases[:cutoff], cases[cutoff:]


def _stable_split_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def default_thresholds(confidence_method: ConfidenceMethod) -> list[float]:
    if confidence_method in {"top_score", "score_with_evidence_veto"}:
        return [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
    return [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _evaluate_normalized(
    memory: EventMemory,
    cases: list[tuple[str, str | None, str]],
    *,
    k: int,
    threshold: float,
    retriever: RetrieverName,
    confidence_method: ConfidenceMethod,
    selection_mode: SelectionMode,
) -> AbstentionReport:
    true_positives = 0
    false_positives = 0
    negative_false_positives = 0
    false_negatives = 0
    true_negatives = 0
    answered = 0
    positives = sum(1 for _query, expected_id, _category in cases if expected_id is not None)
    negatives = len(cases) - positives

    for query, expected_event_id, _category in cases:
        decision = decide_evidence(
            memory,
            query,
            k=k,
            threshold=threshold,
            retriever=retriever,
            confidence_method=confidence_method,
            selection_mode=selection_mode,
        )
        candidate_ids = set(decision.candidate_event_ids)
        did_answer = bool(candidate_ids)
        answered += int(did_answer)

        if expected_event_id is None:
            if did_answer:
                false_positives += 1
                negative_false_positives += 1
            else:
                true_negatives += 1
            continue

        if expected_event_id in candidate_ids:
            true_positives += 1
        else:
            false_negatives += 1
            if did_answer:
                false_positives += 1

    precision_denominator = true_positives + false_positives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    positive_recall = true_positives / positives if positives else 0.0
    negative_abstention = true_negatives / negatives if negatives else 0.0
    false_positive_rate = negative_false_positives / negatives if negatives else 0.0
    accuracy = (true_positives + true_negatives) / len(cases) if cases else 0.0
    answer_rate = answered / len(cases) if cases else 0.0

    return AbstentionReport(
        cases=len(cases),
        positives=positives,
        negatives=negatives,
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        positive_recall=positive_recall,
        negative_abstention=negative_abstention,
        false_positive_rate=false_positive_rate,
        true_positives=true_positives,
        false_positives=false_positives,
        negative_false_positives=negative_false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        answered=answered,
        answer_rate=answer_rate,
        retriever=retriever,
        confidence_method=confidence_method,
        selection_mode=selection_mode,
    )


def _retrieve(
    memory: EventMemory,
    query: str,
    *,
    k: int,
    retriever: RetrieverName,
) -> list[RetrievalResult]:
    if retriever == "rag":
        return memory.retrieve_rag(query, k=k)
    return memory.retrieve_event_memory(query, k=k)


def _confidence(
    memory: EventMemory,
    query: str,
    results: list[RetrievalResult],
    confidence_method: ConfidenceMethod,
) -> float:
    if confidence_method == "top_score":
        return results[0].score
    if confidence_method == "score_margin":
        second_score = results[1].score if len(results) > 1 else 0.0
        return max(results[0].score - second_score, 0.0)
    if confidence_method == "score_with_evidence_veto":
        evidence_confidence = _evidence_confidence(memory, query, results)
        if evidence_confidence < EVIDENCE_VETO_THRESHOLD:
            return 0.0
        return results[0].score + _support_bonus(results)
    return _evidence_confidence(memory, query, results)


def _support_bonus(results: list[RetrievalResult]) -> float:
    result_ids = {result.event.event_id for result in results}
    for result in results:
        if result.reason.startswith(("causal_link:", "temporal_neighbor:")):
            return EVIDENCE_SUPPORT_BONUS
        if any(causal_id in result_ids for causal_id in result.event.causal_links):
            return EVIDENCE_SUPPORT_BONUS
    return 0.0


def _evidence_confidence(memory: EventMemory, query: str, results: list[RetrievalResult]) -> float:
    query_terms = _content_terms(query)
    if not query_terms:
        return 0.0

    evidence_text = _expanded_evidence_text(memory, results)
    evidence_terms = _content_terms(evidence_text)
    coverage = len(query_terms & evidence_terms) / len(query_terms)
    relation_score = _relation_score(memory, query_terms, results)
    return coverage if relation_score == 1.0 else coverage * 0.35


def _expanded_evidence_text(memory: EventMemory, results: list[RetrievalResult]) -> str:
    parts: list[str] = []
    seen_ids: set[str] = set()
    for result in results:
        if result.event.event_id not in seen_ids:
            parts.append(result.event.searchable_text())
            seen_ids.add(result.event.event_id)

        for prefix in ("causal_link:", "temporal_neighbor:"):
            if not result.reason.startswith(prefix):
                continue
            source_id = result.reason.removeprefix(prefix)
            source_event = memory.get(source_id)
            if source_event is not None and source_event.event_id not in seen_ids:
                parts.append(source_event.searchable_text())
                seen_ids.add(source_event.event_id)

    return " ".join(parts)


def _relation_score(
    memory: EventMemory,
    query_terms: set[str],
    results: list[RetrievalResult],
) -> float:
    all_events = memory.events()
    known_actors = {actor for event in all_events for actor in event.actors}
    known_actions = {event.action_or_state.lower() for event in all_events if event.action_or_state}
    query_actors = {actor for actor in known_actors if _phrase_in_terms(actor, query_terms)}
    query_actions = {
        action
        for action in known_actions
        if _term_matches(action, query_terms)
    }
    payload_terms = _payload_terms(
        query_terms,
        actors=query_actors,
        actions=query_actions,
    )

    if not query_actions:
        return 1.0

    for result in results:
        event = result.event
        action_matches = _term_matches(event.action_or_state, query_terms)
        if not action_matches:
            continue
        if not query_actors:
            if _event_supports_payload(event, payload_terms):
                return 1.0
            continue
        if any(actor in query_actors for actor in event.actors) and _event_supports_payload(
            event,
            payload_terms,
        ):
            return 1.0
    return 0.0


def _payload_terms(
    query_terms: set[str],
    *,
    actors: set[str],
    actions: set[str],
) -> set[str]:
    ignored: set[str] = set()
    for action in actions:
        ignored.update(_stem(part) for part in CONTENT_RE.findall(action.lower()))
    for actor in actors:
        ignored.update(_stem(part) for part in CONTENT_RE.findall(actor.lower()))
    return query_terms - ignored


def _event_supports_payload(event, payload_terms: set[str]) -> bool:
    if not payload_terms:
        return True
    required_overlap = min(2, len(payload_terms))
    for object_phrase in event.objects:
        object_terms = {
            _stem(CANONICAL_TERMS.get(part, part))
            for part in CONTENT_RE.findall(object_phrase.lower())
        }
        if len(object_terms & payload_terms) >= required_overlap:
            return True
    return False


def _content_terms(text: str) -> set[str]:
    return {
        _stem(CANONICAL_TERMS.get(term, term))
        for term in CONTENT_RE.findall(text.lower())
        if term not in STOPWORDS and len(term) > 1
    }


def _term_matches(term: str, query_terms: set[str]) -> bool:
    if not term:
        return False
    parts = [_stem(part) for part in CONTENT_RE.findall(term.lower())]
    if not parts:
        return False
    return any(part in query_terms for part in parts)


def _phrase_in_terms(phrase: str, query_terms: set[str]) -> bool:
    parts = [_stem(part) for part in CONTENT_RE.findall(phrase.lower())]
    return bool(parts) and all(part in query_terms for part in parts)


def _stem(term: str) -> str:
    if term.endswith("'s") and len(term) > 3:
        term = term[:-2]
    for suffix in ("ing", "ed", "s"):
        if len(term) > len(suffix) + 2 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term
