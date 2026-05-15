"""State tracking over event-memory timelines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from .schema import Event


class StateResolutionCaseLike(Protocol):
    query: str
    object_name: str
    expected_location: str | None
    category: str
    before_event_id: str | None
    at_or_before_event_id: str | None


@dataclass(frozen=True)
class ObjectLocationObservation:
    object_name: str
    location: str
    event_id: str
    time_index: int
    source_span: str
    relation: str


@dataclass(frozen=True)
class StateResolutionResult:
    query: str
    object_name: str
    expected_location: str | None
    predicted_location: str | None
    predicted_event_id: str | None
    correct: bool
    category: str


@dataclass(frozen=True)
class StateResolutionReport:
    cases: int
    accuracy: float
    answer_rate: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    results: list[StateResolutionResult]
    by_category: dict[str, "StateResolutionReport"] | None = None


class ObjectLocationTracker:
    """Resolve simple object-location state from event timelines.

    The tracker deliberately uses explicit event fields instead of free-text
    extraction. If future data does not supply event/state annotations, it
    should fail visibly rather than silently hallucinating locations.
    """

    def __init__(self, events: list[Event] | tuple[Event, ...]) -> None:
        self._events = {event.event_id: event for event in events}
        observations = [_location_observation(event) for event in events]
        self._observations: dict[str, list[ObjectLocationObservation]] = defaultdict(list)
        for observation in observations:
            if observation is None:
                continue
            self._observations[_normalize(observation.object_name)].append(observation)
        for object_observations in self._observations.values():
            object_observations.sort(key=lambda item: (item.time_index, item.event_id))

    def resolve_location(
        self,
        object_name: str,
        *,
        before_event_id: str | None = None,
        at_or_before_event_id: str | None = None,
    ) -> ObjectLocationObservation | None:
        """Return the best known object location under an optional time bound."""

        observations = self._observations.get(_normalize(object_name), [])
        if not observations:
            return None

        upper_bound: int | None = None
        inclusive = True
        if before_event_id:
            event = self._events.get(before_event_id)
            if event is None:
                return None
            upper_bound = event.time_index
            inclusive = False
        elif at_or_before_event_id:
            event = self._events.get(at_or_before_event_id)
            if event is None:
                return None
            upper_bound = event.time_index

        candidates = observations
        if upper_bound is not None:
            if inclusive:
                candidates = [
                    observation
                    for observation in observations
                    if observation.time_index <= upper_bound
                ]
            else:
                candidates = [
                    observation
                    for observation in observations
                    if observation.time_index < upper_bound
                ]
        return candidates[-1] if candidates else None


def evaluate_state_resolution(
    events: list[Event] | tuple[Event, ...],
    cases: list[StateResolutionCaseLike],
) -> StateResolutionReport:
    """Evaluate explicit state resolution cases against event annotations."""

    tracker = ObjectLocationTracker(events)
    results = [
        _evaluate_case(tracker, case)
        for case in cases
    ]
    report = _summarize_results(results)

    buckets: dict[str, list[StateResolutionResult]] = defaultdict(list)
    for result in results:
        buckets[result.category].append(result)

    return StateResolutionReport(
        cases=report.cases,
        accuracy=report.accuracy,
        answer_rate=report.answer_rate,
        false_positive_rate=report.false_positive_rate,
        true_positives=report.true_positives,
        false_positives=report.false_positives,
        false_negatives=report.false_negatives,
        true_negatives=report.true_negatives,
        results=results,
        by_category={
            category: _summarize_results(bucket)
            for category, bucket in sorted(buckets.items())
        },
    )


def _evaluate_case(
    tracker: ObjectLocationTracker,
    case: StateResolutionCaseLike,
) -> StateResolutionResult:
    observation = tracker.resolve_location(
        case.object_name,
        before_event_id=case.before_event_id,
        at_or_before_event_id=case.at_or_before_event_id,
    )
    predicted = observation.location if observation else None
    expected = _normalize_location(case.expected_location)
    normalized_predicted = _normalize_location(predicted)
    return StateResolutionResult(
        query=case.query,
        object_name=case.object_name,
        expected_location=case.expected_location,
        predicted_location=predicted,
        predicted_event_id=observation.event_id if observation else None,
        correct=normalized_predicted == expected,
        category=case.category,
    )


def _summarize_results(results: list[StateResolutionResult]) -> StateResolutionReport:
    positives = [result for result in results if result.expected_location is not None]
    negatives = [result for result in results if result.expected_location is None]
    true_positives = sum(
        1
        for result in positives
        if result.correct and result.predicted_location is not None
    )
    false_negatives = len(positives) - true_positives
    false_positives = sum(
        1
        for result in negatives
        if result.predicted_location is not None
    )
    true_negatives = len(negatives) - false_positives
    answered = sum(1 for result in results if result.predicted_location is not None)
    return StateResolutionReport(
        cases=len(results),
        accuracy=sum(1 for result in results if result.correct) / len(results) if results else 0.0,
        answer_rate=answered / len(results) if results else 0.0,
        false_positive_rate=false_positives / len(negatives) if negatives else 0.0,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        results=results,
    )


def _location_observation(event: Event) -> ObjectLocationObservation | None:
    state_object = event.metadata.get("state_object")
    location_after = event.metadata.get("location_after")
    if state_object and location_after:
        return ObjectLocationObservation(
            object_name=state_object,
            location=location_after,
            event_id=event.event_id,
            time_index=event.time_index,
            source_span=event.source_span,
            relation="metadata_location_after",
        )

    action = event.action_or_state.lower().strip()
    if action in {"put", "placed", "set"} and len(event.objects) >= 2:
        return ObjectLocationObservation(
            object_name=event.objects[0],
            location=event.objects[1],
            event_id=event.event_id,
            time_index=event.time_index,
            source_span=event.source_span,
            relation=action,
        )
    if action in {"moved", "move"} and len(event.objects) >= 3:
        return ObjectLocationObservation(
            object_name=event.objects[0],
            location=event.objects[-1],
            event_id=event.event_id,
            time_index=event.time_index,
            source_span=event.source_span,
            relation=action,
        )
    return None


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _normalize_location(value: str | None) -> str | None:
    return None if value is None else _normalize(value)
