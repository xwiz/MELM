"""State-assisted evaluation for BabyLM entity tracking."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from melm.benchmarks import MultipleChoiceCase, predict_entity_tracking_option


@dataclass(frozen=True)
class StateAssistedPrediction:
    case_id: str
    category: str
    label_index: int
    lm_predicted_index: int | None
    state_predicted_index: int | None
    predicted_index: int | None
    source: str
    correct: bool


@dataclass(frozen=True)
class StateAssistedEntityCategoryReport:
    category: str
    cases: int
    correct: int
    accuracy: float
    lm_correct: int
    lm_accuracy: float
    state_answered: int
    state_answer_rate: float


@dataclass(frozen=True)
class StateAssistedEntityReport:
    cases: int
    correct: int
    accuracy: float
    lm_correct: int
    lm_accuracy: float
    accuracy_lift: float
    state_answered: int
    state_answer_rate: float
    state_correct_when_answered: int
    state_accuracy_when_answered: float
    lm_fallbacks: int
    abstentions: int
    by_category: dict[str, StateAssistedEntityCategoryReport]
    predictions: list[StateAssistedPrediction]


def evaluate_state_assisted_entity_tracking(
    cases: list[MultipleChoiceCase],
    lm_predictions: list[dict[str, Any]],
) -> StateAssistedEntityReport:
    """Evaluate state-first, LM-fallback predictions for entity tracking."""

    lm_by_case: dict[str, deque[int]] = defaultdict(deque)
    for item in lm_predictions:
        lm_by_case[str(item["case_id"])].append(int(item["predicted_index"]))
    predictions: list[StateAssistedPrediction] = []
    for case in cases:
        lm_queue = lm_by_case.get(case.case_id)
        lm_index = lm_queue.popleft() if lm_queue else None
        state_index = predict_entity_tracking_option(case)
        if state_index is not None:
            predicted_index = state_index
            source = "state_memory"
        elif lm_index is not None:
            predicted_index = lm_index
            source = "lm_fallback"
        else:
            predicted_index = None
            source = "abstain"
        predictions.append(
            StateAssistedPrediction(
                case_id=case.case_id,
                category=case.category,
                label_index=case.label_index,
                lm_predicted_index=lm_index,
                state_predicted_index=state_index,
                predicted_index=predicted_index,
                source=source,
                correct=predicted_index == case.label_index,
            )
        )
    return _summarize(predictions)


def _summarize(predictions: list[StateAssistedPrediction]) -> StateAssistedEntityReport:
    cases = len(predictions)
    correct = sum(1 for prediction in predictions if prediction.correct)
    lm_correct = sum(
        1
        for prediction in predictions
        if prediction.lm_predicted_index == prediction.label_index
    )
    state_answered_items = [
        prediction
        for prediction in predictions
        if prediction.state_predicted_index is not None
    ]
    state_correct_when_answered = sum(
        1
        for prediction in state_answered_items
        if prediction.state_predicted_index == prediction.label_index
    )
    lm_fallbacks = sum(1 for prediction in predictions if prediction.source == "lm_fallback")
    abstentions = sum(1 for prediction in predictions if prediction.source == "abstain")
    accuracy = correct / cases if cases else 0.0
    lm_accuracy = lm_correct / cases if cases else 0.0
    return StateAssistedEntityReport(
        cases=cases,
        correct=correct,
        accuracy=accuracy,
        lm_correct=lm_correct,
        lm_accuracy=lm_accuracy,
        accuracy_lift=accuracy - lm_accuracy,
        state_answered=len(state_answered_items),
        state_answer_rate=len(state_answered_items) / cases if cases else 0.0,
        state_correct_when_answered=state_correct_when_answered,
        state_accuracy_when_answered=(
            state_correct_when_answered / len(state_answered_items)
            if state_answered_items
            else 0.0
        ),
        lm_fallbacks=lm_fallbacks,
        abstentions=abstentions,
        by_category=_category_reports(predictions),
        predictions=predictions,
    )


def _category_reports(
    predictions: list[StateAssistedPrediction],
) -> dict[str, StateAssistedEntityCategoryReport]:
    buckets: dict[str, list[StateAssistedPrediction]] = defaultdict(list)
    for prediction in predictions:
        buckets[prediction.category].append(prediction)
    reports: dict[str, StateAssistedEntityCategoryReport] = {}
    for category, items in sorted(buckets.items()):
        correct = sum(1 for prediction in items if prediction.correct)
        lm_correct = sum(
            1
            for prediction in items
            if prediction.lm_predicted_index == prediction.label_index
        )
        state_answered = sum(
            1 for prediction in items if prediction.state_predicted_index is not None
        )
        reports[category] = StateAssistedEntityCategoryReport(
            category=category,
            cases=len(items),
            correct=correct,
            accuracy=correct / len(items) if items else 0.0,
            lm_correct=lm_correct,
            lm_accuracy=lm_correct / len(items) if items else 0.0,
            state_answered=state_answered,
            state_answer_rate=state_answered / len(items) if items else 0.0,
        )
    return reports
