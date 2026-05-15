"""Evaluate multiple-choice language-model ranking tasks."""

from __future__ import annotations

from dataclasses import dataclass

from melm.benchmarks import MultipleChoiceCase


@dataclass(frozen=True)
class MultipleChoicePrediction:
    case_id: str
    category: str
    label_index: int
    predicted_index: int
    correct: bool
    option_scores: tuple[float, ...]


@dataclass(frozen=True)
class MultipleChoiceCategoryReport:
    category: str
    cases: int
    correct: int
    accuracy: float


@dataclass(frozen=True)
class MultipleChoiceReport:
    cases: int
    correct: int
    accuracy: float
    by_category: dict[str, MultipleChoiceCategoryReport]
    predictions: list[MultipleChoicePrediction]


def evaluate_multiple_choice_scores(
    cases: list[MultipleChoiceCase],
    score_by_text: dict[str, float],
) -> MultipleChoiceReport:
    """Evaluate multiple-choice cases where lower scores are better."""

    predictions: list[MultipleChoicePrediction] = []
    for case in cases:
        option_scores = tuple(score_by_text[text] for text in case.option_texts())
        predicted_index = min(range(len(option_scores)), key=lambda index: option_scores[index])
        predictions.append(
            MultipleChoicePrediction(
                case_id=case.case_id,
                category=case.category,
                label_index=case.label_index,
                predicted_index=predicted_index,
                correct=predicted_index == case.label_index,
                option_scores=option_scores,
            )
        )

    correct = sum(1 for prediction in predictions if prediction.correct)
    return MultipleChoiceReport(
        cases=len(cases),
        correct=correct,
        accuracy=correct / len(cases) if cases else 0.0,
        by_category=_category_reports(predictions),
        predictions=predictions,
    )


def _category_reports(
    predictions: list[MultipleChoicePrediction],
) -> dict[str, MultipleChoiceCategoryReport]:
    categories = sorted({prediction.category for prediction in predictions})
    reports: dict[str, MultipleChoiceCategoryReport] = {}
    for category in categories:
        items = [prediction for prediction in predictions if prediction.category == category]
        correct = sum(1 for prediction in items if prediction.correct)
        reports[category] = MultipleChoiceCategoryReport(
            category=category,
            cases=len(items),
            correct=correct,
            accuracy=correct / len(items) if items else 0.0,
        )
    return reports
