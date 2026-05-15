"""Evaluate sentence minimal-pair ranking results."""

from __future__ import annotations

from dataclasses import dataclass

from melm.benchmarks import MinimalPairCase


@dataclass(frozen=True)
class MinimalPairPrediction:
    case_id: str
    category: str
    good_score: float
    bad_score: float
    margin: float
    chose_good: bool


@dataclass(frozen=True)
class MinimalPairCategoryReport:
    category: str
    cases: int
    correct: int
    accuracy: float


@dataclass(frozen=True)
class MinimalPairReport:
    cases: int
    correct: int
    accuracy: float
    by_category: dict[str, MinimalPairCategoryReport]
    predictions: list[MinimalPairPrediction]


def evaluate_minimal_pair_scores(
    cases: list[MinimalPairCase],
    score_by_text: dict[str, float],
) -> MinimalPairReport:
    """Evaluate minimal pairs where lower scores are better."""

    predictions: list[MinimalPairPrediction] = []
    for case in cases:
        good_score = score_by_text[case.good]
        bad_score = score_by_text[case.bad]
        margin = bad_score - good_score
        predictions.append(
            MinimalPairPrediction(
                case_id=case.case_id,
                category=case.category,
                good_score=good_score,
                bad_score=bad_score,
                margin=margin,
                chose_good=margin > 0.0,
            )
        )

    correct = sum(1 for prediction in predictions if prediction.chose_good)
    by_category = _category_reports(predictions)
    return MinimalPairReport(
        cases=len(cases),
        correct=correct,
        accuracy=correct / len(cases) if cases else 0.0,
        by_category=by_category,
        predictions=predictions,
    )


def _category_reports(
    predictions: list[MinimalPairPrediction],
) -> dict[str, MinimalPairCategoryReport]:
    categories = sorted({prediction.category for prediction in predictions})
    reports: dict[str, MinimalPairCategoryReport] = {}
    for category in categories:
        items = [prediction for prediction in predictions if prediction.category == category]
        correct = sum(1 for prediction in items if prediction.chose_good)
        reports[category] = MinimalPairCategoryReport(
            category=category,
            cases=len(items),
            correct=correct,
            accuracy=correct / len(items) if items else 0.0,
        )
    return reports
