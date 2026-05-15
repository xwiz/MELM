"""Deterministic statistical helpers for benchmark reports."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    samples: int


def bootstrap_mean_ci(
    values: list[float] | tuple[float, ...],
    *,
    samples: int = 1000,
    seed: int = 13,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Return a percentile bootstrap confidence interval for a mean."""

    if not values:
        return ConfidenceInterval(0.0, 0.0, 0.0, samples)
    rng = random.Random(seed)
    observed = sum(values) / len(values)
    estimates: list[float] = []
    values = tuple(float(value) for value in values)
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    low_index = max(0, min(len(estimates) - 1, int((alpha / 2.0) * len(estimates))))
    high_index = max(0, min(len(estimates) - 1, int((1.0 - alpha / 2.0) * len(estimates)) - 1))
    return ConfidenceInterval(
        estimate=observed,
        low=estimates[low_index],
        high=estimates[high_index],
        samples=samples,
    )


def bootstrap_paired_difference_ci(
    candidate: list[bool] | tuple[bool, ...],
    baseline: list[bool] | tuple[bool, ...],
    *,
    samples: int = 1000,
    seed: int = 13,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Bootstrap the paired mean difference candidate-baseline."""

    if len(candidate) != len(baseline):
        raise ValueError("candidate and baseline must have the same length")
    differences = [
        float(candidate_value) - float(baseline_value)
        for candidate_value, baseline_value in zip(candidate, baseline)
    ]
    return bootstrap_mean_ci(differences, samples=samples, seed=seed, alpha=alpha)
