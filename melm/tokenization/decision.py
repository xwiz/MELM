"""Tokenizer decision helpers for validation gates."""

from __future__ import annotations

from dataclasses import dataclass

from .boundary import BoundaryReport
from .language_model import TokenLMReport


@dataclass(frozen=True)
class TokenizerDecisionReport:
    morph_tokenizer: str
    best_lm_tokenizer: str
    morph_nll_per_token: float
    best_baseline_nll_per_token: float
    lm_nll_gain: float
    best_boundary_tokenizer: str
    morph_boundary_f1: float
    best_baseline_boundary_f1: float
    boundary_f1_gain: float
    decision: str
    gate_passed: bool
    recommendation: str


def decide_tokenizer_strategy(
    lm_reports: list[TokenLMReport],
    boundary_reports: list[BoundaryReport],
    *,
    morph_tokenizer: str = "heuristic_morpheme",
    downstream_margin: float = 0.0,
) -> TokenizerDecisionReport:
    """Decide whether morphology should be primary, auxiliary, or rejected."""

    morph_lm = _find_lm(lm_reports, morph_tokenizer)
    morph_boundary = _find_boundary(boundary_reports, morph_tokenizer)
    baseline_lms = [report for report in lm_reports if report.tokenizer != morph_tokenizer]
    baseline_boundaries = [
        report for report in boundary_reports
        if report.tokenizer != morph_tokenizer
    ]
    if not baseline_lms:
        raise ValueError("Tokenizer decision requires at least one non-morphology LM baseline")
    if not baseline_boundaries:
        raise ValueError("Tokenizer decision requires at least one non-morphology boundary baseline")

    best_lm = min(baseline_lms, key=lambda report: report.nll_per_token)
    best_boundary = max(baseline_boundaries, key=lambda report: report.f1)
    lm_gain = best_lm.nll_per_token - morph_lm.nll_per_token
    boundary_gain = morph_boundary.f1 - best_boundary.f1

    downstream_passed = lm_gain > downstream_margin
    boundary_passed = boundary_gain > 0.0

    if downstream_passed:
        decision = "primary_candidate"
        recommendation = (
            "Keep morphology as a primary tokenizer candidate and verify on BabyLM-scale tasks."
        )
    elif boundary_passed:
        decision = "auxiliary_only"
        recommendation = (
            "Use morphology for auxiliary supervision or analysis, not as the primary tokenizer yet."
        )
    else:
        decision = "reject_for_now"
        recommendation = (
            "Do not invest in morphology-specific tokenizer work until stronger evidence appears."
        )

    return TokenizerDecisionReport(
        morph_tokenizer=morph_tokenizer,
        best_lm_tokenizer=best_lm.tokenizer,
        morph_nll_per_token=morph_lm.nll_per_token,
        best_baseline_nll_per_token=best_lm.nll_per_token,
        lm_nll_gain=lm_gain,
        best_boundary_tokenizer=best_boundary.tokenizer,
        morph_boundary_f1=morph_boundary.f1,
        best_baseline_boundary_f1=best_boundary.f1,
        boundary_f1_gain=boundary_gain,
        decision=decision,
        gate_passed=downstream_passed,
        recommendation=recommendation,
    )


def _find_lm(reports: list[TokenLMReport], tokenizer: str) -> TokenLMReport:
    for report in reports:
        if report.tokenizer == tokenizer:
            return report
    raise ValueError(f"Missing LM report for tokenizer {tokenizer!r}")


def _find_boundary(reports: list[BoundaryReport], tokenizer: str) -> BoundaryReport:
    for report in reports:
        if report.tokenizer == tokenizer:
            return report
    raise ValueError(f"Missing boundary report for tokenizer {tokenizer!r}")
