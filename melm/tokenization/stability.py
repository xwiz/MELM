"""Cross-split stability checks for tokenizer LM probes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

from .factory import build_default_tokenizers
from .language_model import TokenLMReport, compare_unigram_lms


@dataclass(frozen=True)
class TokenizerFoldReport:
    fold: int
    train_documents: int
    validation_documents: int
    best_tokenizer: str
    best_baseline_tokenizer: str
    morph_nll_per_token: float
    best_baseline_nll_per_token: float
    lm_nll_gain: float
    reports: list[TokenLMReport]


@dataclass(frozen=True)
class TokenizerStabilityReport:
    folds: int
    documents: int
    morph_tokenizer: str
    winner_counts: dict[str, int]
    average_nll_per_token: dict[str, float]
    morph_average_nll_per_token: float
    best_baseline_tokenizer: str
    best_baseline_average_nll_per_token: float
    average_lm_nll_gain: float
    morph_win_rate: float
    stable_primary_candidate: bool
    fold_reports: list[TokenizerFoldReport]


def make_lm_folds(
    texts: list[str],
    *,
    folds: int = 5,
    seed: int = 13,
) -> list[tuple[list[str], list[str]]]:
    """Create deterministic document-level folds for tokenizer probes."""

    documents = _expand_documents(texts)
    if not documents:
        return []
    if len(documents) == 1:
        return [(documents, documents)]

    fold_count = max(2, min(folds, len(documents)))
    shuffled = documents[:]
    random.Random(seed).shuffle(shuffled)

    splits: list[tuple[list[str], list[str]]] = []
    for fold in range(fold_count):
        validation = [
            document
            for index, document in enumerate(shuffled)
            if index % fold_count == fold
        ]
        train = [
            document
            for index, document in enumerate(shuffled)
            if index % fold_count != fold
        ]
        if validation:
            splits.append((train or validation, validation))
    return splits


def evaluate_tokenizer_lm_stability(
    texts: list[str],
    *,
    folds: int = 5,
    seed: int = 13,
    morph_tokenizer: str = "heuristic_morpheme",
    downstream_margin: float = 0.0,
    min_win_rate: float = 0.5,
) -> TokenizerStabilityReport:
    """Evaluate whether tokenizer LM ranking is stable across document folds."""

    documents = _expand_documents(texts)
    if not documents:
        raise ValueError("Tokenizer stability requires at least one text document")

    fold_splits = make_lm_folds(documents, folds=folds, seed=seed)
    if not fold_splits:
        raise ValueError("Tokenizer stability could not create any folds")

    fold_reports: list[TokenizerFoldReport] = []
    winner_counts: dict[str, int] = defaultdict(int)
    nll_values: dict[str, list[float]] = defaultdict(list)
    morph_wins = 0

    for fold_index, (train_texts, validation_texts) in enumerate(fold_splits, start=1):
        tokenizers = build_default_tokenizers(documents, train_texts=train_texts)
        reports = compare_unigram_lms(tokenizers, train_texts, validation_texts)
        best = reports[0]
        morph = _find_lm(reports, morph_tokenizer)
        baselines = [
            report for report in reports if report.tokenizer != morph_tokenizer
        ]
        if not baselines:
            raise ValueError("Tokenizer stability requires at least one non-morphology baseline")
        best_baseline = min(baselines, key=lambda report: report.nll_per_token)

        winner_counts[best.tokenizer] += 1
        if best.tokenizer == morph_tokenizer:
            morph_wins += 1
        for report in reports:
            nll_values[report.tokenizer].append(report.nll_per_token)

        fold_reports.append(
            TokenizerFoldReport(
                fold=fold_index,
                train_documents=len(train_texts),
                validation_documents=len(validation_texts),
                best_tokenizer=best.tokenizer,
                best_baseline_tokenizer=best_baseline.tokenizer,
                morph_nll_per_token=morph.nll_per_token,
                best_baseline_nll_per_token=best_baseline.nll_per_token,
                lm_nll_gain=best_baseline.nll_per_token - morph.nll_per_token,
                reports=reports,
            )
        )

    average_nll = {
        tokenizer: sum(values) / len(values)
        for tokenizer, values in sorted(nll_values.items())
    }
    morph_average = average_nll[morph_tokenizer]
    baseline_average = {
        tokenizer: value
        for tokenizer, value in average_nll.items()
        if tokenizer != morph_tokenizer
    }
    best_baseline_tokenizer = min(
        baseline_average,
        key=lambda tokenizer: baseline_average[tokenizer],
    )
    best_baseline_average = baseline_average[best_baseline_tokenizer]
    average_gain = best_baseline_average - morph_average
    morph_win_rate = morph_wins / len(fold_reports)

    return TokenizerStabilityReport(
        folds=len(fold_reports),
        documents=len(documents),
        morph_tokenizer=morph_tokenizer,
        winner_counts=dict(sorted(winner_counts.items())),
        average_nll_per_token=average_nll,
        morph_average_nll_per_token=morph_average,
        best_baseline_tokenizer=best_baseline_tokenizer,
        best_baseline_average_nll_per_token=best_baseline_average,
        average_lm_nll_gain=average_gain,
        morph_win_rate=morph_win_rate,
        stable_primary_candidate=(
            average_gain > downstream_margin and morph_win_rate >= min_win_rate
        ),
        fold_reports=fold_reports,
    )


def _find_lm(reports: list[TokenLMReport], tokenizer: str) -> TokenLMReport:
    for report in reports:
        if report.tokenizer == tokenizer:
            return report
    raise ValueError(f"Missing LM report for tokenizer {tokenizer!r}")


def _expand_documents(texts: list[str]) -> list[str]:
    documents: list[str] = []
    for text in texts:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        documents.extend(paragraphs if len(paragraphs) > 1 else [text])
    return documents
