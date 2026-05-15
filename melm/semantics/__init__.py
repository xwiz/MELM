"""Morpheme/root/meaning MVP utilities."""

from .morpheme_meaning import (
    MeaningComponent,
    MeaningCorpus,
    MeaningInference,
    MeaningMvpReport,
    UtteranceInference,
    evaluate_meaning_mvp,
    load_meaning_corpus,
)

__all__ = [
    "MeaningComponent",
    "MeaningCorpus",
    "MeaningInference",
    "MeaningMvpReport",
    "UtteranceInference",
    "evaluate_meaning_mvp",
    "load_meaning_corpus",
]
