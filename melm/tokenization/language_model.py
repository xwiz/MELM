"""Tiny token-level language-model probe for tokenizer comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import Counter

from .simple_tokenizers import Tokenizer


@dataclass(frozen=True)
class TokenLMReport:
    tokenizer: str
    train_tokens: int
    validation_tokens: int
    vocabulary: int
    nll_per_token: float
    perplexity: float
    validation_bytes: int = 0
    bits_per_byte: float = 0.0


def evaluate_unigram_lm(
    tokenizer: Tokenizer,
    train_texts: list[str],
    validation_texts: list[str],
    *,
    alpha: float = 0.1,
) -> TokenLMReport:
    """Train a smoothed unigram LM and evaluate held-out token NLL."""

    counts, train_token_count = _token_counts(tokenizer, train_texts)
    validation_counts, validation_token_count = _token_counts(tokenizer, validation_texts)
    vocab = set(counts)
    vocab.update(validation_counts)
    vocabulary = max(len(vocab), 1)
    denominator = train_token_count + alpha * vocabulary

    total_nll = 0.0
    for token, count in validation_counts.items():
        probability = (counts.get(token, 0) + alpha) / denominator
        total_nll += -math.log(probability) * count

    nll_per_token = total_nll / validation_token_count if validation_token_count else 0.0
    perplexity = math.exp(nll_per_token) if validation_token_count else 0.0
    validation_bytes = sum(len(text.encode("utf-8")) for text in validation_texts)
    bits_per_byte = (
        total_nll / (validation_bytes * math.log(2.0))
        if validation_bytes
        else 0.0
    )
    return TokenLMReport(
        tokenizer=tokenizer.name,
        train_tokens=train_token_count,
        validation_tokens=validation_token_count,
        vocabulary=vocabulary,
        nll_per_token=nll_per_token,
        perplexity=perplexity,
        validation_bytes=validation_bytes,
        bits_per_byte=bits_per_byte,
    )


def compare_unigram_lms(
    tokenizers: list[Tokenizer],
    train_texts: list[str],
    validation_texts: list[str],
) -> list[TokenLMReport]:
    reports = [
        evaluate_unigram_lm(tokenizer, train_texts, validation_texts)
        for tokenizer in tokenizers
    ]
    return sorted(reports, key=lambda report: report.nll_per_token)


def split_for_lm(texts: list[str], *, validation_fraction: float = 0.2) -> tuple[list[str], list[str]]:
    """Split docs/paragraphs into train and validation sets."""

    docs = _expand_docs(texts)
    if not docs:
        return [], []
    if len(docs) == 1:
        return docs, docs
    validation_count = max(1, round(len(docs) * validation_fraction))
    train = docs[:-validation_count]
    validation = docs[-validation_count:]
    return train or docs[:1], validation


def _token_counts(tokenizer: Tokenizer, texts: list[str]) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    total = 0
    for text in texts:
        tokens = tokenizer.tokenize(text)
        counts.update(tokens)
        total += len(tokens)
    return counts, total


def _expand_docs(texts: list[str]) -> list[str]:
    docs: list[str] = []
    for text in texts:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        docs.extend(paragraphs if len(paragraphs) > 1 else [text])
    return docs
