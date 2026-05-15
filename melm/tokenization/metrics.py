"""Tokenization metrics for early MELM gates."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .simple_tokenizers import Tokenizer, WORD_RE


@dataclass(frozen=True)
class TokenizationReport:
    tokenizer: str
    documents: int
    words: int
    tokens: int
    unique_tokens: int
    tokens_per_word: float
    compression_vs_chars: float
    fallback_rate: float


def evaluate_tokenizer(tokenizer: Tokenizer, texts: list[str]) -> TokenizationReport:
    """Evaluate a tokenizer on simple corpus-level metrics."""

    all_tokens: list[str] = []
    word_count = 0
    char_count = 0
    fallback_count = 0

    for text in texts:
        tokens = tokenizer.tokenize(text)
        all_tokens.extend(tokens)
        word_count += sum(1 for part in WORD_RE.findall(text) if part.strip())
        char_count += len(text)
        fallback_count += sum(1 for token in tokens if token.startswith("<unk:"))

    token_count = len(all_tokens)
    tokens_per_word = token_count / word_count if word_count else 0.0
    compression_vs_chars = token_count / char_count if char_count else 0.0
    fallback_rate = fallback_count / token_count if token_count else 0.0

    return TokenizationReport(
        tokenizer=tokenizer.name,
        documents=len(texts),
        words=word_count,
        tokens=token_count,
        unique_tokens=len(set(all_tokens)),
        tokens_per_word=tokens_per_word,
        compression_vs_chars=compression_vs_chars,
        fallback_rate=fallback_rate,
    )


def compare_tokenizers(tokenizers: list[Tokenizer], texts: list[str]) -> list[TokenizationReport]:
    """Return reports sorted by tokens per word, then fallback rate."""

    reports = [evaluate_tokenizer(tokenizer, texts) for tokenizer in tokenizers]
    return sorted(reports, key=lambda item: (item.tokens_per_word, item.fallback_rate))


def average_tokens_per_word(tokenizer: Tokenizer, texts: list[str]) -> float:
    """Small helper for quick experiments."""

    ratios: list[float] = []
    for text in texts:
        words = WORD_RE.findall(text)
        if not words:
            continue
        ratios.append(len(tokenizer.tokenize(text)) / len(words))
    return mean(ratios) if ratios else 0.0
