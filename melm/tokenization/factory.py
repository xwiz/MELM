"""Tokenizer factory used by CLI scripts and reports."""

from __future__ import annotations

from .bpe import train_bpe
from .simple_tokenizers import (
    BytePatchTokenizer,
    HeuristicMorphemeTokenizer,
    Tokenizer,
    UnigramLikeTokenizer,
    WhitespaceTokenizer,
)


def build_default_tokenizers(
    texts: list[str],
    *,
    train_texts: list[str] | None = None,
    unigram_vocab_size: int = 512,
    bpe_vocab_size: int = 512,
    byte_patch_size: int = 4,
) -> list[Tokenizer]:
    """Build the current default tokenizer comparison set."""

    training_source = train_texts or texts
    return [
        WhitespaceTokenizer(),
        train_bpe(training_source, vocab_size=bpe_vocab_size),
        HeuristicMorphemeTokenizer(),
        BytePatchTokenizer(patch_size=byte_patch_size),
        UnigramLikeTokenizer(frozenset(simple_vocab(training_source, limit=unigram_vocab_size))),
    ]


def simple_vocab(texts: list[str], limit: int) -> set[str]:
    counts: dict[str, int] = {}
    for text in texts:
        for raw in text.lower().replace("-", " ").split():
            token = "".join(ch for ch in raw if ch.isalpha())
            if token:
                counts[token] = counts.get(token, 0) + 1
    return {
        token
        for token, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    }
