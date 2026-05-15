"""Tokenizer vocabulary wrappers for fairer ablations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .simple_tokenizers import Tokenizer


@dataclass(frozen=True)
class VocabCappedTokenizer:
    base: Tokenizer
    vocab: frozenset[str]
    name: str
    unk_token: str = "<unk>"

    def tokenize(self, text: str) -> list[str]:
        return [
            token if token in self.vocab else self.unk_token
            for token in self.base.tokenize(text)
        ]


def cap_tokenizer_vocab(
    tokenizer: Tokenizer,
    train_texts: list[str],
    *,
    vocab_size: int,
    name: str | None = None,
    unk_token: str = "<unk>",
) -> VocabCappedTokenizer:
    """Wrap a tokenizer with a deterministic top-k training vocabulary."""

    if vocab_size < 1:
        raise ValueError("vocab_size must be at least 1")

    counts: Counter[str] = Counter()
    for text in train_texts:
        counts.update(tokenizer.tokenize(text))
    limit = max(0, vocab_size - 1)
    vocab = {
        token
        for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    }
    vocab.add(unk_token)
    return VocabCappedTokenizer(
        base=tokenizer,
        vocab=frozenset(vocab),
        name=name or f"capped_{tokenizer.name}",
        unk_token=unk_token,
    )
