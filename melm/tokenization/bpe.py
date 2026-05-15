"""Dependency-free byte-pair encoding baseline.

This is intentionally small and deterministic. It is not a replacement for
Hugging Face `tokenizers`, but it gives MELM a learned subword baseline before
external dependencies enter the project.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .simple_tokenizers import WORD_RE


END = "</w>"


@dataclass(frozen=True)
class SimpleBPETokenizer:
    merges: tuple[tuple[str, str], ...]
    name: str = "simple_bpe"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in WORD_RE.findall(text):
            lower = raw.lower()
            if not lower.isalpha():
                tokens.append(raw)
                continue
            pieces = tuple(lower) + (END,)
            for left, right in self.merges:
                pieces = _merge_pair(pieces, left, right)
            tokens.extend(_strip_end_marker(pieces))
        return tokens


def train_bpe(texts: list[str], *, vocab_size: int = 512, min_pair_frequency: int = 2) -> SimpleBPETokenizer:
    """Train a tiny BPE tokenizer from local texts."""

    word_counts: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        for raw in WORD_RE.findall(text):
            word = raw.lower()
            if word.isalpha():
                word_counts[tuple(word) + (END,)] += 1

    base_vocab = {piece for word in word_counts for piece in word}
    target_merges = max(0, vocab_size - len(base_vocab))
    merges: list[tuple[str, str]] = []

    for _ in range(target_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, count in word_counts.items():
            for left, right in zip(word, word[1:]):
                pair_counts[(left, right)] += count
        if not pair_counts:
            break
        best_pair, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if best_count < min_pair_frequency:
            break
        merges.append(best_pair)
        word_counts = Counter(
            {
                _merge_pair(word, best_pair[0], best_pair[1]): count
                for word, count in word_counts.items()
            }
        )

    return SimpleBPETokenizer(tuple(merges))


def _merge_pair(pieces: tuple[str, ...], left: str, right: str) -> tuple[str, ...]:
    merged: list[str] = []
    index = 0
    while index < len(pieces):
        if index < len(pieces) - 1 and pieces[index] == left and pieces[index + 1] == right:
            merged.append(left + right)
            index += 2
        else:
            merged.append(pieces[index])
            index += 1
    return tuple(merged)


def _strip_end_marker(pieces: tuple[str, ...]) -> list[str]:
    output: list[str] = []
    for piece in pieces:
        if piece == END:
            continue
        if piece.endswith(END):
            piece = piece[: -len(END)]
        if piece:
            output.append(piece)
    return output
