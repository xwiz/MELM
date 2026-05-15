"""Morphological boundary alignment metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .simple_tokenizers import Tokenizer


@dataclass(frozen=True)
class MorphemeExample:
    word: str
    morphemes: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryReport:
    tokenizer: str
    examples: int
    precision: float
    recall: float
    f1: float
    exact_match: float


def evaluate_boundary_f1(tokenizer: Tokenizer, examples: list[MorphemeExample]) -> BoundaryReport:
    """Compare tokenizer-internal word boundaries with gold morpheme boundaries."""

    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_matches = 0

    for example in examples:
        gold = boundary_offsets(example.morphemes)
        predicted = predicted_offsets(tokenizer, example.word)

        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
        exact_matches += int(gold == predicted)

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    exact_match = _safe_div(exact_matches, len(examples))

    return BoundaryReport(
        tokenizer=tokenizer.name,
        examples=len(examples),
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=exact_match,
    )


def boundary_offsets(parts: tuple[str, ...]) -> set[int]:
    offsets: set[int] = set()
    cursor = 0
    for part in parts[:-1]:
        cursor += len(part)
        offsets.add(cursor)
    return offsets


def predicted_offsets(tokenizer: Tokenizer, word: str) -> set[int]:
    pieces = tokenizer.tokenize(word)
    offsets: set[int] = set()
    cursor = 0
    normalized_word = word.lower()

    for piece in pieces[:-1]:
        clean = _clean_piece(piece)
        if not clean:
            continue
        next_cursor = cursor + len(clean)
        if normalized_word[cursor:next_cursor] == clean:
            cursor = next_cursor
            offsets.add(cursor)
        else:
            return set()
    return offsets


def _clean_piece(piece: str) -> str:
    if piece.startswith("<unk:"):
        return piece.removeprefix("<unk:").removesuffix(">")
    return piece.replace("+", "").lower()


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
