"""Small minimal-pair fixtures for checkpoint sanity checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimalPairCase:
    case_id: str
    category: str
    good: str
    bad: str


def child_language_minimal_pairs_fixture() -> list[MinimalPairCase]:
    """Return a tiny child-level sentence-ranking fixture."""

    return [
        MinimalPairCase(
            "agreement_singular_1",
            "agreement",
            "The dog is running.",
            "The dog are running.",
        ),
        MinimalPairCase(
            "agreement_plural_1",
            "agreement",
            "Two cats are sleeping.",
            "Two cats is sleeping.",
        ),
        MinimalPairCase(
            "determiner_1",
            "determiner",
            "Maya found a red ball.",
            "Maya found red ball.",
        ),
        MinimalPairCase(
            "past_tense_1",
            "tense",
            "Yesterday Leo opened the box.",
            "Yesterday Leo open the box.",
        ),
        MinimalPairCase(
            "auxiliary_1",
            "auxiliary",
            "The baby can see the moon.",
            "The baby can sees the moon.",
        ),
        MinimalPairCase(
            "preposition_1",
            "preposition",
            "Maya put the cup on the table.",
            "Maya put the cup the table.",
        ),
        MinimalPairCase(
            "pronoun_1",
            "pronoun",
            "She gave the toy to him.",
            "She gave the toy to he.",
        ),
        MinimalPairCase(
            "word_order_1",
            "word_order",
            "The little bird ate the seed.",
            "The little bird the seed ate.",
        ),
    ]
