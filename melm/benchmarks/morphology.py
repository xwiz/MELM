"""Small gold morphology fixture for tokenizer boundary checks."""

from __future__ import annotations

from melm.tokenization.boundary import MorphemeExample


def morphology_fixture() -> list[MorphemeExample]:
    return [
        MorphemeExample("unbreakable", ("un", "break", "able")),
        MorphemeExample("replaying", ("re", "play", "ing")),
        MorphemeExample("kindness", ("kind", "ness")),
        MorphemeExample("careless", ("care", "less")),
        MorphemeExample("misread", ("mis", "read")),
        MorphemeExample("teacher", ("teach", "er")),
        MorphemeExample("happily", ("happi", "ly")),
        MorphemeExample("preheat", ("pre", "heat")),
    ]
