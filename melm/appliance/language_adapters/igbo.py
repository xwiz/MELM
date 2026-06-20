"""Igbo language adapter."""

from __future__ import annotations

import re
from typing import Any

from . import build_syntax_graph, coverage_score, register_adapter

_IGBO_DIACRITIC_MAP: dict[str, str] = {
    "\u1eb9": "e",
    "\u1ecb": "i",
    "\u1ecd": "o",
    "\u1ee5": "u",
    "\u1e3f": "m",
    "\u0144": "n",
    "\u1e45": "n",
    "\u00c9": "E",
    "\u00c8": "E",
    "\u00e9": "e",
    "\u00e8": "e",
    "\u00ed": "i",
    "\u00ec": "i",
    "\u00f3": "o",
    "\u00f2": "o",
    "\u00fa": "u",
    "\u00f9": "u",
    "\u00e1": "a",
    "\u00e0": "a",
}

_IGBO_DIACRITIC_RE = re.compile("|".join(re.escape(k) for k in _IGBO_DIACRITIC_MAP))
_IGBO_VERB_PREFIXES: tuple[str, ...] = ("ga",)


class IgboAdapter:
    language_code = "ig"

    _IGBO_INDICATORS: frozenset[str] = frozenset(
        {
            "gini",
            "g\u1ecbn\u1ecb",
            "onye",
            "ebee",
            "mgbe",
            "kedu",
            "ked\u1ee5",
            "ndeewo",
            "nnoo",
            "nn\u1ecd\u1ecd",
            "\u1ee5t\u1ee5t\u1ee5",
            "ututu",
            "any\u1ecb",
            "anyi",
            "unu",
            "nke",
            "ah\u1ee5",
            "ahu",
            "eri",
            "\u1e45\u1ee5",
            "nu",
            "g\u1ee5",
            "gu",
            "mara",
            "ch\u1ecdo",
            "choo",
            "gwa",
            "b\u1ecba",
            "bia",
            "aga",
            "k\u1ee5",
            "ku",
            "kwa",
            "h\u1ee5",
            "hu",
            "nwe",
            "b\u1ee5",
            "bu",
        }
    )

    def correct(self, text: str) -> str:
        """No-op surface repair for Igbo (Layer 0 expansion is English-only)."""
        return text

    def detect(self, text: str) -> float:
        raw = text.lower()
        tokens = self.tokenize(raw)
        if not tokens:
            return 0.0
        cov = coverage_score(tokens, "ig")
        indicator_hits = sum(1 for t in tokens if t in self._IGBO_INDICATORS)
        indicator_ratio = indicator_hits / len(tokens)
        has_igbo_chars = any(ch in raw for ch in ("\u1ecb", "\u1ecd", "\u1ee5", "\u1eb9", "\u1e45", "\u1e3f", "\u0144"))
        char_boost = 0.15 if has_igbo_chars else 0.0
        return min(1.0, cov * 0.4 + indicator_ratio * 0.5 + char_boost)

    def normalize(self, text: str) -> str:
        raw = text.lower().strip()
        if not raw:
            return ""

        def _replace(match: re.Match[str]) -> str:
            return _IGBO_DIACRITIC_MAP[match.group(0)]

        return _IGBO_DIACRITIC_RE.sub(_replace, raw)

    def tokenize(self, text: str) -> tuple[str, ...]:
        raw = self.normalize(text)
        if not raw:
            return ()
        tokens: list[str] = []
        for word in raw.split():
            while word and word[-1] in ".,;:!?":
                word = word[:-1]
            if not word:
                continue
            if "-" in word:
                parts = word.split("-", 1)
                if parts[0] in _IGBO_VERB_PREFIXES:
                    tokens.extend(parts)
                    continue
            tokens.append(word)
        return tuple(tokens)

    def lemmatize(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(tokens)

    def tag(self, tokens: tuple[str, ...]):
        lemmas = self.lemmatize(tokens)
        return build_syntax_graph("ig", tokens, lemmas)


def seed_igbo_lexicon(
    lexicon: dict[str, frozenset[str]],
    entries: list[dict[str, Any]],
) -> None:
    """Inject Igbo contract entries into the runtime lexicon."""
    for entry in entries:
        lemma = str(entry.get("lemma", "")).strip().lower()
        class_id = str(entry.get("semantic_class", "")).strip()
        if not lemma or not class_id:
            continue
        existing = lexicon.get(lemma, frozenset())
        lexicon[lemma] = frozenset(existing | {class_id})


def strip_igbo_diacritics(text: str) -> str:
    """Return text with Igbo diacritics stripped."""

    def _replace(match: re.Match[str]) -> str:
        return _IGBO_DIACRITIC_MAP[match.group(0)]

    return _IGBO_DIACRITIC_RE.sub(_replace, text)


def tokenize_igbo(text: str) -> tuple[str, ...]:
    """Whitespace-tokenize Igbo text after normalization."""
    return IgboAdapter().tokenize(text)


def translate_igbo_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return Igbo tokens unchanged (no-op shim)."""
    return tuple(tokens)


def normalise_igbo_for_uol(text: str) -> tuple[str, ...]:
    """Normalize Igbo text into Igbo lemma tokens for UOL parsing."""
    return IgboAdapter().tokenize(text)


register_adapter(IgboAdapter())
