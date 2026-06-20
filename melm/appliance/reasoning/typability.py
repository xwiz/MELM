"""Token typability classifier — gibberish vs abbreviation vs word.

Deterministic heuristic using vowel/consonant ratios, length, and an
abbreviation lookup.  No ML, no spell-check, stdlib only.
"""

import re
from typing import Any

_CACHE: dict[str, Any] | None = None


def _ensure_cache() -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    from melm.contracts.validation import load_token_typability
    _CACHE = load_token_typability()
    return _CACHE


def classify_token_string(text: str) -> str:
    """Classify a single token as ``"word"``, ``"abbreviation"``, or ``"gibberish"``.

    Classification rules (all thresholds from ``token_typability.v1.json``):

    1. If the token matches a known abbreviation → ``"abbreviation"``.
    2. If the token is short (≤ max_abbrev_len) and consonant-heavy → ``"abbreviation"``.
    3. If the token's vowel ratio < min_vowel_ratio **and** its longest
       consecutive consonant run ≥ max_consonant_run **and** its length ≥
       min_gibberish_len → ``"gibberish"``.
    4. Otherwise → ``"word"``.
    """
    config = _ensure_cache()
    known_abbrevs: set[str] = set(config.get("known_abbreviations", []))
    clean = text.strip().lower()
    if not clean:
        return "word"

    if clean in known_abbrevs:
        return "abbreviation"

    max_abbrev = config.get("max_abbreviation_len", 5)
    if len(clean) <= max_abbrev and _is_consonant_heavy(clean):
        return "abbreviation"

    min_gibber = config.get("min_gibberish_len", 6)
    min_vowel = config.get("min_vowel_ratio", 0.15)
    max_cons = config.get("max_consonant_run", 5)

    if len(clean) >= min_gibber:
        vowel_ratio = _vowel_ratio(clean)
        cons_run = _max_consonant_run(clean)
        if vowel_ratio < min_vowel and cons_run >= max_cons:
            return "gibberish"

    return "word"


def classify_utterance_tokens(tokens: list[str]) -> str:
    """Classify a full utterance by its majority token type.

    Returns the most common classification across tokens, or ``"word"`` if
    all tokens are words.
    """
    counts: dict[str, int] = {}
    for token in tokens:
        cls = classify_token_string(token)
        counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return "word"
    return max(counts, key=counts.get)


def _vowel_ratio(text: str) -> float:
    vowels = sum(1 for ch in text if ch in "aeiou")
    return vowels / len(text) if text else 0.0


def _max_consonant_run(text: str) -> int:
    runs = re.findall(r"[^aeiou\W_]+", text)
    return max((len(r) for r in runs), default=0)


def _is_consonant_heavy(text: str) -> bool:
    if len(text) < 2:
        return False
    consonants = sum(1 for ch in text if ch.isalpha() and ch not in "aeiou")
    return consonants / len(text) > 0.7
