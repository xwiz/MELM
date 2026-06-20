"""Deterministic proper-noun / number protection mask (Tier 1.5b).

No ML: capitalization + digit heuristics, plus an optional caller-supplied
known-name set (e.g. profile contacts/places). Protected tokens must never be
altered by any downstream corrector. This is the guard that stops SymSpell/
Harper from mangling names ("nneka"->"sneak", "france"->"franc").
"""

from __future__ import annotations

import re

_HAS_DIGIT = re.compile(r"\d")
_PUNCT = ".,!?;:\"'()[]{}"


def protected_indices(
    tokens: tuple[str, ...],
    *,
    known_names: frozenset[str] = frozenset(),
) -> set[int]:
    """Return indices of whitespace-split *tokens* that must not be corrected.

    Protected when the token is:
      * a known name/place (case-insensitive match against *known_names*),
      * capitalized and not sentence-initial (likely a proper noun),
      * an all-caps token of length >= 2 (acronym),
      * containing a digit (times/codes/quantities like "8", "7am", "v2").
    """
    protected: set[int] = set()
    for i, tok in enumerate(tokens):
        core = tok.strip(_PUNCT)
        if not core:
            continue
        if core.lower() in known_names:
            protected.add(i)
        elif _HAS_DIGIT.search(core):
            protected.add(i)
        elif core.isupper() and len(core) >= 2:
            protected.add(i)
        elif i > 0 and core[0].isupper():
            protected.add(i)
    return protected
