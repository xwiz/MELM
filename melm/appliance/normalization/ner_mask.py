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
_POSSESSIVE_MARKERS = frozenset({"my", "our", "your", "his", "her", "their", "its"})


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


def syntactic_entity_indices(
    tokens: tuple[str, ...],
    *,
    language_code: str = "en",
) -> set[int]:
    """Return likely entity argument indices from the lightweight syntax graph.

    This protects tokens that are syntactic subjects/objects and have entity
    evidence the spellchecker cannot infer: capitalization or possessive
    attachment ("my zorbulator"). It intentionally does not protect every noun
    object, so ordinary object-position typos can still be corrected.
    """
    if not tokens:
        return set()
    try:
        from melm.appliance.functional_grammar import _lemma
        from melm.appliance.language_adapters import build_syntax_graph

        cores = tuple(tok.strip(_PUNCT) for tok in tokens)
        lowered = tuple(core.lower() for core in cores)
        lemmas = tuple(_lemma(token, language=language_code) for token in lowered)
        graph = build_syntax_graph(language_code, lowered, lemmas)
    except Exception:
        return set()

    argument_indices = {
        edge.dependent
        for edge in graph.dependencies
        if edge.relation in {"nsubj", "obj", "obl"}
    }
    protected: set[int] = set()
    for index in argument_indices:
        if index < 0 or index >= len(tokens):
            continue
        core = cores[index]
        if not core:
            continue
        previous = lowered[index - 1] if index > 0 else ""
        if core[:1].isupper() or previous in _POSSESSIVE_MARKERS:
            protected.add(index)
    for index, core in enumerate(cores[:-1]):
        if core[:1].isupper() and lowered[index + 1] in {"am", "is", "are", "was", "were", "be"}:
            protected.add(index)
    for entity in graph.entities:
        protected.update(range(entity.start, entity.end))
    return protected
