"""Lexicon-backed SymSpell typo correction (Tier 1).

Uses symspellpy's bundled English frequency dictionary (a real ~82k-word base,
NOT a hand-coded list) plus the MELM runtime lexicon for domain/learned
vocabulary, so neither valid common words nor known domain terms get
"corrected". Import-guarded: degrades to a no-op when symspellpy is absent
(the device provisioning script installs it).

Safety levers (see docs/human-friendly-NLG-pipeline.md §14):
  * never corrects a token already in the dictionary (distance 0),
  * callers exclude proper nouns/numbers via the NER mask (ner_mask.py),
  * only alphabetic tokens of length >= 4 are eligible (short tokens are
    handled by Tier 0 expansion and are too ambiguous to edit safely).
"""

from __future__ import annotations

from typing import Optional

_MIN_LEN = 4
_corrector: "Optional[_SymSpellCorrector]" = None
_init_done = False


class _SymSpellCorrector:
    def __init__(self, sym, max_edit: int = 2) -> None:
        self._sym = sym
        self._max_edit = max_edit

    def correct(self, token: str) -> Optional[str]:
        """Return a confident correction for *token* (lowercased), or None."""
        if len(token) < _MIN_LEN or not token.isalpha():
            return None
        from symspellpy import Verbosity

        sugg = self._sym.lookup(
            token, Verbosity.TOP, max_edit_distance=self._max_edit, include_unknown=False
        )
        if not sugg:
            return None
        best = sugg[0]
        if best.distance == 0 or best.term == token:
            return None
        return str(best.term)


def _build() -> "Optional[_SymSpellCorrector]":
    try:
        from importlib.resources import files
        from symspellpy import SymSpell
    except Exception:
        return None
    try:
        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        sym.load_dictionary(str(dict_path), 0, 1)
        # Inject MELM domain/learned vocabulary so it is never mis-corrected.
        # Deferred import (runtime) avoids a module-load circular import.
        try:
            from melm.appliance.local_assistant_router import _IN_MEMORY_LEXICON

            for lemma in _IN_MEMORY_LEXICON:
                if lemma and lemma.isalpha() and len(lemma) >= _MIN_LEN:
                    sym.create_dictionary_entry(lemma, 1)
        except Exception:
            pass
        return _SymSpellCorrector(sym)
    except Exception:
        return None


def get_corrector() -> "Optional[_SymSpellCorrector]":
    global _corrector, _init_done
    if not _init_done:
        _corrector = _build()
        _init_done = True
    return _corrector


def available() -> bool:
    return get_corrector() is not None
