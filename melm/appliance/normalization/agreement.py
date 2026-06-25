"""Deterministic subject-verb agreement / tense corrector (Tier 1.5a).

No ML, no LLM: a conservative, rule-based pass that repairs the highest-value,
unambiguous agreement slips after de-slanging (Tier 0) and de-typo (Tier 1) have
run. See docs/human-friendly-NLG-pipeline.md (§4 T1.5a, §4.4 finding 4, §16
trace B).

Core rule: a 1st/2nd-person or plural subject pronoun ({i, you, we, they}),
optionally followed by an adverb, that is immediately followed by a
3rd-person-singular verb form ("goes", "was", "has", "wants") -> the agreeing
base/plural form ("go", "were", "have", "want"). Examples:

    "i goes to school" -> "i go to school"
    "they was here"    -> "they were here"
    "we has it"        -> "we have it"

It is deliberately conservative:
  * only rewrites on a confident subject + verb adjacency match;
  * never touches NER/proper-noun/number protected tokens;
  * leaves 3rd-person-singular subjects (he/she/it/<name>) alone;
  * leaves already-correct input ("you go", "they were", "she goes") unchanged.

Rules are loaded from ``agreement_rules.v1.json`` at import time, falling back
to the module-level constants below if the contract is unavailable.
"""

from __future__ import annotations

import re

from melm.appliance.functional_grammar import _VERBS, _lemma
from melm.appliance.normalization.ner_mask import protected_indices

_PUNCT = ".,!?;:\"'()[]{}"

# Fallback constants — overridden by contract at module load below.
_NON_3SG_SUBJECTS: frozenset[str] = frozenset({"i", "you", "we", "they"})
_IRREGULAR_3SG: dict[str, str] = {
    "is": "are",
    "was": "were",
    "has": "have",
    "does": "do",
    "goes": "go",
}
_INTERVENING_ADVERBS: frozenset[str] = frozenset(
    {"really", "always", "never", "still", "just", "also", "often", "sometimes",
     "usually", "now", "then", "already"}
)


def _load_rules_from_contract() -> None:
    """Populate module globals from agreement_rules.v1.json if available."""
    global _NON_3SG_SUBJECTS, _IRREGULAR_3SG, _INTERVENING_ADVERBS
    try:
        from melm.contracts.validation import load_agreement_rules

        data = load_agreement_rules()
        _NON_3SG_SUBJECTS = frozenset(data["non_3sg_subjects"])
        _IRREGULAR_3SG = dict(data["irregular_3sg_map"])
        _INTERVENING_ADVERBS = frozenset(data["intervening_adverbs"])
    except Exception:
        pass  # contract unavailable — keep module-level defaults


_load_rules_from_contract()


def _split_affixes(tok: str) -> tuple[str, str, str]:
    """Return (leading_punct, core, trailing_punct) for *tok*."""
    prefix = tok[: len(tok) - len(tok.lstrip(_PUNCT))]
    suffix = tok[len(tok.rstrip(_PUNCT)):]
    core = tok[len(prefix): len(tok) - len(suffix)]
    return prefix, core, suffix


def _is_known_verb(stem: str) -> bool:
    """True when *stem* is a recognised verb lemma (read-only over _VERBS)."""
    return stem in _VERBS


def _deinflect_3sg(core_lower: str) -> str | None:
    """Return the agreeing (non-3sg) form of a 3rd-person-singular verb form.

    Handles the small irregular map first, then regular "-s" 3sg verbs whose
    stem is a known verb. Returns ``None`` when *core_lower* is not a 3sg verb
    form we are confident about (leave it untouched).
    """
    if core_lower in _IRREGULAR_3SG:
        return _IRREGULAR_3SG[core_lower]
    # Regular "-es" form: "wishes" -> "wish", "fixes" -> "fix".
    if core_lower.endswith("es") and len(core_lower) > 3:
        stem = core_lower[:-2]
        if _is_known_verb(stem):
            return stem
    # Regular "-s" form: "wants" -> "want", "runs" -> "run".
    if core_lower.endswith("s") and len(core_lower) > 2:
        stem = core_lower[:-1]
        # Confirm via _VERBS directly and via the shared lemmatizer, which only
        # strips trailing "s" when the stem is a known verb -- a second guard
        # against de-inflecting nouns ("its", "this", "bus").
        if _is_known_verb(stem) and _lemma(core_lower) == stem:
            return stem
    return None


def _match_surface_case(corrected: str, original_core: str) -> str:
    """Carry the surface casing of *original_core* onto *corrected*."""
    if original_core.isupper() and len(original_core) >= 2:
        return corrected.upper()
    if original_core[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def correct_agreement(text: str, *, known_names: frozenset[str] = frozenset()) -> str:
    """Apply the deterministic subject-verb agreement fix to *text*.

    Operates on whitespace tokens, case-insensitively for matching, preserving
    surface punctuation and casing where possible. Returns *text* unchanged when
    no confident subject+verb mismatch is found.
    """
    if not text:
        return text
    toks = text.split()
    if len(toks) < 2:
        return text

    protected = protected_indices(tuple(toks), known_names=known_names)
    cores = [_split_affixes(t)[1] for t in toks]
    lowers = [c.lower() for c in cores]

    changed = False
    i = 0
    while i < len(toks) - 1:
        # Subject must be a non-3sg pronoun and not a protected token.
        if i in protected or lowers[i] not in _NON_3SG_SUBJECTS:
            i += 1
            continue

        # Allow at most one optional intervening adverb between subject & verb.
        verb_pos = i + 1
        if (
            lowers[verb_pos] in _INTERVENING_ADVERBS
            and verb_pos + 1 < len(toks)
        ):
            verb_pos += 1

        if verb_pos >= len(toks) or verb_pos in protected:
            i += 1
            continue

        corrected_core = _deinflect_3sg(lowers[verb_pos])
        if corrected_core is None:
            i += 1
            continue

        prefix, core, suffix = _split_affixes(toks[verb_pos])
        surfaced = _match_surface_case(corrected_core, core)
        toks[verb_pos] = f"{prefix}{surfaced}{suffix}"
        cores[verb_pos] = surfaced
        lowers[verb_pos] = surfaced.lower()
        changed = True
        # Advance past the verb we just fixed.
        i = verb_pos + 1

    return " ".join(toks) if changed else text


# Backwards-friendly aliases (the layer wires to whichever name it prefers).
agreement_correct = correct_agreement
