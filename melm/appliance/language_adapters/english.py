"""English language adapter - default fallback."""

from __future__ import annotations

import re

from melm.appliance.functional_grammar import _lemma

from . import build_syntax_graph, register_adapter

_EXPANSION_CACHE: dict[str, str] | None = None
_NOISE_TOKEN_CACHE: set[str] | None = None
_PUNCT = ".,!?;:\"'()[]{}"


def _expansion_map() -> dict[str, str]:
    """Lazily load the raw->standard surface-expansion map (Layer 0).

    Degrades to an empty map (no-op correction) if the contract is missing or
    invalid, matching the contract's declared failure_behavior.
    """
    global _EXPANSION_CACHE
    if _EXPANSION_CACHE is None:
        try:
            from melm.contracts import load_normalization_expansions

            data = load_normalization_expansions()
            _EXPANSION_CACHE = {
                str(e["raw"]).strip().lower(): str(e["standard"])
                for e in data.get("entries", [])
                if e.get("raw") and e.get("standard")
            }
        except Exception:
            _EXPANSION_CACHE = {}
    return _EXPANSION_CACHE


def _noise_tokens() -> set[str]:
    global _NOISE_TOKEN_CACHE
    if _NOISE_TOKEN_CACHE is None:
        try:
            from melm.contracts import load_noise_tokens

            payload = load_noise_tokens()
            entries = payload.get("entries", payload) if isinstance(payload, dict) else {}
            _NOISE_TOKEN_CACHE = {
                str(token).lower()
                for token, entry in entries.items()
                if isinstance(entry, dict) and entry.get("strip_from_parse")
            }
        except Exception:
            _NOISE_TOKEN_CACHE = set()
    return _NOISE_TOKEN_CACHE


def _known_runtime_lexeme(token: str) -> bool:
    try:
        from melm.appliance.local_assistant_router import _IN_MEMORY_LEXICON

        return token in _IN_MEMORY_LEXICON
    except Exception:
        return False


class EnglishAdapter:
    language_code = "en"

    def detect(self, text: str) -> float:
        return 0.1

    def correct(self, text: str, *, known_names: frozenset[str] = frozenset()) -> str:
        """Pre-UOL surface-repair cascade, run before normalize()/tokenize().

        Tier 0: contract-driven expansion of contractions/abbreviations/slang
        and a few unambiguous typos (``normalization_expansions.v1``).
        Tier 1: lexicon-backed SymSpell typo correction (optional dependency),
        with proper nouns and numbers protected by the NER mask.
        Tier 1.5a: deterministic subject-verb agreement / tense fix, run after
        the subjects/verbs have been de-slanged and de-typo'd ("i goes"->"i go").

        ``known_names`` feeds the NER mask so profile contacts are never mangled
        by SymSpell ("leo"→"let") or agreement rules.

        Unknown tokens, proper nouns, and numbers pass through untouched.
        Surrounding punctuation is preserved. Degrades to a no-op when a
        resource is unavailable.
        """
        if not text:
            return text
        text = self._layer0_expand(text)
        text = self._layer1_symspell(text, known_names=known_names)
        text = self._layer15_agreement(text, known_names=known_names)
        return text

    def _layer15_agreement(self, text: str, *, known_names: frozenset[str] = frozenset()) -> str:
        """Tier 1.5a: deterministic subject-verb agreement / tense correction.

        Conservative, rule-based, no ML. Degrades to a no-op on any failure.
        """
        try:
            from melm.appliance.normalization.agreement import correct_agreement

            return correct_agreement(text, known_names=known_names)
        except Exception:
            return text

    def _layer0_expand(self, text: str) -> str:
        mapping = _expansion_map()
        if not mapping:
            return text
        out: list[str] = []
        for tok in text.split():
            core = tok.strip(_PUNCT)
            if not core:
                out.append(tok)
                continue
            std = mapping.get(core.lower())
            if std is None:
                out.append(tok)
                continue
            prefix = tok[: len(tok) - len(tok.lstrip(_PUNCT))]
            suffix = tok[len(tok.rstrip(_PUNCT)):]
            out.append(f"{prefix}{std}{suffix}")
        return " ".join(out)

    def _layer1_symspell(self, text: str, *, known_names: frozenset[str] = frozenset()) -> str:
        from melm.appliance.normalization.symspell import get_corrector
        from melm.appliance.normalization.ner_mask import (
            protected_indices,
            syntactic_entity_indices,
        )

        corrector = get_corrector()
        if corrector is None:
            return text
        toks = text.split()
        if not toks:
            return text
        protected = (
            protected_indices(tuple(toks), known_names=known_names)
            | syntactic_entity_indices(tuple(toks))
        )
        out: list[str] = []
        noise = _noise_tokens()
        for i, tok in enumerate(toks):
            if i in protected:
                out.append(tok)
                continue
            core = tok.strip(_PUNCT)
            if core.lower() in noise:
                out.append(tok)
                continue
            if _known_runtime_lexeme(core.lower()):
                out.append(tok)
                continue
            corrected = corrector.correct(core.lower())
            if not corrected or corrected == core.lower():
                out.append(tok)
                continue
            prefix = tok[: len(tok) - len(tok.lstrip(_PUNCT))]
            suffix = tok[len(tok.rstrip(_PUNCT)):]
            out.append(f"{prefix}{corrected}{suffix}")
        return " ".join(out)

    def normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9']+", text.lower()))

    def lemmatize(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_lemma(token, language="en") for token in tokens)

    def tag(self, tokens: tuple[str, ...]):
        lemmas = self.lemmatize(tokens)
        return build_syntax_graph("en", tokens, lemmas)


register_adapter(EnglishAdapter())
