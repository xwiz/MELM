"""Pluggable language adapters for UOL-first multilingual routing.

Each adapter normalizes surface text into canonical lemmas and can emit a
lightweight rule-backed SyntaxGraph without introducing a heavy parser
dependency into the MVP path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from melm.contracts import load_function_words, load_predicate_inventory


@dataclass(frozen=True)
class DepEdge:
    head: int
    dependent: int
    relation: str
    confidence: float = 1.0


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    label: str
    text: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class SyntaxGraph:
    tokens: tuple[str, ...]
    lemmas: tuple[str, ...]
    pos_tags: tuple[str, ...]
    morph_features: tuple[dict[str, str], ...]
    dependencies: tuple[DepEdge, ...]
    entities: tuple[EntitySpan, ...] = ()
    language: str = "en"


@runtime_checkable
class LanguageAdapter(Protocol):
    """Normalize, lemmatize, and tag text for a specific language."""

    language_code: str

    def detect(self, text: str) -> float:
        ...

    def correct(self, text: str) -> str:
        """Surface-repair *text* (slang/abbreviation/typo expansion) before
        ``normalize()``/``tokenize()``. Implementations may return *text*
        unchanged (no-op) when no repair applies."""
        ...

    def normalize(self, text: str) -> str:
        ...

    def tokenize(self, text: str) -> tuple[str, ...]:
        ...

    def lemmatize(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        ...

    def tag(self, tokens: tuple[str, ...]) -> SyntaxGraph:
        ...


_ADAPTER_REGISTRY: dict[str, LanguageAdapter] = {}


def register_adapter(adapter: LanguageAdapter) -> None:
    _ADAPTER_REGISTRY[adapter.language_code] = adapter


def get_adapter(language_code: str) -> LanguageAdapter | None:
    return _ADAPTER_REGISTRY.get(language_code)


def detect_language(text: str) -> tuple[str, float]:
    """Return the best-matching (language_code, confidence) for *text*."""
    best_code = "en"
    best_conf = 0.0
    english_conf = 0.0
    for code, adapter in _ADAPTER_REGISTRY.items():
        conf = adapter.detect(text)
        if code == "en":
            english_conf = conf
        if conf > best_conf:
            best_conf = conf
            best_code = code
    if best_code != "en" and best_conf < 0.2 and english_conf:
        return "en", english_conf
    return best_code, best_conf


_FUNCTION_WORDS_CACHE: dict[str, set[str]] | None = None
_FUNCTION_WORD_ENTRIES: dict[tuple[str, str], dict[str, Any]] | None = None
_PREDICATE_ENTRIES: dict[tuple[str, str], dict[str, Any]] | None = None

_POS_BY_ROLE = {
    "greeting": "INTJ",
    "wh_word": "PRON",
    "modal": "AUX",
    "auxiliary": "AUX",
    "negation": "PART",
    "determiner": "DET",
    "preposition": "ADP",
    "conjunction": "CCONJ",
    "frequency": "ADV",
    "equivalence": "ADJ",
    "politeness": "INTJ",
    "discourse_particle": "PART",
    "pronoun": "PRON",
}


def _cached_function_word_lemmas() -> dict[str, set[str]]:
    global _FUNCTION_WORDS_CACHE
    if _FUNCTION_WORDS_CACHE is None:
        data = load_function_words()
        by_lang: dict[str, set[str]] = {}
        for entry in data.get("entries", []):
            lang = str(entry.get("language", "en")).strip().lower()
            lemma = str(entry.get("lemma", "")).strip().lower()
            if lang and lemma:
                by_lang.setdefault(lang, set()).add(lemma)
        _FUNCTION_WORDS_CACHE = by_lang
    return _FUNCTION_WORDS_CACHE


def _cached_function_word_entries() -> dict[tuple[str, str], dict[str, Any]]:
    global _FUNCTION_WORD_ENTRIES
    if _FUNCTION_WORD_ENTRIES is None:
        data = load_function_words()
        _FUNCTION_WORD_ENTRIES = {}
        for entry in data.get("entries", []):
            lang = str(entry.get("language", "en")).strip().lower()
            lemma = str(entry.get("lemma", "")).strip().lower()
            if lang and lemma:
                _FUNCTION_WORD_ENTRIES[(lang, lemma)] = dict(entry)
    return _FUNCTION_WORD_ENTRIES


def _cached_predicate_entries() -> dict[tuple[str, str], dict[str, Any]]:
    global _PREDICATE_ENTRIES
    if _PREDICATE_ENTRIES is None:
        data = load_predicate_inventory()
        _PREDICATE_ENTRIES = {}
        for entry in data.get("predicates", []):
            lang = str(entry.get("language", "en")).strip().lower()
            lemma = str(entry.get("lemma", "")).strip().lower()
            if lang and lemma:
                _PREDICATE_ENTRIES[(lang, lemma)] = dict(entry)
    return _PREDICATE_ENTRIES


def coverage_score(tokens: tuple[str, ...], language_code: str) -> float:
    fw = _cached_function_word_lemmas().get(language_code, set())
    if not fw or not tokens:
        return 0.0
    covered = sum(1 for t in tokens if t in fw)
    return covered / len(tokens)


def function_word_entry(language_code: str, lemma: str) -> dict[str, Any]:
    return _cached_function_word_entries().get((language_code, lemma), {})


def predicate_entry(language_code: str, lemma: str) -> dict[str, Any]:
    return _cached_predicate_entries().get((language_code, lemma), {})


def pos_tag_for(language_code: str, lemma: str) -> str:
    role = str(function_word_entry(language_code, lemma).get("role", "")).strip().lower()
    if role:
        return _POS_BY_ROLE.get(role, "X")
    if predicate_entry(language_code, lemma):
        return "VERB"
    if lemma.isdigit():
        return "NUM"
    return "NOUN"


def morph_features_for(language_code: str, lemma: str, pos_tag: str) -> dict[str, str]:
    entry = function_word_entry(language_code, lemma)
    role = str(entry.get("role", "")).strip().lower()
    subrole = str(entry.get("subrole", "")).strip().lower()
    features: dict[str, str] = {}
    if pos_tag == "AUX" and subrole == "future":
        features["Tense"] = "Fut"
    if role == "pronoun":
        if subrole in {"agent", "agent_or_patient"}:
            features["Case"] = "Nom"
        elif subrole in {"patient", "human_collective", "human_indefinite"}:
            features["Case"] = "Acc"
    return features


def simple_dependencies(
    language_code: str,
    lemmas: tuple[str, ...],
    pos_tags: tuple[str, ...],
) -> tuple[DepEdge, ...]:
    if not lemmas:
        return ()
    root_index = 0
    for index, lemma in enumerate(lemmas):
        role = str(function_word_entry(language_code, lemma).get("role", "")).strip().lower()
        if predicate_entry(language_code, lemma) and role not in {"modal", "auxiliary"}:
            root_index = index
            break
    edges: list[DepEdge] = [DepEdge(head=root_index, dependent=root_index, relation="root")]
    for index, lemma in enumerate(lemmas):
        if index == root_index:
            continue
        role = str(function_word_entry(language_code, lemma).get("role", "")).strip().lower()
        subrole = str(function_word_entry(language_code, lemma).get("subrole", "")).strip().lower()
        relation = "dep"
        if role == "pronoun" and subrole in {"agent", "agent_or_patient"} and index < root_index:
            relation = "nsubj"
        elif role == "wh_word":
            relation = "obj" if index < root_index else "obl"
        elif role == "pronoun" and subrole in {"patient", "human_collective", "human_indefinite"}:
            relation = "obj"
        elif pos_tags[index] == "NOUN":
            relation = "obj" if index > root_index else "nsubj"
        elif role == "preposition":
            relation = "case"
        elif role == "modal":
            relation = "aux"
        elif role == "auxiliary":
            relation = "cop" if lemma == "be" else "aux"
        elif role == "negation":
            relation = "advmod"
        elif role == "determiner":
            relation = "det"
        elif role == "conjunction":
            relation = "cc"
        edges.append(DepEdge(head=root_index, dependent=index, relation=relation))
    return tuple(edges)


def build_syntax_graph(
    language_code: str,
    tokens: tuple[str, ...],
    lemmas: tuple[str, ...],
) -> SyntaxGraph:
    pos_tags = tuple(pos_tag_for(language_code, lemma) for lemma in lemmas)
    morph_features = tuple(
        morph_features_for(language_code, lemma, pos_tag)
        for lemma, pos_tag in zip(lemmas, pos_tags)
    )
    dependencies = simple_dependencies(language_code, lemmas, pos_tags)
    return SyntaxGraph(
        tokens=tokens,
        lemmas=lemmas,
        pos_tags=pos_tags,
        morph_features=morph_features,
        dependencies=dependencies,
        entities=(),
        language=language_code,
    )


from . import english  # noqa: E402
from . import igbo  # noqa: E402
