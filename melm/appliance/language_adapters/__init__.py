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

_MODIFIER_ATOM_LEMMAS: set[str] | None = None
_NOUN_ATOM_LEMMAS: set[str] | None = None

_ADJECTIVE_SUFFIXES: dict[str, frozenset[str]] = {
    "en": frozenset({"able", "ible", "al", "ful", "ic", "ive", "less",
                     "ous", "ish", "like", "proof", "ward"}),
}


def _load_modifier_atom_lemmas() -> set[str]:
    global _MODIFIER_ATOM_LEMMAS
    if _MODIFIER_ATOM_LEMMAS is not None:
        return _MODIFIER_ATOM_LEMMAS
    try:
        from melm.contracts import load_modifier_atoms
        payload = load_modifier_atoms()
        _MODIFIER_ATOM_LEMMAS = {
            str(e["canonical_lemma"]).strip().lower()
            for e in payload.get("entries", [])
            if e.get("canonical_lemma")
        }
    except Exception:
        _MODIFIER_ATOM_LEMMAS = set()
    return _MODIFIER_ATOM_LEMMAS


def _load_noun_atom_lemmas() -> set[str]:
    global _NOUN_ATOM_LEMMAS
    if _NOUN_ATOM_LEMMAS is not None:
        return _NOUN_ATOM_LEMMAS
    try:
        from melm.contracts import load_noun_atoms as _load_na
        payload = _load_na()
        _NOUN_ATOM_LEMMAS = {
            str(e["canonical_lemma"]).strip().lower()
            for e in payload.get("entities", [])
            if e.get("canonical_lemma")
        }
    except Exception:
        _NOUN_ATOM_LEMMAS = set()
    return _NOUN_ATOM_LEMMAS


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


def _lemma_in_modifier_atoms(lemma: str) -> bool:
    return lemma.strip().lower() in _load_modifier_atom_lemmas()


def _lemma_in_noun_atoms(lemma: str) -> bool:
    return lemma.strip().lower() in _load_noun_atom_lemmas()


def _guess_is_adjective(language_code: str, lemma: str) -> bool:
    """Heuristic: check modifier_atoms contract first, then suffix patterns."""
    if _lemma_in_modifier_atoms(lemma):
        return True
    if _lemma_in_noun_atoms(lemma):
        return False
    suffixes = _ADJECTIVE_SUFFIXES.get(language_code, frozenset())
    if not suffixes:
        return False
    stem = lemma.strip().lower()
    return any(stem.endswith(suf) for suf in suffixes)


def pos_tag_for(language_code: str, lemma: str) -> str:
    role = str(function_word_entry(language_code, lemma).get("role", "")).strip().lower()
    if role:
        return _POS_BY_ROLE.get(role, "X")
    if predicate_entry(language_code, lemma):
        return "VERB"
    if lemma.isdigit():
        return "NUM"
    if _guess_is_adjective(language_code, lemma):
        return "ADJ"
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


def _load_causal_link_markers() -> dict[str, dict[str, Any]]:
    try:
        from melm.contracts import load_causal_link_markers
        return load_causal_link_markers()
    except Exception:
        return {}


def _is_predicate_token(language_code: str, lemma: str, allow_copula: bool = False) -> bool:
    role = str(function_word_entry(language_code, lemma).get("role", "")).strip().lower()
    subrole = str(function_word_entry(language_code, lemma).get("subrole", "")).strip().lower()
    if not predicate_entry(language_code, lemma):
        return False
    if role in {"modal", "auxiliary"}:
        return allow_copula and subrole == "copula"
    return True


def _default_root_index(language_code: str, lemmas: tuple[str, ...]) -> int:
    for index, lemma in enumerate(lemmas):
        if _is_predicate_token(language_code, lemma, allow_copula=False):
            return index
    return 0


def _find_causal_clause_roots(
    language_code: str,
    lemmas: tuple[str, ...],
) -> tuple[int, int, str] | None:
    """Return (main_root_index, subordinate_root_index, marker_lemma) or None."""
    markers = _load_causal_link_markers()
    marker_indices = [(i, lemmas[i]) for i in range(len(lemmas)) if lemmas[i] in markers]
    if not marker_indices:
        return None

    predicate_indices = [
        i for i, lemma in enumerate(lemmas)
        if _is_predicate_token(language_code, lemma, allow_copula=True)
    ]
    if len(predicate_indices) < 2:
        return None

    marker_idx, marker_lemma = marker_indices[0]
    direction = markers[marker_lemma].get("direction", "")

    # Subordinate predicate is the predicate in the clause introduced by the marker.
    subordinate_index = None
    for pred_idx in predicate_indices:
        if pred_idx > marker_idx:
            subordinate_index = pred_idx
            break
    if subordinate_index is None:
        return None

    main_index = None
    if direction == "effect_to_cause" or marker_lemma in {"so", "therefore"}:
        # Main clause is the predicate before the marker.
        for pred_idx in reversed(predicate_indices):
            if pred_idx < marker_idx:
                main_index = pred_idx
                break
    else:
        # cause_to_effect: main clause follows the subordinate clause.
        found_subordinate = False
        for pred_idx in predicate_indices:
            if found_subordinate:
                main_index = pred_idx
                break
            if pred_idx == subordinate_index:
                found_subordinate = True

    if main_index is None or main_index == subordinate_index:
        return None
    return (main_index, subordinate_index, marker_lemma)


def _find_marker_index(
    lemmas: tuple[str, ...],
    marker_lemma: str,
    subordinate_index: int,
) -> int | None:
    best: int | None = None
    for i, lemma in enumerate(lemmas):
        if lemma == marker_lemma:
            if best is None or abs(i - subordinate_index) < abs(best - subordinate_index):
                best = i
    return best


def simple_dependencies(
    language_code: str,
    lemmas: tuple[str, ...],
    pos_tags: tuple[str, ...],
) -> tuple[DepEdge, ...]:
    if not lemmas:
        return ()

    causal_roots = _find_causal_clause_roots(language_code, lemmas)
    subordinate_index: int | None = None
    marker_index: int | None = None
    marker_lemma: str = ""
    if causal_roots is not None:
        root_index, subordinate_index, marker_lemma = causal_roots
    else:
        root_index = _default_root_index(language_code, lemmas)

    edges: list[DepEdge] = [DepEdge(head=root_index, dependent=root_index, relation="root")]

    subordinate_clause_indices: set[int] = set()
    if subordinate_index is not None and subordinate_index != root_index:
        marker_index = _find_marker_index(lemmas, marker_lemma, subordinate_index)
        if marker_index is not None:
            edges.append(DepEdge(head=subordinate_index, dependent=marker_index, relation="mark"))
            edges.append(DepEdge(head=root_index, dependent=subordinate_index, relation="advcl"))
            start, end = sorted([marker_index, subordinate_index])
            subordinate_clause_indices = set(range(start, end + 1))
        else:
            subordinate_clause_indices = {subordinate_index}

    for index, lemma in enumerate(lemmas):
        if index == root_index or index == subordinate_index:
            continue
        if index == marker_index:
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
        elif pos_tags[index] == "ADJ":
            relation = "amod"
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
        # Attach subordinate clause tokens to the subordinate predicate so that
        # each clause gets its own local roles in the atomizer.
        head = subordinate_index if subordinate_index is not None and index in subordinate_clause_indices else root_index
        edges.append(DepEdge(head=head, dependent=index, relation=relation))
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
