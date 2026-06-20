"""Weighted functional grammar for foundational UOL relation parsing.

This layer does not decide whether the device can answer. It assigns functional
roles, proposes predicate structures, and ranks them from token relations.
Capability routing happens after this parse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Mutable reference to the runtime lexicon (set by the router at init time).
# Acquired verbs that are not in _VERBS can still lemmatize and get a semantic
# class through this back-reference.
_UOL_LEXICON: dict[str, frozenset[str]] = {}


def set_uol_lexicon(lexicon: dict[str, frozenset[str]]) -> None:
    global _UOL_LEXICON
    _UOL_LEXICON = lexicon


# ---------------------------------------------------------------------------
# Contract-based role / predicate lookups (language-agnostic)
# ---------------------------------------------------------------------------
# These helpers read from melm.contracts.  The exported legacy names
# (``_VERBS``, ``_WH_WORDS``, etc.) are compatibility projections from the
# contracts, not independent vocabulary sources.

_FUNCTION_WORDS: dict[str, dict[str, Any]] | None = None
_PREDICATE_INVENTORY: dict[str, dict[str, Any]] | None = None
_CONTENT_DOMAINS: dict[str, dict[str, Any]] | None = None


def _load_function_words() -> dict[str, dict[str, Any]]:
    global _FUNCTION_WORDS
    if _FUNCTION_WORDS is not None:
        return _FUNCTION_WORDS
    try:
        from melm.contracts import load_function_words

        payload = load_function_words()
        _FUNCTION_WORDS = {}
        for entry in payload.get("entries", []):
            lang = str(entry.get("language", "en")).strip().lower()
            lemma = str(entry.get("lemma", "")).strip().lower()
            _FUNCTION_WORDS[(lang, lemma)] = dict(entry)
        return _FUNCTION_WORDS
    except Exception:
        _FUNCTION_WORDS = {}
        return _FUNCTION_WORDS


def _load_predicate_inventory() -> dict[str, dict[str, Any]]:
    global _PREDICATE_INVENTORY, _CONTENT_DOMAINS
    if _PREDICATE_INVENTORY is not None:
        return _PREDICATE_INVENTORY
    try:
        from melm.contracts import load_predicate_inventory

        payload = load_predicate_inventory()
        _PREDICATE_INVENTORY = {}
        for entry in payload.get("predicates", []):
            lang = str(entry.get("language", "en")).strip().lower()
            lemma = str(entry.get("lemma", "")).strip().lower()
            _PREDICATE_INVENTORY[(lang, lemma)] = dict(entry)
        _CONTENT_DOMAINS = {}
        for entry in payload.get("content_domains", []):
            lemma = str(entry.get("lemma", "")).strip().lower()
            _CONTENT_DOMAINS[lemma] = dict(entry)
        return _PREDICATE_INVENTORY
    except Exception:
        _PREDICATE_INVENTORY = {}
        _CONTENT_DOMAINS = {}
        return _PREDICATE_INVENTORY


def _get_role(lemma: str, language: str = "en") -> str:
    """Return the universal UOL role for a lemma, e.g. 'wh_word', 'modal', 'pronoun'."""
    fw = _load_function_words()
    entry = fw.get((language, lemma))
    if entry is not None:
        return str(entry.get("role", "")).strip().lower()
    # Compatibility fallback to contract-projected English constants.
    if lemma in _GREETINGS:
        return "greeting"
    if lemma in _WH_WORDS:
        return "wh_word"
    if lemma in _MODALS:
        return "modal"
    if lemma in _AUXILIARIES:
        return "auxiliary"
    if lemma in _NEGATIONS:
        return "negation"
    if lemma in _DETERMINERS:
        return "determiner"
    if lemma in _PREPOSITIONS:
        return "preposition"
    if lemma in _CONJUNCTIONS:
        return "conjunction"
    if lemma in _FREQUENCY:
        return "frequency"
    if lemma in _EQUIVALENCE:
        return "equivalence"
    if lemma in _POLITENESS:
        return "politeness"
    if lemma in _DISCOURSE_PARTICLES:
        return "discourse_particle"
    if lemma in _PRONOUNS:
        return "pronoun"
    return ""


def _get_subrole(lemma: str, language: str = "en") -> str:
    """Return the subrole for a lemma, e.g. 'agent', 'possessor', 'possibility'."""
    fw = _load_function_words()
    entry = fw.get((language, lemma))
    if entry is not None:
        return str(entry.get("subrole", "")).strip().lower()
    # Compatibility fallback for projected pronouns only.
    pronoun = _PRONOUNS.get(lemma)
    if pronoun is not None:
        return pronoun[1]
    return ""


def _get_referent(lemma: str, language: str = "en") -> str:
    """Return the canonical referent for a pronoun, e.g. 'user', 'assistant'."""
    fw = _load_function_words()
    entry = fw.get((language, lemma))
    if entry is not None:
        return str(entry.get("referent", "")).strip().lower()
    # Transitional fallback
    pronoun = _PRONOUNS.get(lemma)
    if pronoun is not None:
        return pronoun[0]
    return ""


def _get_predicate(lemma: str, language: str = "en") -> dict[str, Any] | None:
    """Return the predicate entry for a lemma, or None if not a known predicate."""
    pi = _load_predicate_inventory()
    entry = pi.get((language, lemma))
    if entry is not None:
        return entry
    # Compatibility fallback to the contract-projected English predicate export.
    verb = _VERBS.get(lemma)
    if verb is not None:
        canonical, semantic_class = verb
        return {
            "lemma": lemma,
            "predicate_id": canonical,
            "kind": "event",
            "semantic_class": semantic_class,
            "language": "en",
        }
    return None


def _get_predicate_id(lemma: str, language: str = "en") -> str:
    pred = _get_predicate(lemma, language=language)
    return str(pred["predicate_id"]) if pred is not None else ""


def _get_content_domain(lemma: str) -> str:
    """Return the content domain label for a nominal lemma, or None."""
    _load_predicate_inventory()  # ensures _CONTENT_DOMAINS is loaded
    entry = _CONTENT_DOMAINS.get(lemma) if _CONTENT_DOMAINS is not None else None
    if entry is not None:
        return str(entry.get("domain", ""))
    # Compatibility fallback to the contract-projected English content-domain export.
    return _KNOWN_NOMINAL_DOMAINS.get(lemma)


def _role_lemmas(role: str) -> set[str]:
    return {
        lemma
        for (language, lemma), entry in _load_function_words().items()
        if language == "en" and entry.get("role") == role
    }


def _pronoun_projection() -> dict[str, tuple[str, str]]:
    return {
        lemma: (
            str(entry.get("referent", "")).strip().lower(),
            str(entry.get("subrole", "")).strip().lower(),
        )
        for (language, lemma), entry in _load_function_words().items()
        if language == "en" and entry.get("role") == "pronoun"
    }


def _verb_projection() -> dict[str, tuple[str, str]]:
    return {
        lemma: (
            str(entry.get("predicate_id", "")).strip(),
            str(entry.get("semantic_class", "")).strip(),
        )
        for (language, lemma), entry in _load_predicate_inventory().items()
        if language == "en"
    }


def _content_domain_projection() -> dict[str, str]:
    _load_predicate_inventory()
    if _CONTENT_DOMAINS is None:
        return {}
    return {
        lemma: str(entry.get("domain", "")).strip()
        for lemma, entry in _CONTENT_DOMAINS.items()
    }


_GREETINGS = _role_lemmas("greeting")
_WH_WORDS = _role_lemmas("wh_word")
_MODALS = _role_lemmas("modal")
_AUXILIARIES = _role_lemmas("auxiliary")
_NEGATIONS = _role_lemmas("negation")
_DETERMINERS = _role_lemmas("determiner")
_PREPOSITIONS = _role_lemmas("preposition")
_CONJUNCTIONS = _role_lemmas("conjunction")
_FREQUENCY = _role_lemmas("frequency")
_EQUIVALENCE = _role_lemmas("equivalence")
_POLITENESS = _role_lemmas("politeness")
_DISCOURSE_PARTICLES = _role_lemmas("discourse_particle")
_PRONOUNS = _pronoun_projection()
_VERBS = _verb_projection()
_KNOWN_NOMINAL_DOMAINS = _content_domain_projection()


def _verb_info(token: str) -> tuple[str, str] | None:
    """Return (canonical, semantic_class) for a verb.

    Checks the contract-projected ``_VERBS`` compatibility export first, then
    falls back to the runtime lexicon (``_UOL_LEXICON``), which contains
    acquired / bulk-seeded verbs. Returns ``None`` when *token* is not known
    in either source.

    Only matches lexical entries whose semantic class starts with ``verb.``
    (pure verb senses).  Noun entries like ``narrative_content`` or
    ``weather_phenomenon`` are excluded to avoid false predicate matches.
    """
    entry = _VERBS.get(token)
    if entry is not None:
        return entry
    classes = _UOL_LEXICON.get(token)
    if classes:
        first = next(iter(classes))
        if first.startswith("verb."):
            return (token, first)
    return None


@dataclass(frozen=True)
class FunctionalParse:
    speech_act: str
    subject: str
    action: str
    object: str
    target: str
    complement_action: str
    indirect_object: str
    modifiers: dict[str, tuple[str, ...]]
    relations: tuple[dict[str, Any], ...]
    token_roles: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    parse_score: float
    syntactic_coverage: float
    semantic_unknown_tokens: tuple[str, ...]
    pattern: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "melm.weighted_functional_grammar.v1",
            "speech_act": self.speech_act,
            "subject": self.subject,
            "action": self.action,
            "object": self.object,
            "target": self.target,
            "complement_action": self.complement_action,
            "indirect_object": self.indirect_object,
            "modifiers": {key: list(value) for key, value in self.modifiers.items()},
            "relations": [dict(item) for item in self.relations],
            "token_roles": [dict(item) for item in self.token_roles],
            "candidates": [dict(item) for item in self.candidates],
            "parse_score": self.parse_score,
            "syntactic_coverage": self.syntactic_coverage,
            "semantic_unknown_tokens": list(self.semantic_unknown_tokens),
            "pattern": self.pattern,
        }


def parse_functional_relations(
    tokens: tuple[str, ...],
    *,
    question_mark: bool = False,
    language: str = "en",
) -> FunctionalParse | None:
    """Produce a ranked relation parse from functional and lexical biases."""

    if not tokens:
        return None
    lemmas = tuple(_lemma(token, language=language) for token in tokens)
    if all(_get_role(token, language=language) in {"greeting", "politeness", "discourse_particle"} for token in lemmas):
        roles = tuple(
            _role(index, token, lemma, "discourse_greeting", "social_opening", 1.0)
            for index, (token, lemma) in enumerate(zip(tokens, lemmas))
        )
        return FunctionalParse(
            speech_act="greeting",
            subject="user",
            action="greet",
            object="",
            target="assistant",
            complement_action="",
            indirect_object="",
            modifiers={},
            relations=(
                {"type": "agent", "head": "greet", "value": "user", "weight": 1.0},
                {"type": "target", "head": "greet", "value": "assistant", "weight": 1.0},
            ),
            token_roles=roles,
            candidates=(
                {"action": "greet", "semantic_class": "social", "score": 1.0, "index": 0},
            ),
            parse_score=1.0,
            syntactic_coverage=1.0,
            semantic_unknown_tokens=(),
            pattern="greeting_interjection",
        )

    speech_act = _speech_act(lemmas, question_mark=question_mark, language=language)
    subject_index, subject = _subject(lemmas, speech_act, language=language)
    predicate_candidates = _predicate_candidates(lemmas, subject_index, speech_act, language=language)
    if not predicate_candidates:
        return None
    primary = predicate_candidates[0]
    predicate_index = int(primary["index"])
    action = str(primary["action"])
    semantic_class = str(primary["semantic_class"])
    complement = _complement_predicate(lemmas, predicate_index, language=language)
    complement_index = int(complement["index"]) if complement else -1
    complement_action = str(complement["action"]) if complement else ""
    indirect_object = _indirect_object(lemmas, predicate_index, complement_index, language=language)
    object_value, object_index, possessor, nominal_relations = _object(
        tokens,
        lemmas,
        predicate_index,
        complement_index,
        action=action,
        complement_action=complement_action,
        indirect_object=indirect_object,
        language=language,
    )
    target = _target(subject, speech_act)
    modifiers = {
        "frequency": tuple(token for token in lemmas if _get_role(token, language=language) == "frequency"),
        "equivalence": tuple(token for token in lemmas if _get_role(token, language=language) == "equivalence"),
        "negation": tuple(token for token in lemmas if _get_role(token, language=language) == "negation"),
    }
    token_roles, semantic_unknown = _token_roles(
        tokens,
        lemmas,
        subject_index=subject_index,
        predicate_index=predicate_index,
        complement_index=complement_index,
        object_index=object_index,
        indirect_object=indirect_object,
        possessor=possessor,
        language=language,
    )
    covered = sum(1 for item in token_roles if item["role"] != "unresolved_token")
    syntactic_coverage = round(covered / max(1, len(tokens)), 3)
    parse_score = round(
        min(
            1.0,
            0.25
            + 0.25 * float(primary["score"])
            + 0.18 * (1.0 if subject else 0.0)
            + 0.17 * (1.0 if object_value or complement_action else 0.0)
            + 0.15 * syntactic_coverage,
        ),
        3,
    )
    relations = [
        {"type": "agent", "head": action, "value": subject, "weight": 0.95},
        {"type": "target", "head": action, "value": target, "weight": 0.8},
    ]
    if object_value:
        relations.append({"type": "patient", "head": complement_action or action, "value": object_value, "weight": 0.88})
    if indirect_object:
        relations.append({"type": "recipient", "head": action, "value": indirect_object, "weight": 0.84})
    if complement_action:
        relations.append({"type": "complement", "head": action, "value": complement_action, "weight": 0.9})
    if possessor:
        relations.append({"type": "possessor", "head": object_value, "value": possessor, "weight": 0.92})
    relations.extend(nominal_relations)
    for kind, values in modifiers.items():
        for value in values:
            relations.append({"type": kind, "head": action, "value": value, "weight": 0.8})
    pattern = _pattern(
        speech_act,
        subject=subject,
        semantic_class=semantic_class,
        complement_action=complement_action,
        object_value=object_value,
    )
    return FunctionalParse(
        speech_act=speech_act,
        subject=subject,
        action=action,
        object=object_value,
        target=target,
        complement_action=complement_action,
        indirect_object=indirect_object,
        modifiers={key: value for key, value in modifiers.items() if value},
        relations=tuple(relations),
        token_roles=tuple(token_roles),
        candidates=tuple(predicate_candidates),
        parse_score=parse_score,
        syntactic_coverage=syntactic_coverage,
        semantic_unknown_tokens=tuple(semantic_unknown),
        pattern=pattern,
    )


def functional_frame_kind(parse: FunctionalParse | None) -> str:
    if parse is None:
        return ""
    if parse.speech_act == "greeting":
        return "social_greeting"
    semantic_class = _semantic_class(parse.action)
    complement_class = _semantic_class(parse.complement_action)
    if (
        parse.subject == "assistant"
        and parse.speech_act in {"yes_no_question", "wh_question"}
        and (
            semantic_class in {"verb.communicate", "verb.emotion", "action"}
            or complement_class in {"verb.communicate", "action"}
        )
    ):
        return "assistant_behavior"
    object_domain = _get_content_domain(parse.object)
    if (
        (
            (
                parse.subject in {"user", "user_group"}
                and semantic_class in {"verb.stative", "verb.social", "verb.change"}
            )
            or (
                parse.subject == "assistant"
                and semantic_class == "verb.social"
                and parse.indirect_object in {"user", "user_group"}
            )
            or complement_class == "verb.change"
        )
        and object_domain in {"career", "work"}
    ):
        return "personal_goal_advice"
    action_predicate_id = _get_predicate_id(parse.action)
    if (
        parse.speech_act in {"yes_no_question", "wh_question", "request", "statement"}
        and (
            parse.object
            or parse.complement_action
            or action_predicate_id in {"be", "work"}
            or (
                parse.speech_act in {"yes_no_question", "wh_question"}
                and parse.action
                and parse.subject not in {"user", "user_group"}
            )
        )
    ):
        return "open_domain"
    return ""


def _speech_act(lemmas: tuple[str, ...], *, question_mark: bool, language: str = "en") -> str:
    first = lemmas[0]
    first_role = _get_role(first, language=language)
    if first_role == "wh_word":
        return "wh_question"
    if first_role in {"modal", "auxiliary"}:
        return "yes_no_question"
    if question_mark:
        return "question"
    if _get_predicate(first, language=language) is not None or first_role == "politeness":
        return "request"
    return "statement"


def _subject(lemmas: tuple[str, ...], speech_act: str, language: str = "en") -> tuple[int, str]:
    # Find the first agent pronoun
    if speech_act in {"yes_no_question", "wh_question"}:
        search_range = enumerate(lemmas[1:], start=1)
    else:
        search_range = enumerate(lemmas)
    for index, token in search_range:
        if _get_role(token, language=language) == "pronoun" and _get_subrole(token, language=language) in {"agent", "agent_or_patient"}:
            return index, _get_referent(token, language=language) or token

    # No agent pronoun: find the first predicate, then look for a nominal subject before it
    predicate_indexes = [
        index
        for index, token in enumerate(lemmas)
        if _get_predicate(token, language=language) is not None
        and not (
            _get_role(token, language=language) == "auxiliary"
            and any(_get_predicate(item, language=language) is not None and item != token for item in lemmas[index + 1 :])
        )
        and not (
            index + 1 < len(lemmas)
            and lemmas[index + 1] == "of"
            and any(item == "be" for item in lemmas[index + 2 :])
        )
    ]
    if predicate_indexes:
        first_pred_role = _get_role(lemmas[predicate_indexes[0]], language=language)
        predicate_index = predicate_indexes[-1] if first_pred_role == "auxiliary" else predicate_indexes[0]
        first_role = _get_role(lemmas[0], language=language)
        search_start = 1 if first_role in {"wh_word", "modal", "auxiliary"} else 0
        for index in range(search_start, predicate_index):
            token = lemmas[index]
            token_role = _get_role(token, language=language)
            if token_role in {"wh_word", "modal", "auxiliary", "determiner", "preposition", "conjunction", "frequency", "equivalence", "negation", "politeness", "pronoun"}:
                continue
            return index, _get_content_domain(token) or token
    if speech_act == "request":
        return -1, "user"
    return -1, "user"


def _predicate_candidates(
    lemmas: tuple[str, ...],
    subject_index: int,
    speech_act: str,
    language: str = "en",
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, token in enumerate(lemmas):
        pred = _get_predicate(token, language=language)
        if pred is None:
            continue
        if (
            index + 1 < len(lemmas)
            and lemmas[index + 1] == "of"
            and any(item == "be" for item in lemmas[index + 2 :])
        ):
            continue
        # English-specific: "be" before another predicate is usually auxiliary
        # (e.g. "What is he eating?"). Skip for other languages where copula
        # can precede predicate-derived nouns (e.g. Igbo "gini bu eri?").
        if language == "en" and _get_predicate_id(token, language=language) == "be" and any(_get_predicate(item, language=language) is not None and item != "be" for item in lemmas[index + 1 :]):
            continue
        if _get_role(token, language=language) == "auxiliary" and any(_get_predicate(item, language=language) is not None and item != token for item in lemmas[index + 1 :]):
            continue
        score = 0.52
        if index > subject_index:
            score += 0.18
        if index > 0 and lemmas[index - 1] == "to":
            score -= 0.2
        if not candidates:
            score += 0.18
        if speech_act == "request" and index <= 1:
            score += 0.12
        candidates.append(
            {
                "index": index,
                "token": token,
                "action": pred["predicate_id"],
                "semantic_class": pred["semantic_class"],
                "score": round(min(1.0, max(0.0, score)), 3),
            }
        )
    return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["index"])))


def _complement_predicate(lemmas: tuple[str, ...], predicate_index: int, language: str = "en") -> dict[str, Any] | None:
    for index in range(predicate_index + 1, len(lemmas)):
        token = lemmas[index]
        pred = _get_predicate(token, language=language)
        if pred is None:
            continue
        previous = lemmas[index - 1] if index > 0 else ""
        prev_role = _get_role(previous, language=language)
        prev_subrole = _get_subrole(previous, language=language)
        intervening_pronoun = (
            index == predicate_index + 2
            and prev_role == "pronoun"
            and prev_subrole in {"patient", "agent_or_patient"}
        )
        if previous == "to" or intervening_pronoun or _get_predicate_id(lemmas[predicate_index], language=language) in {"help", "like"}:
            return {
                "index": index,
                "action": pred["predicate_id"],
                "semantic_class": pred["semantic_class"],
            }
    return None


def _indirect_object(lemmas: tuple[str, ...], predicate_index: int, complement_index: int, language: str = "en") -> str:
    end = complement_index if complement_index > predicate_index else len(lemmas)
    for token in lemmas[predicate_index + 1 : end]:
        if _get_role(token, language=language) == "pronoun" and _get_subrole(token, language=language) in {"patient", "agent_or_patient", "human_collective", "human_indefinite"}:
            return _get_referent(token, language=language) or token
    # "tell" can take a human collective / indefinite as indirect object; already caught above
    return ""


def _object(
    tokens: tuple[str, ...],
    lemmas: tuple[str, ...],
    predicate_index: int,
    complement_index: int,
    *,
    action: str,
    complement_action: str,
    indirect_object: str,
    language: str = "en",
) -> tuple[str, int, str, list[dict[str, Any]]]:
    start = complement_index + 1 if complement_index >= 0 else predicate_index + 1
    candidates: list[tuple[int, str, str, str]] = []
    possessor = ""
    pending_possessor = ""
    active_relation = ""
    for index in range(start, len(lemmas)):
        token = lemmas[index]
        token_role = _get_role(token, language=language)
        token_subrole = _get_subrole(token, language=language)
        if token_role == "pronoun":
            referent = _get_referent(token, language=language)
            if token_subrole == "possessor":
                pending_possessor = referent
                continue
            if token_subrole == "reflexive":
                candidates.append((index, referent, pending_possessor, active_relation))
                continue
            if referent == indirect_object:
                continue
        if token_role in {"modal", "auxiliary", "negation"}:
            continue
        if token_role == "preposition":
            active_relation = token
            continue
        if token_role in {"determiner", "conjunction"}:
            continue
        if token_role in {"frequency", "equivalence", "politeness"}:
            continue
        if _get_predicate(token, language=language) is not None:
            previous = lemmas[index - 1] if index > 0 else ""
            prev_role = _get_role(previous, language=language)
            nominal_by_relation = prev_role == "preposition" or prev_role == "determiner"
            nominal_after_communication = action in {"answer", "ask", "describe", "explain", "say", "tell"} and index > start
            gerund_nominal = tokens[index].endswith("ing") and index != complement_index
            if not nominal_by_relation and not nominal_after_communication and not gerund_nominal:
                continue
        canonical = _get_content_domain(token) or token
        candidates.append((index, canonical, pending_possessor, active_relation))
        pending_possessor = ""
    if not candidates:
        return "", -1, "", []
    direct_candidates = [item for item in candidates if not item[3]]
    selected = direct_candidates[-1] if direct_candidates else candidates[-1]
    object_index, object_value, possessor, _ = selected
    relation_types = {
        "about": "topic",
        "at": "context",
        "for": "purpose",
        "from": "source",
        "in": "context",
        "into": "destination",
        "of": "qualifier",
        "on": "topic",
        "over": "topic",
        "through": "path",
        "to": "destination",
        "with": "accompaniment",
    }
    nominal_relations = [
        {
            "type": relation_types.get(relation, "nominal_relation"),
            "head": complement_action or action,
            "value": value,
            "marker": relation,
            "weight": 0.78,
        }
        for _, value, _, relation in candidates
        if relation
    ]
    return object_value, object_index, possessor, nominal_relations


def _target(subject: str, speech_act: str) -> str:
    if speech_act in {"yes_no_question", "wh_question", "question", "request"}:
        return "user" if subject == "assistant" else "assistant"
    return "assistant"


def _token_roles(
    tokens: tuple[str, ...],
    lemmas: tuple[str, ...],
    *,
    subject_index: int,
    predicate_index: int,
    complement_index: int,
    object_index: int,
    indirect_object: str,
    possessor: str,
    language: str = "en",
) -> tuple[list[dict[str, Any]], list[str]]:
    roles: list[dict[str, Any]] = []
    semantic_unknown: list[str] = []
    for index, (token, lemma) in enumerate(zip(tokens, lemmas)):
        if index == subject_index:
            role, meaning, weight = (
                "grammatical_subject",
                _get_referent(lemma, language=language) or _get_content_domain(lemma) or lemma,
                0.98,
            )
        elif index == predicate_index:
            pred = _get_predicate(lemma, language=language)
            canonical, semantic_class = (pred["predicate_id"], pred["semantic_class"]) if pred else (lemma, "unknown")
            role, meaning, weight = "main_predicate", f"{canonical}:{semantic_class}", 0.98
        elif index == complement_index:
            pred = _get_predicate(lemma, language=language)
            canonical, semantic_class = (pred["predicate_id"], pred["semantic_class"]) if pred else (lemma, "unknown")
            role, meaning, weight = "complement_predicate", f"{canonical}:{semantic_class}", 0.92
        elif index == object_index:
            role, meaning, weight = "semantic_object", _get_content_domain(lemma) or lemma, 0.88
            if _get_content_domain(lemma) is None and _get_role(lemma, language=language) != "pronoun":
                semantic_unknown.append(lemma)
        else:
            lemma_role = _get_role(lemma, language=language)
            if lemma_role == "greeting":
                role, meaning, weight = "discourse_greeting", "social_opening", 1.0
            elif lemma_role == "wh_word":
                role, meaning, weight = "interrogative", "requested_information_dimension", 0.96
            elif lemma_role == "modal":
                role, meaning, weight = "modal", "possibility_or_obligation", 0.94
            elif lemma_role == "auxiliary":
                role, meaning, weight = "auxiliary", "tense_aspect_or_inversion", 0.94
            elif lemma_role == "negation":
                role, meaning, weight = "negation", "predicate_polarity", 0.94
            elif lemma_role == "frequency":
                role, meaning, weight = "frequency_modifier", lemma, 0.9
            elif lemma_role == "equivalence":
                role, meaning, weight = "equivalence_modifier", lemma, 0.88
            elif lemma_role == "determiner":
                role, meaning, weight = "determiner", "nominal_scope", 0.9
            elif lemma_role == "preposition":
                role, meaning, weight = "relation_marker", lemma, 0.88
            elif lemma_role == "conjunction":
                role, meaning, weight = "clause_link", lemma, 0.86
            elif lemma_role == "politeness":
                role, meaning, weight = "politeness", "request_softener", 0.9
            elif lemma_role == "discourse_particle":
                role, meaning, weight = "discourse_particle", lemma, 0.86
            elif lemma_role == "pronoun":
                referent = _get_referent(lemma, language=language)
                pronoun_role = _get_subrole(lemma, language=language)
                if pronoun_role == "possessor" and referent == possessor:
                    role, meaning, weight = "possessor", referent, 0.92
                elif referent == indirect_object and pronoun_role in {"patient", "agent_or_patient"}:
                    role, meaning, weight = "indirect_object", referent, 0.92
                elif pronoun_role == "reflexive":
                    role, meaning, weight = "reflexive_object", referent, 0.94
                else:
                    role, meaning, weight = "pronoun_relation", f"{referent}:{pronoun_role}", 0.82
            else:
                pred = _get_predicate(lemma, language=language)
                if pred is not None:
                    role, meaning, weight = "secondary_predicate_candidate", f"{pred['predicate_id']}:{pred['semantic_class']}", 0.62
                else:
                    domain = _get_content_domain(lemma)
                    if domain is not None:
                        role, meaning, weight = "content_nominal", domain, 0.58
                    else:
                        role, meaning, weight = "content_nominal", "semantic_class_unknown", 0.58
                        semantic_unknown.append(lemma)
        roles.append(_role(index, token, lemma, role, meaning, weight))
    return roles, list(dict.fromkeys(semantic_unknown))


def _pattern(
    speech_act: str,
    *,
    subject: str,
    semantic_class: str,
    complement_action: str,
    object_value: str,
) -> str:
    parts = [speech_act, subject, semantic_class]
    if complement_action:
        parts.append("with_complement")
    if object_value:
        parts.append("with_object")
    return "_".join(parts)


def _semantic_class(action: str) -> str:
    pred = _get_predicate(action)
    if pred is not None:
        return pred["semantic_class"]
    if action in _UOL_LEXICON:
        classes = _UOL_LEXICON[action]
        if classes:
            return next(iter(classes))
    return ""


def _lemma(token: str, language: str = "en") -> str:
    if language != "en":
        return token
    # n't → not expansion (Phase 2)
    if token.endswith("n't"):
        base = token[:-3]
        nt_irregular = {
            "wo": "will", "ca": "can", "sha": "shall", "ma": "may",
            "ai": "be", "does": "do", "doe": "do", "is": "be",
            "are": "be", "were": "be", "was": "be", "have": "have",
            "has": "have", "had": "have", "did": "do",
        }
        return nt_irregular.get(base, base)

    irregular = {
        "am": "be",
        "are": "be",
        "bought": "buy",
        "did": "do",
        "does": "do",
        "grew": "grow",
        "had": "have",
        "has": "have",
        "is": "be",
        "said": "say",
        "told": "tell",
        "took": "take",
        "was": "be",
        "were": "be",
        "written": "write",
        "wrote": "write",
    }
    if token in irregular:
        return irregular[token]

    def _known_verb(stem: str) -> bool:
        return stem in _VERBS or stem in _UOL_LEXICON

    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if stem.endswith(stem[-1:] * 2):
            stem = stem[:-1]
        if _known_verb(stem):
            return stem
        if _known_verb(f"{stem}e"):
            return f"{stem}e"
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if _known_verb(stem):
            return stem
        if _known_verb(f"{stem}e"):
            return f"{stem}e"
    if token.endswith("s") and _known_verb(token[:-1]):
        return token[:-1]
    return token


def _role(index: int, token: str, lemma: str, role: str, meaning: str, weight: float) -> dict[str, Any]:
    return {
        "index": index,
        "token": token,
        "lemma": lemma,
        "role": role,
        "meaning": meaning,
        "weight": round(weight, 3),
    }
