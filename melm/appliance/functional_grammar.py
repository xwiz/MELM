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


_GREETINGS = {"hello", "hey", "hi", "hiya"}
_WH_WORDS = {"how", "what", "when", "where", "which", "who", "why"}
_MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
_AUXILIARIES = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "was",
    "were",
}
_NEGATIONS = {"never", "no", "not"}
_DETERMINERS = {"a", "an", "any", "some", "the", "this", "that", "these", "those"}
_PREPOSITIONS = {
    "about",
    "at",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "over",
    "through",
    "to",
    "with",
}
_CONJUNCTIONS = {"and", "but", "or"}
_FREQUENCY = {"always", "never", "often", "rarely", "sometimes", "usually"}
_EQUIVALENCE = {"same", "similar"}
_POLITENESS = {"kindly", "please"}
_DISCOURSE_PARTICLES = {"there"}
_PRONOUNS = {
    "i": ("user", "agent"),
    "me": ("user", "patient"),
    "my": ("user", "possessor"),
    "myself": ("user", "reflexive"),
    "we": ("user_group", "agent"),
    "us": ("user_group", "patient"),
    "our": ("user_group", "possessor"),
    "ourselves": ("user_group", "reflexive"),
    "you": ("assistant", "agent_or_patient"),
    "your": ("assistant", "possessor"),
    "yourself": ("assistant", "reflexive"),
    "people": ("people", "human_collective"),
    "someone": ("someone", "human_indefinite"),
}
_VERBS = {
    "acknowledge": ("acknowledge", "verb.communicate"),
    "advise": ("advise", "verb.communicate"),
    "answer": ("answer", "verb.communicate"),
    "ask": ("ask", "verb.communicate"),
    "be": ("be", "verb.stative"),
    "bring": ("bring", "verb.move"),
    "buy": ("buy", "verb.possess"),
    "breathe": ("breathe", "verb.stative"),
    "call": ("call", "verb.communicate"),
    "cancel": ("cancel", "verb.communicate"),
    "cook": ("cook", "verb.create"),
    "define": ("define", "verb.communicate"),
    "describe": ("describe", "verb.communicate"),
    "do": ("do", "action"),
    "eat": ("eat", "verb.consume"),
    "explain": ("explain", "verb.communicate"),
    "feel": ("feel", "verb.cognition"),
    "fly": ("fly", "verb.move"),
    "forget": ("forget", "verb.cognition"),
    "give": ("give", "verb.move"),
    "go": ("go", "verb.move"),
    "grow": ("grow", "verb.change"),
    "have": ("have", "verb.possess"),
    "help": ("help", "verb.social"),
    "improve": ("improve", "verb.change"),
    "know": ("know", "verb.cognition"),
    "like": ("like", "verb.emotion"),
    "list": ("list", "verb.communicate"),
    "live": ("live", "verb.stative"),
    "make": ("make", "verb.create"),
    "need": ("need", "verb.stative"),
    "play": ("play", "action"),
    "rain": ("rain", "verb.stative"),
    "reach": ("reach", "verb.communicate"),
    "read": ("read", "action"),
    "recall": ("recall", "verb.cognition"),
    "recap": ("recap", "verb.communicate"),
    "recommend": ("recommend", "verb.communicate"),
    "remember": ("remember", "verb.cognition"),
    "ring": ("ring", "verb.communicate"),
    "repeat": ("repeat", "action"),
    "say": ("say", "verb.communicate"),
    "see": ("see", "verb.cognition"),
    "share": ("share", "verb.communicate"),
    "show": ("show", "verb.communicate"),
    "sleep": ("sleep", "verb.stative"),
    "speak": ("speak", "verb.communicate"),
    "start": ("start", "verb.change"),
    "summarize": ("summarize", "verb.communicate"),
    "suggest": ("suggest", "verb.communicate"),
    "swallow": ("swallow", "verb.consume"),
    "tell": ("tell", "verb.communicate"),
    "talk": ("talk", "verb.communicate"),
    "upload": ("upload", "verb.move"),
    "use": ("use", "verb.consume"),
    "take": ("take", "verb.possess"),
    "walk": ("walk", "verb.move"),
    "want": ("want", "verb.stative"),
    "work": ("work", "verb.stative"),
    "write": ("write", "verb.create"),
}
def _verb_info(token: str) -> tuple[str, str] | None:
    """Return (canonical, semantic_class) for a verb.

    Checks ``_VERBS`` first (the hardcoded seed set), then falls back to the
    runtime lexicon (``_UOL_LEXICON``) which contains acquired / bulk-seeded
    verbs.  Returns ``None`` when *token* is not known in either source.

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


_KNOWN_NOMINAL_DOMAINS = {
    "career": "career",
    "health": "health",
    "job": "career",
    "memory": "memory",
    "people": "people",
    "person": "person",
    "purpose": "purpose",
    "thing": "thing",
    "work": "work",
}


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


def parse_functional_relations(tokens: tuple[str, ...], *, question_mark: bool = False) -> FunctionalParse | None:
    """Produce a ranked relation parse from functional and lexical biases."""

    if not tokens:
        return None
    lemmas = tuple(_lemma(token) for token in tokens)
    if all(token in _GREETINGS or token in _POLITENESS or token in _DISCOURSE_PARTICLES for token in lemmas):
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

    speech_act = _speech_act(lemmas, question_mark=question_mark)
    subject_index, subject = _subject(lemmas, speech_act)
    predicate_candidates = _predicate_candidates(lemmas, subject_index, speech_act)
    if not predicate_candidates:
        return None
    primary = predicate_candidates[0]
    predicate_index = int(primary["index"])
    action = str(primary["action"])
    semantic_class = str(primary["semantic_class"])
    complement = _complement_predicate(lemmas, predicate_index)
    complement_index = int(complement["index"]) if complement else -1
    complement_action = str(complement["action"]) if complement else ""
    indirect_object = _indirect_object(lemmas, predicate_index, complement_index)
    object_value, object_index, possessor, nominal_relations = _object(
        tokens,
        lemmas,
        predicate_index,
        complement_index,
        action=action,
        complement_action=complement_action,
        indirect_object=indirect_object,
    )
    target = _target(subject, speech_act)
    modifiers = {
        "frequency": tuple(token for token in lemmas if token in _FREQUENCY),
        "equivalence": tuple(token for token in lemmas if token in _EQUIVALENCE),
        "negation": tuple(token for token in lemmas if token in _NEGATIONS),
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
        and parse.object in {"career", "work"}
    ):
        return "personal_goal_advice"
    if (
        parse.speech_act in {"yes_no_question", "wh_question", "request", "statement"}
        and (
            parse.object
            or parse.complement_action
            or parse.action in {"be", "work"}
            or (
                parse.speech_act in {"yes_no_question", "wh_question"}
                and parse.action
                and parse.subject not in {"user", "user_group"}
            )
        )
    ):
        return "open_domain"
    return ""


def _speech_act(lemmas: tuple[str, ...], *, question_mark: bool) -> str:
    first = lemmas[0]
    if first in _WH_WORDS:
        return "wh_question"
    if first in _MODALS or first in _AUXILIARIES:
        return "yes_no_question"
    if question_mark:
        return "question"
    if _verb_info(first) is not None or first in _POLITENESS:
        return "request"
    return "statement"


def _subject(lemmas: tuple[str, ...], speech_act: str) -> tuple[int, str]:
    if speech_act in {"yes_no_question", "wh_question"}:
        for index, token in enumerate(lemmas[1:], start=1):
            if token in _PRONOUNS and _PRONOUNS[token][1] in {"agent", "agent_or_patient"}:
                return index, _PRONOUNS[token][0]
    for index, token in enumerate(lemmas):
        if token in _PRONOUNS and _PRONOUNS[token][1] in {"agent", "agent_or_patient"}:
            return index, _PRONOUNS[token][0]
    predicate_indexes = [
        index
        for index, token in enumerate(lemmas)
        if _verb_info(token) is not None
        and not (
            token in _AUXILIARIES
            and any(_verb_info(item) is not None and item != token for item in lemmas[index + 1 :])
        )
        and not (
            index + 1 < len(lemmas)
            and lemmas[index + 1] == "of"
            and any(item == "be" for item in lemmas[index + 2 :])
        )
    ]
    if predicate_indexes:
        predicate_index = predicate_indexes[-1] if lemmas[predicate_indexes[0]] in _AUXILIARIES else predicate_indexes[0]
        search_start = 1 if lemmas[0] in _WH_WORDS or lemmas[0] in _MODALS or lemmas[0] in _AUXILIARIES else 0
        for index in range(search_start, predicate_index):
            token = lemmas[index]
            if token in _WH_WORDS or token in _MODALS or token in _AUXILIARIES:
                continue
            if token in _DETERMINERS or token in _PREPOSITIONS or token in _CONJUNCTIONS:
                continue
            if token in _FREQUENCY or token in _EQUIVALENCE or token in _NEGATIONS or token in _POLITENESS:
                continue
            if token in _PRONOUNS:
                continue
            return index, _KNOWN_NOMINAL_DOMAINS.get(token, token)
    if speech_act == "request":
        return -1, "user"
    return -1, "user"


def _predicate_candidates(
    lemmas: tuple[str, ...],
    subject_index: int,
    speech_act: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, token in enumerate(lemmas):
        verb = _verb_info(token)
        if verb is None:
            continue
        if (
            index + 1 < len(lemmas)
            and lemmas[index + 1] == "of"
            and any(item == "be" for item in lemmas[index + 2 :])
        ):
            continue
        if token == "be" and any(_verb_info(item) and item != "be" for item in lemmas[index + 1 :]):
            continue
        if token in _AUXILIARIES and any(_verb_info(item) and item != token for item in lemmas[index + 1 :]):
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
        canonical, semantic_class = verb
        candidates.append(
            {
                "index": index,
                "token": token,
                "action": canonical,
                "semantic_class": semantic_class,
                "score": round(min(1.0, max(0.0, score)), 3),
            }
        )
    return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["index"])))


def _complement_predicate(lemmas: tuple[str, ...], predicate_index: int) -> dict[str, Any] | None:
    for index in range(predicate_index + 1, len(lemmas)):
        token = lemmas[index]
        verb = _verb_info(token)
        if verb is None:
            continue
        previous = lemmas[index - 1] if index > 0 else ""
        intervening_pronoun = (
            index == predicate_index + 2
            and previous in _PRONOUNS
            and _PRONOUNS[previous][1] in {"patient", "agent_or_patient"}
        )
        if previous == "to" or intervening_pronoun or lemmas[predicate_index] in {"help", "like"}:
            canonical, semantic_class = verb
            return {
                "index": index,
                "action": canonical,
                "semantic_class": semantic_class,
            }
    return None


def _indirect_object(lemmas: tuple[str, ...], predicate_index: int, complement_index: int) -> str:
    end = complement_index if complement_index > predicate_index else len(lemmas)
    for token in lemmas[predicate_index + 1 : end]:
        pronoun = _PRONOUNS.get(token)
        if pronoun and pronoun[1] in {"patient", "agent_or_patient", "human_collective", "human_indefinite"}:
            return pronoun[0]
    if lemmas[predicate_index] == "tell":
        for token in lemmas[predicate_index + 1 :]:
            if token in {"people", "person", "someone"}:
                return token
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
) -> tuple[str, int, str, list[dict[str, Any]]]:
    start = complement_index + 1 if complement_index >= 0 else predicate_index + 1
    candidates: list[tuple[int, str, str, str]] = []
    possessor = ""
    pending_possessor = ""
    active_relation = ""
    for index in range(start, len(lemmas)):
        token = lemmas[index]
        if token in _PRONOUNS:
            referent, role = _PRONOUNS[token]
            if role == "possessor":
                pending_possessor = referent
                continue
            if role == "reflexive":
                candidates.append((index, referent, pending_possessor, active_relation))
                continue
            if referent == indirect_object:
                continue
        if token in _MODALS or token in _AUXILIARIES or token in _NEGATIONS:
            continue
        if token in _PREPOSITIONS:
            active_relation = token
            continue
        if token in _DETERMINERS or token in _CONJUNCTIONS:
            continue
        if token in _FREQUENCY or token in _EQUIVALENCE or token in _POLITENESS:
            continue
        if _verb_info(token) is not None:
            previous = lemmas[index - 1] if index > 0 else ""
            nominal_by_relation = previous in _PREPOSITIONS or previous in _DETERMINERS
            nominal_after_communication = action in {"answer", "ask", "describe", "explain", "say", "tell"} and index > start
            gerund_nominal = tokens[index].endswith("ing") and index != complement_index
            if not nominal_by_relation and not nominal_after_communication and not gerund_nominal:
                continue
        canonical = _KNOWN_NOMINAL_DOMAINS.get(token, token)
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
) -> tuple[list[dict[str, Any]], list[str]]:
    roles: list[dict[str, Any]] = []
    semantic_unknown: list[str] = []
    for index, (token, lemma) in enumerate(zip(tokens, lemmas)):
        if index == subject_index:
            role, meaning, weight = (
                "grammatical_subject",
                _PRONOUNS.get(lemma, (_KNOWN_NOMINAL_DOMAINS.get(lemma, lemma), ""))[0],
                0.98,
            )
        elif index == predicate_index:
            verb = _verb_info(lemma)
            canonical, semantic_class = verb or (lemma, "unknown")
            role, meaning, weight = "main_predicate", f"{canonical}:{semantic_class}", 0.98
        elif index == complement_index:
            verb = _verb_info(lemma)
            canonical, semantic_class = verb or (lemma, "unknown")
            role, meaning, weight = "complement_predicate", f"{canonical}:{semantic_class}", 0.92
        elif index == object_index:
            role, meaning, weight = "semantic_object", _KNOWN_NOMINAL_DOMAINS.get(lemma, lemma), 0.88
            if lemma not in _KNOWN_NOMINAL_DOMAINS and lemma not in _PRONOUNS:
                semantic_unknown.append(lemma)
        elif lemma in _GREETINGS:
            role, meaning, weight = "discourse_greeting", "social_opening", 1.0
        elif lemma in _WH_WORDS:
            role, meaning, weight = "interrogative", "requested_information_dimension", 0.96
        elif lemma in _MODALS:
            role, meaning, weight = "modal", "possibility_or_obligation", 0.94
        elif lemma in _AUXILIARIES:
            role, meaning, weight = "auxiliary", "tense_aspect_or_inversion", 0.94
        elif lemma in _NEGATIONS:
            role, meaning, weight = "negation", "predicate_polarity", 0.94
        elif lemma in _FREQUENCY:
            role, meaning, weight = "frequency_modifier", lemma, 0.9
        elif lemma in _EQUIVALENCE:
            role, meaning, weight = "equivalence_modifier", lemma, 0.88
        elif lemma in _DETERMINERS:
            role, meaning, weight = "determiner", "nominal_scope", 0.9
        elif lemma in _PREPOSITIONS:
            role, meaning, weight = "relation_marker", lemma, 0.88
        elif lemma in _CONJUNCTIONS:
            role, meaning, weight = "clause_link", lemma, 0.86
        elif lemma in _POLITENESS:
            role, meaning, weight = "politeness", "request_softener", 0.9
        elif lemma in _DISCOURSE_PARTICLES:
            role, meaning, weight = "discourse_particle", lemma, 0.86
        elif lemma in _PRONOUNS:
            referent, pronoun_role = _PRONOUNS[lemma]
            if pronoun_role == "possessor" and referent == possessor:
                role, meaning, weight = "possessor", referent, 0.92
            elif referent == indirect_object and pronoun_role in {"patient", "agent_or_patient"}:
                role, meaning, weight = "indirect_object", referent, 0.92
            elif pronoun_role == "reflexive":
                role, meaning, weight = "reflexive_object", referent, 0.94
            else:
                role, meaning, weight = "pronoun_relation", f"{referent}:{pronoun_role}", 0.82
        elif (verb := _verb_info(lemma)) is not None:
            canonical, semantic_class = verb
            role, meaning, weight = "secondary_predicate_candidate", f"{canonical}:{semantic_class}", 0.62
        else:
            role, meaning, weight = "content_nominal", _KNOWN_NOMINAL_DOMAINS.get(lemma, "semantic_class_unknown"), 0.58
            if lemma not in _KNOWN_NOMINAL_DOMAINS:
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
    if action in _VERBS:
        return _VERBS[action][1]
    if action in _UOL_LEXICON:
        classes = _UOL_LEXICON[action]
        if classes:
            return next(iter(classes))
    return ""


def _lemma(token: str) -> str:
    irregular = {
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
