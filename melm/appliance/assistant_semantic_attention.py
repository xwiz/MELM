"""Semantic attention packet — post-UOL, pre-synthesis semantic workspace.

This module builds a compact SemanticAttentionPacket from the current utterance
data (UOL atoms, functional parse, learned facts, noun/modifier contracts).
The packet is consumed by assistant_nlg_renderer.py for contract-backed
deterministic NLG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from melm.contracts import load_modifier_atoms, load_noun_atoms
from melm.contracts import load_semantic_attention_rules
from melm.appliance._utils import tokenize as _tokenize

__all__ = [
    "ConstraintBinding",
    "CapabilityBinding",
    "SemanticAttentionPacket",
    "build_attention_packet",
    "resolve_task_topic",
    "extract_constraints",
]


@dataclass(frozen=True)
class ConstraintBinding:
    lemma: str
    constraint_type: str = "modifier"


@dataclass(frozen=True)
class CapabilityBinding:
    installed: bool = False
    missing_skill: str = ""
    fallback: str = "open_domain"


@dataclass(frozen=True)
class SemanticAttentionPacket:
    input_text: str
    speech_act: str = ""
    predicate: str = ""
    task_topic: str = ""
    task_topic_class: str = ""
    task_topic_source: str = ""
    output_type: str = ""
    content_entities: tuple[dict[str, str], ...] = ()
    constraints: tuple[ConstraintBinding, ...] = ()
    learned_summary: str = ""
    capability: CapabilityBinding = field(default_factory=CapabilityBinding)
    normalization_alerts: tuple[str, ...] = ()
    reasoning_protected: bool = False
    entity_properties: Mapping[str, Any] = field(default_factory=dict)
    uol_atoms: tuple[Mapping[str, Any], ...] = ()
    entity_fragile: bool = False


_ADVERB_RE = re.compile(r"^[a-z]+ly$")

_NounIndex: dict[str, dict[str, str]] | None = None
_ModifierIndex: dict[str, str] | None = None
_RulesCache: dict[str, Any] | None = None


def _get_noun_index() -> dict[str, dict[str, str]]:
    global _NounIndex
    if _NounIndex is not None:
        return _NounIndex
    try:
        payload = load_noun_atoms()
        idx: dict[str, dict[str, str]] = {}
        for entry in payload.get("entities", []):
            lemma = str(entry.get("canonical_lemma", "")).strip().lower()
            if lemma:
                idx[lemma] = {
                    "semantic_class": str(entry.get("semantic_class_id", "")),
                    "entity_id": str(entry.get("entity_id", "")),
                    "definition": str(entry.get("definition", "")),
                }
        _NounIndex = idx
    except Exception:
        _NounIndex = {}
    return _NounIndex


def _get_modifier_index() -> dict[str, str]:
    global _ModifierIndex
    if _ModifierIndex is not None:
        return _ModifierIndex
    try:
        payload = load_modifier_atoms()
        idx: dict[str, str] = {}
        for entry in payload.get("entries", []):
            lemma = str(entry.get("canonical_lemma", "")).strip().lower()
            if lemma:
                idx[lemma] = str(entry.get("modifier_type", ""))
        _ModifierIndex = idx
    except Exception:
        _ModifierIndex = {}
    return _ModifierIndex


def _get_rules() -> dict[str, Any]:
    global _RulesCache
    if _RulesCache is not None:
        return _RulesCache
    try:
        _RulesCache = load_semantic_attention_rules()
    except Exception:
        _RulesCache = {}
    return _RulesCache




def _lookup_lexical_entry(
    store: Any,
    term: str,
) -> dict[str, str] | None:
    if store is None:
        return None
    try:
        from melm.appliance.assistant_lexicon import lookup_lexical_senses
        senses = lookup_lexical_senses(
            store,
            term,
            statuses=("active", "dormant", "quarantined"),
        )
        if senses:
            sense = senses[0]
            return {
                "lemma": term,
                "source": "lexical_senses",
                "semantic_class": str(sense.get("semantic_class_id", "")),
                "entity_id": str(sense.get("concept_id", "")),
            }
    except Exception:
        pass
    return None


def _find_learned_summary(store: Any, term: str) -> str:
    if store is None:
        return ""
    try:
        from melm.appliance.assistant_skill_research import find_learned_fact
        fact = find_learned_fact(store, term)
        return str(fact.get("summary", "")) if fact else ""
    except Exception:
        return ""


def _known_candidate(
    store: Any,
    nouns: dict[str, dict[str, str]],
    term: str,
) -> dict[str, str] | None:
    if term in nouns:
        noun = nouns[term]
        return {
            "lemma": term,
            "source": "noun_atoms",
            "semantic_class": str(noun.get("semantic_class", "")),
            "entity_id": str(noun.get("entity_id", "")),
        }
    lexical = _lookup_lexical_entry(store, term)
    if lexical:
        return lexical
    if _find_learned_summary(store, term):
        return {
            "lemma": term,
            "source": "learned_facts",
            "semantic_class": "",
            "entity_id": "",
        }
    return None


def _detect_technical_tokens(
    tokens: list[str],
    rules: dict[str, Any],
) -> list[str]:
    tech_terms = set(rules.get("technical_token_terms", []))
    alerts: list[str] = []
    for token in tokens:
        if token in tech_terms:
            alerts.append(f"{token}_normalized_alert")
    return alerts


def resolve_task_topic(
    tokens: list[str],
    parse: Any,
    candidates: list[dict[str, str]],
    constraints: set[str],
    rules: dict[str, Any],
    decisions: Any = None,
) -> str:
    stopwords = set(rules.get("stopwords", []))
    artifact_terms = set(rules.get("response_artifact_terms", []))
    output_type_terms = set(rules.get("output_type_terms", {}))

    if "riddle" in tokens:
        return "riddle"

    obj = None
    if parse is not None:
        obj = getattr(parse, "object", None)
        if obj and obj not in stopwords and obj not in constraints and obj not in output_type_terms:
            topic = obj
            return topic

    if candidates:
        return candidates[0]["lemma"]

    for token in tokens:
        if token not in stopwords and token not in constraints and token not in output_type_terms:
            return token

    return ""


def _resolve_topic_from_decision(
    tokens: list[str],
    parse: Any,
    candidates: list[dict[str, str]],
    constraints: set[str],
    rules: dict[str, Any],
    decisions: Any = None,
) -> str:
    topic = resolve_task_topic(tokens, parse, candidates, constraints, rules, decisions)
    artifact_terms = set(rules.get("response_artifact_terms", []))
    candidate_lemmas = {c["lemma"] for c in candidates}
    if topic in artifact_terms:
        for candidate in reversed(candidates):
            if candidate["lemma"] not in artifact_terms and candidate["lemma"] not in constraints:
                topic = candidate["lemma"]
                break
        else:
            topic = ""
    if not topic and candidates:
        topic = candidates[0]["lemma"]
    elif topic and topic not in candidate_lemmas and candidates:
        topic = candidates[0]["lemma"]
    return topic


def extract_constraints(
    tokens: list[str],
    modifiers: dict[str, str],
    rules: dict[str, Any],
    noun_index: dict[str, dict[str, str]],
    topic: str,
) -> list[ConstraintBinding]:
    output_terms = rules.get("output_type_terms", {})
    seen: set[str] = set()
    constraints: list[ConstraintBinding] = []

    for token in tokens:
        if token in seen:
            continue
        ctype = output_terms.get(token)
        if ctype:
            constraints.append(ConstraintBinding(lemma=token, constraint_type=ctype))
            seen.add(token)
        elif token in modifiers:
            constraints.append(ConstraintBinding(lemma=token, constraint_type=modifiers[token]))
            seen.add(token)
        elif _ADVERB_RE.match(token) and len(token) > 4:
            constraints.append(ConstraintBinding(lemma=token, constraint_type="adverb"))
            seen.add(token)

    return constraints


def build_attention_packet(
    text: str,
    decisions: Any = None,
    store: Any = None,
    rules: dict[str, Any] | None = None,
) -> SemanticAttentionPacket:
    """Build a SemanticAttentionPacket from utterance data and optional decision context."""

    if rules is None:
        rules = _get_rules()
    tokens = _tokenize(text)
    nouns = _get_noun_index()
    modifiers = _get_modifier_index()

    uol_act = None
    parse = None
    speech_act = ""
    predicate = ""
    if decisions is not None:
        uol_act = getattr(decisions, "uol_act", None)
        parse = getattr(decisions, "functional_parse", None)
        if parse is not None:
            speech_act = str(getattr(parse, "speech_act", ""))
            predicate = str(getattr(parse, "complement_action", None) or getattr(parse, "action", ""))
        elif uol_act is not None:
            content = uol_act.get("content", []) if isinstance(uol_act, dict) else []
            if content:
                first = content[0] if isinstance(content, (list, tuple)) else {}
                pred = first.get("predicate", {}) if isinstance(first, dict) else {}
                predicate = str(pred.get("id", "")) if isinstance(pred, dict) else ""

    alerts = _detect_technical_tokens(tokens, rules)
    stopwords = set(rules.get("stopwords", []))

    candidates = []
    for token in tokens:
        if token in stopwords:
            continue
        candidate = _known_candidate(store, nouns, token)
        if candidate:
            candidates.append(candidate)

    constraints = extract_constraints(tokens, modifiers, rules, nouns, "")
    constraint_lemmas = {c.lemma for c in constraints}

    topic = _resolve_topic_from_decision(tokens, parse, candidates, constraint_lemmas, rules, decisions)

    topic_candidate = _known_candidate(store, nouns, topic) if topic else None
    topic_class = topic_candidate.get("semantic_class", "") if topic_candidate else ""
    topic_source = topic_candidate.get("source", "") if topic_candidate else ""

    learned_summary = _find_learned_summary(store, topic)

    entity_props: dict[str, Any] = {}
    if topic and topic in nouns:
        noun_entry = _get_noun_index().get(topic, {})
        try:
            payload = load_noun_atoms()
            for entry in payload.get("entities", []):
                if str(entry.get("canonical_lemma", "")).strip().lower() == topic:
                    entity_props = dict(entry.get("slots", {}))
                    break
        except Exception:
            pass

    entity_fragile = entity_props.get("build_strength") == "fragile"

    content_entities: list[dict[str, str]] = [
        c for c in candidates
        if c["lemma"] != topic
        and c["lemma"] not in constraint_lemmas
        and c["lemma"] not in set(rules.get("response_artifact_terms", []))
    ]

    output_type = ""
    if "riddle" in tokens:
        output_type = "riddle"
    elif topic in tokens:
        pass
    else:
        for token in tokens:
            if token in rules.get("output_type_terms", {}):
                output_type = token
                break

    reasoning_cues = set(rules.get("reasoning_cues", []))
    reasoning_protected = any(cue in text.lower() for cue in reasoning_cues)
    if tokens and tokens[0] == "why":
        reasoning_protected = True

    capability = CapabilityBinding(
        installed=False,
        missing_skill=f"{topic}_skill" if topic else "unknown_skill",
        fallback="reasoning:causal_explanation" if reasoning_protected else "open_domain",
    )

    uol_atoms: tuple[Mapping[str, Any], ...] = ()
    if isinstance(uol_act, dict):
        content = uol_act.get("content")
        if isinstance(content, (list, tuple)):
            uol_atoms = tuple(content)

    return SemanticAttentionPacket(
        input_text=text,
        speech_act=speech_act,
        predicate=predicate,
        task_topic=topic,
        task_topic_class=topic_class,
        task_topic_source=topic_source,
        output_type=output_type,
        content_entities=tuple(content_entities),
        constraints=tuple(constraints),
        learned_summary=learned_summary,
        capability=capability,
        normalization_alerts=tuple(alerts),
        reasoning_protected=reasoning_protected,
        entity_properties=entity_props,
        uol_atoms=uol_atoms,
        entity_fragile=entity_fragile,
    )
