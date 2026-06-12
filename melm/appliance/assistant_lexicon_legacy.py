"""Export legacy parser vocabulary as ordinary seed-authored candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from melm.contracts import load_contract_json, validate_reserved_lexemes
from melm.contracts import validate_router_lexicon_families

from .functional_grammar import _KNOWN_NOMINAL_DOMAINS, _VERBS


LEGACY_VERB_CLASS_MAP = {
    "acquisition": "verb.possess",
    "activity": "action",
    "cognition": "verb.cognition",
    "communication": "verb.communicate",
    "consumption": "verb.consume",
    "creation": "verb.create",
    "desire": "verb.stative",
    "development": "verb.change",
    "function": "verb.stative",
    "generic_action": "action",
    "guidance": "verb.communicate",
    "iteration": "action",
    "knowledge_transfer": "verb.communicate",
    "motion": "verb.move",
    "necessity": "verb.stative",
    "possession_or_experience": "verb.possess",
    "preference": "verb.emotion",
    "state": "verb.stative",
    "support": "verb.social",
    "transfer": "verb.move",
}

LEGACY_NOMINAL_CLASS_MAP = {
    "career": "abstract",
    "health": "abstract",
    "memory": "cognition",
    "people": "group",
    "person": "person",
    "purpose": "cognition",
    "thing": "entity",
    "work": "action",
}

LEGACY_ROUTER_TERM_CLASSES: dict[str, str] = {
    "accessibility": "personal_attribute",
    "adult": "social_relation",
    "advice": "advice_action",
    "advise": "advice_action",
    "afraid": "health_condition",
    "age": "personal_attribute",
    "all": "temporal_descriptor",
    "amazing": "evaluative_expression",
    "answer": "autobiographical_event",
    "answered": "autobiographical_action",
    "anxious": "health_condition",
    "ask": "autobiographical_action",
    "asked": "autobiographical_action",
    "audio": "media_content",
    "awesome": "evaluative_expression",
    "bad": "evaluative_expression",
    "bedtime": "routine_concept",
    "better": "evaluative_expression",
    "bleeding": "health_condition",
    "boring": "evaluative_expression",
    "breathe": "wellness_activity",
    "breakfast": "food_item",
    "brother": "social_relation",
    "bye": "social_greeting",
    "call": "contact_action",
    "caregiver": "social_relation",
    "child": "child_relation",
    "child's": "child_relation",
    "class": "public_place",
    "clothe": "clothing_item",
    "clothes": "clothing_item",
    "coat": "clothing_item",
    "contact": "social_relation",
    "conversation": "autobiographical_event",
    "conversations": "autobiographical_event",
    "cook": "food_item",
    "cool": "evaluative_expression",
    "dad": "social_relation",
    "daughter": "child_relation",
    "daughter's": "child_relation",
    "day": "temporal_descriptor",
    "define": "definition_request",
    "device": "hardware_entity",
    "diagnose": "health_condition",
    "diet": "wellness_activity",
    "different": "evaluative_expression",
    "dinner": "food_item",
    "doctor": "health_condition",
    "dress": "clothing_item",
    "dressed": "clothing_item",
    "earlier": "temporal_descriptor",
    "easy": "evaluative_expression",
    "emergency": "health_condition",
    "export": "contact_action",
    "eat": "food_item",
    "exercise": "wellness_activity",
    "explain": "definition_request",
    "fable": "narrative_content",
    "fables": "narrative_content",
    "faint": "health_condition",
    "family": "household_concept",
    "favorite": "personal_attribute",
    "fever": "health_condition",
    "fitness": "wellness_activity",
    "food": "food_item",
    "forecast": "weather_phenomenon",
    "forget": "memory_recall",
    "fun": "evaluative_expression",
    "funny": "evaluative_expression",
    "go": "movement_action",
    "goal": "goal_concept",
    "goals": "goal_concept",
    "going": "movement_action",
    "good": "evaluative_expression",
    "goodbye": "social_greeting",
    "great": "evaluative_expression",
    "happened": "autobiographical_action",
    "hard": "evaluative_expression",
    "health": "health_domain",
    "healthier": "health_domain",
    "healthy": "health_domain",
    "hello": "social_greeting",
    "help": "advice_action",
    "hey": "social_greeting",
    "hi": "social_greeting",
    "history": "temporal_descriptor",
    "hiya": "social_greeting",
    "household": "household_concept",
    "hurt": "health_condition",
    "hurts": "health_condition",
    "ill": "health_condition",
    "improve": "advice_action",
    "interesting": "evaluative_expression",
    "job": "personal_attribute",
    "kid": "child_relation",
    "kid's": "child_relation",
    "know": "memory_recall",
    "last": "temporal_descriptor",
    "list": "autobiographical_action",
    "location": "personal_attribute",
    "lofi": "media_descriptor",
    "lonely": "health_condition",
    "long": "temporal_descriptor",
    "lunch": "food_item",
    "meal": "food_item",
    "mean": "definition_request",
    "means": "definition_request",
    "medicine": "health_condition",
    "medication": "health_condition",
    "memory": "memory_recall",
    "milestone": "goal_concept",
    "milestones": "goal_concept",
    "mom": "social_relation",
    "morning": "routine_concept",
    "name": "personal_attribute",
    "music": "media_content",
    "naked": "undress_state",
    "need": "request_softener",
    "nice": "evaluative_expression",
    "now": "temporal_descriptor",
    "objective": "goal_concept",
    "objectives": "goal_concept",
    "outside": "public_place",
    "over": "temporal_descriptor",
    "pain": "health_condition",
    "past": "temporal_descriptor",
    "poison": "health_condition",
    "person": "social_relation",
    "phone": "contact_action",
    "piano": "physical_object.instrument",
    "please": "request_softener",
    "preference": "personal_attribute",
    "previous": "temporal_descriptor",
    "profile": "personal_attribute",
    "public": "public_place",
    "question": "autobiographical_event",
    "questions": "autobiographical_event",
    "radio": "physical_object.media_source",
    "rain": "weather_phenomenon",
    "raincoat": "clothing_item",
    "rash": "health_condition",
    "recommend": "advice_action",
    "report": "communication_action",
    "reach": "contact_action",
    "recent": "temporal_descriptor",
    "recall": "memory_recall",
    "recap": "autobiographical_action",
    "remember": "memory_recall",
    "rest": "wellness_activity",
    "ring": "contact_action",
    "routine": "routine_concept",
    "sad": "health_condition",
    "scare": "health_condition",
    "see": "advice_action",
    "send": "contact_action",
    "share": "contact_action",
    "scared": "health_condition",
    "schedule": "routine_concept",
    "school": "public_place",
    "session": "autobiographical_event",
    "sessions": "autobiographical_event",
    "shared": "owner_concept",
    "show": "autobiographical_action",
    "sick": "health_condition",
    "silly": "evaluative_expression",
    "sister": "social_relation",
    "sleep": "wellness_activity",
    "snack": "food_item",
    "snacks": "food_item",
    "someone": "social_relation",
    "son": "child_relation",
    "son's": "child_relation",
    "song": "media_content",
    "sound": "media_content",
    "suggest": "advice_action",
    "sounds": "media_content",
    "speak": "communication_action",
    "speaking": "communication_action",
    "stories": "narrative_content",
    "story": "narrative_content",
    "summarize": "autobiographical_action",
    "system": "abstract_concept",
    "systems": "abstract_concept",
    "tale": "narrative_content",
    "tales": "narrative_content",
    "take": "advice_action",
    "upload": "contact_action",
    "talk": "communication_action",
    "talked": "autobiographical_action",
    "talking": "communication_action",
    "tell": "communication_action",
    "give": "action_verb",
    "made": "action_verb",
    "make": "action_verb",
    "read": "action_verb",
    "temperature": "weather_phenomenon",
    "thanks": "social_greeting",
    "tired": "health_condition",
    "today": "temporal_descriptor",
    "tomorrow": "temporal_descriptor",
    "tonight": "temporal_descriptor",
    "track": "media_content",
    "trip": "personal_attribute",
    "tummy": "health_condition",
    "undressed": "undress_state",
    "use": "action_verb",
    "uses": "action_verb",
    "using": "action_verb",
    "walk": "movement_action",
    "wear": "clothing_item",
    "weather": "weather_phenomenon",
    "week": "temporal_descriptor",
    "weekly": "temporal_descriptor",
    "weird": "evaluative_expression",
    "wellness": "health_domain",
    "work": "action_verb",
    "workout": "wellness_activity",
    "worse": "evaluative_expression",
}


def build_legacy_lexicon_candidates() -> list[dict[str, Any]]:
    reserved, policy = _controlled_lexemes()
    candidates = [
        _candidate(
            lemma=lemma,
            pos="verb",
            semantic_class_id=LEGACY_VERB_CLASS_MAP[legacy_class],
            definition=f"legacy parser verb: {canonical} ({legacy_class})",
            source_ref=f"functional_grammar._VERBS:{lemma}",
            reserved=lemma in reserved,
            policy_overlap=lemma in policy,
        )
        for lemma, (canonical, legacy_class) in sorted(_VERBS.items())
    ]
    candidates.extend(
        _candidate(
            lemma=lemma,
            pos="noun",
            semantic_class_id=LEGACY_NOMINAL_CLASS_MAP[domain],
            definition=f"legacy parser nominal domain: {domain}",
            source_ref=f"functional_grammar._KNOWN_NOMINAL_DOMAINS:{lemma}",
            reserved=lemma in reserved,
            policy_overlap=lemma in policy,
        )
        for lemma, domain in sorted(_KNOWN_NOMINAL_DOMAINS.items())
    )
    return candidates


def build_legacy_router_candidates(
    families: tuple[str, ...] = ("media", "story", "weather"),
    seed_all: bool = False,
) -> list[dict[str, Any]]:
    reserved, policy = _controlled_lexemes()
    candidates: list[dict[str, Any]] = []
    definitions = _router_family_definitions()
    for family in families:
        try:
            definition = definitions[family]
        except KeyError as exc:
            raise ValueError(f"unknown legacy router vocabulary family: {family}") from exc
        candidates.extend(
            _candidate(
                lemma=lemma,
                pos="noun",
                semantic_class_id=LEGACY_ROUTER_TERM_CLASSES[lemma],
                definition=f"legacy router {family} vocabulary: {lemma}",
                source_ref=f"local_assistant_router.{family}:{lemma}",
                reserved=lemma in reserved,
                policy_overlap=lemma in policy,
            )
            for lemma in definition["required_terms"]
        )
    if seed_all:
        seen = {c["lemma"] for c in candidates}
        candidates.extend(
            _candidate(
                lemma=lemma,
                pos="noun",
                semantic_class_id=semantic_class,
                definition=f"legacy router global vocabulary: {lemma}",
                source_ref=f"local_assistant_router.legacy_vocabulary:{lemma}",
                reserved=lemma in reserved,
                policy_overlap=lemma in policy,
            )
            for lemma, semantic_class in sorted(LEGACY_ROUTER_TERM_CLASSES.items())
            if lemma not in seen
        )
    return candidates


def write_legacy_lexicon_candidates(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(candidate, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for candidate in (
            build_legacy_lexicon_candidates() + build_legacy_router_candidates(seed_all=True)
        )
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _candidate(
    *,
    lemma: str,
    pos: str,
    semantic_class_id: str,
    definition: str,
    source_ref: str,
    reserved: bool,
    policy_overlap: bool,
) -> dict[str, Any]:
    return {
        "schema_id": "melm.sense_candidate.v1",
        "batch_id": "legacy_functional_grammar_v1",
        "lemma": lemma,
        "language": "en",
        "pos": pos,
        "source": {
            "provenance": "seed_authored",
            "source_ref": source_ref,
            "license": "MELM project source",
        },
        "definition": definition,
        "semantic_class_candidates": [
            {
                "class_id": semantic_class_id,
                "method": "seed_authored",
                "confidence": 0.95,
            }
        ],
        "forms": [],
        "relations": [],
        "safety": {
            "reserved_conflict": reserved,
            "policy_term_overlap": policy_overlap,
        },
        "suggested_status": "active",
        "confidence_prior": 0.95,
    }


def _controlled_lexemes() -> tuple[set[str], set[str]]:
    payload = load_contract_json("reserved_lexemes.v1.json")
    validate_reserved_lexemes(payload)
    return set(payload["lexemes"]), set(payload["policy_lexemes"])


def build_legacy_in_memory_lexicon() -> dict[str, frozenset[str]]:
    """Build the in-memory term→classes cache from LEGACY_ROUTER_TERM_CLASSES.

    Returns a dict mapping each term to a frozenset of its single semantic
    class ID. (Each legacy term maps to exactly one class.)
    """
    return {
        term: frozenset([cls])
        for term, cls in LEGACY_ROUTER_TERM_CLASSES.items()
    }


def _router_family_definitions() -> dict[str, dict[str, list[str]]]:
    payload = load_contract_json("router_lexicon_families.v1.json")
    validate_router_lexicon_families(payload)
    return {
        str(family): {
            "required_terms": list(definition["required_terms"]),
            "allowed_classes": list(definition["allowed_classes"]),
        }
        for family, definition in payload["families"].items()
    }
