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

LEGACY_ROUTER_TERM_CLASSES = {
    "audio": "media_content",
    "fable": "narrative_content",
    "fables": "narrative_content",
    "forecast": "weather_phenomenon",
    "lofi": "media_descriptor",
    "music": "media_content",
    "piano": "physical_object.instrument",
    "radio": "physical_object.media_source",
    "rain": "weather_phenomenon",
    "song": "media_content",
    "sound": "media_content",
    "sounds": "media_content",
    "stories": "narrative_content",
    "story": "narrative_content",
    "tale": "narrative_content",
    "tales": "narrative_content",
    "temperature": "weather_phenomenon",
    "track": "media_content",
    "weather": "weather_phenomenon",
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
    return candidates


def write_legacy_lexicon_candidates(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(candidate, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for candidate in (
            build_legacy_lexicon_candidates() + build_legacy_router_candidates()
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
