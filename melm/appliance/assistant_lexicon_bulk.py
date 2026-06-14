"""Bulk lexicon seeders — WordNet supersense and VerbNet adapters.

Each adapter reads a word→class mapping data source and produces
``sense_candidate.v1`` objects for ingestion through ``lexicon_ingest()``.

Data files live under ``melm/contracts/``:
  - ``wn_supersense_map.v1.json``  — supersense → MELM class mapping
  - ``verbnet_map.v1.json``        — VerbNet class → MELM class mapping
  - ``word_supersense_data.v1.jsonl``  — word → supersense entries
  - ``verb_data.v1.jsonl``             — verb → VerbNet-class entries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from melm.contracts import (
    ContractValidationError,
    load_contract_json,
    load_semantic_class_ids,
)

from .assistant_lexicon import _normalize_term, lexicon_ingest
from .assistant_os_store import AssistantOSStore


_WN_SUPERSENSE_MAP: dict[str, str] | None = None
_VERBNET_MAP: dict[str, str] | None = None
_KNOWN_CLASS_IDS: set[str] | None = None


def _wn_supersense_map() -> dict[str, str]:
    global _WN_SUPERSENSE_MAP
    if _WN_SUPERSENSE_MAP is None:
        payload = load_contract_json("wn_supersense_map.v1.json")
        _WN_SUPERSENSE_MAP = dict(payload.get("mappings", {}))
    return _WN_SUPERSENSE_MAP


def _verbnet_map() -> dict[str, str]:
    global _VERBNET_MAP
    if _VERBNET_MAP is None:
        payload = load_contract_json("verbnet_map.v1.json")
        _VERBNET_MAP = dict(payload.get("mappings", {}))
    return _VERBNET_MAP


def _class_ids() -> set[str]:
    global _KNOWN_CLASS_IDS
    if _KNOWN_CLASS_IDS is None:
        _KNOWN_CLASS_IDS = load_semantic_class_ids()
    return _KNOWN_CLASS_IDS


_RESERVED, _POLICY = set(), set()


def _load_controlled() -> None:
    if _RESERVED or _POLICY:
        return
    from .assistant_lexicon import _controlled_lexemes
    _RESERVED.update(_controlled_lexemes()[0])
    _POLICY.update(_controlled_lexemes()[1])


_CONTRACT_ROOT = Path(__file__).resolve().parent.parent / "contracts"


def _candidate(
    lemma: str,
    pos: str,
    class_id: str,
    definition: str,
    source_ref: str,
    provenance: str,
) -> dict[str, Any]:
    _load_controlled()
    lemma_norm = _normalize_term(lemma)
    return {
        "schema_id": "melm.sense_candidate.v1",
        "lemma": lemma,
        "language": "en",
        "pos": pos,
        "source": {
            "provenance": provenance,
            "source_ref": source_ref,
            "license": "public_domain_or_bulk_seeded",
        },
        "definition": definition,
        "semantic_class_candidates": [
            {
                "class_id": class_id,
                "method": "supersense_map" if provenance == "wordnet" else "verbnet_map",
                "confidence": 0.85 if provenance in ("wordnet", "verbnet") else 0.80,
            }
        ],
        "forms": [],
        "relations": [],
        "safety": {
            "reserved_conflict": lemma_norm in _RESERVED,
            "policy_term_overlap": lemma_norm in _POLICY,
        },
        "suggested_status": "dormant",
        "confidence_prior": 0.85 if provenance in ("wordnet", "verbnet") else 0.80,
    }


def _load_word_supersense_data(path: Path | None = None) -> list[dict[str, str]]:
    """Load word→supersense entries from a JSONL file.
    
    Each line: {"word": ..., "supersense": ..., "pos": ...}
    """
    if path is None:
        path = _CONTRACT_ROOT / "word_supersense_data.v1.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "word" in entry and "supersense" in entry:
            entries.append(entry)
    return entries


def _load_verb_data(path: Path | None = None) -> list[dict[str, str]]:
    """Load verb→verbnet-class entries from a JSONL file."""
    if path is None:
        path = _CONTRACT_ROOT / "verb_data.v1.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "verb" in entry and "verbnet_class" in entry:
            entries.append(entry)
    return entries


def _make_wn_definition(word: str, supersense: str) -> str:
    return f"wordnet supersense {supersense}: {word}"


def _make_verbnet_definition(verb: str, vn_class: str) -> str:
    return f"verbnet class {vn_class}: {verb}"


# ── Public API ───────────────────────────────────────────────────────────────


def seed_wordnet_supersenses(
    store: AssistantOSStore,
    *,
    data_path: Path | None = None,
    provenance: str = "wordnet",
) -> int:
    """Seed word→supersense→MELM-class entries through the ingestion gate.
    
    Reads a word→supersense JSONL file, maps each supersense to a MELM
    semantic class via ``wn_supersense_map.v1.json``, and ingests through
    ``lexicon_ingest()`` with ``provenance=wordnet`` and
    ``suggested_status=dormant``.
    
    Returns the count of successfully ingested entries.
    """
    mapping = _wn_supersense_map()
    known = _class_ids()
    entries = _load_word_supersense_data(data_path)
    applied = 0
    for entry in entries:
        word = str(entry["word"]).strip().lower()
        supersense = str(entry["supersense"]).strip().lower()
        pos = str(entry.get("pos", "noun")).strip().lower()
        if not word or not supersense:
            continue
        melm_class = mapping.get(supersense)
        if melm_class is None or melm_class not in known:
            continue
        candidate = _candidate(
            lemma=word,
            pos=pos,
            class_id=melm_class,
            definition=_make_wn_definition(word, supersense),
            source_ref=f"wordnet:supersense:{supersense}:{word}",
            provenance=provenance,
        )
        try:
            lexicon_ingest(store, candidate, expected_provenance=provenance)
            applied += 1
        except ContractValidationError:
            pass
    return applied


def seed_verbnet_classes(
    store: AssistantOSStore,
    *,
    data_path: Path | None = None,
    provenance: str = "verbnet",
) -> int:
    """Seed verb→verbnet-class→MELM-class entries through the ingestion gate.
    
    Reads a verb→verbnet-class JSONL file, maps each VerbNet class to a MELM
    semantic class via ``verbnet_map.v1.json``, and ingests through
    ``lexicon_ingest()`` with ``provenance=verbnet`` and
    ``suggested_status=dormant``.
    """
    mapping = _verbnet_map()
    known = _class_ids()
    entries = _load_verb_data(data_path)
    applied = 0
    for entry in entries:
        verb = str(entry["verb"]).strip().lower()
        vn_class = str(entry["verbnet_class"]).strip()
        pos = str(entry.get("pos", "verb")).strip().lower()
        if not verb or not vn_class:
            continue
        melm_class = mapping.get(vn_class)
        if melm_class is None or melm_class not in known:
            continue
        candidate = _candidate(
            lemma=verb,
            pos=pos,
            class_id=melm_class,
            definition=_make_verbnet_definition(verb, vn_class),
            source_ref=f"verbnet:class:{vn_class}:{verb}",
            provenance=provenance,
        )
        try:
            lexicon_ingest(store, candidate, expected_provenance=provenance)
            applied += 1
        except ContractValidationError:
            pass
    return applied


def seed_bulk_lexicon(
    store: AssistantOSStore,
    *,
    wordnet_data: Path | None = None,
    verbnet_data: Path | None = None,
) -> dict[str, int]:
    """Run all bulk lexicon seeders and return counts per provenance."""
    return {
        "wordnet": seed_wordnet_supersenses(store, data_path=wordnet_data),
        "verbnet": seed_verbnet_classes(store, data_path=verbnet_data),
    }


