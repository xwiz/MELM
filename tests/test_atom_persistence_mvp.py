"""Tests for UOL atom persistence (T1→T2 aggregation)."""

import json
from pathlib import Path

from melm.appliance.assistant_atom_persistence import record_uol_parse
from melm.appliance.assistant_os_store import AssistantOSStore
from melm.appliance.assistant_os_store import seed_class_schemas


def _make_store(tmp_path):
    db = tmp_path / "test_atom_persist.db"
    store = AssistantOSStore(db)
    seed_class_schemas(store)
    return store


def _make_uol_act():
    return {
        "act": "request",
        "content": [
            {
                "kind": "state",
                "predicate": {"id": "play", "lemma": "play", "semantic_class": "action.play"},
                "roles": [
                    {"role": "agent", "value": "user"},
                    {"role": "theme", "value": "music"},
                ],
                "context": {"polarity": "positive", "modality": "assertive"},
            }
        ],
    }


def test_record_uol_parse_creates_entity(tmp_path):
    store = _make_store(tmp_path)
    entity_id = record_uol_parse(store, "e_test_001", _make_uol_act())
    assert entity_id is not None
    entity = store.get_entity(entity_id)
    assert entity is not None
    assert entity.kind == "uol_parse"


def test_record_uol_parse_stores_content(tmp_path):
    store = _make_store(tmp_path)
    entity_id = record_uol_parse(store, "e_test_002", _make_uol_act())
    slot = store.get_entity_slot(entity_id, "uol_json")
    assert slot is not None
    stored = json.loads(slot.value_json)
    assert stored["content"][0]["predicate"]["id"] == "play"


def test_record_uol_parse_links_to_event(tmp_path):
    store = _make_store(tmp_path)
    entity_id = record_uol_parse(store, "e_test_003", _make_uol_act())
    slot = store.get_entity_slot(entity_id, "event_id")
    assert slot is not None
    assert json.loads(slot.value_json) == "e_test_003"


def test_record_uol_parse_none_safe(tmp_path):
    store = _make_store(tmp_path)
    assert record_uol_parse(store, "e_test_004", None) is None


def test_record_uol_parse_empty_content(tmp_path):
    store = _make_store(tmp_path)
    assert record_uol_parse(store, "e_test_005", {"act": "unknown", "content": []}) is None


def test_seed_class_schemas_includes_uol_parse(tmp_path):
    store = _make_store(tmp_path)
    row = store.connection.execute(
        "SELECT semantic_class_id FROM class_schemas WHERE semantic_class_id='uol_parse'"
    ).fetchone()
    assert row is not None
