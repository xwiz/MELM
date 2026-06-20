"""Tests for knowledge typing (MVP3 fact-negation layer)."""

import json
from pathlib import Path

from melm.appliance.assistant_knowledge import classify_knowledge, extract_proposition
from melm.appliance.assistant_os_store import AssistantOSStore
from melm.appliance.assistant_os_store import seed_class_schemas


# ── classify_knowledge tests ──────────────────────────────────────────────

def test_classify_knowledge_claim_is_a():
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "be", "lemma": "be"},
            "roles": [
                {"role": "subject", "value": "Abuja"},
                {"role": "theme", "value": "capital"},
            ],
            "context": {"polarity": "positive"},
        }],
    }
    assert classify_knowledge(uol, "Abuja is the capital") == "static_fact"


def test_classify_knowledge_negated():
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "be"},
            "roles": [
                {"role": "subject", "value": "Abuja"},
                {"role": "theme", "value": "capital"},
            ],
            "context": {"polarity": "negative", "negation_scope": True},
        }],
    }
    assert classify_knowledge(uol, "Abuja is not the capital") == "negated_fact"


def test_classify_knowledge_opinion():
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "be"},
            "roles": [
                {"role": "subject", "value": "Jollof"},
                {"role": "theme", "value": "best food"},
            ],
            "context": {"polarity": "positive"},
        }],
    }
    assert classify_knowledge(uol, "Jollof is the best food") == "opinion"


def test_classify_knowledge_question_ignored():
    uol = {"act": "question", "content": [{"predicate": {"id": "be"}, "roles": [], "context": {}}]}
    assert classify_knowledge(uol, "what is") is None


def test_classify_knowledge_personal_ignored():
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "be"},
            "roles": [{"role": "subject", "value": "I"}],
            "context": {"polarity": "positive"},
        }],
    }
    assert classify_knowledge(uol, "I am tired") is None


def test_classify_knowledge_literary_device():
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "have"},
            "roles": [
                {"role": "subject", "value": "key"},
                {"role": "theme", "value": "locks"},
            ],
            "context": {"polarity": "positive"},
        }],
    }
    assert classify_knowledge(uol, "What has keys but can't open locks?") == "literary_device"


def test_extract_proposition():
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "be", "lemma": "be"},
            "roles": [
                {"role": "subject", "value": "Abuja"},
                {"role": "theme", "value": "capital"},
            ],
            "context": {"polarity": "positive"},
        }],
    }
    prop = extract_proposition(uol)
    assert prop is not None
    assert prop["subject"] == "abuja"
    assert prop["relation"] == "is_a"
    assert prop["object"] == "capital"


def test_extract_proposition_no_roles():
    uol = {"act": "claim", "content": [{"predicate": {"id": "be"}, "roles": [], "context": {}}]}
    assert extract_proposition(uol) is None


# ── Store method tests ────────────────────────────────────────────────────

def _make_store(tmp_path):
    db = tmp_path / "test_knowledge_store.db"
    store = AssistantOSStore(db)
    seed_class_schemas(store)
    return store


def test_set_world_fact_creates_entity(tmp_path):
    store = _make_store(tmp_path)
    eid = "wf_test_001"
    store.set_world_fact(eid, "abuja", "is_a", "capital", "asserted", "user", 0.6)
    entity = store.get_entity(eid)
    assert entity is not None
    assert entity.kind == "world_fact"
    assert entity.semantic_class_id == "world_fact"


def test_set_world_fact_stores_slots(tmp_path):
    store = _make_store(tmp_path)
    eid = "wf_test_002"
    store.set_world_fact(eid, "abuja", "is_a", "capital", "asserted", "user", 0.6, "Abuja is capital")
    slot = store.get_entity_slot(eid, "subject")
    assert slot is not None
    assert json.loads(slot.value_json) == "abuja"
    slot = store.get_entity_slot(eid, "polarity")
    assert json.loads(slot.value_json) == "asserted"
    slot = store.get_entity_slot(eid, "source_utterance")
    assert json.loads(slot.value_json) == "Abuja is capital"


def test_query_world_fact_matches_subject_relation(tmp_path):
    store = _make_store(tmp_path)
    store.set_world_fact("wf_q1", "abuja", "is_a", "capital", "asserted", "user", 0.6)
    store.set_world_fact("wf_q2", "lagos", "is_a", "city", "asserted", "user", 0.6)
    results = store.query_world_fact("abuja", "is_a")
    assert len(results) == 1, f"Expected 1, got {len(results)}: {results}"
    assert results[0].get("subject") == "abuja"
    results2 = store.query_world_fact("lagos", "is_a")
    assert len(results2) == 1
    assert results2[0].get("subject") == "lagos"


def test_set_world_fact_empty_source(tmp_path):
    store = _make_store(tmp_path)
    eid = "wf_test_003"
    store.set_world_fact(eid, "test", "is_a", "fact", "asserted", "user", 0.5)
    slot = store.get_entity_slot(eid, "source_utterance")
    assert slot is not None
    assert json.loads(slot.value_json) == ""


def test_find_contradicting_facts_finds_opposite(tmp_path):
    store = _make_store(tmp_path)
    store.set_world_fact("wf_c1", "abuja", "is_a", "capital", "asserted", "user", 0.6)
    store.set_world_fact("wf_c2", "abuja", "is_a", "capital", "negated", "user", 0.4)
    contradictions = store.find_contradicting_facts("abuja", "is_a", "capital", "asserted")
    assert len(contradictions) >= 1
    assert contradictions[0]["polarity"] == "negated"


def test_find_contradicting_facts_no_match(tmp_path):
    store = _make_store(tmp_path)
    store.set_world_fact("wf_nc1", "abuja", "is_a", "capital", "asserted", "user", 0.6)
    contradictions = store.find_contradicting_facts("abuja", "is_a", "capital", "asserted")
    assert len(contradictions) == 0
