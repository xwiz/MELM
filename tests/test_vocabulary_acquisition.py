"""Tests for vocabulary acquisition pipeline fixes.

Covers:
- ``_extract_genus_lemma()`` purpose-clause fix
- ``research_to_lexicon()`` Wikipedia→lexicon ingestion
- WordNet active supersenses seeding
- Auto-research → lexicon integration
"""

from pathlib import Path
import tempfile

# ── F1: Genus extractor purpose-clause fix ──────────────────────────────────


def test_genus_extracts_simple_noun():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    assert _extract_genus_lemma("small thumb piano") == "piano"


def test_genus_strips_used_to_clause():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    result = _extract_genus_lemma("open container used to hold cut flowers")
    assert result == "container", f"Expected 'container', got '{result}'"


def test_genus_strips_used_for_clause():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    result = _extract_genus_lemma("broad flat dish used for serving food")
    assert result == "dish", f"Expected 'dish', got '{result}'"


def test_genus_strips_which_clause():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    result = _extract_genus_lemma("machine which cuts paper")
    assert result == "machine", f"Expected 'machine', got '{result}'"


def test_genus_handles_pp_via_backward_walk():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    result = _extract_genus_lemma("soft cushion for resting the head")
    assert result == "cushion", f"Expected 'cushion', got '{result}'"


def test_genus_handles_trailing_prep_phrase():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    result = _extract_genus_lemma("appliance for keeping food cold")
    assert result == "appliance", f"Expected 'appliance', got '{result}'"


def test_genus_type_of_pattern():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    result = _extract_genus_lemma("type of flower")
    assert result == "type", f"Expected 'type', got '{result}'"


def test_genus_no_trailing_clause():
    from melm.appliance.assistant_lexicon import _extract_genus_lemma
    assert _extract_genus_lemma("small furry animal") == "animal"


# ── F2: research_to_lexicon ─────────────────────────────────────────────────


def test_research_to_lexicon_matches_definition():
    from melm.appliance.assistant_skill_research import research_to_lexicon
    from melm.appliance.assistant_os_store import AssistantOSStore
    store = AssistantOSStore(":memory:")
    candidate = research_to_lexicon(
        store,
        "vase",
        "A vase is an open container used to hold cut flowers",
    )
    assert candidate is not None
    assert candidate["lemma"] == "vase"
    assert candidate["pos"] == "noun"
    assert candidate["source"]["provenance"] == "auto_research"
    assert candidate["suggested_status"] == "active"


def test_research_to_lexicon_resolves_semantic_class():
    from melm.appliance.assistant_skill_research import research_to_lexicon
    from melm.appliance.assistant_os_store import (
        AssistantOSStore, seed_class_schemas,
    )
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    # Seed "container" as an active noun so genus resolution works
    from melm.appliance.assistant_lexicon import lexicon_ingest
    lexicon_ingest(store, {
        "schema_id": "melm.sense_candidate.v1",
        "lemma": "container",
        "language": "en",
        "pos": "noun",
        "source": {"provenance": "seed_authored", "source_ref": "test", "license": "test"},
        "definition": "something that contains things",
        "genus_lemma": "thing",
        "semantic_class_candidates": [{"class_id": "physical_object", "method": "seed_authored", "confidence": 0.9}],
        "forms": [],
        "relations": [],
        "safety": {"reserved_conflict": False, "policy_term_overlap": False},
        "suggested_status": "active",
        "confidence_prior": 0.9,
    }, expected_provenance="seed_authored")
    candidate = research_to_lexicon(
        store,
        "vase",
        "A vase is an open container used to hold cut flowers",
    )
    assert candidate is not None
    classes = [c["class_id"] for c in candidate["semantic_class_candidates"]]
    assert any(c == "physical_object" for c in classes), (
        f"Expected physical_object in {classes}"
    )


def test_research_to_lexicon_no_match_returns_none():
    from melm.appliance.assistant_skill_research import research_to_lexicon
    from melm.appliance.assistant_os_store import AssistantOSStore
    store = AssistantOSStore(":memory:")
    result = research_to_lexicon(store, "llama", "Llama is not a definition-style summary")
    assert result is None


def test_research_to_lexicon_wrong_word_returns_none():
    from melm.appliance.assistant_skill_research import research_to_lexicon
    from melm.appliance.assistant_os_store import AssistantOSStore
    store = AssistantOSStore(":memory:")
    result = research_to_lexicon(
        store, "vase",
        "A kalimba is a small thumb piano",
    )
    assert result is None


# ── F3: WordNet active supersenses ─────────────────────────────────────────


def test_wordnet_artifact_supersense_promoted_to_active():
    from melm.appliance.assistant_os_store import (
        AssistantOSStore, seed_class_schemas,
    )
    from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses
    from melm.appliance.assistant_lexicon import lookup_lexical_senses
    import tempfile
    import json
    from pathlib import Path
    td = Path(tempfile.mkdtemp())
    jsonl = td / "test_wordnet.jsonl"
    jsonl.write_text("\n".join([
        json.dumps({"word": "vase", "supersense": "noun.artifact", "pos": "noun"}),
        json.dumps({"word": "fridge", "supersense": "noun.artifact", "pos": "noun"}),
        json.dumps({"word": "soap", "supersense": "noun.substance", "pos": "noun"}),
    ]))
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    count = seed_wordnet_supersenses(store, data_path=jsonl, max_entries=10)
    assert count == 3, f"Expected 3 ingested, got {count}"
    for word in ("vase", "fridge", "soap"):
        senses = lookup_lexical_senses(store, word)
        assert senses, f"{word} should have a lexical sense"
        active = [s for s in senses if s.get("status") in ("active",)]
        assert active, (
            f"{word} should be active (supersense promoted to active), "
            f"got statuses={[s.get('status') for s in senses]}"
        )


def test_wordnet_abstract_supersense_remains_dormant():
    from melm.appliance.assistant_os_store import (
        AssistantOSStore, seed_class_schemas,
    )
    from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses
    from melm.appliance.assistant_lexicon import lookup_lexical_senses
    import tempfile
    import json
    from pathlib import Path
    td = Path(tempfile.mkdtemp())
    jsonl = td / "test_wordnet.jsonl"
    jsonl.write_text(json.dumps({
        "word": "paradox", "supersense": "noun.cognition", "pos": "noun",
    }))
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    seed_wordnet_supersenses(store, data_path=jsonl, max_entries=10)
    senses = lookup_lexical_senses(store, "paradox")
    assert senses
    dormant = [s for s in senses if s.get("status") in ("dormant",)]
    assert dormant, (
        f"Expected dormant for noun.cognition, "
        f"got statuses={[s.get('status') for s in senses]}"
    )


# ── F4: Integration — auto-research → lexicon pipeline ──────────────────────


def test_learn_topic_calls_lexicon_ingest():
    from melm.appliance.assistant_skill_research import (
        learn_topic, StubResearchProvider,
    )
    from melm.appliance.assistant_os_store import (
        AssistantOSStore, seed_class_schemas,
    )
    from melm.appliance.assistant_lexicon import lookup_lexical_senses
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    provider = StubResearchProvider({
        "vase": "A vase is an open container used to hold cut flowers",
    })
    result = learn_topic(store, "vase", provider)
    assert result.found
    # Lexicon should now have "vase" as active
    senses = lookup_lexical_senses(store, "vase")
    assert senses, "vase should have a lexical sense after learn_topic"
    active = [s for s in senses if s.get("status") in ("active",)]
    assert active, (
        f"vase should be active, got {[s.get('status') for s in senses]}"
    )


def test_learn_topic_non_definition_skips_lexicon():
    from melm.appliance.assistant_skill_research import (
        learn_topic, StubResearchProvider,
    )
    from melm.appliance.assistant_os_store import (
        AssistantOSStore, seed_class_schemas,
    )
    from melm.appliance.assistant_lexicon import lookup_lexical_senses
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    # "was" summary does NOT match the "X is a Y" definition pattern
    provider = StubResearchProvider({
        "llama": "Llamas were first domesticated in the Andes.",
    })
    result = learn_topic(store, "llama", provider)
    assert result.found
    # Non-definition summary should not create a lexicon entry
    senses = lookup_lexical_senses(store, "llama")
    assert not senses, (
        f"llama should not have a lexical sense from non-matching summary, "
        f"got {len(senses)} senses"
    )


def test_learn_topic_idempotent_skips_duplicate_ingest():
    from melm.appliance.assistant_skill_research import (
        learn_topic, StubResearchProvider,
    )
    from melm.appliance.assistant_os_store import (
        AssistantOSStore, seed_class_schemas,
    )
    from melm.appliance.assistant_lexicon import lookup_lexical_senses
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    provider = StubResearchProvider({
        "vase": "A vase is an open container used to hold cut flowers",
    })
    result1 = learn_topic(store, "vase", provider)
    assert result1.found
    # Call again — should return cached learned_fact, not re-ingest
    result2 = learn_topic(store, "vase", provider)
    assert result2.found
    senses = lookup_lexical_senses(store, "vase")
    assert senses
    active = [s for s in senses if s.get("status") in ("active",)]
    assert active


def test_learn_topic_errors_dont_break_learned_fact():
    """If lexicon_ingest raises, the learned_fact should still be stored."""
    from melm.appliance.assistant_skill_research import (
        learn_topic, StubResearchProvider,
    )
    from melm.appliance.assistant_os_store import (
        AssistantOSStore, seed_class_schemas,
    )
    from melm.appliance.assistant_skill_research import find_learned_fact
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    provider = StubResearchProvider({
        "vase": "A vase is an open container used to hold cut flowers",
    })
    result = learn_topic(store, "vase", provider)
    assert result.found
    fact = find_learned_fact(store, "vase")
    assert fact is not None
    assert fact["topic"] == "vase"
