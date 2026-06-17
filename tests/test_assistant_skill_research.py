"""Tests for assistant_skill_research — open-domain topic extraction and learned-fact management."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.assistant_skill_research import (
    StubResearchProvider,
    extract_action,
    extract_topic,
    find_learned_fact,
    format_open_domain_answer,
    learn_topic,
    record_learned_fact,
)
from melm.appliance.functional_grammar import FunctionalParse


class ExtractTopicTests(unittest.TestCase):
    def test_extracts_object(self):
        parse = FunctionalParse(
            speech_act="wh_question",
            subject="",
            action="tell",
            object="mars",
            target="",
            complement_action="",
            indirect_object="",
            modifiers={},
            relations=(),
            token_roles=(),
            candidates=(),
            parse_score=0.0,
            syntactic_coverage=0.0,
            semantic_unknown_tokens=(),
            pattern="",
        )
        self.assertEqual(extract_topic(parse), "mars")

    def test_extracts_from_dict(self):
        parse = {
            "speech_act": "wh_question",
            "subject": "",
            "action": "tell",
            "object": "mars",
            "target": "",
            "complement_action": "",
            "semantic_unknown_tokens": [],
        }
        self.assertEqual(extract_topic(parse), "mars")

    def test_first_match_priority(self):
        parse = {
            "object": "dog",
            "target": "golden retriever",
            "semantic_unknown_tokens": [],
        }
        # object has priority over target, so "dog" wins even though target is longer
        self.assertEqual(extract_topic(parse), "dog")

    def test_returns_none_for_empty(self):
        self.assertIsNone(extract_topic(None))
        self.assertIsNone(extract_topic({}))


class ExtractActionTests(unittest.TestCase):
    def test_extracts_action(self):
        parse = {"action": "explain"}
        self.assertEqual(extract_action(parse), "explain")

    def test_fallback(self):
        self.assertEqual(extract_action(None), "learn about")
        self.assertEqual(extract_action({}), "learn about")


class LearnedFactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_record_and_find(self):
        entity_id = record_learned_fact(self.store, "Mars", "Mars is the fourth planet.", source="test")
        self.assertTrue(entity_id.startswith("learned_fact:mars:"))
        found = find_learned_fact(self.store, "Mars")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["topic"], "Mars")
        self.assertEqual(found["summary"], "Mars is the fourth planet.")
        self.assertEqual(found["source"], "test")

    def test_find_no_match(self):
        self.assertIsNone(find_learned_fact(self.store, "Venus"))

    def test_find_partial_match(self):
        record_learned_fact(self.store, "solar system", "The solar system has 8 planets.")
        found = find_learned_fact(self.store, "solar")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["topic"], "solar system")


class FormatOpenDomainAnswerTests(unittest.TestCase):
    def test_learned_fact_template(self):
        fact = {"summary": "Mars is red."}
        answer = format_open_domain_answer("Mars", learned_fact=fact)
        self.assertIn("Mars", answer)
        self.assertIn("Mars is red.", answer)

    def test_handoff_template(self):
        answer = format_open_domain_answer("Mars")
        self.assertIn("Mars", answer)


class StubResearchProviderTests(unittest.TestCase):
    def test_found(self):
        p = StubResearchProvider(canned={"mars": "Mars is red."})
        result = p.research("Mars")
        self.assertTrue(result.found)
        self.assertEqual(result.summary, "Mars is red.")

    def test_not_found(self):
        p = StubResearchProvider(canned={})
        result = p.research("Venus")
        self.assertFalse(result.found)
        self.assertEqual(result.summary, "")


class LearnTopicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_learns_and_stores(self):
        provider = StubResearchProvider(canned={"mars": "Mars is the fourth planet."})
        result = learn_topic(self.store, "Mars", provider)
        self.assertTrue(result.found)
        self.assertEqual(result.summary, "Mars is the fourth planet.")
        # Verify stored
        found = find_learned_fact(self.store, "Mars")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["summary"], "Mars is the fourth planet.")

    def test_idempotent_skips_provider(self):
        provider = StubResearchProvider(canned={"mars": "Original."})
        learn_topic(self.store, "Mars", provider)
        provider.canned = {"mars": "CHANGED"}
        result = learn_topic(self.store, "Mars", provider)
        self.assertEqual(result.summary, "Original.")

    def test_no_store_returns_provider_result(self):
        provider = StubResearchProvider(canned={"mars": "Mars."})
        result = learn_topic(None, "Mars", provider)
        # When store is None, provider result is returned directly without storage
        self.assertTrue(result.found)
        self.assertEqual(result.summary, "Mars.")


if __name__ == "__main__":
    unittest.main()
