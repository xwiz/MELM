"""Fixes: informal-token affect wiring + complaint-not-greeting routing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import (
    LocalAssistantProfile,
    _aggregate_informal_affect,
)


class AggregateInformalAffectTests(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_aggregate_informal_affect(()))

    def test_aggregates_valence_and_source(self):
        sig = _aggregate_informal_affect((
            {"valence": 0.2, "arousal": 0.3, "tags": ["amused"], "confidence": 0.4},
            {"valence": 0.4, "arousal": 0.5, "tags": ["positive"], "confidence": 0.4},
        ))
        self.assertIsNotNone(sig)
        self.assertAlmostEqual(sig.valence, 0.3, places=3)
        self.assertEqual(sig.source, "informal")


class InformalAffectKernelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=self.store)

    def tearDown(self):
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _affect(self, utterance):
        return getattr(self.kernel.handle(utterance), "utterance_affect", None)

    def test_haha_registers_positive_mood(self):
        a = self._affect("haha")
        self.assertIsNotNone(a)
        self.assertGreater(a.valence, 0.0)
        self.assertEqual(a.source, "informal")

    def test_ugh_registers_negative_mood(self):
        a = self._affect("ugh")
        self.assertIsNotNone(a)
        self.assertLess(a.valence, 0.0)

    def test_lexicon_affect_wins_over_informal(self):
        # "great" (lexicon) should dominate the stripped "lol".
        a = self._affect("lol that was great")
        self.assertEqual(a.source, "lexicon")
        self.assertGreater(a.valence, 0.3)


class ComplaintRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=self.store)

    def tearDown(self):
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_insult_routes_to_assistant_behavior_not_greeting(self):
        d = self.kernel.handle("you are useless")
        self.assertEqual(d.intent, "assistant_behavior")
        self.assertEqual(d.reason, "complaint_acknowledged")
        self.assertNotIn("what would you like help with", d.answer.lower())
        self.assertIn("went wrong", d.answer.lower())

    def test_complaint_survives_trailing_informal_token(self):
        d = self.kernel.handle("you are useless lol")
        self.assertEqual(d.intent, "assistant_behavior")
        self.assertEqual(d.reason, "complaint_acknowledged")

    def test_genuine_greeting_still_greets(self):
        d = self.kernel.handle("hello")
        self.assertEqual(d.intent, "social_greeting")

    def test_meal_question_unaffected(self):
        d = self.kernel.handle("what should I eat today?")
        self.assertEqual(d.intent, "meal_suggestion")


if __name__ == "__main__":
    unittest.main()
