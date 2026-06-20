"""Slice 10: end-to-end regression lock + anti-hallucination/refusal probes.

Replaces the manual _stress_reasoning.py driver with assertions over the kernel,
covering every headline case across the four reasoning dimensions, plus refusal
behaviour and an O(1) per-turn counter check.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile


def _kernel():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = AssistantOSStore(tmp.name)
    seed_class_schemas(store)
    return AssistantOSKernel(profile=LocalAssistantProfile(), store=store), store, tmp.name


class HeadlineCasesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel, self.store, self.path = _kernel()

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.path).unlink(missing_ok=True)

    def _ans(self, utterance: str):
        return self.kernel.handle(utterance)

    def test_decomposition(self):
        self.assertIn("3", self._ans("How many r's in strawberry?").answer)
        self.assertIn("3", self._ans("How many letter a in banana?").answer)

    def test_arithmetic_and_meal_preserved(self):
        self.assertIn("2", self._ans("I have 3 apples and eat one, how many are left?").answer)
        self.assertEqual(self._ans("What should I eat today?").intent, "meal_suggestion")

    def test_self_awareness(self):
        self.assertIn("not conscious", self._ans("Are you conscious?").answer.lower())
        self.assertIn("device", self._ans("Where are you right now?").answer.lower())
        self.assertEqual(self._ans("What are you?").intent, "assistant_identity")

    def test_temporal(self):
        with mock.patch("melm.appliance.reasoning.clock.now", return_value=datetime(2026, 6, 19, 14, 30)):
            self.assertIn("It is", self._ans("What time is it?").answer)
            self.assertIn("Monday", self._ans("What day will it be in 3 days?").answer)

    def test_geo_decision_entity_purpose(self):
        d = self._ans("The car wash is 50m away, should I drive or walk?")
        self.assertEqual(d.intent, "reasoning:geo_decision")
        self.assertIn("drive", d.answer.lower())

    def test_no_cloud_for_reasoning(self):
        for q in ("How many r's in strawberry?", "Are you conscious?",
                  "The car wash is 50m away, should I drive or walk?"):
            self.assertFalse(self._ans(q).cloud_needed, q)


class RefusalProbeTests(unittest.TestCase):
    """No fabrication: impossible/missing facts refuse rather than invent."""

    def setUp(self) -> None:
        self.kernel, self.store, self.path = _kernel()

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.path).unlink(missing_ok=True)

    def test_negative_quantity_refuses_not_fabricates(self):
        d = self.kernel.handle("I have 1 apple and eat three, how many are left?")
        self.assertEqual(d.intent, "reasoning:quantity_arithmetic")
        self.assertEqual(d.refusal_signal, "negative_quantity")
        self.assertNotIn("-2", d.answer)

    def test_projection_flags_assumption(self):
        with mock.patch("melm.appliance.reasoning.clock.now", return_value=datetime(2026, 6, 18, 10, 0)):
            self.kernel.handle(
                "You're in Enugu, moving to Lagos on saturday, then Ikeja, then Nsukka.")
            d = self.kernel.handle("Where will you be tomorrow at about 4pm?")
        self.assertIn("assuming", d.answer.lower())


class HotPathCounterTests(unittest.TestCase):
    """Per-turn occurrence uses the O(1) in-memory tally, correct at scale."""

    def setUp(self) -> None:
        self.kernel, self.store, self.path = _kernel()

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.path).unlink(missing_ok=True)

    def test_occurrence_correct_after_many_turns(self):
        last = None
        for _ in range(60):
            last = self.kernel.handle("How many r's in strawberry?")
        self.assertEqual(last.intent_occurrence, 60)
        self.assertIn("3", last.answer)


if __name__ == "__main__":
    unittest.main()
