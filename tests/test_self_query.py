"""Slice 6: self-query reasoner (conscious / where / feeling), answered locally."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile
from melm.appliance.reasoning.task_router import detect_reasoning_task


class SelfQueryDetectionTests(unittest.TestCase):
    def test_consciousness(self):
        self.assertEqual(detect_reasoning_task("Are you conscious?"),
                         {"task": "self_query", "category": "consciousness"})

    def test_sentient_alive(self):
        self.assertEqual(detect_reasoning_task("Are you sentient?")["category"], "consciousness")
        self.assertEqual(detect_reasoning_task("Are you alive?")["category"], "consciousness")

    def test_location(self):
        self.assertEqual(detect_reasoning_task("Where are you right now?")["category"], "location")

    def test_feeling(self):
        self.assertEqual(detect_reasoning_task("How do you feel?")["category"], "feeling")
        self.assertEqual(detect_reasoning_task("How are you feeling?")["category"], "feeling")

    def test_dated_name_before_runtime(self):
        self.assertEqual(
            detect_reasoning_task("What was your name on 1st June 2002?"),
            {
                "task": "self_query",
                "category": "dated_name",
                "date": "2002-06-01",
            },
        )

    def test_what_are_you_not_self_query(self):
        # Identity question — handled by assistant_identity, not self_query.
        self.assertIsNone(detect_reasoning_task("What are you?"))

    def test_third_person_alive_not_self_query(self):
        self.assertIsNone(detect_reasoning_task("Is the project still alive?"))


class SelfQueryKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_conscious_answered_locally(self):
        d = self.kernel.handle("Are you conscious?")
        self.assertEqual(d.intent, "reasoning:self_query")
        self.assertEqual(d.route, "local_answer")
        self.assertFalse(d.cloud_needed)
        self.assertIn("not conscious", d.answer.lower())

    def test_where_answered_locally(self):
        d = self.kernel.handle("Where are you right now?")
        self.assertEqual(d.intent, "reasoning:self_query")
        self.assertIn("device", d.answer.lower())

    def test_feeling_grounded_in_mood(self):
        d = self.kernel.handle("How do you feel?")
        self.assertEqual(d.intent, "reasoning:self_query")
        self.assertIn("operating state", d.answer.lower())

    def test_what_are_you_still_identity(self):
        d = self.kernel.handle("What are you?")
        self.assertEqual(d.intent, "assistant_identity")

    def test_dated_name_before_runtime_says_not_created_yet(self):
        d = self.kernel.handle("What was your name on 1st June 2002?")
        self.assertEqual(d.intent, "reasoning:self_query")
        self.assertIn("did not exist", d.answer.lower())
        self.assertIn("2002", d.answer)


if __name__ == "__main__":
    unittest.main()
