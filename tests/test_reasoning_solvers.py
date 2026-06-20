"""Slice 5: value extraction, task detection, solvers, and end-to-end fixes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile
from melm.appliance.reasoning.task_router import detect_reasoning_task
from melm.appliance.reasoning.solvers import solve


class TaskDetectionTests(unittest.TestCase):
    def test_metalinguistic_detected(self):
        t = detect_reasoning_task("How many r's in strawberry?")
        self.assertEqual(t, {"task": "metalinguistic_count", "char": "r", "word": "strawberry"})

    def test_letter_phrasing_detected(self):
        t = detect_reasoning_task("How many letter a in banana?")
        self.assertEqual(t["char"], "a")
        self.assertEqual(t["word"], "banana")

    def test_arithmetic_detected(self):
        t = detect_reasoning_task("I have 3 apples and eat one, how many are left?")
        self.assertEqual(t["task"], "quantity_arithmetic")
        self.assertEqual(t["start"], 3.0)
        self.assertEqual(t["delta"], 1.0)
        self.assertEqual(t["sign"], -1)
        self.assertEqual(t["noun"], "apples")

    def test_plain_meal_question_not_a_task(self):
        self.assertIsNone(detect_reasoning_task("What should I eat today?"))

    def test_non_count_question_not_a_task(self):
        self.assertIsNone(detect_reasoning_task("Tell me a story."))


class SolverTests(unittest.TestCase):
    def test_metalinguistic_count(self):
        result, answer, refusal = solve({"task": "metalinguistic_count", "char": "r", "word": "strawberry"})
        self.assertIsNone(refusal)
        self.assertEqual(result["count"], 3)
        self.assertIn("3", answer)
        self.assertIn("strawberry", answer)

    def test_banana_a_count(self):
        result, _, _ = solve({"task": "metalinguistic_count", "char": "a", "word": "banana"})
        self.assertEqual(result["count"], 3)

    def test_arithmetic(self):
        result, answer, refusal = solve(
            {"task": "quantity_arithmetic", "start": 3.0, "delta": 1.0, "sign": -1, "noun": "apples"})
        self.assertIsNone(refusal)
        self.assertEqual(result["value"], 2)
        self.assertEqual(answer, "2 apples.")

    def test_arithmetic_negative_refuses(self):
        result, _, refusal = solve(
            {"task": "quantity_arithmetic", "start": 1.0, "delta": 3.0, "sign": -1, "noun": "apples"})
        self.assertIsNone(result)
        self.assertEqual(refusal, "negative_quantity")


class EndToEndKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_strawberry(self):
        d = self.kernel.handle("How many r's in strawberry?")
        self.assertEqual(d.intent, "reasoning:metalinguistic_count")
        self.assertEqual(d.route, "local_answer")
        self.assertFalse(d.cloud_needed)
        self.assertIn("3", d.answer)

    def test_apples_not_meal_misroute(self):
        d = self.kernel.handle("I have 3 apples and eat one, how many are left?")
        self.assertEqual(d.intent, "reasoning:quantity_arithmetic")
        self.assertNotEqual(d.intent, "meal_suggestion")
        self.assertIn("2", d.answer)

    def test_meal_question_still_routes_to_meal(self):
        d = self.kernel.handle("What should I eat today?")
        self.assertEqual(d.intent, "meal_suggestion")


if __name__ == "__main__":
    unittest.main()
