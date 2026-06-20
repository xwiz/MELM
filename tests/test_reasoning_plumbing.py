"""Slice 4: reasoning plumbing — evidence-bound results + typed refusal."""

from __future__ import annotations

import unittest

import melm.appliance.local_assistant_router as lar
from melm.appliance.assistant_mood_engine import MoodState
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.assistant_synthesis import BoundedLocalSynthesizer
from melm.appliance.local_assistant_router import AssistantDecision, LocalAssistantProfile


def _synth(store=None):
    return BoundedLocalSynthesizer(LocalAssistantProfile(), store=store)


class ReasoningRenderTests(unittest.TestCase):
    def test_reasoning_result_renders_and_binds_evidence(self):
        d = AssistantDecision(
            utterance="how many r in strawberry?", intent="reasoning:metalinguistic_count",
            route="local_answer", answer="There are 3 r's in 'strawberry'.",
            reasoning_result={"task": "metalinguistic_count", "count": 3},
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertFalse(res.refused)
        self.assertEqual(res.answer, "There are 3 r's in 'strawberry'.")
        self.assertIn("reasoning.result", res.citations)
        self.assertEqual(res.reason, "reasoning:metalinguistic_count")

    def test_reasoning_bypasses_no_bound_evidence_with_empty_keys(self):
        d = AssistantDecision(
            utterance="x", intent="reasoning:quantity_arithmetic", route="local_answer",
            answer="2 apples.", reasoning_result={"task": "quantity_arithmetic", "value": 2},
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertNotEqual(res.reason, "bounded_synthesis:no_bound_evidence")

    def test_refusal_renders_clarification(self):
        d = AssistantDecision(
            utterance="how long?", intent="reasoning:itinerary", route="local_answer",
            answer="Which day do you start?", refusal_signal="missing_start_time",
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertEqual(res.answer, "Which day do you start?")
        self.assertIn("reasoning.refusal", res.citations)
        self.assertEqual(res.reason, "reasoning_refusal:missing_start_time")


class ReasoningBehaviorProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = lar._CAPABILITY_PAYLOAD
        lar._CAPABILITY_PAYLOAD = {
            "families": {"mood_affect": {"installed": True, "creative_behaviors": True}}
        }
        self.store = AssistantOSStore(":memory:")
        seed_class_schemas(self.store)

    def tearDown(self) -> None:
        lar._CAPABILITY_PAYLOAD = self._saved
        self.store.connection.close()

    def test_behaviors_never_replace_reasoning_answer(self):
        # ambient_mood_narrative would replace_answer; protect must block it.
        d = AssistantDecision(
            utterance="hi", intent="social_greeting", route="local_answer",
            answer="2 apples.",
            reasoning_result={"task": "quantity_arithmetic", "value": 2},
            intent_occurrence=1, ambient_valence_delta=-0.3,
            prev_mood=MoodState(mood_id="annoyed", valence=-0.4),
            session_mood=MoodState(mood_id="neutral"),
        )
        res = _synth(self.store).synthesize(d, boundary_crossed="none", membrane_allowed=True)
        # The faithful solver answer survives; not overwritten by the greeting template.
        self.assertIn("2 apples.", res.answer)


if __name__ == "__main__":
    unittest.main()
