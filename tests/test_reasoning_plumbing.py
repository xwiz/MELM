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
            reasoning_result={"task": "metalinguistic_count", "char": "r",
                              "word": "strawberry", "count": 3,
                              "count_word": "3", "plural": "s"},
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertFalse(res.refused)
        self.assertEqual(res.answer, 'There are 3 "r"s in "strawberry".')
        self.assertIn("reasoning.result", res.citations)
        self.assertEqual(res.reason, "reasoning:metalinguistic_count")

    def test_reasoning_bypasses_no_bound_evidence_with_empty_keys(self):
        d = AssistantDecision(
            utterance="x", intent="reasoning:quantity_arithmetic", route="local_answer",
            answer="2 apples.", reasoning_result={"task": "quantity_arithmetic", "value": 2,
                                                  "value_str": "2", "noun": "apples"},
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertEqual(res.answer, "2 apples.")
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
            reasoning_result={"task": "quantity_arithmetic", "value": 2,
                              "value_str": "2", "noun": "apples"},
            intent_occurrence=1, ambient_valence_delta=-0.3,
            prev_mood=MoodState(mood_id="annoyed", valence=-0.4),
            session_mood=MoodState(mood_id="neutral"),
        )
        res = _synth(self.store).synthesize(d, boundary_crossed="none", membrane_allowed=True)
        # The faithful solver answer survives; not overwritten by the greeting template.
        self.assertIn("2 apples.", res.answer)




class CausalReasoningSynthesisTests(unittest.TestCase):
    def test_causal_explanation_synthesis_preserves_answer(self):
        d = AssistantDecision(
            utterance="Why is the ground wet?", intent="reasoning:causal_explanation",
            route="local_answer",
            answer="A likely cause is rain.",
            reasoning_result={"task": "causal_explanation", "effect": "wet",
                              "selected_cause": "rain", "candidate_causes": [],
                              "state_definition": {"definition": "covered with water"}},
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertIn("rain", res.answer)
        self.assertEqual(res.reason, "reasoning:causal_explanation")

    def test_causal_prediction_synthesis_preserves_answer(self):
        d = AssistantDecision(
            utterance="What happens if it rains?", intent="reasoning:causal_prediction",
            route="local_answer",
            answer="Likely effects include wet, cooler.",
            reasoning_result={"task": "causal_prediction", "cause": "rain",
                              "effects": [{"state": "wet"}, {"state": "cooler"}]},
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertIn("wet", res.answer)
        self.assertEqual(res.reason, "reasoning:causal_prediction")

    def test_causal_contrast_synthesis_preserves_answer(self):
        d = AssistantDecision(
            utterance="What happens if I eat vs sleep?", intent="reasoning:causal_contrast",
            route="local_answer",
            answer="If eat happens, satisfied; if sleep happens, rested.",
            reasoning_result={"task": "causal_contrast", "cause_a": "eat", "cause_b": "sleep",
                              "effects_a": ["satisfied"], "effects_b": ["rested"]},
        )
        res = _synth().synthesize(d, boundary_crossed="none", membrane_allowed=True)
        self.assertTrue(res.applied)
        self.assertIn("eat", res.answer)
        self.assertIn("sleep", res.answer)
        self.assertEqual(res.reason, "reasoning:causal_contrast")




class NonCausalNlgRendererTests(unittest.TestCase):
    """Test that non-causal reasoning templates render correctly via nlg.render_reasoning_result."""

    def setUp(self):
        from melm.contracts import load_reasoning_templates
        self.templates = load_reasoning_templates()

    def test_geo_decision_walk(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "geo_decision", "decision": "walk",
                  "distance_text": "0.5 km", "purpose": None, "note": ""}
        rendered = render_reasoning_result(result, self.templates)
        self.assertIsNotNone(rendered)
        self.assertIn("walk", rendered.lower())
        self.assertIn("0.5 km", rendered)

    def test_geo_decision_drive(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "geo_decision", "decision": "drive",
                  "distance_text": "5 km", "purpose": None, "note": ""}
        rendered = render_reasoning_result(result, self.templates)
        self.assertIsNotNone(rendered)
        self.assertIn("drive", rendered.lower())
        self.assertIn("5 km", rendered)

    def test_geo_decision_drive_with_purpose(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "geo_decision", "decision": "drive",
                  "distance_text": "1.2 km", "purpose": "car_wash",
                  "note": "it needs your vehicle"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertIsNotNone(rendered)
        self.assertIn("vehicle", rendered)

    def test_temporal_time(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "temporal", "op": "time", "display": "3:45 PM"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertEqual(rendered, "It is 3:45 PM.")

    def test_temporal_date_today(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "temporal", "op": "date_today", "display": "June 21, 2026"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertEqual(rendered, "Today is June 21, 2026.")

    def test_temporal_absolute_date(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "temporal", "op": "absolute_date",
                  "display": "June 21, 2026", "verb": "is", "weekday": "Sunday"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertEqual(rendered, "June 21, 2026 is a Sunday.")

    def test_temporal_day_offset_ago(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "temporal", "op": "day_offset", "direction": "ago",
                  "magnitude": 3, "unit": "days", "weekday": "Thursday",
                  "date": "June 18, 2026"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertEqual(rendered, "3 days ago it was Thursday (June 18, 2026).")

    def test_temporal_day_offset_from_now(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "temporal", "op": "day_offset", "direction": "from_now",
                  "magnitude": 1, "unit": "day", "weekday": "Monday",
                  "date": "June 22, 2026"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertEqual(rendered, "In 1 day it will be Monday (June 22, 2026).")

    def test_metalinguistic_count(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "metalinguistic_count", "char": "r",
                  "word": "strawberry", "count_word": "3", "plural": "s"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertEqual(rendered, 'There are 3 "r"s in "strawberry".')

    def test_quantity_arithmetic(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "quantity_arithmetic", "value_str": "7", "noun": "apples"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertEqual(rendered, "7 apples.")

    def test_unknown_task_returns_none(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "nonexistent_task", "some": "data"}
        rendered = render_reasoning_result(result, self.templates)
        self.assertIsNone(rendered)


if __name__ == "__main__":
    unittest.main()
