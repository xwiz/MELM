"""Slice 3: creative behaviors engine + gated synthesis post-processor."""

from __future__ import annotations

import unittest

import melm.appliance.local_assistant_router as lar
from melm.appliance.assistant_behavior_engine import (
    BehaviorContext,
    BehaviorEngine,
    BehaviorResult,
    ConditionEvaluator,
    apply_behaviors,
    build_behavior_context,
)
from melm.appliance.assistant_mood_engine import MoodState
from melm.appliance.assistant_synthesis import BoundedLocalSynthesizer
from melm.appliance.local_assistant_router import AssistantDecision, LocalAssistantProfile


class ConditionEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ev = ConditionEvaluator()

    def test_eq_string(self):
        ctx = BehaviorContext(intent="social_greeting")
        self.assertTrue(self.ev.evaluate("intent == 'social_greeting'", ctx))
        self.assertFalse(self.ev.evaluate("intent == 'weather'", ctx))

    def test_gt_float(self):
        self.assertTrue(self.ev.evaluate("engagement > 0.8", BehaviorContext(engagement=0.9)))
        self.assertFalse(self.ev.evaluate("engagement > 0.8", BehaviorContext(engagement=0.5)))

    def test_in_tuple(self):
        cond = "prev_mood_id in ('hurt','sad','annoyed')"
        self.assertTrue(self.ev.evaluate(cond, BehaviorContext(prev_mood_id="sad")))
        self.assertFalse(self.ev.evaluate(cond, BehaviorContext(prev_mood_id="neutral")))

    def test_not_in_tuple(self):
        cond = "intent NOT IN ('social_greeting','assistant_identity')"
        self.assertTrue(self.ev.evaluate(cond, BehaviorContext(intent="weather")))
        self.assertFalse(self.ev.evaluate(cond, BehaviorContext(intent="social_greeting")))

    def test_abs_and_occurrence_and_chain(self):
        cond = "abs(ambient_valence_delta) > 0.04 AND intent == 'social_greeting' AND occurrence == 1"
        self.assertTrue(self.ev.evaluate(
            cond, BehaviorContext(ambient_valence_delta=-0.1, intent="social_greeting", occurrence=1)))
        self.assertFalse(self.ev.evaluate(
            cond, BehaviorContext(ambient_valence_delta=-0.01, intent="social_greeting", occurrence=1)))

    def test_bare_boolean_var(self):
        self.assertTrue(self.ev.evaluate("prev_affect_has_pain", BehaviorContext(prev_affect_has_pain=True)))
        self.assertFalse(self.ev.evaluate("prev_affect_has_pain", BehaviorContext(prev_affect_has_pain=False)))

    def test_unknown_variable_raises(self):
        with self.assertRaises(ValueError):
            self.ev.evaluate("bogus == 'x'", BehaviorContext())


class BehaviorEngineTests(unittest.TestCase):
    def test_curiosity_fires_and_cooldown_suppresses(self):
        engine = BehaviorEngine()  # real shipped behaviors
        ctx = BehaviorContext(engagement=0.9, intent="weather")
        first = engine.evaluate(ctx)
        ids = {r.behavior_id for r in first}
        self.assertIn("curiosity_follow_up", ids)
        # Cooldown is 5; immediate re-eval suppresses it.
        second = engine.evaluate(ctx)
        self.assertNotIn("curiosity_follow_up", {r.behavior_id for r in second})

    def test_mood_narrative_is_preamble(self):
        engine = BehaviorEngine()
        ctx = BehaviorContext(prev_mood_id="sad", current_mood_id="neutral", intent="weather")
        results = {r.behavior_id: r for r in engine.evaluate(ctx)}
        self.assertIn("mood_narrative", results)
        self.assertTrue(results["mood_narrative"].preamble)
        self.assertIsNone(results["mood_narrative"].replace_answer)

    def test_ambient_narrative_replaces(self):
        engine = BehaviorEngine()
        ctx = BehaviorContext(ambient_valence_delta=-0.2, intent="social_greeting",
                              occurrence=1, prev_mood_id="annoyed")
        results = {r.behavior_id: r for r in engine.evaluate(ctx)}
        self.assertIn("ambient_mood_narrative", results)
        self.assertTrue(results["ambient_mood_narrative"].replace_answer)
        self.assertIn("annoyed", results["ambient_mood_narrative"].replace_answer)

    def test_fatigue_sets_max_words(self):
        engine = BehaviorEngine()
        ctx = BehaviorContext(affect_has_fatigue=True, response_mode="normal", intent="weather")
        results = {r.behavior_id: r for r in engine.evaluate(ctx)}
        self.assertIn("fatigue_pacing", results)
        self.assertEqual(results["fatigue_pacing"].max_words_override, 20)


class ApplyBehaviorsTests(unittest.TestCase):
    def test_preamble_and_postamble(self):
        out = apply_behaviors("MAIN", [
            BehaviorResult("a", preamble="PRE"),
            BehaviorResult("b", postamble="POST"),
        ])
        self.assertEqual(out, "PRE MAIN POST")

    def test_replace_wins(self):
        out = apply_behaviors("MAIN", [BehaviorResult("a", replace_answer="NEW")])
        self.assertEqual(out, "NEW")

    def test_max_words_truncates(self):
        out = apply_behaviors("one two three four five", [BehaviorResult("a", max_words_override=2)])
        self.assertEqual(out, "one two")

    def test_protect_blocks_replace_and_truncate_keeps_postamble(self):
        out = apply_behaviors("SOLVER ANSWER", [
            BehaviorResult("a", replace_answer="NEW", max_words_override=1, postamble="P"),
        ], protect=True)
        self.assertEqual(out, "SOLVER ANSWER P")


class BuildContextTests(unittest.TestCase):
    def test_maps_decision_fields(self):
        d = AssistantDecision(
            utterance="x", intent="social_greeting", route="local_answer", answer="",
            intent_occurrence=3,
            session_mood=MoodState(mood_id="neutral", engagement_level=0.9, response_mode="normal"),
            prev_mood=MoodState(mood_id="sad", valence=-0.5),
            ambient_valence_delta=-0.2,
        )
        ctx = build_behavior_context(d)
        self.assertEqual(ctx.current_mood_id, "neutral")
        self.assertEqual(ctx.prev_mood_id, "sad")
        self.assertEqual(ctx.intent, "social_greeting")
        self.assertEqual(ctx.occurrence, 3)
        self.assertAlmostEqual(ctx.engagement, 0.9)
        self.assertTrue(ctx.prev_affect_has_pain)  # prev valence -0.5 < -0.3


class GatedSynthesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = lar._CAPABILITY_PAYLOAD
        self.synth = BoundedLocalSynthesizer(LocalAssistantProfile(), store=None)
        self.decision = AssistantDecision(
            utterance="hi", intent="weather", route="local_answer", answer="",
            intent_occurrence=1,
            session_mood=MoodState(mood_id="neutral", engagement_level=0.95, response_mode="normal"),
        )

    def tearDown(self) -> None:
        lar._CAPABILITY_PAYLOAD = self._saved

    def _set_flag(self, enabled: bool) -> None:
        lar._CAPABILITY_PAYLOAD = {
            "families": {"mood_affect": {"installed": True, "creative_behaviors": enabled}}
        }

    def test_disabled_default_no_change(self):
        self._set_flag(False)
        self.assertEqual(self.synth._apply_creative_behaviors(self.decision, "ANSWER"), "ANSWER")

    def test_enabled_with_store_applies_postamble(self):
        from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
        store = AssistantOSStore(":memory:")
        seed_class_schemas(store)
        synth = BoundedLocalSynthesizer(LocalAssistantProfile(), store=store)
        self._set_flag(True)
        out = synth._apply_creative_behaviors(self.decision, "ANSWER")
        # curiosity_follow_up (engagement 0.95, intent weather) appends a postamble.
        self.assertTrue(out.startswith("ANSWER"))
        self.assertGreater(len(out), len("ANSWER"))
        store.connection.close()


if __name__ == "__main__":
    unittest.main()
