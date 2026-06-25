"""Tests for UOL-pattern trigger responses."""

import hashlib
import unittest

from melm.appliance import AssistantOSKernel, BoundedLocalSynthesizer, LocalAssistantProfile
from melm.appliance.assistant_mood_engine import compute_utterance_affect, load_affect_lexicon
from melm.appliance.assistant_skill_uol_triggers import (
    UolTriggerMatch,
    detect_uol_trigger,
    render_trigger_response,
    _seed_for_response,
)
from melm.appliance.local_assistant_router import _build_parse_bundle
from melm.contracts import load_uol_trigger_responses
from melm.contracts.validation import validate_uol_trigger_responses


class UolTriggerResponsesContractTests(unittest.TestCase):
    def test_contract_validates(self) -> None:
        payload = load_uol_trigger_responses()
        validate_uol_trigger_responses(payload)
        self.assertIn("triggers", payload)
        self.assertTrue(payload["triggers"])


class UolTriggerDetectionTests(unittest.TestCase):
    def _match(self, text: str) -> UolTriggerMatch | None:
        bundle = _build_parse_bundle(text)
        lex = load_affect_lexicon()
        affect = compute_utterance_affect(bundle.lemmas, bundle.uol_act, lex)
        return detect_uol_trigger(bundle.uol_act, affect, bundle.tokens)

    def test_negative_assertion_matches(self) -> None:
        match = self._match("You are nameless")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.trigger_id, "assistant_negative_assertion")
        self.assertEqual(match.variables.get("modifier"), "nameless")

    def test_negated_state_matches(self) -> None:
        match = self._match("You are not real")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.trigger_id, "assistant_negative_assertion")
        self.assertEqual(match.variables.get("polarity"), "negative")

    def test_contradiction_matches(self) -> None:
        match = self._match("But you are")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.trigger_id, "assistant_contradiction")

    def test_excluded_insult_does_not_match(self) -> None:
        match = self._match("You are stupid")
        self.assertIsNone(match)

    def test_negative_assertion_with_non_excluded_word(self) -> None:
        match = self._match("You are nameless")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.trigger_id, "assistant_negative_assertion")

    def test_non_targeted_does_not_match(self) -> None:
        match = self._match("The weather is bad")
        self.assertIsNone(match)

    def test_positive_assertion_does_not_match(self) -> None:
        match = self._match("You are helpful")
        self.assertIsNone(match)


class UolTriggerRenderingTests(unittest.TestCase):
    def test_fallback_pool_used_when_no_store(self) -> None:
        match = UolTriggerMatch(
            trigger_id="assistant_negative_assertion",
            fallback_pool=("No I am not {modifier}.", "Stop saying that."),
            variables={"modifier": "nameless"},
        )
        text = render_trigger_response(match, "You are nameless", None, None)
        seed = _seed_for_response("assistant_negative_assertion", "You are nameless", "fallback")
        expected = ("No I am not {modifier}.", "Stop saying that.")[seed % 2]
        self.assertEqual(text, expected)

    def test_generic_template_used_when_no_modifier(self) -> None:
        match = UolTriggerMatch(
            trigger_id="assistant_contradiction",
            fallback_pool=("How dare you?", "Yada yada yada."),
            variables={"agent": "assistant"},
        )
        text = render_trigger_response(match, "But you are", None, None)
        seed = _seed_for_response("assistant_contradiction", "But you are", "fallback")
        expected = ("How dare you?", "Yada yada yada.")[seed % 2]
        self.assertEqual(text, expected)

    def test_response_is_deterministic_per_utterance(self) -> None:
        match = UolTriggerMatch(
            trigger_id="assistant_contradiction",
            fallback_pool=("How dare you?", "Why are you trying to annoy me?", "No."),
            variables={},
        )
        first = render_trigger_response(match, "But you are", None, None)
        second = render_trigger_response(match, "But you are", None, None)
        self.assertEqual(first, second)

    def test_variables_are_filled(self) -> None:
        match = UolTriggerMatch(
            trigger_id="assistant_contradiction",
            fallback_pool=("What {agent}?", "Okay."),
            variables={"agent": "assistant"},
        )
        text = render_trigger_response(match, "But you are", None, None)
        seed = _seed_for_response("assistant_contradiction", "But you are", "fallback")
        expected = ("What {agent}?", "Okay.")[seed % 2]
        self.assertEqual(text, expected)


class UolTriggerIntegrationTests(unittest.TestCase):
    def test_router_short_circuits_negative_assertion(self) -> None:
        kernel = AssistantOSKernel()
        decision = kernel.handle("You are nameless")
        self.assertEqual(decision.intent, "assistant_behavior")
        self.assertEqual(decision.reason, "uol_trigger_detected")
        seed = _seed_for_response("assistant_negative_assertion", "You are nameless", "fallback")
        expected = ("I disagree.", "That is not correct.")[seed % 2]
        self.assertEqual(decision.answer, expected)

    def test_router_short_circuits_contradiction(self) -> None:
        kernel = AssistantOSKernel()
        decision = kernel.handle("But you are")
        self.assertEqual(decision.intent, "assistant_behavior")
        self.assertEqual(decision.reason, "uol_trigger_detected")
        seed = _seed_for_response("assistant_contradiction", "But you are", "fallback")
        expected = ("I disagree.", "That is not correct.")[seed % 2]
        self.assertEqual(decision.answer, expected)

    def test_synthesis_uses_trigger_response(self) -> None:
        profile = LocalAssistantProfile()
        kernel = AssistantOSKernel(profile=profile)
        decision = kernel.handle("You are nameless")
        synthesis = kernel.last_synthesis
        self.assertIsNotNone(synthesis)
        assert synthesis is not None
        self.assertTrue(synthesis.applied)
        seed = _seed_for_response("assistant_negative_assertion", "You are nameless", "fallback")
        expected = ("I disagree.", "That is not correct.")[seed % 2]
        self.assertEqual(synthesis.answer, expected)

    def test_synthesizer_uses_trigger_response(self) -> None:
        router = AssistantOSKernel()
        synth = BoundedLocalSynthesizer(router.profile)
        decision = router.handle("You are not real")
        result = synth.synthesize(decision, boundary_crossed="local", membrane_allowed=True)
        self.assertTrue(result.applied)
        seed = _seed_for_response("assistant_negative_assertion", "You are not real", "fallback")
        expected = ("I disagree.", "That is not correct.")[seed % 2]
        self.assertEqual(result.answer, expected)


if __name__ == "__main__":
    unittest.main()
