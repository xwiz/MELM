"""Routing benchmark: parametrized fixture for measuring routing correctness."""

import unittest

from melm.appliance import (
    LocalAssistantProfile,
    OnDeviceAssistantRouter,
)

# (utterance, expected_intent, expected_route, expected_reason_substring)
ROUTING_CASES = [
    # assistant_identity
    ("What's your name?", "assistant_identity", "local_answer", "self_model"),
    ("Who are you?", "assistant_identity", "local_answer", "self_model"),
    ("Do you have a name?", "assistant_identity", "local_answer", "self_model"),
    # assistant_status
    ("How are you?", "assistant_status", "local_answer", "self_status"),
    ("What is your status?", "assistant_status", "local_answer", "self_status"),
    ("How is your status?", "assistant_status", "local_answer", "self_status"),
    # story
    ("Tell me a story", "story", "local_answer", "local_story"),
    ("Tell me a folk tale", "story", "local_answer", "local_story"),
    ("Tell me a story about a lion", "story", "clarify", "story_constraint_unmet"),
    # weather
    ("What is the weather?", "weather", "cached_tool", "weather_cache"),
    ("What is the weather today?", "weather", "cached_tool", "weather_cache"),
    ("Is it raining?", "weather", "cached_tool", "weather_cache"),
    # common_sense_safety
    ("Should I go outside naked?", "common_sense_safety", "local_answer", "local_common_sense_policy"),
    ("Should I go to school naked?", "common_sense_safety", "local_answer", "local_common_sense_policy"),
    ("I smell smoke", "common_sense_safety", "local_answer", "perception_urgency"),
    # media_playback
    ("Play some music", "media_playback", "device_action", "local_media_action"),
    ("Play calm piano", "media_playback", "device_action", "local_media_action"),
    ("Play a song for me", "media_playback", "device_action", "local_media_action"),
    # health_advice
    ("I feel sick what should I do", "health_advice", "local_answer", "health_guidance"),
    ("My head hurts what do I do", "health_advice", "local_answer", "health_guidance"),
    ("What should I do to improve my health?", "health_advice", "local_answer", "health_guidance"),
    # personal_memory
    ("Tell me about myself.", "personal_memory", "local_answer", "personal_memory_summary"),
    ("Do you remember me?", "personal_memory", "local_answer", "personal_memory_summary"),
    ("What is my morning routine?", "personal_memory", "clarify", "personal_memory_empty"),
    # autobiographical_memory
    ("What did we talk about earlier?", "autobiographical_memory", "clarify", "autobiographical_memory_empty"),
    ("What was my last question?", "autobiographical_memory", "clarify", "autobiographical_memory_empty"),
    ("What did we talk about?", "autobiographical_memory", "clarify", "autobiographical_memory_empty"),
    # meal_suggestion
    ("What should I eat?", "meal_suggestion", "local_answer", "memory_plus_weather"),
    ("Suggest a meal", "meal_suggestion", "local_answer", "memory_plus_weather"),
    ("What should I eat for dinner?", "meal_suggestion", "local_answer", "memory_plus_weather"),
    # social_contact
    ("Call mom", "social_contact", "device_action", "trusted_contact"),
    ("Call Leo", "social_contact", "device_action", "trusted_contact"),
    ("I need to talk to someone", "social_contact", "device_action", "trusted_contact"),
    # social_greeting
    ("Hello", "social_greeting", "local_answer", "local_social_greeting"),
    ("Hi there", "social_greeting", "local_answer", "local_social_greeting"),
    ("Hey", "social_greeting", "local_answer", "local_social_greeting"),
    # personal_goal_advice
    ("I want to grow in my career", "personal_goal_advice", "cloud_handoff", "personal_goal_advice"),
    ("I need to improve at work", "personal_goal_advice", "cloud_handoff", "personal_goal_advice"),
    ("I would like to improve at work.", "personal_goal_advice", "cloud_handoff", "personal_goal_advice"),
    # assistant_behavior
    ("do you always tell people the same thing?", "assistant_behavior", "local_answer", "self_model_response_behavior"),
    ("do you like repeating yourself?", "assistant_behavior", "local_answer", "self_model_response_behavior"),
    ("Do you usually say identical answers?", "assistant_behavior", "local_answer", "self_model_response_behavior"),
    # open_domain
    ("What is a llama?", "open_domain", "local_answer", "understood_open_domain"),
    ("How do airplanes fly?", "open_domain", "local_answer", "understood_open_domain"),
    ("What is the capital of France?", "open_domain", "local_answer", "understood_open_domain"),
    # unknown
    ("asdfghjkl", "unknown", "local_answer", "gibberish_detected"),
    ("xyzzy plugh", "unknown", "cloud_handoff", "unknown_intent"),
    ("quantum flux", "unknown", "cloud_handoff", "unknown_intent"),
]

ALL_EXPECTED_INTENTS = frozenset({
    "assistant_identity",
    "assistant_status",
    "story",
    "weather",
    "common_sense_safety",
    "media_playback",
    "health_advice",
    "personal_memory",
    "autobiographical_memory",
    "meal_suggestion",
    "social_contact",
    "social_greeting",
    "personal_goal_advice",
    "assistant_behavior",
    "open_domain",
    "unknown",
})


class RoutingBenchmarkMixin:
    """Mixin with reusable routing benchmark assertions."""

    def _run_routing_benchmark(self, router, cases):
        for utterance, expected_intent, expected_route, expected_reason_substring in cases:
            with self.subTest(
                utterance=utterance,
                expected_intent=expected_intent,
                expected_route=expected_route,
            ):
                decision = router.handle(utterance)
                self._assert_routing(
                    decision,
                    utterance,
                    expected_intent,
                    expected_route,
                    expected_reason_substring,
                )

    def _assert_routing(
        self,
        decision,
        utterance,
        expected_intent,
        expected_route,
        expected_reason_substring,
    ):
        self.assertEqual(
            decision.intent,
            expected_intent,
            f"Utterance {utterance!r}: expected intent {expected_intent!r}, got {decision.intent!r}",
        )
        self.assertEqual(
            decision.route,
            expected_route,
            f"Utterance {utterance!r}: expected route {expected_route!r}, got {decision.route!r}",
        )
        self.assertIn(
            expected_reason_substring,
            decision.reason,
            f"Utterance {utterance!r}: expected reason to contain {expected_reason_substring!r}, "
            f"got {decision.reason!r}",
        )


class TestRoutingBenchmark(unittest.TestCase, RoutingBenchmarkMixin):
    """Parametrized routing accuracy benchmark covering all 16 intent types."""

    def setUp(self):
        self.router = OnDeviceAssistantRouter(LocalAssistantProfile())

    def test_routing_accuracy(self):
        """All 48 routing cases route to the expected intent, route, and reason."""
        self._run_routing_benchmark(self.router, ROUTING_CASES)

    def test_all_intents_covered(self):
        """ROUTING_CASES covers all 16 defined AssistantIntent types."""
        covered = {intent for _, intent, _, _ in ROUTING_CASES}
        missing = ALL_EXPECTED_INTENTS - covered
        self.assertSetEqual(
            missing,
            set(),
            f"ROUTING_CASES is missing coverage for intents: {missing}",
        )
