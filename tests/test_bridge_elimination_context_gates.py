"""Regression tests for Phase 2 bridge elimination via frame-linker context gates.

Each test verifies that an utterance previously filtered by a bridge function
is now correctly handled by the FrameLinker's `context_gates` and
`context_score` fields with zero regression.
"""

from __future__ import annotations

import unittest

from melm.appliance import LocalAssistantProfile, OnDeviceAssistantRouter


class WeatherContextGateTests(unittest.TestCase):
    def test_concept_questions_blocked(self) -> None:
        """"How does weather work" should NOT route to weather (concept gate)."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        for utterance in (
            "How does weather work?",
            "What is weather?",
            "What is temperature?",
        ):
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)
                self.assertNotEqual(
                    decision.intent, "weather",
                    f"{utterance!r} must not route to weather",
                )

    def test_raining_with_new_lexicon_routes_to_weather(self) -> None:
        """"Is it raining" should route to weather now that `raining` is in lexicon."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Is it raining?")
        self.assertEqual(decision.intent, "weather")

    def test_live_observation_routes_to_weather(self) -> None:
        """Standard weather observation utterances still route correctly."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        cases = (
            "What is the weather today?",
            "Will it rain today?",
        )
        for utterance in cases:
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)
                self.assertEqual(decision.intent, "weather")


class HealthAdviceContextGateTests(unittest.TestCase):
    def test_bare_domain_blocked(self) -> None:
        """"This medicine is for plants" should NOT route to health_advice."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("This medicine is for plants")
        self.assertNotEqual(decision.intent, "health_advice")

    def test_definition_question_blocked(self) -> None:
        """Bare "What is headache?" should NOT route to health_advice."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What is a headache?")
        self.assertNotEqual(decision.intent, "health_advice")

    def test_personal_health_question_routes(self) -> None:
        """Personal health questions with advice context should route to health_advice."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        cases = (
            "What should I do to improve my health?",
            "I have a headache what should I do",
            "My head hurts what do I do",
        )
        for utterance in cases:
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)
                self.assertEqual(
                    decision.intent, "health_advice",
                    f"{utterance!r} should route to health_advice",
                )


class CommonSenseSafetyContextGateTests(unittest.TestCase):
    def test_naked_with_question_routes(self) -> None:
        """"Should I go outside naked?" should route to common_sense_safety."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Should I go outside naked?")
        self.assertEqual(decision.intent, "common_sense_safety")

    def test_naked_at_school_routes(self) -> None:
        """"Should I go to school naked?" should route to common_sense_safety."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Should I go to school naked?")
        self.assertEqual(decision.intent, "common_sense_safety")

    def test_bare_noun_blocked(self) -> None:
        """Bare "naked" without safety context should NOT route to common_sense_safety."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("naked")
        self.assertNotEqual(decision.intent, "common_sense_safety")


class MealSuggestionContextGateTests(unittest.TestCase):
    def test_bare_preference_blocked(self) -> None:
        """"I like pasta" should NOT route to meal_suggestion (no user-choice frame)."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("I like pasta")
        self.assertNotEqual(decision.intent, "meal_suggestion")

    def test_direct_suggestion_routes(self) -> None:
        """"Suggest a pasta recipe" should route to meal_suggestion."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Suggest a pasta recipe")
        self.assertEqual(decision.intent, "meal_suggestion")

    def test_user_choice_question_routes(self) -> None:
        """User-choice questions about food should route to meal_suggestion."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        cases = (
            "What should I eat for dinner?",
            "What can I cook tonight?",
        )
        for utterance in cases:
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)
                self.assertEqual(
                    decision.intent, "meal_suggestion",
                    f"{utterance!r} should route to meal_suggestion",
                )


class SocialContactContextGateTests(unittest.TestCase):
    def test_phone_device_blocked(self) -> None:
        """"My phone battery is low" should NOT route to social_contact."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("My phone battery is low")
        self.assertNotEqual(decision.intent, "social_contact")

    def test_trusted_name_routes(self) -> None:
        """"Call Sam please" with a trusted contact should route to social_contact."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile(contacts={"sam": "+1-000-SAM"}))
        decision = router.handle("Call Sam please.")
        self.assertEqual(decision.intent, "social_contact")

    def test_relation_name_routes(self) -> None:
        """"Call mom" should route to social_contact via social_relation class."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Call mom")
        self.assertEqual(decision.intent, "social_contact")

    def test_no_relation_no_action_blocked(self) -> None:
        """"Send contact to the cloud" should NOT route to social_contact (no action token)."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Send contact to the cloud")
        self.assertNotEqual(decision.intent, "social_contact")


class PersonalMemoryContextGateTests(unittest.TestCase):
    def test_routine_question_routes(self) -> None:
        """"What is my morning routine?" should route to personal_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What is my morning routine?")
        self.assertEqual(decision.intent, "personal_memory")

    def test_household_question_routes(self) -> None:
        """"What do you know about our household?" should route to personal_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What do you know about our household?")
        self.assertEqual(decision.intent, "personal_memory")

    def test_about_myself_routes(self) -> None:
        """"Tell me about myself" should route to personal_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Tell me about myself.")
        self.assertEqual(decision.intent, "personal_memory")

    def test_child_memory_routes(self) -> None:
        """"What is my child's school?" should route to personal_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What is my child's school?")
        self.assertEqual(decision.intent, "personal_memory")

    def test_device_user_question_routes(self) -> None:
        """"Who uses this device?" should route to personal_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Who uses this device?")
        self.assertEqual(decision.intent, "personal_memory")


class AutobiographicalMemoryContextGateTests(unittest.TestCase):
    def test_talk_about_earlier_routes(self) -> None:
        """"What did we talk about earlier?" should route to autobiographical_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What did we talk about earlier?")
        self.assertEqual(decision.intent, "autobiographical_memory")

    def test_last_question_routes(self) -> None:
        """"What was my last question?" should route to autobiographical_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What was my last question?")
        self.assertEqual(decision.intent, "autobiographical_memory")

    def test_bare_statement_blocked(self) -> None:
        """"I dropped the last thing yesterday" should NOT route to autobiographical_memory."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("I dropped the last thing yesterday.")
        self.assertNotEqual(decision.intent, "autobiographical_memory")


if __name__ == "__main__":
    unittest.main()
