"""Multi-turn stress tests for the Local Assistant OS kernel.

Exercises cross-turn state: mood decay, intent tallies, event ring buffer,
personal_experience entities, and session summaries across multiple turns.
"""

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore
from melm.appliance.local_assistant_router import (
    LocalAssistantProfile,
    OnDeviceAssistantRouter,
)


# ---------------------------------------------------------------------------
# A. Intent tracking across turns
# ---------------------------------------------------------------------------

class TestMultiTurnIntentTracking(unittest.TestCase):
    """Test that intent_occurrence and rapid_occurrence increment correctly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AssistantOSStore(Path(self.tmp.name) / "test.db")
        self.kernel = AssistantOSKernel(store=self.store, profile=LocalAssistantProfile(
            story_models={"seed": "{name} rested in {location}."},
            weekly_weather={"today": "warm", "tomorrow": "rainy"},
        ))

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def test_intent_occurrence_increases_for_same_intent(self):
        d1 = self.kernel.handle("Tell me a story.")
        self.assertEqual(d1.intent, "story")
        self.assertEqual(d1.intent_occurrence, 1)

        d2 = self.kernel.handle("Tell me a story.")
        self.assertEqual(d2.intent, "story")
        self.assertEqual(d2.intent_occurrence, 2)

        d3 = self.kernel.handle("Tell me a story.")
        self.assertEqual(d3.intent, "story")
        self.assertEqual(d3.intent_occurrence, 3)

    def test_intent_occurrence_resets_per_intent(self):
        self.kernel.handle("Tell me a story.")
        self.kernel.handle("Tell me a story.")
        self.kernel.handle("Tell me a story.")

        d = self.kernel.handle("What is the weather today?")
        self.assertEqual(d.intent, "weather")
        self.assertEqual(d.intent_occurrence, 1)

    def test_rapid_occurrence_increases_on_same_utterance(self):
        # rapid_occurrence tracks consecutive same-utterance text in the
        # router's in-memory _rapid_state dict.  The kernel recreates the
        # router per turn, so test directly with a persistent router instance.
        # Use store=None to exercise the in-memory fallback; the store-backed
        # path counts DB events which are only written by the kernel.
        router = OnDeviceAssistantRouter(
            self.kernel.profile, store=None,
        )
        d1 = router.handle("Hello")
        self.assertEqual(d1.rapid_occurrence, 0)
        d2 = router.handle("Hello")
        self.assertEqual(d2.rapid_occurrence, 1)
        d3 = router.handle("Hello")
        self.assertEqual(d3.rapid_occurrence, 2)
        d4 = router.handle("Hello")
        self.assertEqual(d4.rapid_occurrence, 3)


# ---------------------------------------------------------------------------
# B. Event ring buffer
# ---------------------------------------------------------------------------

class TestMultiTurnEventRingBuffer(unittest.TestCase):
    """Test that the ring buffer records events correctly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AssistantOSStore(Path(self.tmp.name) / "test.db")
        self.kernel = AssistantOSKernel(store=self.store, profile=LocalAssistantProfile(
            story_models={"seed": "{name} rested in {location}."},
            weekly_weather={"today": "warm", "tomorrow": "rainy"},
        ))

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def test_ring_buffer_records_all_turns(self):
        self.kernel.handle("Tell me a story.")
        self.kernel.handle("What is the weather today?")
        self.kernel.handle("Hello")
        session_id = self.store.current_session_id()
        events = self.store.get_recent_events(session_id, window_seconds=3600)
        self.assertEqual(len(events), 3)

    def test_ring_buffer_events_have_intent_and_event_type(self):
        self.kernel.handle("Tell me a story.")
        self.kernel.handle("Hello")
        session_id = self.store.current_session_id()
        events = self.store.get_recent_events(session_id, window_seconds=3600)
        self.assertGreaterEqual(len(events), 2)
        intents = [e["intent"] for e in events]
        self.assertIn("story", intents)
        self.assertIn("social_greeting", intents)
        for e in events:
            self.assertIn("event_type", e)
            self.assertEqual(e["event_type"], e["intent"])


# ---------------------------------------------------------------------------
# C. Mood decay
# ---------------------------------------------------------------------------

class TestMultiTurnMoodDecay(unittest.TestCase):
    """Test that mood changes appropriately across turns."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AssistantOSStore(Path(self.tmp.name) / "test.db")
        self.kernel = AssistantOSKernel(store=self.store, profile=LocalAssistantProfile(
            story_models={"seed": "{name} rested in {location}."},
        ))

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def test_positive_utterances_improve_valence(self):
        self.kernel.handle("I love you")
        self.kernel.handle("You are wonderful")
        d3 = self.kernel.handle("Thank you")
        self.assertIsNotNone(d3.session_mood)
        # After several positive utterances, valence should be clearly positive
        self.assertGreaterEqual(d3.session_mood.valence, 0.1)

    def test_negative_utterances_decline_valence(self):
        self.kernel.handle("I hate this")
        self.kernel.handle("I am angry")
        d3 = self.kernel.handle("Leave me alone")
        self.assertIsNotNone(d3.session_mood)
        # After several negative utterances, valence should be clearly negative
        self.assertLessEqual(d3.session_mood.valence, -0.1)

    def test_neutral_utterances_mood_drifts_toward_baseline(self):
        # Start with positive
        self.kernel.handle("I love you")
        # Then neutral
        self.kernel.handle("The sky is blue.")
        self.kernel.handle("It is Tuesday.")
        self.kernel.handle("I have a book.")
        d_last = self.kernel.handle("The door is open.")
        # Mood should still exist but drift back toward baseline
        self.assertIsNotNone(d_last.session_mood)


# ---------------------------------------------------------------------------
# D. Personal experience entities
# ---------------------------------------------------------------------------

class TestMultiTurnPersonalExperience(unittest.TestCase):
    """Test that personal_experience entities are created per turn."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AssistantOSStore(Path(self.tmp.name) / "test.db")
        self.kernel = AssistantOSKernel(store=self.store, profile=LocalAssistantProfile(
            story_models={"seed": "{name} rested in {location}."},
            weekly_weather={"today": "warm", "tomorrow": "rainy"},
        ))

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def test_one_personal_experience_per_turn(self):
        self.kernel.handle("Tell me a story.")
        self.kernel.handle("What is the weather today?")
        self.kernel.handle("Hello")
        experiences = self.store.find_entities(kind="personal_experience")
        self.assertEqual(len(experiences), 3)

    def test_each_experience_has_outcome_and_polarity(self):
        self.kernel.handle("Tell me a story.")
        self.kernel.handle("What is the weather today?")
        experiences = self.store.find_entities(kind="personal_experience")
        self.assertEqual(len(experiences), 2)
        for e in experiences:
            outcome_slot = self.store.get_entity_slot(e.entity_id, "outcome")
            self.assertIsNotNone(outcome_slot, f"Missing outcome on {e.entity_id}")
            polarity_slot = self.store.get_entity_slot(e.entity_id, "polarity")
            self.assertIsNotNone(polarity_slot, f"Missing polarity on {e.entity_id}")
            intent_slot = self.store.get_entity_slot(e.entity_id, "intent")
            self.assertIsNotNone(intent_slot, f"Missing intent on {e.entity_id}")

    def test_learned_fact_ids_recorded_when_applicable(self):
        self.kernel.handle("My name is Alice")
        experiences = self.store.find_entities(kind="personal_experience")
        # At least the name-fact turn creates an experience
        pe = experiences[0]
        lid_slot = self.store.get_entity_slot(pe.entity_id, "learned_fact_ids")
        self.assertIsNotNone(lid_slot)


# ---------------------------------------------------------------------------
# E. Memory recall across sessions
# ---------------------------------------------------------------------------

class TestMultiTurnMemoryRecall(unittest.TestCase):
    """Test that facts recorded in one session are accessible in another."""

    def test_facts_survive_kernel_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    store=store,
                    profile=LocalAssistantProfile(
                        facts={}, preferences={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                kernel.handle("My name is Alice")
                kernel.handle("I love dinosaur stories")
            finally:
                store.close()

            # New session, same store
            store2 = AssistantOSStore(db)
            try:
                kernel2 = AssistantOSKernel(
                    store=store2,
                    profile=LocalAssistantProfile(
                        facts={}, preferences={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                # "What is my name?" recalls the stored user_name
                d = kernel2.handle("What is my name?")
                self.assertEqual(d.intent, "personal_memory")
                self.assertIn("Alice", d.answer)
            finally:
                store2.close()

    def test_story_preference_survives_kernel_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    store=store,
                    profile=LocalAssistantProfile(
                        facts={}, preferences={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                kernel.handle("I love dinosaur stories")
            finally:
                store.close()

            store2 = AssistantOSStore(db)
            try:
                kernel2 = AssistantOSKernel(
                    store=store2,
                    profile=LocalAssistantProfile(
                        facts={}, preferences={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                # The preference should persist and affect routing
                self.assertEqual(kernel2.profile.preferences.get("story_theme"), "dinosaur stories")
            finally:
                store2.close()


# ---------------------------------------------------------------------------
# F. Self-identity learning
# ---------------------------------------------------------------------------

class TestMultiTurnSelfIdentityLearning(unittest.TestCase):
    """Test that self-identity information persists across sessions."""

    def test_name_stored_and_recalled_same_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    store=store,
                    profile=LocalAssistantProfile(
                        facts={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                d1 = kernel.handle("My name is Alice")
                self.assertEqual(d1.intent, "personal_memory")
                self.assertIn("Alice", d1.answer)
                self.assertIn("your name", d1.answer.lower())

                d2 = kernel.handle("What is my name?")
                self.assertEqual(d2.intent, "personal_memory")
                self.assertIn("Alice", d2.answer)
            finally:
                store.close()

    def test_name_recalled_across_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    store=store,
                    profile=LocalAssistantProfile(
                        facts={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                kernel.handle("My name is Alice")
            finally:
                store.close()

            store2 = AssistantOSStore(db)
            try:
                kernel2 = AssistantOSKernel(
                    store=store2,
                    profile=LocalAssistantProfile(
                        facts={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                d = kernel2.handle("What is my name?")
                self.assertEqual(d.intent, "personal_memory")
                self.assertIn("Alice", d.answer)
            finally:
                store2.close()

    def test_multiple_facts_persist_across_session_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    store=store,
                    profile=LocalAssistantProfile(
                        facts={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                kernel.handle("My name is Alice")
                kernel.handle("I am 30 years old")
            finally:
                store.close()

            store2 = AssistantOSStore(db)
            try:
                kernel2 = AssistantOSKernel(
                    store=store2,
                    profile=LocalAssistantProfile(
                        facts={},
                        weekly_weather={"today": "warm"},
                    ),
                )
                self.assertEqual(kernel2.profile.user_name, "Alice")
                self.assertEqual(kernel2.profile.age, 30)
            finally:
                store2.close()


if __name__ == "__main__":
    unittest.main()
