"""Slice 1: boundary + counters.

Validates the merged-plan P0 boundary fields and correct per-(session,intent)
occurrence semantics, public session-id plumbing, and prev_mood/prev_intent
carried across turns via the persistent store (the router is rebuilt per turn,
so cross-turn state must come from the store, not router instance state).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import AssistantDecision, LocalAssistantProfile


class BoundaryFieldDefaultsTests(unittest.TestCase):
    def test_new_optional_fields_default_backward_compat(self):
        d = AssistantDecision(
            utterance="x", intent="unknown", route="local_answer", answer="",
        )
        self.assertIsNone(d.prev_mood)
        self.assertEqual(d.prev_intent, "")
        self.assertEqual(d.ambient_valence, 0.0)
        self.assertEqual(d.ambient_valence_delta, 0.0)


class SessionIdPlumbingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        self.store.initialize()
        seed_class_schemas(self.store)
        self.store.connection.commit()
        self.store.start_new_session()

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_current_session_id_public_nonempty_stable(self):
        sid = self.store.current_session_id()
        self.assertTrue(sid)
        self.assertEqual(sid, self.store.current_session_id())

    def test_previous_intent_empty_then_tracks_last_turn(self):
        sid = self.store.current_session_id()
        self.assertEqual(self.store.previous_intent(sid), "")


class CounterSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(), store=self.store,
        )

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_per_intent_occurrence_increments_not_turn_count(self):
        d1 = self.kernel.handle("What are you?")
        d2 = self.kernel.handle("What are you?")
        d3 = self.kernel.handle("What are you?")
        self.assertEqual(d1.intent, "assistant_identity")
        self.assertEqual(d1.intent_occurrence, 1)
        self.assertEqual(d2.intent_occurrence, 2)
        self.assertEqual(d3.intent_occurrence, 3)

    def test_occurrence_is_per_intent_independent(self):
        # First identity turn → occurrence 1.
        a1 = self.kernel.handle("What are you?")
        # A story turn in between does not advance the identity counter.
        self.kernel.handle("Tell me a story.")
        a2 = self.kernel.handle("What are you?")
        self.assertEqual(a1.intent_occurrence, 1)
        self.assertEqual(a2.intent_occurrence, 2)

    def test_prev_mood_none_first_turn_then_populated(self):
        d1 = self.kernel.handle("What are you?")
        d2 = self.kernel.handle("What are you?")
        self.assertIsNone(d1.prev_mood)
        self.assertIsNotNone(d2.prev_mood)

    def test_prev_intent_tracks_last_turn(self):
        d1 = self.kernel.handle("What are you?")
        d2 = self.kernel.handle("Tell me a story.")
        self.assertEqual(d1.prev_intent, "")
        self.assertEqual(d2.prev_intent, "assistant_identity")

    def test_ambient_valence_delta_is_float(self):
        d = self.kernel.handle("What are you?")
        self.assertIsInstance(d.ambient_valence, float)
        self.assertIsInstance(d.ambient_valence_delta, float)


if __name__ == "__main__":
    unittest.main()
