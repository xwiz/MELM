"""Issue 6 — Sensory understanding natural-language end-to-end.

Proves the full utterance -> UOL perception priming -> routed response path,
not just engine-level affect on a synthetic perception atom.

A natural-language perception utterance ("I smell smoke") must:
  * carry a perception-derived AffectSignal on the decision
    (``utterance_affect.source == "perception"`` with elevated arousal), and
  * route to the perception-urgency safety response (``common_sense_safety``),
    never plain ``open_domain`` and never the cloud.

A neutral control ("the meeting is at noon") must NOT pick up perception affect.

Arousal/confidence assertions match the inline ``_PERCEPTION_AFFECT_MAP`` in
``assistant_mood_engine`` (the map the live code actually uses): high-urgency
stimuli get confidence 0.9; smoke/burnt arousal 0.8, bang arousal 0.9.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile


class SensoryE2ETests(unittest.TestCase):
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

    # -- perception utterances ------------------------------------------

    def test_smell_smoke_carries_perception_affect_and_routes_to_safety(self):
        d = self.kernel.handle("I smell smoke")
        aff = d.utterance_affect
        self.assertIsNotNone(aff, "perception affect must reach the decision")
        self.assertEqual(aff.source, "perception")
        self.assertGreaterEqual(aff.arousal, 0.7)
        # Routed to the perception-urgency safety response, not open_domain.
        self.assertEqual(d.intent, "common_sense_safety")
        self.assertNotEqual(d.intent, "open_domain")
        self.assertFalse(d.cloud_needed)
        self.assertEqual(d.reason, "perception_urgency_high")

    def test_hear_loud_bang_perception_affect_present(self):
        d = self.kernel.handle("I hear a loud bang")
        aff = d.utterance_affect
        self.assertIsNotNone(aff)
        self.assertEqual(aff.source, "perception")
        self.assertGreaterEqual(aff.arousal, 0.7)
        self.assertEqual(d.intent, "common_sense_safety")
        self.assertNotEqual(d.intent, "open_domain")

    def test_room_smells_burnt_perception_affect_present(self):
        d = self.kernel.handle("the room smells burnt")
        aff = d.utterance_affect
        self.assertIsNotNone(aff)
        self.assertEqual(aff.source, "perception")
        self.assertGreaterEqual(aff.arousal, 0.7)
        self.assertEqual(d.intent, "common_sense_safety")
        self.assertNotEqual(d.intent, "open_domain")

    # -- neutral control ------------------------------------------------

    def test_neutral_utterance_has_no_perception_affect(self):
        d = self.kernel.handle("the meeting is at noon")
        aff = d.utterance_affect
        # Either no affect, or affect that is not perception-sourced.
        if aff is not None:
            self.assertNotEqual(aff.source, "perception")
        self.assertNotEqual(d.intent, "common_sense_safety")


if __name__ == "__main__":
    unittest.main()
