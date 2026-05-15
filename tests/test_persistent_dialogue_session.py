import tempfile
import unittest
from pathlib import Path

from melm.demo import (
    PersistentDialogueSession,
    load_session_events,
    save_session_events,
)
from melm.memory import Event


class PersistentDialogueSessionTests(unittest.TestCase):
    def test_session_persists_events_and_answers_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo_session.jsonl"
            session = PersistentDialogueSession(
                path,
                threshold=0.0,
                k=1,
                confidence_method="top_score",
            )

            event = session.remember_observation(
                "Maya put the red shell in the blue cup.",
                actors=("maya",),
                action_or_state="put",
                objects=("red shell", "blue cup"),
                location="play table",
            )

            reloaded = PersistentDialogueSession(
                path,
                threshold=0.0,
                k=1,
                confidence_method="top_score",
            )
            response = reloaded.ask("Where did Maya put the red shell?")

            self.assertEqual(len(reloaded.events()), 1)
            self.assertEqual(reloaded.events()[0].event_id, event.event_id)
            self.assertEqual(response.status, "answered")
            self.assertIn(event.event_id, response.evidence_event_ids)

    def test_session_links_sequential_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo_session.jsonl"
            session = PersistentDialogueSession(path)

            first = session.remember_observation(
                "Leo carried the purple lunchbox to the reading rug.",
                actors=("leo",),
                action_or_state="carried",
                objects=("purple lunchbox", "reading rug"),
            )
            second = session.remember_observation(
                "Sam said he saw Leo place the purple lunchbox beside the reading rug.",
                actors=("sam", "leo"),
                action_or_state="said",
                objects=("purple lunchbox", "reading rug"),
                causal_links=(first.event_id,),
            )

            reloaded_events = load_session_events(path)

            self.assertEqual(reloaded_events[0].next_event_id, second.event_id)
            self.assertEqual(reloaded_events[1].previous_event_id, first.event_id)

    def test_session_rejects_duplicate_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo_session.jsonl"
            session = PersistentDialogueSession(path)
            event = Event(
                event_id="fixed",
                source_span="Nora built a tower.",
                time_index=1,
            )
            session.remember(event)

            with self.assertRaises(ValueError):
                session.remember(event)

    def test_session_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            events = [
                Event(
                    event_id="e1",
                    source_span="Owen put the shell in the bucket.",
                    time_index=1,
                    actors=("owen",),
                    action_or_state="put",
                    objects=("shell", "bucket"),
                    metadata={"source": "unit"},
                )
            ]

            save_session_events(path, events)
            loaded = load_session_events(path)

            self.assertEqual(loaded, events)


if __name__ == "__main__":
    unittest.main()
