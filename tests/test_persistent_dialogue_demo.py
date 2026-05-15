import unittest

from melm.benchmarks import authored_child_dialogue_fixture
from melm.demo import PersistentDialogueDemo, evaluate_dialogue_demo
from melm.memory import Event


class PersistentDialogueDemoTests(unittest.TestCase):
    def test_demo_answers_after_remembering_event(self) -> None:
        demo = PersistentDialogueDemo(threshold=0.0, k=1, confidence_method="top_score")
        demo.remember(
            Event(
                event_id="e1",
                source_span="Maya put the red shell in the blue cup.",
                time_index=1,
                actors=("maya",),
                action_or_state="put",
                objects=("red shell", "blue cup"),
            )
        )

        response = demo.ask("Where did Maya put the red shell?")

        self.assertEqual(response.status, "answered")
        self.assertIn("e1", response.evidence_event_ids)
        self.assertIn("blue cup", response.answer)

    def test_demo_abstains_without_evidence(self) -> None:
        demo = PersistentDialogueDemo(threshold=1.25)

        response = demo.ask("Where is the silver train?")

        self.assertEqual(response.status, "abstained")
        self.assertEqual(response.evidence_event_ids, ())

    def test_authored_dialogue_demo_preserves_negative_abstention(self) -> None:
        events, _recall_cases, evidence_cases = authored_child_dialogue_fixture()
        demo = PersistentDialogueDemo(events, threshold=1.25, k=2)

        report = evaluate_dialogue_demo(demo, evidence_cases)

        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.positive_recall, 1.0)
        self.assertEqual(report.negative_abstention, 1.0)

    def test_demo_resolves_causal_source_from_linked_witness(self) -> None:
        events, _recall_cases, _evidence_cases = authored_child_dialogue_fixture()
        demo = PersistentDialogueDemo(events, threshold=1.25, k=2)

        response = demo.ask(
            "What earlier event explains why Sam knew about the purple lunchbox?"
        )

        self.assertEqual(response.status, "answered")
        self.assertIn("dialogue1_e2", response.evidence_event_ids)
        self.assertIn("dialogue1_e4", response.evidence_event_ids)

    def test_demo_resolves_later_state_change_without_lowering_threshold(self) -> None:
        events, _recall_cases, _evidence_cases = authored_child_dialogue_fixture()
        demo = PersistentDialogueDemo(events, threshold=1.25, k=2)

        response = demo.ask("What changed where the tiny shell was later?")

        self.assertEqual(response.status, "answered")
        self.assertEqual(response.evidence_event_ids, ("dialogue3_e3",))
        self.assertGreaterEqual(response.confidence, 1.25)

    def test_demo_abstains_on_false_presupposed_move(self) -> None:
        events, _recall_cases, _evidence_cases = authored_child_dialogue_fixture()
        demo = PersistentDialogueDemo(events, threshold=1.25, k=2)

        response = demo.ask("Where was the tiny shell after Owen moved it to the shelf?")

        self.assertEqual(response.status, "abstained")
        self.assertEqual(response.evidence_event_ids, ())


if __name__ == "__main__":
    unittest.main()
