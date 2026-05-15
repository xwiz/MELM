import unittest

from melm.benchmarks import authored_child_dialogue_fixture
from melm.evaluation import abstention_gate, memory_gate
from melm.memory import EventMemory, evaluate_abstention, evaluate_memory


class AuthoredDialogueTests(unittest.TestCase):
    def test_fixture_contains_recall_and_negative_cases(self) -> None:
        events, recall_cases, evidence_cases = authored_child_dialogue_fixture()

        self.assertEqual(len(events), 12)
        self.assertEqual(len(recall_cases), 10)
        self.assertEqual(len(evidence_cases), 15)
        self.assertEqual(
            sum(case.expected_event_id is None for case in evidence_cases),
            5,
        )

    def test_event_memory_passes_authored_dialogue_recall_gate(self) -> None:
        events, recall_cases, _evidence_cases = authored_child_dialogue_fixture()
        report = evaluate_memory(EventMemory(events), recall_cases, k=2)

        self.assertTrue(memory_gate(report.event_memory_recall_at_k, report.rag_recall_at_k).passed)
        self.assertGreaterEqual(report.event_memory_recall_at_k, 0.90)

    def test_hybrid_abstention_transfers_to_authored_dialogue(self) -> None:
        events, _recall_cases, evidence_cases = authored_child_dialogue_fixture()
        report = evaluate_abstention(
            EventMemory(events),
            evidence_cases,
            k=2,
            threshold=1.25,
            retriever="event_memory",
            confidence_method="score_with_evidence_veto",
        )

        self.assertTrue(
            abstention_gate(
                report.negative_abstention,
                positive_recall=report.positive_recall,
            ).passed
        )
        self.assertEqual(report.negative_abstention, 1.0)


if __name__ == "__main__":
    unittest.main()
