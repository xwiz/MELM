import tempfile
import unittest
from pathlib import Path

from melm.benchmarks import (
    load_annotated_transcript_benchmark,
    sample_transcript_distractor_events,
    sample_transcript_noisy_evidence_cases,
)
from melm.demo import PersistentDialogueDemo, PersistentDialogueSession, evaluate_dialogue_demo
from melm.memory import EventMemory, evaluate_memory, evaluate_state_resolution


SAMPLE_ANNOTATIONS = Path("benchmarks/sample_transcript_annotations.jsonl")


class TranscriptSessionDemoTests(unittest.TestCase):
    def test_transcript_session_preserves_dialogue_state_and_memory_after_reload(self) -> None:
        benchmark = load_annotated_transcript_benchmark(SAMPLE_ANNOTATIONS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_session.jsonl"
            session = PersistentDialogueSession(path, threshold=1.25, k=2)
            session.replace_events(
                [*benchmark.events, *sample_transcript_distractor_events()]
            )

            reloaded = PersistentDialogueSession(path, threshold=1.25, k=2)
            events = list(reloaded.events())
            demo = PersistentDialogueDemo(events, threshold=1.25, k=2)
            dialogue_report = evaluate_dialogue_demo(demo, benchmark.evidence_cases)
            noisy_report = evaluate_dialogue_demo(
                demo,
                sample_transcript_noisy_evidence_cases(),
            )
            memory_report = evaluate_memory(EventMemory(events), benchmark.recall_cases, k=2)
            state_report = evaluate_state_resolution(events, benchmark.state_cases)

        self.assertEqual(
            len(events),
            len(benchmark.events) + len(sample_transcript_distractor_events()),
        )
        self.assertEqual(dialogue_report.accuracy, 1.0)
        self.assertEqual(dialogue_report.positive_recall, 1.0)
        self.assertEqual(dialogue_report.negative_abstention, 1.0)
        self.assertEqual(noisy_report.accuracy, 1.0)
        self.assertEqual(noisy_report.positive_recall, 1.0)
        self.assertEqual(noisy_report.negative_abstention, 1.0)
        self.assertGreaterEqual(
            memory_report.event_memory_recall_at_k,
            memory_report.rag_recall_at_k + 0.15,
        )
        self.assertEqual(state_report.accuracy, 1.0)
        self.assertEqual(state_report.false_positive_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
