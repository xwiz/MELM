import tempfile
from dataclasses import replace
from pathlib import Path
import unittest

from melm.benchmarks import (
    load_annotated_transcript_benchmark,
    load_dialogue_benchmark,
    save_dialogue_benchmark,
    validate_annotated_transcript_benchmark,
)
from melm.memory import EventMemory, evaluate_memory
from melm.memory import evaluate_state_resolution


SAMPLE_ANNOTATIONS = Path("benchmarks/sample_transcript_annotations.jsonl")


class TranscriptAnnotationTests(unittest.TestCase):
    def test_load_sample_annotated_transcript(self) -> None:
        benchmark = load_annotated_transcript_benchmark(SAMPLE_ANNOTATIONS)

        self.assertEqual(len(benchmark.turns), 8)
        self.assertEqual(len(benchmark.events), 7)
        self.assertEqual(len(benchmark.recall_cases), 6)
        self.assertEqual(len(benchmark.evidence_cases), 10)
        self.assertEqual(len(benchmark.state_cases), 4)
        self.assertEqual(validate_annotated_transcript_benchmark(benchmark), [])

    def test_sample_transcript_compiles_to_dialogue_benchmark(self) -> None:
        benchmark = load_annotated_transcript_benchmark(SAMPLE_ANNOTATIONS)

        with tempfile.TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "events.jsonl"
            recall_path = Path(temp_dir) / "recall.jsonl"
            evidence_path = Path(temp_dir) / "evidence.jsonl"
            save_dialogue_benchmark(
                benchmark.events,
                benchmark.recall_cases,
                benchmark.evidence_cases,
                events_path=events_path,
                recall_cases_path=recall_path,
                evidence_cases_path=evidence_path,
            )
            events, recall_cases, evidence_cases = load_dialogue_benchmark(
                events_path=events_path,
                recall_cases_path=recall_path,
                evidence_cases_path=evidence_path,
            )

        self.assertEqual(benchmark.events, events)
        self.assertEqual(benchmark.recall_cases, recall_cases)
        self.assertEqual(benchmark.evidence_cases, evidence_cases)

    def test_sample_transcript_event_memory_beats_rag(self) -> None:
        benchmark = load_annotated_transcript_benchmark(SAMPLE_ANNOTATIONS)
        report = evaluate_memory(EventMemory(benchmark.events), benchmark.recall_cases, k=2)

        self.assertGreaterEqual(report.event_memory_recall_at_k, report.rag_recall_at_k)

    def test_sample_transcript_state_resolution(self) -> None:
        benchmark = load_annotated_transcript_benchmark(SAMPLE_ANNOTATIONS)

        report = evaluate_state_resolution(benchmark.events, benchmark.state_cases)

        self.assertEqual(report.cases, 4)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.false_positive_rate, 0.0)

    def test_validation_flags_missing_source_turn_id(self) -> None:
        benchmark = load_annotated_transcript_benchmark(SAMPLE_ANNOTATIONS)
        broken_event = replace(benchmark.events[0], metadata={})
        broken = replace(benchmark, events=[broken_event, *benchmark.events[1:]])

        errors = validate_annotated_transcript_benchmark(broken)

        self.assertTrue(any("metadata.source_turn_id" in error for error in errors))

    def test_validation_flags_missing_temporal_link(self) -> None:
        benchmark = load_annotated_transcript_benchmark(SAMPLE_ANNOTATIONS)
        broken_event = replace(benchmark.events[0], next_event_id="missing")
        broken = replace(benchmark, events=[broken_event, *benchmark.events[1:]])

        errors = validate_annotated_transcript_benchmark(broken)

        self.assertTrue(any("missing next_event_id" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
