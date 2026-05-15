import tempfile
from pathlib import Path
import unittest

from melm.benchmarks import (
    EpisodicCase,
    authored_child_dialogue_fixture,
    generate_synthetic_episodic_benchmark,
    load_dialogue_benchmark,
    load_episodic_benchmark,
    save_dialogue_benchmark,
    save_episodic_benchmark,
    validate_dialogue_benchmark,
    validate_episodic_benchmark,
    validate_evidence_benchmark,
)
from melm.memory import Event


class BenchmarkIOTests(unittest.TestCase):
    def test_round_trip_synthetic_benchmark(self) -> None:
        events, cases = generate_synthetic_episodic_benchmark(stories=2, seed=3)

        with tempfile.TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "events.jsonl"
            cases_path = Path(temp_dir) / "cases.jsonl"
            save_episodic_benchmark(
                events,
                cases,
                events_path=events_path,
                cases_path=cases_path,
            )
            loaded_events, loaded_cases = load_episodic_benchmark(
                events_path=events_path,
                cases_path=cases_path,
            )

        self.assertEqual(events, loaded_events)
        self.assertEqual(cases, loaded_cases)

    def test_round_trip_dialogue_benchmark(self) -> None:
        events, recall_cases, evidence_cases = authored_child_dialogue_fixture()

        with tempfile.TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "events.jsonl"
            recall_path = Path(temp_dir) / "recall.jsonl"
            evidence_path = Path(temp_dir) / "evidence.jsonl"
            save_dialogue_benchmark(
                events,
                recall_cases,
                evidence_cases,
                events_path=events_path,
                recall_cases_path=recall_path,
                evidence_cases_path=evidence_path,
            )
            loaded_events, loaded_recall, loaded_evidence = load_dialogue_benchmark(
                events_path=events_path,
                recall_cases_path=recall_path,
                evidence_cases_path=evidence_path,
            )

        self.assertEqual(events, loaded_events)
        self.assertEqual(recall_cases, loaded_recall)
        self.assertEqual(evidence_cases, loaded_evidence)

    def test_validation_flags_missing_expected_event(self) -> None:
        events = [
            Event(
                event_id="e1",
                source_span="Maya put the cup on the table.",
                time_index=1,
            )
        ]
        cases = [EpisodicCase("Where is the cup?", "missing", "direct")]

        errors = validate_episodic_benchmark(events, cases)

        self.assertTrue(any("missing" in error for error in errors))

    def test_validation_flags_duplicate_event_ids(self) -> None:
        events = [
            Event(event_id="e1", source_span="first", time_index=1),
            Event(event_id="e1", source_span="second", time_index=2),
        ]
        cases = [EpisodicCase("What happened?", "e1", "direct")]

        errors = validate_episodic_benchmark(events, cases)

        self.assertTrue(any("duplicate event_id" in error for error in errors))

    def test_validation_flags_missing_event_links(self) -> None:
        events = [
            Event(
                event_id="e1",
                source_span="Maya put the cup on the table.",
                time_index=1,
                next_event_id="missing_next",
                causal_links=("missing_cause",),
            )
        ]
        cases = [EpisodicCase("Where is the cup?", "e1", "direct")]

        errors = validate_episodic_benchmark(events, cases)

        self.assertTrue(any("missing next_event_id" in error for error in errors))
        self.assertTrue(any("missing causal_link" in error for error in errors))

    def test_validation_flags_bad_temporal_order(self) -> None:
        events = [
            Event(event_id="e1", source_span="late", time_index=2),
            Event(event_id="e2", source_span="early", time_index=1, previous_event_id="e1"),
        ]
        cases = [EpisodicCase("What happened?", "e2", "direct")]

        errors = validate_episodic_benchmark(events, cases)

        self.assertTrue(any("is not earlier" in error for error in errors))

    def test_evidence_validation_allows_unanswerable_cases(self) -> None:
        events, _recall_cases, evidence_cases = authored_child_dialogue_fixture()

        errors = validate_evidence_benchmark(events, evidence_cases)

        self.assertEqual(errors, [])

    def test_dialogue_validation_combines_recall_and_evidence_checks(self) -> None:
        events, recall_cases, evidence_cases = authored_child_dialogue_fixture()
        bad_recall = [
            EpisodicCase(
                query=recall_cases[0].query,
                expected_event_id="missing",
                category=recall_cases[0].category,
            )
        ]

        errors = validate_dialogue_benchmark(events, bad_recall, evidence_cases)

        self.assertTrue(any("missing" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
