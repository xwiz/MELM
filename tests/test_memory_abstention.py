import unittest

from melm.benchmarks import generate_synthetic_evidence_benchmark
from melm.memory import (
    EventMemory,
    best_abstention_report,
    calibrate_abstention_threshold,
    decide_evidence,
    evaluate_abstention,
    split_evidence_cases,
    sweep_abstention_thresholds,
)
from melm.evaluation import abstention_gate


class MemoryAbstentionTests(unittest.TestCase):
    def test_evidence_benchmark_contains_positive_and_negative_cases(self) -> None:
        events, cases = generate_synthetic_evidence_benchmark(stories=3)

        positives = [case for case in cases if case.expected_event_id is not None]
        negatives = [case for case in cases if case.expected_event_id is None]

        self.assertEqual(len(events), 18)
        self.assertEqual(len(positives), 24)
        self.assertEqual(len(negatives), 15)

    def test_abstention_report_tracks_false_positives(self) -> None:
        events, cases = generate_synthetic_evidence_benchmark(stories=3)
        report = evaluate_abstention(
            EventMemory(events),
            cases,
            threshold=0.0,
            retriever="event_memory",
        )

        self.assertEqual(report.cases, len(cases))
        self.assertGreater(report.false_positives, 0)
        self.assertEqual(report.negative_abstention, 0.0)

    def test_decide_evidence_reports_confidence_and_candidates(self) -> None:
        events, cases = generate_synthetic_evidence_benchmark(stories=3)
        positive = next(case for case in cases if case.expected_event_id is not None)
        decision = decide_evidence(
            EventMemory(events),
            positive.query,
            threshold=0.0,
            retriever="event_memory",
            confidence_method="score_with_evidence_veto",
        )

        self.assertTrue(decision.answered)
        self.assertGreater(decision.confidence, 0.0)
        self.assertTrue(decision.candidate_event_ids)

    def test_threshold_sweep_selects_best_report(self) -> None:
        events, cases = generate_synthetic_evidence_benchmark(stories=3)
        reports = sweep_abstention_thresholds(
            EventMemory(events),
            cases,
            thresholds=[0.0, 1.0, 2.0],
            retriever="event_memory",
        )
        best = best_abstention_report(reports)

        self.assertIn(best.threshold, {0.0, 1.0, 2.0})
        self.assertGreaterEqual(best.accuracy, min(report.accuracy for report in reports))

    def test_calibration_uses_heldout_story_split(self) -> None:
        events, cases = generate_synthetic_evidence_benchmark(stories=5)
        calibration_cases, evaluation_cases = split_evidence_cases(cases)
        run = calibrate_abstention_threshold(
            EventMemory(events),
            calibration_cases,
            evaluation_cases,
            thresholds=[0.0, 1.0, 1.5],
            retriever="event_memory",
            confidence_method="top_score",
        )

        self.assertGreater(len(calibration_cases), 0)
        self.assertGreater(len(evaluation_cases), 0)
        self.assertIn(run.threshold, {0.0, 1.0, 1.5})
        self.assertEqual(run.evaluation_report.cases, len(evaluation_cases))

    def test_hybrid_calibration_passes_current_synthetic_gate(self) -> None:
        events, cases = generate_synthetic_evidence_benchmark(stories=25)
        calibration_cases, evaluation_cases = split_evidence_cases(cases)
        run = calibrate_abstention_threshold(
            EventMemory(events),
            calibration_cases,
            evaluation_cases,
            retriever="event_memory",
            confidence_method="score_with_evidence_veto",
        )
        report = run.evaluation_report

        self.assertGreaterEqual(report.positive_recall, 0.75)
        self.assertGreaterEqual(report.negative_abstention, 0.80)
        self.assertTrue(
            abstention_gate(
                report.negative_abstention,
                positive_recall=report.positive_recall,
            ).passed
        )

    def test_abstention_gate_requires_high_negative_abstention(self) -> None:
        self.assertFalse(abstention_gate(0.50).passed)
        self.assertTrue(abstention_gate(0.90).passed)
        self.assertFalse(abstention_gate(0.90, positive_recall=0.50).passed)


if __name__ == "__main__":
    unittest.main()
