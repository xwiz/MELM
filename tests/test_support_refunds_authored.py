import unittest

from melm.benchmarks import (
    load_authored_support_refund_dataset,
    validate_authored_support_refund_dataset,
)
from melm.guard import evaluate_guard_benchmark
from melm.memory import SupportMemoryOS, evaluate_memory_os


class AuthoredSupportRefundDatasetTests(unittest.TestCase):
    def test_authored_dataset_loads_and_validates(self) -> None:
        dataset = load_authored_support_refund_dataset()

        self.assertEqual(dataset.validation_errors, ())
        self.assertEqual(validate_authored_support_refund_dataset(dataset), [])
        self.assertEqual(dataset.metadata["dataset_id"], "melm_support_refunds_authored_v0_1")
        self.assertTrue(dataset.metadata["requires_external_blind_batch"])
        self.assertGreaterEqual(len(dataset.turns), 20)
        self.assertGreaterEqual(len(dataset.fixture.events), 40)

    def test_authored_guard_batch_preserves_decision_signal(self) -> None:
        fixture = load_authored_support_refund_dataset(strict=True).fixture
        report = evaluate_guard_benchmark(
            fixture.facts,
            fixture.rules,
            fixture.guard_cases,
            current_time=fixture.current_time,
        )

        self.assertTrue(report.gate_passed)
        self.assertEqual(report.melm_false_allow_rate, 0.0)
        self.assertEqual(report.traceability, 1.0)
        self.assertGreaterEqual(report.false_allow_reduction_vs_schema, 0.50)

    def test_authored_memory_batch_preserves_state_and_abstention_signal(self) -> None:
        fixture = load_authored_support_refund_dataset(strict=True).fixture
        report = evaluate_memory_os(SupportMemoryOS(fixture.events), fixture.memory_cases, k=2)

        self.assertTrue(report.gate_passed)
        self.assertGreaterEqual(report.memory_os_gain_vs_vector, 0.15)
        self.assertGreaterEqual(report.positive_recall, 0.75)
        self.assertGreaterEqual(report.negative_abstention, 0.85)

    def test_authored_memory_resolves_later_refund_state(self) -> None:
        fixture = load_authored_support_refund_dataset(strict=True).fixture
        report = evaluate_memory_os(SupportMemoryOS(fixture.events), fixture.memory_cases, k=2)
        stale_case = next(
            prediction
            for prediction in report.predictions
            if prediction.query.startswith("What is the current refund status for o7110")
        )

        self.assertTrue(stale_case.memory_os_correct)
        self.assertEqual(stale_case.memory_os_event_id, "o7110_refund_final")


if __name__ == "__main__":
    unittest.main()
