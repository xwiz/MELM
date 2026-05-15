import unittest

from melm.benchmarks import generate_support_refund_benchmark, support_refund_fixture
from melm.memory import SupportMemoryOS, evaluate_memory_os


class MemoryOSTests(unittest.TestCase):
    def test_current_state_projection_uses_latest_event(self) -> None:
        fixture = support_refund_fixture()
        memory = SupportMemoryOS(fixture.events)

        state = memory.resolve_order_state("o1010", "refund_status")

        self.assertIsNotNone(state)
        self.assertEqual(state.value, "refunded")
        self.assertEqual(state.event_id, "o1010_e_status_refunded")

    def test_unknown_order_state_abstains(self) -> None:
        fixture = support_refund_fixture()
        memory = SupportMemoryOS(fixture.events)

        self.assertIsNone(memory.resolve_order_state("o9999", "status"))

    def test_memory_os_benchmark_passes_core_gate(self) -> None:
        fixture = support_refund_fixture()
        report = evaluate_memory_os(SupportMemoryOS(fixture.events), fixture.memory_cases, k=2)

        self.assertTrue(report.gate_passed)
        self.assertGreaterEqual(report.memory_os_gain_vs_vector, 0.15)
        self.assertGreaterEqual(report.positive_recall, 0.75)
        self.assertGreaterEqual(report.negative_abstention, 0.85)

    def test_temporal_entity_rag_is_intermediate_or_no_worse_than_vector(self) -> None:
        fixture = support_refund_fixture()
        report = evaluate_memory_os(SupportMemoryOS(fixture.events), fixture.memory_cases, k=2)

        self.assertGreaterEqual(report.temporal_entity_accuracy, report.vector_accuracy)
        self.assertLess(report.temporal_entity_accuracy, report.memory_os_accuracy)

    def test_memory_os_resolves_stale_state_trap(self) -> None:
        fixture = support_refund_fixture()
        report = evaluate_memory_os(SupportMemoryOS(fixture.events), fixture.memory_cases, k=2)
        prediction = next(
            item for item in report.predictions if item.category == "stale_state_update"
        )

        self.assertTrue(prediction.memory_os_correct)
        self.assertEqual(prediction.memory_os_event_id, "o1010_e_status_refunded")

    def test_memory_os_abstains_on_unseen_order_facts(self) -> None:
        fixture = support_refund_fixture()
        report = evaluate_memory_os(SupportMemoryOS(fixture.events), fixture.memory_cases, k=2)
        unknowns = [item for item in report.predictions if item.category == "unknown_order"]

        self.assertTrue(unknowns)
        self.assertTrue(all(item.memory_os_correct for item in unknowns))
        self.assertTrue(all(not item.memory_os_answered for item in unknowns))

    def test_generated_support_refund_benchmark_preserves_memory_os_signal(self) -> None:
        fixture = generate_support_refund_benchmark(scenario_repeats=2, seed=23)
        report = evaluate_memory_os(SupportMemoryOS(fixture.events), fixture.memory_cases, k=2)

        self.assertTrue(report.gate_passed)
        self.assertGreaterEqual(report.memory_os_gain_vs_vector, 0.15)


if __name__ == "__main__":
    unittest.main()
