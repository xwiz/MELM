import unittest

from melm.benchmarks import episodic_memory_fixture, generate_synthetic_episodic_benchmark
from melm.memory import EventMemory, RetrievalConfig, evaluate_memory, evaluate_memory_variants
from melm.evaluation import memory_gate


class MemoryTests(unittest.TestCase):
    def test_event_memory_retrieves_expected_events(self) -> None:
        events, cases = episodic_memory_fixture()
        memory = EventMemory(events)
        comparison = evaluate_memory(memory, cases, k=2)
        self.assertEqual(comparison.cases, len(cases))
        self.assertGreaterEqual(comparison.event_memory_recall_at_k, comparison.rag_recall_at_k)
        self.assertGreaterEqual(comparison.event_memory_mrr_at_k, comparison.rag_mrr_at_k)

    def test_temporal_neighbors_are_expanded(self) -> None:
        events, _ = episodic_memory_fixture()
        memory = EventMemory(events)
        results = memory.retrieve_event_memory("What happened after e1?", k=2)
        ids = {result.event.event_id for result in results}
        self.assertIn("e2", ids)

    def test_temporal_before_keeps_target_neighbor_in_top_k(self) -> None:
        events, cases = generate_synthetic_episodic_benchmark(stories=1)
        memory = EventMemory(events)
        case = next(case for case in cases if case.category == "temporal_before")

        results = memory.retrieve_event_memory(case.query, k=2)
        ids = {result.event.event_id for result in results}

        self.assertIn(case.expected_event_id, ids)

    def test_causal_links_are_expanded_for_why_queries(self) -> None:
        events, _ = episodic_memory_fixture()
        memory = EventMemory(events)
        results = memory.retrieve_event_memory("Why did Nora know where the red cup was?", k=2)
        ids = {result.event.event_id for result in results}
        self.assertIn("e1", ids)

    def test_retrieval_config_can_disable_temporal_expansion(self) -> None:
        events, _ = episodic_memory_fixture()
        memory = EventMemory(events)
        results = memory.retrieve_structured(
            "What happened after e1?",
            k=2,
            config=RetrievalConfig(name="no_temporal", expand_temporal_neighbors=False),
        )
        self.assertTrue(results)

    def test_access_history_can_bias_recently_recalled_events(self) -> None:
        events, _ = episodic_memory_fixture()
        memory = EventMemory(events)
        memory.mark_access(["e1", "e1"])

        results = memory.retrieve_structured(
            "red cup",
            k=2,
            config=RetrievalConfig.with_access_history(),
        )

        self.assertGreaterEqual(memory.access_count("e1"), 2)
        self.assertIn("access", results[0].reason)

    def test_memory_variants_report_component_gains(self) -> None:
        events, cases = episodic_memory_fixture()
        memory = EventMemory(events)
        variants = evaluate_memory_variants(memory, cases, k=2)
        self.assertIn("event_memory", variants)
        self.assertIn("entity_action_temporal", variants)

    def test_memory_gate_matches_validation_plan(self) -> None:
        result = memory_gate(event_memory_recall=0.70, rag_recall=0.50)
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "event_memory_vs_rag")


if __name__ == "__main__":
    unittest.main()
