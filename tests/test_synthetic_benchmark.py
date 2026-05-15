import unittest

from melm.benchmarks import generate_synthetic_episodic_benchmark
from melm.memory import EventMemory, evaluate_memory


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_generator_is_deterministic(self) -> None:
        events_a, cases_a = generate_synthetic_episodic_benchmark(stories=3, seed=7)
        events_b, cases_b = generate_synthetic_episodic_benchmark(stories=3, seed=7)
        self.assertEqual(events_a, events_b)
        self.assertEqual(cases_a, cases_b)

    def test_generator_sizes_match_story_count(self) -> None:
        events, cases = generate_synthetic_episodic_benchmark(stories=5, distractors_per_story=0)
        self.assertEqual(len(events), 25)
        self.assertEqual(len(cases), 40)

    def test_generator_adds_distractors(self) -> None:
        events, cases = generate_synthetic_episodic_benchmark(stories=5, distractors_per_story=2)
        self.assertEqual(len(events), 35)
        self.assertEqual(len(cases), 40)

    def test_evaluation_reports_categories(self) -> None:
        events, cases = generate_synthetic_episodic_benchmark(stories=2)
        comparison = evaluate_memory(EventMemory(events), cases, k=2)
        self.assertIsNotNone(comparison.by_category)
        assert comparison.by_category is not None
        self.assertIn("temporal_after", comparison.by_category)
        self.assertGreaterEqual(comparison.event_memory_mrr_at_k, 0.0)
        self.assertGreater(comparison.event_memory_mrr_at_k, 0.0)


if __name__ == "__main__":
    unittest.main()
