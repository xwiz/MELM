import unittest

from melm.benchmarks import child_language_minimal_pairs_fixture
from melm.evaluation import evaluate_minimal_pair_scores


class MinimalPairTests(unittest.TestCase):
    def test_fixture_has_unique_cases(self) -> None:
        cases = child_language_minimal_pairs_fixture()

        self.assertGreaterEqual(len(cases), 4)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    def test_evaluate_minimal_pair_scores(self) -> None:
        cases = child_language_minimal_pairs_fixture()[:2]
        scores = {}
        for case in cases:
            scores[case.good] = 1.0
            scores[case.bad] = 2.0

        report = evaluate_minimal_pair_scores(cases, scores)

        self.assertEqual(report.cases, 2)
        self.assertEqual(report.correct, 2)
        self.assertEqual(report.accuracy, 1.0)
        self.assertIn("agreement", report.by_category)


if __name__ == "__main__":
    unittest.main()
