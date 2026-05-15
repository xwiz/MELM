import unittest

from melm.evaluation import bootstrap_mean_ci, bootstrap_paired_difference_ci


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_mean_ci_is_deterministic(self) -> None:
        ci_a = bootstrap_mean_ci([1, 1, 0, 1], samples=100, seed=7)
        ci_b = bootstrap_mean_ci([1, 1, 0, 1], samples=100, seed=7)

        self.assertEqual(ci_a, ci_b)
        self.assertEqual(ci_a.estimate, 0.75)
        self.assertLessEqual(ci_a.low, ci_a.estimate)
        self.assertGreaterEqual(ci_a.high, ci_a.estimate)

    def test_paired_difference_ci_requires_equal_lengths(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_paired_difference_ci([True], [True, False])

    def test_paired_difference_ci_reports_candidate_gain(self) -> None:
        ci = bootstrap_paired_difference_ci(
            [True, True, True, False],
            [False, True, False, False],
            samples=100,
            seed=9,
        )

        self.assertEqual(ci.estimate, 0.5)


if __name__ == "__main__":
    unittest.main()
