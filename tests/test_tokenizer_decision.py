import unittest

from melm.tokenization import (
    BoundaryReport,
    TokenLMReport,
    decide_tokenizer_strategy,
)


class TokenizerDecisionTests(unittest.TestCase):
    def test_morphology_becomes_auxiliary_when_boundary_wins_but_lm_loses(self) -> None:
        report = decide_tokenizer_strategy(
            [
                TokenLMReport("unigram_like", 100, 20, 50, 4.0, 55.0),
                TokenLMReport("heuristic_morpheme", 100, 20, 50, 5.0, 150.0),
            ],
            [
                BoundaryReport("unigram_like", 10, 0.3, 0.3, 0.3, 0.0),
                BoundaryReport("heuristic_morpheme", 10, 1.0, 1.0, 1.0, 1.0),
            ],
        )

        self.assertEqual(report.decision, "auxiliary_only")
        self.assertFalse(report.gate_passed)
        self.assertLess(report.lm_nll_gain, 0.0)
        self.assertGreater(report.boundary_f1_gain, 0.0)

    def test_morphology_can_be_primary_when_lm_wins(self) -> None:
        report = decide_tokenizer_strategy(
            [
                TokenLMReport("simple_bpe", 100, 20, 50, 5.0, 150.0),
                TokenLMReport("heuristic_morpheme", 100, 20, 50, 4.5, 90.0),
            ],
            [
                BoundaryReport("simple_bpe", 10, 0.4, 0.4, 0.4, 0.0),
                BoundaryReport("heuristic_morpheme", 10, 1.0, 1.0, 1.0, 1.0),
            ],
        )

        self.assertEqual(report.decision, "primary_candidate")
        self.assertTrue(report.gate_passed)


if __name__ == "__main__":
    unittest.main()
