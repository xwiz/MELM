import unittest

from scripts.run_melm_letta_style_eval import (
    _answer_token_recall,
    _contains_score,
    _evidence_recall,
)


class LettaStyleEvalMetricTests(unittest.TestCase):
    def test_contains_score_accepts_order_independent_gold_terms(self) -> None:
        self.assertEqual(
            _contains_score("She visited the red clinic after lunch.", "red clinic"),
            1.0,
        )

    def test_answer_token_recall_ignores_common_words(self) -> None:
        self.assertEqual(
            _answer_token_recall("Caroline met Maya at the clinic.", "Maya at the clinic"),
            1.0,
        )

    def test_evidence_recall_scores_partial_session_coverage(self) -> None:
        self.assertEqual(
            _evidence_recall(["sample_1_session_1"], ["sample_1_session_1", "sample_1_session_2"]),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
