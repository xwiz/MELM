import unittest

from melm.tokenization import evaluate_tokenizer_lm_stability, make_lm_folds


class TokenizerStabilityTests(unittest.TestCase):
    def test_make_lm_folds_splits_documents_deterministically(self) -> None:
        texts = ["doc one\ndoc two\ndoc three\ndoc four"]

        first = make_lm_folds(texts, folds=3, seed=7)
        second = make_lm_folds(texts, folds=3, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(train for train, _validation in first))
        self.assertTrue(all(validation for _train, validation in first))

    def test_evaluate_tokenizer_lm_stability_aggregates_fold_results(self) -> None:
        texts = [
            "\n".join(
                [
                    "maya replayed the memory game",
                    "leo replaying the memory game",
                    "maya unpacks the toy carefully",
                    "leo unpacked the toy carefully",
                    "nina remembers the red cup",
                    "nina remembered the blue cup",
                ]
            )
        ]

        report = evaluate_tokenizer_lm_stability(texts, folds=3, seed=11)

        self.assertEqual(report.folds, 3)
        self.assertEqual(report.documents, 6)
        self.assertEqual(sum(report.winner_counts.values()), 3)
        self.assertIn("heuristic_morpheme", report.average_nll_per_token)
        self.assertGreaterEqual(report.morph_win_rate, 0.0)
        self.assertLessEqual(report.morph_win_rate, 1.0)
        self.assertEqual(len(report.fold_reports), 3)


if __name__ == "__main__":
    unittest.main()
