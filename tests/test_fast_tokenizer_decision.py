import unittest

from melm.tokenization import decide_fast_tokenizer_ablation


class FastTokenizerDecisionTests(unittest.TestCase):
    def test_promotes_candidate_when_bits_per_byte_wins(self) -> None:
        decision = decide_fast_tokenizer_ablation(
            [
                {"tokenizer": "capped_morpheme", "bits_per_byte": 2.3},
                {"tokenizer": "hf_bpe", "bits_per_byte": 2.5},
                {"tokenizer": "hf_unigram", "bits_per_byte": 2.6},
            ]
        )

        self.assertEqual(decision.decision, "promote_to_neural_ablation")
        self.assertEqual(decision.best_baseline_tokenizer, "hf_bpe")
        self.assertGreater(decision.bits_per_byte_gain, 0.0)

    def test_keeps_candidate_auxiliary_when_baseline_wins(self) -> None:
        decision = decide_fast_tokenizer_ablation(
            [
                {"tokenizer": "capped_morpheme", "bits_per_byte": 2.7},
                {"tokenizer": "hf_bpe", "bits_per_byte": 2.5},
            ]
        )

        self.assertEqual(decision.decision, "auxiliary_only")
        self.assertLess(decision.bits_per_byte_gain, 0.0)


if __name__ == "__main__":
    unittest.main()
