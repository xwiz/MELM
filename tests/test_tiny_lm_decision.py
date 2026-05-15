import unittest

from melm.training import decide_tiny_lm_tokenizer_ablation


class TinyLMDecisionTests(unittest.TestCase):
    def test_promotes_candidate_when_tiny_lm_bits_per_byte_wins(self) -> None:
        decision = decide_tiny_lm_tokenizer_ablation(
            [
                {"tokenizer": "capped_morpheme", "validation_bits_per_byte": 3.0},
                {"tokenizer": "hf_bpe", "validation_bits_per_byte": 3.5},
                {"tokenizer": "hf_unigram", "validation_bits_per_byte": 3.8},
            ]
        )

        self.assertEqual(decision.decision, "promote_to_scaled_neural_ablation")
        self.assertEqual(decision.best_baseline_tokenizer, "hf_bpe")
        self.assertGreater(decision.relative_bits_per_byte_gain, 0.0)

    def test_does_not_scale_when_baseline_wins(self) -> None:
        decision = decide_tiny_lm_tokenizer_ablation(
            [
                {"tokenizer": "capped_morpheme", "validation_bits_per_byte": 4.0},
                {"tokenizer": "hf_bpe", "validation_bits_per_byte": 3.5},
            ]
        )

        self.assertEqual(decision.decision, "do_not_scale_yet")
        self.assertLess(decision.bits_per_byte_gain, 0.0)


if __name__ == "__main__":
    unittest.main()
