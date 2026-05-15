import unittest

from melm.evaluation import decide_checkpoint_tokenizer_validation


class CheckpointDecisionTests(unittest.TestCase):
    def test_holds_when_quality_smoke_lags(self) -> None:
        artifact_runs = [
            {"evaluation": {"tokenizer": "capped_morpheme", "validation_bits_per_byte": 2.0}},
            {"evaluation": {"tokenizer": "hf_bpe", "validation_bits_per_byte": 3.0}},
        ]
        minimal_pair_runs = [
            {"tokenizer": "capped_morpheme", "report": {"accuracy": 0.4}},
            {"tokenizer": "hf_bpe", "report": {"accuracy": 0.7}},
        ]

        decision = decide_checkpoint_tokenizer_validation(
            artifact_runs,
            minimal_pair_runs,
        )

        self.assertEqual(decision.decision, "hold_for_quality_evidence")
        self.assertGreater(decision.relative_bits_per_byte_gain, 0.0)
        self.assertLess(decision.minimal_pair_accuracy_delta, 0.0)

    def test_promotes_when_loss_and_quality_both_win(self) -> None:
        artifact_runs = [
            {"evaluation": {"tokenizer": "capped_morpheme", "validation_bits_per_byte": 2.0}},
            {"evaluation": {"tokenizer": "hf_bpe", "validation_bits_per_byte": 3.0}},
        ]
        minimal_pair_runs = [
            {"tokenizer": "capped_morpheme", "report": {"accuracy": 0.8}},
            {"tokenizer": "hf_bpe", "report": {"accuracy": 0.7}},
        ]

        decision = decide_checkpoint_tokenizer_validation(
            artifact_runs,
            minimal_pair_runs,
        )

        self.assertEqual(decision.decision, "promote_to_small_model_training")


if __name__ == "__main__":
    unittest.main()
