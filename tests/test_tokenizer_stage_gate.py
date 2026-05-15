import unittest

from melm.evaluation import decide_tokenizer_stage_gate


class TokenizerStageGateTests(unittest.TestCase):
    def test_advances_when_loss_blimp_and_entity_tolerances_pass(self) -> None:
        progression = {
            "progressions": [
                {
                    "tokenizer": "tiered_morph_unigram",
                    "points": [{"steps": 200, "mean_bits_per_byte": 2.6}],
                },
                {
                    "tokenizer": "hf_bpe",
                    "points": [{"steps": 200, "mean_bits_per_byte": 2.8}],
                },
                {
                    "tokenizer": "hf_unigram",
                    "points": [{"steps": 200, "mean_bits_per_byte": 2.9}],
                },
            ]
        }
        blimp = [
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.55}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.50}},
                    {"tokenizer": "hf_unigram", "report": {"accuracy": 0.51}},
                ]
            }
        ]
        entity = {
            "runs": [
                {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.41}},
                {"tokenizer": "hf_bpe", "report": {"accuracy": 0.42}},
                {"tokenizer": "hf_unigram", "report": {"accuracy": 0.39}},
            ]
        }

        decision = decide_tokenizer_stage_gate(progression, blimp, entity)

        self.assertEqual(decision.decision, "advance_to_small_model_ablation")
        self.assertEqual(decision.blimp_wins, 1)
        self.assertIsNone(decision.proxy_supports_scale)

    def test_advances_to_scaled_ablation_when_proxy_supports_scale(self) -> None:
        progression = {
            "progressions": [
                {
                    "tokenizer": "tiered_morph_unigram",
                    "points": [{"steps": 200, "mean_bits_per_byte": 2.6}],
                },
                {
                    "tokenizer": "hf_bpe",
                    "points": [{"steps": 200, "mean_bits_per_byte": 2.8}],
                },
            ]
        }
        blimp = [
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.55}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.50}},
                ]
            }
        ]
        entity = {
            "runs": [
                {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.41}},
                {"tokenizer": "hf_bpe", "report": {"accuracy": 0.42}},
            ]
        }
        proxy = {
            "decision": {
                "candidate_tokenizer": "tiered_morph_unigram",
                "best_baseline_tokenizer": "hf_bpe",
                "relative_bits_per_byte_gain": 0.03,
                "decision": "promote_to_scaled_neural_ablation",
            }
        }

        decision = decide_tokenizer_stage_gate(
            progression,
            blimp,
            entity,
            proxy_decision_payload=proxy,
        )

        self.assertEqual(decision.decision, "advance_to_scaled_neural_ablation")
        self.assertTrue(decision.proxy_supports_scale)

    def test_holds_when_proxy_rejects_scaled_ablation(self) -> None:
        progression = {
            "progressions": [
                {
                    "tokenizer": "tiered_morph_unigram",
                    "points": [{"steps": 200, "mean_bits_per_byte": 2.6}],
                },
                {
                    "tokenizer": "hf_bpe",
                    "points": [{"steps": 200, "mean_bits_per_byte": 2.8}],
                },
            ]
        }
        blimp = [
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.55}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.50}},
                ]
            }
        ]
        entity = {
            "runs": [
                {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.41}},
                {"tokenizer": "hf_bpe", "report": {"accuracy": 0.42}},
            ]
        }
        proxy = {
            "decision": {
                "candidate_tokenizer": "tiered_morph_unigram",
                "best_baseline_tokenizer": "hf_bpe",
                "relative_bits_per_byte_gain": 0.0,
                "decision": "do_not_scale_yet",
            }
        }

        decision = decide_tokenizer_stage_gate(
            progression,
            blimp,
            entity,
            proxy_decision_payload=proxy,
        )

        self.assertEqual(decision.decision, "hold_for_proxy_signal")
        self.assertFalse(decision.proxy_supports_scale)


if __name__ == "__main__":
    unittest.main()
