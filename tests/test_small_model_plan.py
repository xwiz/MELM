import unittest

from melm.training import (
    build_small_model_stage_plan,
    estimate_tiny_decoder_parameters,
    estimate_training_flops,
    small_model_spec_from_mapping,
)


class SmallModelPlanTests(unittest.TestCase):
    def test_estimates_tiny_decoder_parameters(self) -> None:
        parameters = estimate_tiny_decoder_parameters(
            vocab_size=10,
            sequence_length=5,
            embedding_dim=4,
            layers=2,
        )

        self.assertEqual(parameters, 598)

    def test_estimates_lower_bound_training_flops(self) -> None:
        self.assertEqual(
            estimate_training_flops(
                parameters=100,
                steps=2,
                batch_size=3,
                sequence_length=4,
            ),
            14400,
        )

    def test_builds_run_card_with_dependency_status_and_commands(self) -> None:
        spec = small_model_spec_from_mapping(
            {
                "name": "stage",
                "manifest": "reports/manifest.json",
                "tokenizers": ["hf_bpe", "tiered_morph_unigram"],
                "candidate": "tiered_morph_unigram",
                "seeds": [3, 13],
                "max_train_bytes": 1000,
                "max_validation_bytes": 500,
                "steps": 2,
                "sequence_length": 8,
                "embedding_dim": 16,
                "layers": 1,
                "heads": 4,
                "batch_size": 2,
                "max_vocab_size": 64,
                "tokenizer_vocab_size": 64,
                "learning_rate": 0.001,
                "checkpoint_seed": 13,
                "artifact_root": "artifacts/stage",
                "report_prefix": "reports/stage",
            }
        )
        gate = {"decision": {"decision": "advance_to_scaled_neural_ablation"}}
        proxy = {
            "decision": {
                "decision": "promote_to_scaled_neural_ablation",
                "candidate_tokenizer": "tiered_morph_unigram",
            }
        }

        plan = build_small_model_stage_plan(spec, gate_payload=gate, proxy_payload=proxy)

        self.assertTrue(plan["dependency_status"]["pass"])
        self.assertEqual(plan["estimates"]["training_tokens_per_arm"], 32)
        self.assertIn("preflight_small_model_stage.py", plan["commands"][0])
        self.assertIn("--candidate tiered_morph_unigram", plan["commands"][1])
        self.assertIn("--resume", plan["commands"][1])
        self.assertIn("--run-cache-dir artifacts/stage\\multiseed_cache", plan["commands"][1])
        self.assertEqual(len(plan["commands"]), 11)

    def test_rejects_candidate_outside_tokenizer_set(self) -> None:
        with self.assertRaises(ValueError):
            small_model_spec_from_mapping(
                {
                    "name": "stage",
                    "manifest": "reports/manifest.json",
                    "tokenizers": ["hf_bpe"],
                    "candidate": "tiered_morph_unigram",
                    "seeds": [13],
                    "max_train_bytes": 1000,
                    "max_validation_bytes": 500,
                    "steps": 2,
                    "sequence_length": 8,
                    "embedding_dim": 16,
                    "layers": 1,
                    "heads": 4,
                    "batch_size": 2,
                    "max_vocab_size": 64,
                    "tokenizer_vocab_size": 64,
                    "learning_rate": 0.001,
                    "checkpoint_seed": 13,
                    "artifact_root": "artifacts/stage",
                    "report_prefix": "reports/stage",
                }
            )


if __name__ == "__main__":
    unittest.main()
