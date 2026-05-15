import unittest

from melm.evaluation import decide_small_model_stage_gate


class SmallModelStageGateTests(unittest.TestCase):
    def test_advances_when_loss_blimp_and_entity_pass(self) -> None:
        multiseed = {
            "summaries": [
                {"tokenizer": "tiered_morph_unigram", "mean_bits_per_byte": 1.9},
                {"tokenizer": "hf_bpe", "mean_bits_per_byte": 2.0},
                {"tokenizer": "hf_unigram", "mean_bits_per_byte": 2.1},
                {"tokenizer": "capped_morpheme", "mean_bits_per_byte": 1.7},
            ]
        }
        blimp = [
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.53}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.52}},
                ]
            },
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.50}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.51}},
                ]
            },
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.54}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.52}},
                ]
            },
        ]
        entity = {
            "runs": [
                {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.41}},
                {"tokenizer": "hf_bpe", "report": {"accuracy": 0.40}},
            ]
        }
        symbolic = {"report": {"accuracy": 1.0}}

        decision = decide_small_model_stage_gate(
            multiseed,
            blimp,
            entity,
            symbolic_entity_payload=symbolic,
        )

        self.assertEqual(decision.decision, "advance_to_event_memory_integration")
        self.assertEqual(decision.blimp_wins, 2)
        self.assertEqual(decision.compression_control, "capped_morpheme")
        self.assertEqual(decision.symbolic_entity_accuracy, 1.0)

    def test_holds_when_candidate_loses_bits_per_byte(self) -> None:
        multiseed = {
            "summaries": [
                {"tokenizer": "tiered_morph_unigram", "mean_bits_per_byte": 2.1},
                {"tokenizer": "hf_bpe", "mean_bits_per_byte": 2.0},
            ]
        }
        blimp = [
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.53}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.52}},
                ]
            },
            {
                "runs": [
                    {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.53}},
                    {"tokenizer": "hf_bpe", "report": {"accuracy": 0.52}},
                ]
            },
        ]
        entity = {
            "runs": [
                {"tokenizer": "tiered_morph_unigram", "report": {"accuracy": 0.41}},
                {"tokenizer": "hf_bpe", "report": {"accuracy": 0.40}},
            ]
        }

        decision = decide_small_model_stage_gate(multiseed, blimp, entity)

        self.assertEqual(decision.decision, "hold_for_loss_signal")


if __name__ == "__main__":
    unittest.main()
