import unittest

from melm.training import summaries_as_decision_reports, summarize_multiseed_reports


class MultiSeedTrainingTests(unittest.TestCase):
    def test_summarize_multiseed_reports_orders_by_mean_bits_per_byte(self) -> None:
        summaries = summarize_multiseed_reports(
            [
                {"tokenizer": "hf_bpe", "validation_bits_per_byte": 3.0, "validation_nll": 5.0, "parameters": 10, "validation_tokens": 7},
                {"tokenizer": "hf_bpe", "validation_bits_per_byte": 4.0, "validation_nll": 6.0, "parameters": 10, "validation_tokens": 7},
                {"tokenizer": "capped_morpheme", "validation_bits_per_byte": 2.0, "validation_nll": 5.5, "parameters": 10, "validation_tokens": 6},
                {"tokenizer": "capped_morpheme", "validation_bits_per_byte": 2.2, "validation_nll": 5.7, "parameters": 10, "validation_tokens": 6},
            ]
        )

        self.assertEqual([summary.tokenizer for summary in summaries], ["capped_morpheme", "hf_bpe"])
        self.assertAlmostEqual(summaries[0].mean_bits_per_byte, 2.1)
        self.assertGreater(summaries[1].std_bits_per_byte, 0.0)

    def test_summaries_can_feed_decision_helper(self) -> None:
        summaries = summarize_multiseed_reports(
            [
                {"tokenizer": "hf_bpe", "validation_bits_per_byte": 3.0, "validation_nll": 5.0, "parameters": 10, "validation_tokens": 7},
                {"tokenizer": "capped_morpheme", "validation_bits_per_byte": 2.0, "validation_nll": 5.5, "parameters": 10, "validation_tokens": 6},
            ]
        )

        reports = summaries_as_decision_reports(summaries)

        self.assertEqual(reports[0]["tokenizer"], "capped_morpheme")
        self.assertIn("validation_bits_per_byte", reports[0])


if __name__ == "__main__":
    unittest.main()
