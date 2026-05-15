import unittest

from melm.tokenization import (
    HeuristicMorphemeTokenizer,
    WhitespaceTokenizer,
    build_default_tokenizers,
    compare_unigram_lms,
    evaluate_unigram_lm,
    split_for_lm,
)


class TokenLMTests(unittest.TestCase):
    def test_split_for_lm_expands_lines(self) -> None:
        train, validation = split_for_lm(["a\nb\nc\nd"], validation_fraction=0.25)
        self.assertEqual(train, ["a", "b", "c"])
        self.assertEqual(validation, ["d"])

    def test_unigram_lm_report_has_loss(self) -> None:
        report = evaluate_unigram_lm(
            WhitespaceTokenizer(),
            ["maya put cup", "leo moved book"],
            ["maya moved cup"],
        )
        self.assertGreater(report.nll_per_token, 0.0)
        self.assertGreater(report.perplexity, 1.0)
        self.assertGreater(report.bits_per_byte, 0.0)
        self.assertGreater(report.validation_bytes, 0)

    def test_compare_unigram_lms_sorts_by_loss(self) -> None:
        reports = compare_unigram_lms(
            [WhitespaceTokenizer(), HeuristicMorphemeTokenizer()],
            ["maya replaying", "maya replaying"],
            ["maya replaying"],
        )
        self.assertEqual(len(reports), 2)
        self.assertLessEqual(reports[0].nll_per_token, reports[-1].nll_per_token)

    def test_default_tokenizers_include_simple_bpe(self) -> None:
        tokenizers = build_default_tokenizers(["maya replaying"], train_texts=["maya replaying"])
        self.assertIn("simple_bpe", {tokenizer.name for tokenizer in tokenizers})


if __name__ == "__main__":
    unittest.main()
