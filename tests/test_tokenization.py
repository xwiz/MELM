import unittest

from melm.tokenization import (
    BytePatchTokenizer,
    HeuristicMorphemeTokenizer,
    HybridMorphUnigramTokenizer,
    TieredMorphUnigramTokenizer,
    UnigramLikeTokenizer,
    WhitespaceTokenizer,
    compare_tokenizers,
    evaluate_tokenizer,
    train_hybrid_morph_unigram,
    train_tiered_morph_unigram,
)


class TokenizationTests(unittest.TestCase):
    def test_heuristic_morpheme_splits_common_affixes(self) -> None:
        tokenizer = HeuristicMorphemeTokenizer()
        self.assertEqual(tokenizer.tokenize("unbreakable"), ["un+", "break", "+able"])
        self.assertEqual(tokenizer.tokenize("replaying"), ["re+", "play", "+ing"])

    def test_reports_include_required_gate_metrics(self) -> None:
        texts = ["Maya replayed the story.", "The cup is unbreakable."]
        report = evaluate_tokenizer(HeuristicMorphemeTokenizer(), texts)
        self.assertEqual(report.documents, 2)
        self.assertGreater(report.words, 0)
        self.assertGreater(report.tokens, 0)
        self.assertGreaterEqual(report.fallback_rate, 0.0)

    def test_compare_tokenizers_sorts_reports(self) -> None:
        texts = ["Maya put the red cup on the table."]
        reports = compare_tokenizers(
            [
                BytePatchTokenizer(patch_size=2),
                WhitespaceTokenizer(),
                UnigramLikeTokenizer(frozenset({"maya", "put", "red", "cup", "table"})),
            ],
            texts,
        )
        self.assertEqual(len(reports), 3)
        self.assertLessEqual(reports[0].tokens_per_word, reports[-1].tokens_per_word)

    def test_hybrid_morph_unigram_uses_tiers(self) -> None:
        tokenizer = HybridMorphUnigramTokenizer(
            whole_words=frozenset({"running"}),
            morpheme_vocab=frozenset({"+ing", "run", "<char:x>", "<char:y>", "<char:z>"}),
        )

        self.assertEqual(tokenizer.tokenize("running"), ["running"])
        self.assertEqual(tokenizer.tokenize("playing"), ["<char:p>", "<char:l>", "<char:a>", "<char:y>", "+ing"])
        self.assertEqual(tokenizer.tokenize("123"), ["<num>"])

    def test_train_hybrid_morph_unigram(self) -> None:
        tokenizer = train_hybrid_morph_unigram(
            ["Maya is running. Maya is running. Leo is replaying."],
            vocab_size=80,
        )

        self.assertIn("running", tokenizer.whole_words)
        self.assertTrue(tokenizer.tokenize("unfamiliar"))

    def test_train_tiered_morph_unigram(self) -> None:
        tokenizer = train_tiered_morph_unigram(
            ["Maya is running. Maya is running. Leo is replaying."],
            vocab_size=96,
        )

        self.assertIsInstance(tokenizer, TieredMorphUnigramTokenizer)
        self.assertTrue(tokenizer.tokenize("running"))


if __name__ == "__main__":
    unittest.main()
