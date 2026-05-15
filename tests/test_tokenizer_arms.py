import importlib.util
import unittest

from melm.tokenization import build_tokenizer_arms


class TokenizerArmsTests(unittest.TestCase):
    def test_builds_default_capped_and_hybrid_arms_in_order(self) -> None:
        arms = build_tokenizer_arms(
            ["whitespace", "capped_morpheme", "hybrid_morph_unigram"],
            ["maya putting cup"],
            ["maya putting cup"],
            tokenizer_vocab_size=80,
        )

        self.assertEqual(list(arms), ["whitespace", "capped_morpheme", "hybrid_morph_unigram"])
        self.assertEqual(arms["capped_morpheme"].name, "capped_morpheme")
        self.assertEqual(arms["hybrid_morph_unigram"].name, "hybrid_morph_unigram")

    @unittest.skipIf(importlib.util.find_spec("tokenizers") is None, "tokenizers is not installed")
    def test_builds_hf_arms(self) -> None:
        arms = build_tokenizer_arms(
            ["hf_bpe", "hf_unigram", "tiered_morph_unigram"],
            ["maya put cup", "leo moved cup"],
            ["maya put cup", "leo moved cup"],
            tokenizer_vocab_size=96,
        )

        self.assertEqual(list(arms), ["hf_bpe", "hf_unigram", "tiered_morph_unigram"])
        self.assertTrue(arms["hf_bpe"].tokenize("maya moved cup"))
        self.assertTrue(arms["tiered_morph_unigram"].tokenize("maya moved cup"))

    def test_unknown_arm_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_tokenizer_arms(
                ["unknown"],
                ["text"],
                ["text"],
                tokenizer_vocab_size=8,
            )


if __name__ == "__main__":
    unittest.main()
