import importlib.util
import unittest

from melm.tokenization import train_hf_bpe, train_hf_unigram


@unittest.skipIf(importlib.util.find_spec("tokenizers") is None, "tokenizers is not installed")
class HFTokenizerTests(unittest.TestCase):
    def test_hf_bpe_tokenizes_after_training(self) -> None:
        tokenizer = train_hf_bpe(
            ["Maya put the cup", "Leo moved the cup"],
            vocab_size=32,
        )

        tokens = tokenizer.tokenize("Maya moved the cup")

        self.assertEqual(tokenizer.name, "hf_bpe")
        self.assertTrue(tokens)

    def test_hf_unigram_tokenizes_after_training(self) -> None:
        tokenizer = train_hf_unigram(
            ["Maya put the cup", "Leo moved the cup"],
            vocab_size=32,
        )

        tokens = tokenizer.tokenize("Maya moved the cup")

        self.assertEqual(tokenizer.name, "hf_unigram")
        self.assertTrue(tokens)


if __name__ == "__main__":
    unittest.main()
