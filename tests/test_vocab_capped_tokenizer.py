import unittest

from melm.tokenization import WhitespaceTokenizer, cap_tokenizer_vocab


class VocabCappedTokenizerTests(unittest.TestCase):
    def test_capped_tokenizer_maps_rare_tokens_to_unk(self) -> None:
        tokenizer = cap_tokenizer_vocab(
            WhitespaceTokenizer(),
            ["common common rare"],
            vocab_size=2,
            name="capped",
        )

        self.assertEqual(tokenizer.tokenize("common rare other"), ["common", "<unk>", "<unk>"])
        self.assertEqual(tokenizer.name, "capped")


if __name__ == "__main__":
    unittest.main()
