import unittest

from melm.tokenization import SimpleBPETokenizer, train_bpe


class SimpleBPETests(unittest.TestCase):
    def test_train_bpe_learns_merges(self) -> None:
        tokenizer = train_bpe(["low lower lowest", "newer wider"], vocab_size=40)
        self.assertIsInstance(tokenizer, SimpleBPETokenizer)
        self.assertGreater(len(tokenizer.merges), 0)

    def test_bpe_tokenizes_without_end_marker(self) -> None:
        tokenizer = train_bpe(["lower lower lower"], vocab_size=30)
        tokens = tokenizer.tokenize("lower")
        self.assertTrue(tokens)
        self.assertFalse(any("</w>" in token for token in tokens))


if __name__ == "__main__":
    unittest.main()
