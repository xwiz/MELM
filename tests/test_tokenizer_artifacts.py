import tempfile
from pathlib import Path
import unittest

from melm.tokenization import (
    WhitespaceTokenizer,
    cap_tokenizer_vocab,
    load_tokenizer_artifact,
    save_tokenizer_artifact,
    train_hybrid_morph_unigram,
    train_tiered_morph_unigram,
)


class TokenizerArtifactTests(unittest.TestCase):
    def test_save_capped_tokenizer_artifact(self) -> None:
        tokenizer = cap_tokenizer_vocab(
            WhitespaceTokenizer(),
            ["maya put cup"],
            vocab_size=8,
            name="capped",
        )
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = save_tokenizer_artifact(tokenizer, tmp)

            self.assertTrue(metadata_path.exists())
            self.assertTrue((Path(tmp) / "vocab.json").exists())

            loaded = load_tokenizer_artifact(metadata_path)

            self.assertEqual(loaded.name, "capped")
            self.assertEqual(loaded.tokenize("maya put unknown"), ["maya", "put", "<unk>"])

    def test_save_hybrid_tokenizer_artifact(self) -> None:
        tokenizer = train_hybrid_morph_unigram(
            ["Maya is running. Maya is running. Leo is replaying."],
            vocab_size=80,
        )
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = save_tokenizer_artifact(tokenizer, tmp)

            loaded = load_tokenizer_artifact(metadata_path)

            self.assertEqual(loaded.name, "hybrid_morph_unigram")
            self.assertEqual(loaded.tokenize("running"), tokenizer.tokenize("running"))

    def test_save_tiered_tokenizer_artifact(self) -> None:
        tokenizer = train_tiered_morph_unigram(
            ["Maya is running. Maya is running. Leo is replaying."],
            vocab_size=96,
        )
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = save_tokenizer_artifact(tokenizer, tmp)

            loaded = load_tokenizer_artifact(metadata_path)

            self.assertEqual(loaded.name, "tiered_morph_unigram")
            self.assertEqual(loaded.tokenize("running"), tokenizer.tokenize("running"))


if __name__ == "__main__":
    unittest.main()
