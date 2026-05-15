import importlib.util
import tempfile
from pathlib import Path
import unittest

from melm.tokenization import WhitespaceTokenizer
from melm.training import (
    TinyLMConfig,
    build_token_vocabulary,
    evaluate_tiny_lm_checkpoint,
    make_lm_sequences,
    score_tiny_lm_continuations,
    score_tiny_lm_texts,
    train_tiny_lm_baseline,
)


class TinyLMTrainingTests(unittest.TestCase):
    def test_vocabulary_and_sequences_are_created(self) -> None:
        tokenizer = WhitespaceTokenizer()
        vocabulary = build_token_vocabulary(
            tokenizer,
            ["maya put cup", "leo moved book"],
            max_vocab_size=16,
        )
        sequences = make_lm_sequences(
            ["maya put cup"],
            tokenizer,
            vocabulary,
            sequence_length=6,
        )

        self.assertIn("maya", vocabulary.token_to_id)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(sequences[0][0]), 6)
        self.assertEqual(len(sequences[0][1]), 6)

    def test_vocabulary_can_be_padded_for_parameter_matching(self) -> None:
        vocabulary = build_token_vocabulary(
            WhitespaceTokenizer(),
            ["maya put cup"],
            max_vocab_size=16,
            pad_to_size=True,
        )

        self.assertEqual(len(vocabulary), 16)
        self.assertIn("<extra:15>", vocabulary.token_to_id)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_tiny_lm_training_smoke(self) -> None:
        tokenizer = WhitespaceTokenizer()
        config = TinyLMConfig(
            tokenizer_name=tokenizer.name,
            max_vocab_size=32,
            sequence_length=8,
            embedding_dim=16,
            layers=1,
            heads=2,
            batch_size=2,
            steps=2,
            seed=3,
            device="cpu",
        )

        report = train_tiny_lm_baseline(
            ["maya put cup", "leo moved book", "maya found cup"],
            ["leo found book"],
            tokenizer,
            config,
        )

        self.assertEqual(report.tokenizer, "whitespace")
        self.assertEqual(report.steps, 2)
        self.assertGreater(report.parameters, 0)
        self.assertGreater(report.validation_tokens, 0)
        self.assertGreater(report.validation_bytes, 0)
        self.assertGreater(report.validation_nll, 0.0)
        self.assertGreater(report.validation_bits_per_byte, 0.0)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_tiny_lm_training_can_save_checkpoint(self) -> None:
        tokenizer = WhitespaceTokenizer()
        config = TinyLMConfig(
            tokenizer_name=tokenizer.name,
            max_vocab_size=32,
            pad_vocab_to_max_size=True,
            sequence_length=8,
            embedding_dim=16,
            layers=1,
            heads=2,
            batch_size=2,
            steps=1,
            seed=5,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmp:
            report = train_tiny_lm_baseline(
                ["maya put cup", "leo moved book"],
                ["maya moved cup"],
                tokenizer,
                config,
                checkpoint_dir=tmp,
            )

            self.assertTrue((Path(tmp) / "model_state.pt").exists())
            self.assertTrue((Path(tmp) / "vocab.json").exists())
            self.assertTrue((Path(tmp) / "training_report.json").exists())

            evaluation = evaluate_tiny_lm_checkpoint(
                ["maya moved cup"],
                tokenizer,
                tmp,
                device="cpu",
            )

            self.assertEqual(evaluation.tokenizer, "whitespace")
            self.assertEqual(evaluation.parameters, report.parameters)
            self.assertAlmostEqual(
                evaluation.validation_bits_per_byte,
                report.validation_bits_per_byte,
                places=5,
            )

            scores = score_tiny_lm_texts(
                ["maya moved cup", "leo found book"],
                tokenizer,
                tmp,
                device="cpu",
            )

            self.assertEqual(len(scores), 2)
            self.assertGreater(scores[0].tokens, 0)
            self.assertGreater(scores[0].bits_per_byte, 0.0)

            continuation_scores = score_tiny_lm_continuations(
                ["maya moved "],
                ["cup"],
                tokenizer,
                tmp,
                device="cpu",
            )

            self.assertEqual(len(continuation_scores), 1)
            self.assertEqual(continuation_scores[0].text, "maya moved cup")
            self.assertGreater(continuation_scores[0].tokens, 0)


if __name__ == "__main__":
    unittest.main()
