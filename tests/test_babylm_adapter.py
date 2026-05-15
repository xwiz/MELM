import tempfile
from pathlib import Path
import unittest

from melm.data import load_train_validation, save_manifest, scan_babylm_corpus


class BabyLMAdapterTests(unittest.TestCase):
    def test_scan_babylm_corpus_preserves_explicit_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "dev").mkdir()
            (root / "test").mkdir()
            (root / "train" / "a.txt").write_text("train text", encoding="utf-8")
            (root / "dev" / "b.txt").write_text("validation text", encoding="utf-8")
            (root / "test" / "c.txt").write_text("test text", encoding="utf-8")

            summary = scan_babylm_corpus(root, track="10M")

            self.assertEqual(summary.explicit_splits, 3)
            self.assertEqual(summary.fallback_splits, 0)
            self.assertEqual({doc.split for doc in summary.manifest.documents}, {"train", "validation", "test"})
            self.assertEqual({doc.source for doc in summary.manifest.documents}, {"babylm:10M"})

    def test_scan_babylm_corpus_detects_dotted_train_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bnc_spoken.train.txt").write_text("train text", encoding="utf-8")
            (root / "bnc_spoken.dev").write_text("validation text", encoding="utf-8")
            (root / "bnc_spoken.test").write_text("test text", encoding="utf-8")

            summary = scan_babylm_corpus(root)

            self.assertEqual(summary.explicit_splits, 3)
            self.assertEqual(summary.fallback_splits, 0)
            self.assertEqual({doc.split for doc in summary.manifest.documents}, {"train", "validation", "test"})

    def test_scan_babylm_corpus_falls_back_when_split_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(4):
                (root / f"doc_{index}.txt").write_text(f"text {index}", encoding="utf-8")

            summary = scan_babylm_corpus(root)

            self.assertEqual(summary.explicit_splits, 0)
            self.assertEqual(summary.fallback_splits, 4)
            self.assertIn("train", {doc.split for doc in summary.manifest.documents})

    def test_babylm_manifest_loads_train_validation_texts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            (corpus / "train").mkdir(parents=True)
            (corpus / "validation").mkdir()
            (corpus / "train" / "a.txt").write_text("alpha", encoding="utf-8")
            (corpus / "validation" / "b.txt").write_text("beta", encoding="utf-8")
            manifest_path = root / "manifest.json"
            summary = scan_babylm_corpus(corpus)
            save_manifest(summary.manifest, manifest_path)

            all_texts, train, validation, source = load_train_validation(manifest_path=manifest_path)

            self.assertEqual(sorted(all_texts), ["alpha", "beta"])
            self.assertEqual(train, ["alpha"])
            self.assertEqual(validation, ["beta"])
            self.assertTrue(source.startswith("manifest:"))


if __name__ == "__main__":
    unittest.main()
