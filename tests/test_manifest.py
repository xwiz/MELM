import tempfile
from pathlib import Path
import unittest

from melm.data import CorpusManifest, save_manifest, scan_corpus, load_manifest, texts_for_split


class ManifestTests(unittest.TestCase):
    def test_scan_and_round_trip_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.md").write_text("beta", encoding="utf-8")
            manifest = scan_corpus(root, name="test", source="unit")
            self.assertEqual(len(manifest.documents), 2)

            path = root / "manifest.json"
            save_manifest(manifest, path)
            loaded = load_manifest(path)
            self.assertEqual(loaded, manifest)
            self.assertIsInstance(loaded, CorpusManifest)

    def test_texts_for_split_loads_matching_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            manifest = scan_corpus(root, name="test")
            split = manifest.documents[0].split
            self.assertEqual(texts_for_split(manifest, split), ["alpha"])

    def test_small_corpus_gets_validation_and_test_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(12):
                (root / f"{index}.txt").write_text(f"doc {index}", encoding="utf-8")
            manifest = scan_corpus(root, name="test")
            splits = {document.split for document in manifest.documents}
            self.assertEqual(splits, {"train", "validation", "test"})

    def test_scan_excludes_generated_report_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.md").write_text("keep", encoding="utf-8")
            reports = root / "reports"
            reports.mkdir()
            (reports / "skip.md").write_text("skip", encoding="utf-8")
            manifest = scan_corpus(root, name="test")
            self.assertEqual([Path(doc.path).name for doc in manifest.documents], ["keep.md"])


if __name__ == "__main__":
    unittest.main()
