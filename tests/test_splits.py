import tempfile
from pathlib import Path
import unittest

from melm.data import load_train_validation, save_manifest, scan_corpus


class SplitLoadingTests(unittest.TestCase):
    def test_load_train_validation_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.md"
            path.write_text("a\nb\nc", encoding="utf-8")
            all_texts, train, validation, source = load_train_validation(path=path)
            self.assertEqual(source, str(path))
            self.assertTrue(all_texts)
            self.assertTrue(train)
            self.assertTrue(validation)

    def test_load_train_validation_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(12):
                (root / f"{index}.txt").write_text(f"doc {index}", encoding="utf-8")
            manifest = scan_corpus(root, name="unit")
            manifest_path = root / "manifest.json"
            save_manifest(manifest, manifest_path)

            all_texts, train, validation, source = load_train_validation(manifest_path=manifest_path)
            self.assertEqual(source, f"manifest:{manifest_path}")
            self.assertEqual(len(all_texts), 12)
            self.assertTrue(train)
            self.assertTrue(validation)


if __name__ == "__main__":
    unittest.main()
