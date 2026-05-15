import tempfile
from pathlib import Path
import unittest

from melm.data import load_texts


class DataTests(unittest.TestCase):
    def test_load_texts_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello world", encoding="utf-8")
            self.assertEqual(load_texts(path), ["hello world"])

    def test_load_texts_from_directory_filters_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.md").write_text("b", encoding="utf-8")
            (root / "c.json").write_text("c", encoding="utf-8")
            self.assertEqual(load_texts(root), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
