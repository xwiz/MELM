import unittest

from melm.data import limit_texts_by_bytes


class SamplingTests(unittest.TestCase):
    def test_limit_texts_by_bytes_caps_total_size(self) -> None:
        samples = limit_texts_by_bytes(["a" * 100, "b" * 100], 50)

        self.assertLessEqual(sum(len(sample.encode("utf-8")) for sample in samples), 50)
        self.assertEqual(len(samples), 2)

    def test_limit_texts_by_bytes_preserves_uncapped_texts(self) -> None:
        texts = ["alpha", "beta"]

        self.assertEqual(limit_texts_by_bytes(texts, None), texts)
        self.assertEqual(limit_texts_by_bytes(texts, 0), texts)


if __name__ == "__main__":
    unittest.main()
