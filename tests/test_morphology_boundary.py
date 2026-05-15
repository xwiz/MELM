import unittest

from melm.benchmarks import morphology_fixture
from melm.tokenization import (
    HeuristicMorphemeTokenizer,
    WhitespaceTokenizer,
    evaluate_boundary_f1,
)
from melm.tokenization.boundary import boundary_offsets, predicted_offsets


class MorphologyBoundaryTests(unittest.TestCase):
    def test_boundary_offsets(self) -> None:
        self.assertEqual(boundary_offsets(("un", "break", "able")), {2, 7})

    def test_predicted_offsets_for_heuristic_tokenizer(self) -> None:
        tokenizer = HeuristicMorphemeTokenizer()
        self.assertEqual(predicted_offsets(tokenizer, "unbreakable"), {2, 7})

    def test_heuristic_beats_whitespace_on_fixture(self) -> None:
        examples = morphology_fixture()
        heuristic = evaluate_boundary_f1(HeuristicMorphemeTokenizer(), examples)
        whitespace = evaluate_boundary_f1(WhitespaceTokenizer(), examples)
        self.assertGreater(heuristic.f1, whitespace.f1)


if __name__ == "__main__":
    unittest.main()
