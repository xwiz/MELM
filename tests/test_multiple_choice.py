import unittest

from melm.benchmarks import MultipleChoiceCase
from melm.evaluation import evaluate_multiple_choice_scores


class MultipleChoiceTests(unittest.TestCase):
    def test_evaluate_multiple_choice_scores(self) -> None:
        case = MultipleChoiceCase(
            case_id="case_1",
            category="tracking",
            prompt="Box 1 contains ",
            options=("the hat.", "the map."),
            label_index=0,
        )
        report = evaluate_multiple_choice_scores(
            [case],
            {
                "Box 1 contains the hat.": 1.0,
                "Box 1 contains the map.": 2.0,
            },
        )

        self.assertEqual(report.cases, 1)
        self.assertEqual(report.correct, 1)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.predictions[0].predicted_index, 0)


if __name__ == "__main__":
    unittest.main()
