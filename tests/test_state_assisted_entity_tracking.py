import unittest

from melm.benchmarks import MultipleChoiceCase
from melm.evaluation import evaluate_state_assisted_entity_tracking


class StateAssistedEntityTrackingTests(unittest.TestCase):
    def test_state_memory_overrides_wrong_lm_prediction(self) -> None:
        case = MultipleChoiceCase(
            case_id="regular:1",
            category="regular_0_ops",
            prompt="Box 0 contains the key, Box 1 contains nothing. Box 0 contains ",
            options=("the key.", "nothing."),
            label_index=0,
        )

        report = evaluate_state_assisted_entity_tracking(
            [case],
            [{"case_id": "regular:1", "predicted_index": 1}],
        )

        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.lm_accuracy, 0.0)
        self.assertEqual(report.accuracy_lift, 1.0)
        self.assertEqual(report.predictions[0].source, "state_memory")

    def test_falls_back_to_lm_when_state_abstains(self) -> None:
        case = MultipleChoiceCase(
            case_id="regular:2",
            category="unparsed",
            prompt="Box 0 contains the key. What is inside? ",
            options=("the key.", "nothing."),
            label_index=0,
        )

        report = evaluate_state_assisted_entity_tracking(
            [case],
            [{"case_id": "regular:2", "predicted_index": 0}],
        )

        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.state_answer_rate, 0.0)
        self.assertEqual(report.lm_fallbacks, 1)
        self.assertEqual(report.predictions[0].source, "lm_fallback")

    def test_duplicate_case_ids_use_prediction_order(self) -> None:
        cases = [
            MultipleChoiceCase(
                case_id="regular:1",
                category="regular_0_ops",
                prompt="Box 0 contains the key. What is inside? ",
                options=("the key.", "nothing."),
                label_index=0,
            ),
            MultipleChoiceCase(
                case_id="regular:1",
                category="regular_0_ops",
                prompt="Box 0 contains the key. What is inside? ",
                options=("the key.", "nothing."),
                label_index=0,
            ),
        ]

        report = evaluate_state_assisted_entity_tracking(
            cases,
            [
                {"case_id": "regular:1", "predicted_index": 1},
                {"case_id": "regular:1", "predicted_index": 0},
            ],
        )

        self.assertEqual(report.lm_accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
