import unittest

from melm.benchmarks import (
    BoxStateTracker,
    MultipleChoiceCase,
    entity_tracking_events_from_prompt,
    evaluate_entity_tracking_symbolic,
    predict_entity_tracking_option,
)


class EntityTrackingSymbolicTests(unittest.TestCase):
    def test_predicts_initial_box_contents(self) -> None:
        case = MultipleChoiceCase(
            case_id="c1",
            category="entity",
            prompt=(
                "Box 0 contains the apple and the map, Box 1 contains nothing. "
                "Box 0 contains "
            ),
            options=("the apple and the map.", "nothing."),
            label_index=0,
        )

        self.assertEqual(predict_entity_tracking_option(case), 0)

    def test_applies_move_remove_and_put_operations(self) -> None:
        case = MultipleChoiceCase(
            case_id="c1",
            category="entity",
            prompt=(
                "Box 0 contains the apple and the map, Box 1 contains the key. "
                "Move the apple from Box 0 to Box 1. "
                "Remove the key from Box 1. "
                "Put the ring into Box 1. "
                "Box 1 contains "
            ),
            options=("the apple and the ring.", "the key and the ring.", "the map."),
            label_index=0,
        )

        self.assertEqual(predict_entity_tracking_option(case), 0)

    def test_evaluates_report_with_abstention_for_unparsed_query(self) -> None:
        cases = [
            MultipleChoiceCase(
                case_id="ok",
                category="entity",
                prompt="Box 0 contains the key. Box 0 contains ",
                options=("the key.", "nothing."),
                label_index=0,
            ),
            MultipleChoiceCase(
                case_id="bad",
                category="entity",
                prompt="Box 0 contains the key. What is inside? ",
                options=("the key.", "nothing."),
                label_index=0,
            ),
        ]

        report = evaluate_entity_tracking_symbolic(cases)

        self.assertEqual(report.cases, 2)
        self.assertEqual(report.correct, 1)
        self.assertEqual(report.abstentions, 1)

    def test_tracker_sorts_contents_for_option_matching(self) -> None:
        tracker, query_box = BoxStateTracker.from_prompt(
            "Box 0 contains the map. Put the apple into Box 0. Box 0 contains "
        )

        self.assertEqual(query_box, 0)
        self.assertEqual(tracker.contents(0), ("apple", "map"))

    def test_entity_tracking_prompt_compiles_to_events(self) -> None:
        events, query_box = entity_tracking_events_from_prompt(
            (
                "Box 0 contains the map, Box 1 contains nothing. "
                "Move the map from Box 0 to Box 1. "
                "Box 1 contains "
            ),
            case_id="case_a",
        )
        tracker = BoxStateTracker.from_events(events)

        self.assertEqual(query_box, 1)
        self.assertEqual(tracker.contents(1), ("map",))
        self.assertEqual(events[0].event_id, "case_a_e000")
        self.assertEqual(events[0].next_event_id, "case_a_e001")
        self.assertEqual(events[-1].previous_event_id, "case_a_e001")


if __name__ == "__main__":
    unittest.main()
