import unittest

from melm.benchmarks import synthetic_state_resolution_fixture
from melm.memory import ObjectLocationTracker, evaluate_state_resolution


class StateResolutionTests(unittest.TestCase):
    def test_object_location_tracker_resolves_latest_and_before_move(self) -> None:
        events, cases = synthetic_state_resolution_fixture(stories=1)
        tracker = ObjectLocationTracker(events)
        latest_case = next(case for case in cases if case.category == "latest_after_move")
        before_case = next(case for case in cases if case.category == "before_move")

        latest = tracker.resolve_location(latest_case.object_name)
        before = tracker.resolve_location(
            before_case.object_name,
            before_event_id=before_case.before_event_id,
        )

        self.assertIsNotNone(latest)
        self.assertIsNotNone(before)
        self.assertEqual(latest.location, latest_case.expected_location)
        self.assertEqual(before.location, before_case.expected_location)

    def test_unknown_object_abstains(self) -> None:
        events, cases = synthetic_state_resolution_fixture(stories=1)
        tracker = ObjectLocationTracker(events)
        unknown_case = next(case for case in cases if case.category == "unknown_object")

        observation = tracker.resolve_location(unknown_case.object_name)

        self.assertIsNone(observation)

    def test_state_resolution_report_groups_categories(self) -> None:
        events, cases = synthetic_state_resolution_fixture(stories=3)

        report = evaluate_state_resolution(events, cases)

        self.assertEqual(report.cases, 12)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.false_positive_rate, 0.0)
        self.assertIn("latest_after_move", report.by_category or {})
        self.assertIn("unknown_object", report.by_category or {})


if __name__ == "__main__":
    unittest.main()
