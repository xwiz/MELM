import unittest

from melm.benchmarks import state_grounding_fixture
from melm.grounding import StateSet, check_transition


class GroundingTests(unittest.TestCase):
    def test_state_satisfies_required_dimensions(self) -> None:
        current = StateSet.from_mapping({"physical": ["present"], "mental": ["aware"]})
        required = StateSet.from_mapping({"physical": ["present"]})
        self.assertTrue(current.satisfies(required))

    def test_missing_preconditions_are_reported(self) -> None:
        current = StateSet.from_mapping({"physical": ["present"]})
        required = StateSet.from_mapping({"physical": ["present"], "mental": ["aware"]})
        result = check_transition(current, required, StateSet())
        self.assertFalse(result.valid)
        self.assertEqual(result.missing, {"mental": {"aware"}})

    def test_conflicts_are_reported(self) -> None:
        current = StateSet.from_mapping({"physical": ["closed"]})
        effects = StateSet.from_mapping({"physical": ["open"]})
        result = check_transition(current, StateSet(), effects)
        self.assertFalse(result.valid)
        self.assertEqual(result.conflicts, [("physical", "open", "closed")])

    def test_state_grounding_fixture_matches_expected_validity(self) -> None:
        for case in state_grounding_fixture():
            result = check_transition(case.current, case.required, case.effects)
            self.assertEqual(result.valid, case.should_be_valid, case.case_id)


if __name__ == "__main__":
    unittest.main()
