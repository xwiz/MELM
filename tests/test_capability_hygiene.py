"""Slice 2: capability sub-flag accessor + creative-behavior condition validation."""

from __future__ import annotations

import unittest

from melm.appliance.local_assistant_router import _capability_flag
from melm.contracts.validation import (
    ContractValidationError,
    load_creative_behaviors,
    validate_creative_behaviors,
)


class CapabilityFlagTests(unittest.TestCase):
    def test_reads_nested_subflag_default_false(self):
        # mood_affect is installed; creative_behaviors ships false.
        self.assertFalse(_capability_flag("mood_affect", "creative_behaviors", False))

    def test_absent_family_returns_default(self):
        self.assertTrue(_capability_flag("no_such_family", "whatever", True))
        self.assertFalse(_capability_flag("no_such_family", "whatever", False))


class CreativeBehaviorConditionValidationTests(unittest.TestCase):
    def _payload(self, condition: str) -> dict:
        return {
            "schema_id": "melm.creative_behaviors.v1",
            "behaviors": [{
                "id": "b1", "trigger": "t", "condition": condition,
                "template": "x", "cooldown_turns": 1,
            }],
        }

    def test_unknown_condition_variable_rejected(self):
        with self.assertRaises(ContractValidationError):
            validate_creative_behaviors(self._payload("current_intent == 'social_greeting'"))

    def test_canonical_variable_accepted(self):
        validate_creative_behaviors(self._payload("intent == 'social_greeting'"))

    def test_complex_condition_with_abs_and_literals(self):
        validate_creative_behaviors(self._payload(
            "abs(ambient_valence_delta) > 0.04 AND intent == 'social_greeting' AND occurrence == 1"
        ))

    def test_shipped_contract_passes_after_drift_fix(self):
        loaded = load_creative_behaviors()
        # load returns {"behaviors": []} on validation failure; fixed contract → 5 behaviors.
        self.assertEqual(len(loaded.get("behaviors", [])), 5)


if __name__ == "__main__":
    unittest.main()
