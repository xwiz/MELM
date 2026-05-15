"""Tiny state-grounding benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

from melm.grounding import StateSet


@dataclass(frozen=True)
class StateGroundingCase:
    case_id: str
    current: StateSet
    required: StateSet
    effects: StateSet
    should_be_valid: bool
    category: str


def state_grounding_fixture() -> list[StateGroundingCase]:
    return [
        StateGroundingCase(
            case_id="valid_open_door",
            current=StateSet.from_mapping({"physical": ["present"], "positional": ["near"]}),
            required=StateSet.from_mapping({"physical": ["present"]}),
            effects=StateSet.from_mapping({"physical": ["open"]}),
            should_be_valid=True,
            category="valid_transition",
        ),
        StateGroundingCase(
            case_id="missing_precondition",
            current=StateSet.from_mapping({"physical": ["present"]}),
            required=StateSet.from_mapping({"physical": ["present"], "mental": ["knows_code"]}),
            effects=StateSet.from_mapping({"physical": ["open"]}),
            should_be_valid=False,
            category="missing_precondition",
        ),
        StateGroundingCase(
            case_id="open_closed_conflict",
            current=StateSet.from_mapping({"physical": ["closed"]}),
            required=StateSet(),
            effects=StateSet.from_mapping({"physical": ["open"]}),
            should_be_valid=False,
            category="contradiction",
        ),
        StateGroundingCase(
            case_id="awake_asleep_conflict",
            current=StateSet.from_mapping({"mental": ["awake"]}),
            required=StateSet(),
            effects=StateSet.from_mapping({"mental": ["asleep"]}),
            should_be_valid=False,
            category="contradiction",
        ),
    ]
