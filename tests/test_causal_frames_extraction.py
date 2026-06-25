"""Test the LLM extraction script's validation logic.
"""
import json
from pathlib import Path
from scripts.extract_causal_frames_llm import (
    validate_llm_output,
    extract_json_from_llm_response,
    _score_for_verb,
    GROUND_TRUTH,
)

SAMPLE_GOOD = {
    "predicate": {
        "predicate_id": "run",
        "semantic_class": "verb.move",
        "atom_kind": "event",
        "default_cause_kind": "intentional_action",
        "roles": ["agent"],
        "effects": [
            {"target_role": "agent", "state": "moved", "domain": "physical", "relation": "causes", "confidence": 0.90},
            {"target_role": "agent", "state": "tired", "domain": "physical", "relation": "causes", "confidence": 0.70},
            {"target_role": "agent", "state": "fit", "domain": "physical", "relation": "causes", "confidence": 0.65},
        ],
        "surface_aliases": ["run", "runs", "ran", "running"],
    },
    "states": {
        "moved": {"aliases": ["moved"], "definition": "has changed physical position", "opposites": ["stationary"]},
        "tired": {"aliases": ["tired", "fatigued"], "definition": "is in need of rest", "opposites": ["rested"]},
        "fit": {"aliases": ["fit", "healthy"], "definition": "has good physical condition", "opposites": ["unfit"]},
    },
}

SAMPLE_MISSING_STATES = {
    "predicate": {
        "predicate_id": "run",
        "semantic_class": "verb.move",
        "atom_kind": "event",
        "default_cause_kind": "intentional_action",
        "roles": ["agent"],
        "effects": [
            {"target_role": "agent", "state": "moved", "domain": "physical", "relation": "causes", "confidence": 0.90},
        ],
        "surface_aliases": ["run", "runs", "ran", "running"],
    },
    "states": {},  # moved is used but not defined
}

SAMPLE_BAD_CLASS = {
    "predicate": {
        "predicate_id": "run",
        "semantic_class": "verb.bogus",
        "atom_kind": "event",
        "default_cause_kind": "intentional_action",
        "roles": ["agent"],
        "effects": [],
        "surface_aliases": ["run"],
    },
    "states": {},
}

SAMPLE_BAD_CAUSE = {
    "predicate": {
        "predicate_id": "run",
        "semantic_class": "verb.move",
        "atom_kind": "event",
        "default_cause_kind": "magical",
        "roles": ["agent"],
        "effects": [],
        "surface_aliases": ["run"],
    },
    "states": {},
}


def test_validate_llm_output_good():
    passed, errors, cleaned = validate_llm_output(SAMPLE_GOOD, "run")
    assert passed, f"Expected pass, got: {errors}"
    assert cleaned is not None
    assert cleaned["predicate"]["predicate_id"] == "run"
    assert len(cleaned["predicate"]["effects"]) == 3
    assert "moved" in cleaned["states"]


def test_validate_llm_output_missing_states():
    passed, errors, _ = validate_llm_output(SAMPLE_MISSING_STATES, "run")
    assert not passed, "Should fail: 'moved' used but not defined"
    assert any("moved" in e for e in errors), f"Expected error about 'moved', got: {errors}"


def test_validate_llm_output_bad_class():
    passed, errors, _ = validate_llm_output(SAMPLE_BAD_CLASS, "run")
    assert not passed
    assert any("bogus" in e for e in errors)


def test_validate_llm_output_bad_cause_kind():
    passed, errors, _ = validate_llm_output(SAMPLE_BAD_CAUSE, "run")
    assert not passed
    assert any("magical" in e for e in errors)


def test_validate_llm_output_wrong_verb():
    passed, errors, _ = validate_llm_output(SAMPLE_GOOD, "swim")
    assert not passed
    assert any("doesn't match" in e for e in errors)


def test_extract_json_from_llm_response_plain():
    data = extract_json_from_llm_response('{"predicate": {"predicate_id": "test"}}')
    assert data is not None
    assert data["predicate"]["predicate_id"] == "test"


def test_extract_json_from_llm_response_fenced():
    raw = '```json\n{"predicate": {"predicate_id": "test"}}\n```'
    data = extract_json_from_llm_response(raw)
    assert data is not None
    assert data["predicate"]["predicate_id"] == "test"


def test_extract_json_from_llm_response_no_json():
    data = extract_json_from_llm_response("I don't know how to do that.")
    assert data is None


def test_score_for_verb_ground_truth():
    """Ground truth against itself = all 1.0."""
    for verb, expected in GROUND_TRUTH.items():
        scores = _score_for_verb(expected, expected)
        for dim, v in scores.items():
            assert v == 1.0, f"{verb}.{dim} = {v}, expected 1.0"


def test_score_for_verb_none():
    for verb, expected in GROUND_TRUTH.items():
        scores = _score_for_verb(None, expected)
        for dim, v in scores.items():
            assert v == 0.0, f"{verb}.{dim} = {v}, expected 0.0"


def test_score_for_verb_partial():
    """Partial match should produce intermediate scores."""
    eat_expected = GROUND_TRUTH["eat"]
    # Predicted has same class but fewer effects
    predicted = {
        "predicate": {
            "predicate_id": "eat",
            "semantic_class": "verb.consume",
            "default_cause_kind": "intentional_action",
            "effects": [
                {"target_role": "agent", "state": "satisfied", "domain": "emotional", "relation": "causes", "confidence": 0.88},
            ],
        },
        "states": {"satisfied": {"aliases": ["satisfied"], "definition": "has had desire fulfilled"}},
    }
    scores = _score_for_verb(predicted, eat_expected)
    assert scores["schema"] == 1.0
    assert scores["semantic_class"] == 1.0
    assert scores["cause_kind"] == 1.0
    assert 0.0 < scores["effects_coverage"] < 1.0, f"Expected partial effects coverage, got {scores['effects_coverage']}"
    assert scores["states_quality"] == 1.0, f"Single effect state is defined, expected 1.0 got {scores['states_quality']}"
