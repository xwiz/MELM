"""Tests for causal contract validators (invalid input rejection), causal_contrast
detector/solver, and causal_link_markers.v1.json contract loading."""

from __future__ import annotations

import pytest

from melm.contracts import (
    load_causal_link_markers,
    validate_causal_cues,
    validate_causal_effects,
    validate_causal_frames,
    validate_causal_link_markers,
)
from melm.contracts.validation import ContractValidationError


# ── Cross-validator schema_id rejection ──────────────────────────────


@pytest.mark.parametrize("validator,contract_name,base_payload", [
    (validate_causal_cues, "causal_cues", {"cues": []}),
    (validate_causal_effects, "causal_effects", {"rules": {}}),
    (validate_causal_frames, "causal_frames", {"predicate_frames": {}}),
    (validate_causal_link_markers, "causal_link_markers", {"markers": {}}),
])
def test_rejects_wrong_schema_id(validator, contract_name, base_payload):
    candidate = {"schema_id": f"melm.wrong_{contract_name}.v1", **base_payload}
    with pytest.raises(ContractValidationError, match="schema_id"):
        validator(candidate)


# ── Task 12: causal_link_markers contract load test ──────────────────


def test_causal_link_markers_loads():
    markers = load_causal_link_markers()
    assert "because" in markers
    assert "if" in markers
    entry = markers["because"]
    assert entry["relation"] in ("caused_by", "causes")


# ── Task 10: Causal contract validator unit tests ────────────────────


class TestValidateCausalCues:
    def test_valid_passes(self):
        payload = {
            "schema_id": "melm.causal_cues.v1",
            "cues": [
                {"lemma": "why", "cue_type": "causal_explanation",
                 "direction": "effect_to_cause", "language": "en",
                 "confidence": 0.9, "description": "test"}
            ],
        }
        validate_causal_cues(payload)

    @pytest.mark.parametrize("payload,match", [
        ({"schema_id": "melm.causal_cues.v1"}, "cues"),
        ({"schema_id": "melm.causal_cues.v1", "cues": "invalid"}, "cues"),
        ({"schema_id": "melm.causal_cues.v1", "cues": [{"cue_type": "causal_explanation", "direction": "effect_to_cause", "language": "en", "confidence": 0.9}]}, "lemma"),
        ({"schema_id": "melm.causal_cues.v1", "cues": [{"lemma": "why", "direction": "effect_to_cause", "language": "en", "confidence": 0.9}]}, "cue_type"),
    ])
    def test_rejects_missing_field(self, payload, match):
        with pytest.raises(ContractValidationError, match=match):
            validate_causal_cues(payload)

    def test_rejects_invalid_direction(self):
        payload = {
            "schema_id": "melm.causal_cues.v1",
            "cues": [{"lemma": "why", "cue_type": "causal_explanation",
                      "direction": "diagonal", "language": "en", "confidence": 0.9}],
        }
        with pytest.raises(ContractValidationError, match="direction"):
            validate_causal_cues(payload)

    def test_rejects_invalid_cue_type(self):
        payload = {
            "schema_id": "melm.causal_cues.v1",
            "cues": [{"lemma": "why", "cue_type": "not_a_real_type",
                      "direction": "effect_to_cause", "language": "en", "confidence": 0.9}],
        }
        with pytest.raises(ContractValidationError, match="cue_type"):
            validate_causal_cues(payload)


class TestValidateCausalEffects:
    def test_valid_passes(self):
        payload = {
            "schema_id": "melm.causal_effects.v1",
            "rules": {
                "rain": {
                    "patient_types": ["weather_phenomenon"],
                    "effects": {"physical": ["wet", "cooler"]},
                    "confidence": 0.95,
                }
            },
        }
        validate_causal_effects(payload)

    @pytest.mark.parametrize("payload,match", [
        ({"schema_id": "melm.causal_effects.v1"}, "rules"),
        ({"schema_id": "melm.causal_effects.v1", "rules": "invalid"}, "rules"),
        ({"schema_id": "melm.causal_effects.v1", "rules": {"rain": {"effects": {"physical": ["wet"]}}}}, "confidence"),
        ({"schema_id": "melm.causal_effects.v1", "rules": {"rain": {"patient_types": ["weather"], "confidence": 0.9}}}, "effects"),
    ])
    def test_rejects_missing_field(self, payload, match):
        with pytest.raises(ContractValidationError, match=match):
            validate_causal_effects(payload)

    def test_accepts_valid_provenance(self):
        payload = {
            "schema_id": "melm.causal_effects.v1",
            "rules": {
                "rain": {
                    "patient_types": ["weather_phenomenon"],
                    "effects": {"physical": ["wet"]},
                    "confidence": 0.9,
                    "provenance": "offline_extractor",
                    "review_status": "pending",
                }
            },
        }
        validate_causal_effects(payload)

    def test_defaults_provenance_if_omitted(self):
        payload = {
            "schema_id": "melm.causal_effects.v1",
            "rules": {
                "rain": {
                    "patient_types": ["weather_phenomenon"],
                    "effects": {"physical": ["wet"]},
                    "confidence": 0.9,
                }
            },
        }
        validate_causal_effects(payload)

    def test_rejects_unknown_provenance(self):
        payload = {
            "schema_id": "melm.causal_effects.v1",
            "rules": {
                "rain": {
                    "patient_types": ["weather_phenomenon"],
                    "effects": {"physical": ["wet"]},
                    "confidence": 0.9,
                    "provenance": "llm_generated",
                }
            },
        }
        with pytest.raises(ContractValidationError, match="provenance"):
            validate_causal_effects(payload)

    def test_rejects_unknown_review_status(self):
        payload = {
            "schema_id": "melm.causal_effects.v1",
            "rules": {
                "rain": {
                    "patient_types": ["weather_phenomenon"],
                    "effects": {"physical": ["wet"]},
                    "confidence": 0.9,
                    "provenance": "manual_curated",
                    "review_status": "unknown",
                }
            },
        }
        with pytest.raises(ContractValidationError, match="review_status"):
            validate_causal_effects(payload)


class TestValidateCausalFrames:
    def test_valid_passes(self):
        payload = {
            "schema_id": "melm.causal_frames.v1",
            "predicate_frames": {
                "rain": {
                    "predicate_id": "rain",
                    "semantic_class": "verb.weather",
                    "atom_kind": "event",
                    "default_cause_kind": "natural_process",
                    "roles": ["source", "patient"],
                    "effects": [{"target_role": "patient", "state": "wet", "domain": "physical", "relation": "causes", "confidence": 0.95}],
                    "surface_aliases": ["rain"],
                }
            },
            "state_definitions": {
                "wet": {
                    "state_id": "wet", "semantic_class": "state",
                    "aliases": ["wet"], "definition": "has moisture",
                    "definition_atoms": [], "opposites": ["dry"],
                }
            },
            "active_entity_affordances": {},
            "surface_aliases": {},
        }
        validate_causal_frames(payload)

    @pytest.mark.parametrize("payload,match", [
        ({"schema_id": "melm.causal_frames.v1"}, "predicate_frames"),
        ({"schema_id": "melm.causal_frames.v1", "predicate_frames": {"rain": {"semantic_class": "verb.weather"}}}, None),
    ])
    def test_rejects_missing_field(self, payload, match):
        ctx = pytest.raises(ContractValidationError) if match is None else pytest.raises(ContractValidationError, match=match)
        with ctx:
            validate_causal_frames(payload)


class TestValidateCausalLinkMarkers:
    def test_valid_passes(self):
        payload = {
            "schema_id": "melm.causal_link_markers.v1",
            "markers": {
                "because": {"relation": "caused_by", "direction": "effect_to_cause", "introduces": "cause", "description": "Test."},
            },
        }
        validate_causal_link_markers(payload)

    def test_rejects_missing_markers(self):
        payload = {"schema_id": "melm.causal_link_markers.v1"}
        with pytest.raises(ContractValidationError, match="markers"):
            validate_causal_link_markers(payload)

    def test_rejects_invalid_relation(self):
        payload = {
            "schema_id": "melm.causal_link_markers.v1",
            "markers": {"because": {"relation": "sideways", "direction": "effect_to_cause", "introduces": "cause", "description": "test"}},
        }
        with pytest.raises(ContractValidationError, match="relation"):
            validate_causal_link_markers(payload)


# ── Task 9: causal_contrast detector + solver tests ──────────────────


class TestCausalContrastDetection:
    @pytest.mark.parametrize("utterance,expected_a,expected_b", [
        ("What happens if I eat vs sleep?", "eat", "sleep"),
        ("What happens if it rains or shines?", "rain", "shine"),
    ])
    def test_what_happens_if_contrast(self, utterance, expected_a, expected_b):
        from melm.appliance.reasoning.task_router import detect_reasoning_task
        result = detect_reasoning_task(utterance)
        assert result is not None
        assert result["task"] == "causal_contrast"
        assert result["cause_a"] in (expected_a,)
        assert result["cause_b"] in (expected_b,)

    def test_plain_contrast_not_causal(self):
        from melm.appliance.reasoning.task_router import detect_reasoning_task
        assert detect_reasoning_task("I eat vs sleep") is None
        assert detect_reasoning_task("Tea or coffee?") is None


class TestCausalContrastSolver:
    def test_contrast_eat_vs_sleep(self):
        from melm.appliance.reasoning.solvers import solve
        result, answer, refusal = solve({"task": "causal_contrast", "cause_a": "eat", "cause_b": "sleep"})
        assert refusal is None
        assert result is not None
        assert result["task"] == "causal_contrast"
        assert "eat" in result["cause_a"] or "eat" in result["effects_a"]
        assert "sleep" in result["cause_b"] or "sleep" in result["effects_b"]
        assert "eat" in answer.lower()
        assert "sleep" in answer.lower()

    def test_contrast_unknown(self):
        from melm.appliance.reasoning.solvers import solve
        result, answer, refusal = solve({"task": "causal_contrast", "cause_a": "glorp", "cause_b": "blarg"})
        assert refusal == "no_contrast_data"

    def test_contrast_missing_causes(self):
        from melm.appliance.reasoning.solvers import solve
        _, _, refusal = solve({"task": "causal_contrast", "cause_a": "eat"})
        assert refusal == "missing_contrast_causes"


class TestCausalContrastNlg:
    @pytest.mark.parametrize("result,expected_keys", [
        ({"task": "causal_contrast", "cause_a": "eat", "cause_b": "sleep", "effects_a": ["satisfied", "nourished"], "effects_b": ["rested", "energetic"]}, ["eat", "sleep", "satisfied", "rested"]),
        ({"task": "causal_contrast", "cause_a": "glorp", "cause_b": "blarg", "effects_a": [], "effects_b": []}, ["glorp", "blarg"]),
    ])
    def test_render_contrast(self, result, expected_keys):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        rendered = render_reasoning_result(result, _contrast_templates())
        assert rendered is not None
        for key in expected_keys:
            assert key in rendered

    def test_nlg_fallback_to_none(self):
        from melm.appliance.reasoning.nlg import render_reasoning_result
        result = {"task": "causal_contrast", "cause_a": "eat", "cause_b": "sleep"}
        assert render_reasoning_result(result, {}) is None


def _contrast_templates():
    return {
        "causal_contrast": {
            "contrast": "If {cause_a} happens, likely effects include {effects_a}. In contrast, if {cause_b} happens, likely effects include {effects_b}.",
            "unknown": "I do not have enough local causal knowledge to compare {cause_a} vs {cause_b}.",
        }
    }
