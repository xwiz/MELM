"""Tests for T4 moral cognition implication engine."""

import json
import os
import tempfile
import pytest

from melm.appliance.reasoning.implications import (
    MoralContext,
    derive_moral_context,
    record_verb_candidate,
    flush_verb_candidates,
)

# Minimal valid contracts for testing
# NOTE: patient_types is a dict-of-dicts matching derive_moral_context()'s expected structure
_MINIMAL_VERBS = {
    "schema_id": "melm.verb_states.v1",
    "version": "1.0.0",
    "verbs": {
        "hit": {
            "patient_states": {"emotional": ["hurt_if_sentient"]},
            "subject_mental": ["target_identified"],
            "patient_types": ["person", "object"],
        },
        "hug": {
            "patient_states": {"emotional": ["comforted_if_sentient"]},
            "subject_mental": ["consent"],
            "patient_types": ["person"],
        },
        "help": {
            "patient_states": {
                "emotional": ["supported_if_sentient"],
                "mental": ["progress_enabled"],
            },
            "subject_mental": ["task_understood"],
            "subject_emotional": ["willing"],
            "patient_types": ["person", "organization", "object"],
        },
        "kill": {
            "patient_states": {"physical": ["dead"]},
            "subject_mental": ["intentional"],
            "patient_types": ["person", "biological_body", "autonomous_agent"],
        },
        "break": {
            "patient_states": {"physical": ["damaged", "broken"]},
            "subject_mental": ["intentional"],
            "patient_types": ["object"],
        },
        "abuse": {
            "patient_states": {
                "physical": ["harmed"],
                "emotional": ["traumatized_if_sentient"],
                "mental": ["weakened"],
            },
            "subject_emotional": ["malicious"],
            "patient_types": ["person", "biological_body"],
        },
        "wear": {
            "patient_states": {"physical": ["covered", "protected"]},
            "patient_types": ["person", "biological_body"],
        },
    },
}

_MINIMAL_VALENCES = {
    "schema_id": "melm.state_valences.v1",
    "version": "1.0.0",
    "valences": {
        "hurt": -0.6,
        "comforted": 0.5,
        "supported": 0.4,
        "progress_enabled": 0.3,
        "dead": -1.0,
        "damaged": -0.5,
        "broken": -0.4,
        "traumatized": -0.9,
        "harmed": -0.6,
        "weakened": -0.3,
        "covered": 0.1,
        "protected": 0.5,
    },
}

_SENTIENCE_MAP = {
    "person": True,
    "biological_body": True,
    "autonomous_agent": True,
    "organization": False,
    "object": False,
}


class TestMoralContextDefaults:
    def test_default_has_no_implication(self):
        mc = MoralContext()
        assert mc.wrongfulness == 0.0
        assert mc.harm_severity is None
        assert mc.consent_status == "unknown"
        assert mc.policy_triggers == ()
        assert mc.has_implication is False


class TestDeriveMoralContext:
    def test_known_harm_verb_with_sentient_patient(self):
        mc = derive_moral_context(
            "hit", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert mc.wrongfulness > 0
        assert mc.harm_severity is not None
        assert mc.consent_status == "not_consented"

    def test_known_positive_verb(self):
        mc = derive_moral_context(
            "hug", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert mc.wrongfulness == 0.0
        assert mc.harm_severity is None
        assert mc.consent_status == "consented"

    def test_help_verb_no_wrongfulness(self):
        mc = derive_moral_context(
            "help", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert mc.wrongfulness == 0.0
        assert mc.harm_severity is None

    def test_unknown_verb_no_implication(self):
        mc = derive_moral_context(
            "glorp", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert mc.wrongfulness == 0.0
        assert mc.harm_severity is None
        assert mc.has_implication is False

    def test_non_matching_patient_type(self):
        mc = derive_moral_context(
            "hug", "object", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert mc.wrongfulness == 0.0
        assert mc.has_implication is False

    def test_sentient_vs_nonsentient_patient(self):
        """Non-sentient patients should not get emotional harm weight."""
        mc_sentient = derive_moral_context(
            "abuse", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        mc_object = derive_moral_context(
            "abuse", "object", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        # object isn't in patient_types for abuse, so should be no implication
        assert mc_object.has_implication is False
        assert mc_sentient.wrongfulness > mc_object.wrongfulness

    def test_kill_verb_high_severity(self):
        mc = derive_moral_context(
            "kill", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert mc.harm_severity == "high"
        assert mc.wrongfulness >= 0.7
        assert "urgent_harm" in mc.policy_triggers

    def test_kill_verb_policy_triggers(self):
        mc = derive_moral_context(
            "kill", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert "urgent_harm" in mc.policy_triggers
        assert "caution" in mc.policy_triggers

    def test_break_object_property_safety(self):
        mc = derive_moral_context(
            "break", "object", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert "property_safety" in mc.policy_triggers
        # non-sentient, so no emotional weight
        assert mc.harm_severity is not None

    def test_wear_no_harm(self):
        mc = derive_moral_context(
            "wear", "person", _MINIMAL_VERBS,
            _MINIMAL_VALENCES["valences"], _SENTIENCE_MAP,
        )
        assert mc.wrongfulness == 0.0
        assert mc.harm_severity is None

    def test_no_contract_returns_default(self):
        mc = derive_moral_context("hit", "person", None, None, None)
        assert mc.wrongfulness == 0.0
        assert mc.has_implication is False


class TestVerbCandidateRecording:
    def test_record_and_flush(self):
        record_verb_candidate("punch", "person", "i punch the wall")
        record_verb_candidate("kick", "object", "he kick the can")
        # Check buffer has entries (can't easily flush without store)
        from melm.appliance.reasoning.implications import _verb_candidate_buffer
        verbs = [e["verb"] for e in _verb_candidate_buffer]
        assert "punch" in verbs
        assert "kick" in verbs

    def test_ring_buffer_bounded(self):
        from melm.appliance.reasoning.implications import _verb_candidate_buffer
        _verb_candidate_buffer.clear()
        for i in range(600):
            record_verb_candidate(f"verb_{i}", "person")
        assert len(_verb_candidate_buffer) <= 500
        # Oldest entries should be evicted
        verbs = [e["verb"] for e in _verb_candidate_buffer]
        assert "verb_0" not in verbs
        assert "verb_599" in verbs
