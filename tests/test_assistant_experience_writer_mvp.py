"""Tests for T2 personal_experience entity writer."""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from melm.appliance.assistant_experience_writer import (
    _compute_outcome,
    _compute_intent_achieved,
    _compute_polarity,
    record_conversation_experience,
)
from melm.appliance.assistant_synthesis import BoundedSynthesisResult
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import AssistantDecision


# ---------------------------------------------------------------------------
# Helpers: build minimal test objects
# ---------------------------------------------------------------------------

def _make_decision(intent: str = "open_domain") -> AssistantDecision:
    return AssistantDecision(
        utterance="test utterance",
        intent=intent,  # type:ignore[arg-type]
        route="local",
        answer="test answer",
    )


def _make_synthesis(
    applied: bool = False,
    refused: bool = False,
    boundary_crossed: str = "none",
    reason: str = "",
) -> BoundedSynthesisResult:
    return BoundedSynthesisResult(
        route="local",
        applied=applied,
        refused=refused,
        answer="synthesized answer",
        citations=(),
        evidence=(),
        admitted_evidence_count=0,
        reason=reason,
        boundary_crossed=boundary_crossed,
    )


# ===================================================================
# _compute_outcome
# ===================================================================

class TestComputeOutcome:
    def test_none_synthesis_returns_unresolved(self) -> None:
        assert _compute_outcome(None, _make_decision()) == "unresolved"

    def test_boundary_crossed_returns_escalated(self) -> None:
        d = _make_decision()
        s = _make_synthesis(boundary_crossed="privacy_blocked")
        assert _compute_outcome(s, d) == "escalated"

    def test_boundary_crossed_none_not_escalated(self) -> None:
        d = _make_decision()
        s = _make_synthesis(boundary_crossed="none")
        assert _compute_outcome(s, d) != "escalated"

    def test_boundary_crossed_empty_not_escalated(self) -> None:
        d = _make_decision()
        s = _make_synthesis(boundary_crossed="")
        assert _compute_outcome(s, d) != "escalated"

    def test_refused_returns_abandoned(self) -> None:
        d = _make_decision()
        s = _make_synthesis(refused=True)
        assert _compute_outcome(s, d) == "abandoned"

    def test_applied_returns_resolved(self) -> None:
        d = _make_decision()
        s = _make_synthesis(applied=True)
        assert _compute_outcome(s, d) == "resolved"

    def test_not_applied_not_refused_returns_unresolved(self) -> None:
        d = _make_decision()
        s = _make_synthesis(applied=False, refused=False)
        assert _compute_outcome(s, d) == "unresolved"

    def test_boundary_crossed_overrides_refused(self) -> None:
        d = _make_decision()
        s = _make_synthesis(boundary_crossed="blocked", refused=True)
        assert _compute_outcome(s, d) == "escalated"


# ===================================================================
# _compute_intent_achieved
# ===================================================================

class TestComputeIntentAchieved:
    def test_none_synthesis_returns_no(self) -> None:
        assert _compute_intent_achieved(None) == "no"

    def test_applied_not_refused_returns_yes(self) -> None:
        assert _compute_intent_achieved(_make_synthesis(applied=True, refused=False)) == "yes"

    def test_not_applied_returns_no(self) -> None:
        assert _compute_intent_achieved(_make_synthesis(applied=False)) == "no"

    def test_refused_returns_no_even_if_applied(self) -> None:
        assert _compute_intent_achieved(_make_synthesis(applied=True, refused=True)) == "no"


# ===================================================================
# _compute_polarity
# ===================================================================

class TestComputePolarity:
    def test_defaults_to_neutral(self) -> None:
        assert _compute_polarity(_make_synthesis(), _make_decision()) == 0.0

    def test_none_synthesis_is_neutral(self) -> None:
        assert _compute_polarity(None, _make_decision()) == 0.0


# ===================================================================
# record_conversation_experience — integration with entity store
# ===================================================================

class TestRecordConversationExperience:
    @pytest.fixture(autouse=True)
    def _in_memory_store(self) -> None:
        self.store = AssistantOSStore(":memory:")
        seed_class_schemas(self.store)

    def _set_slots(self) -> dict[str, Any]:
        pe_entities = self.store.find_entities(kind="personal_experience")
        assert len(pe_entities) == 1, "expected exactly one personal_experience entity"
        entity_id = pe_entities[0].entity_id
        slots = {}
        for row in self.store.connection.execute(
            "SELECT slot_name, value_json FROM entity_slots WHERE entity_id = ?",
            (entity_id,),
        ):
            slots[row["slot_name"]] = row["value_json"]
        return slots

    # --- happy path ---

    def test_writes_entity_with_correct_kind(self) -> None:
        decision = _make_decision()
        synthesis = _make_synthesis(applied=True)
        entity_id = record_conversation_experience(self.store, decision, synthesis)
        assert entity_id is not None
        entities = self.store.find_entities(kind="personal_experience")
        assert len(entities) == 1
        assert entities[0].entity_id == entity_id

    def test_slots_are_set(self) -> None:
        decision = _make_decision()
        synthesis = _make_synthesis(applied=True)
        record_conversation_experience(self.store, decision, synthesis)
        slots = self._set_slots()
        assert json.loads(slots["outcome"]) == "resolved"
        assert json.loads(slots["intent_achieved"]) == "yes"
        assert json.loads(slots["polarity"]) == 0.0
        assert json.loads(slots["follow_up"]) is None
        assert json.loads(slots["learned_fact_ids"]) == []
        assert json.loads(slots["user_id"]) == "default"

    def test_user_id_slot_is_set(self) -> None:
        decision = _make_decision()
        synthesis = _make_synthesis(applied=True)
        record_conversation_experience(self.store, decision, synthesis, user_id="user_alice")
        slots = self._set_slots()
        assert json.loads(slots["user_id"]) == "user_alice"

    def test_user_id_defaults_to_default(self) -> None:
        decision = _make_decision()
        synthesis = _make_synthesis(applied=True)
        record_conversation_experience(self.store, decision, synthesis)
        slots = self._set_slots()
        assert json.loads(slots["user_id"]) == "default"

    def test_escalated_outcome_when_boundary_crossed(self) -> None:
        decision = _make_decision()
        synthesis = _make_synthesis(boundary_crossed="privacy_blocked")
        record_conversation_experience(self.store, decision, synthesis)
        slots = self._set_slots()
        assert json.loads(slots["outcome"]) == "escalated"
        assert json.loads(slots["intent_achieved"]) == "no"

    def test_abandoned_outcome_when_refused(self) -> None:
        decision = _make_decision()
        synthesis = _make_synthesis(refused=True)
        record_conversation_experience(self.store, decision, synthesis)
        slots = self._set_slots()
        assert json.loads(slots["outcome"]) == "abandoned"
        assert json.loads(slots["intent_achieved"]) == "no"

    def test_unresolved_outcome_when_not_applied(self) -> None:
        decision = _make_decision()
        synthesis = _make_synthesis(applied=False, refused=False)
        record_conversation_experience(self.store, decision, synthesis)
        slots = self._set_slots()
        assert json.loads(slots["outcome"]) == "unresolved"
        assert json.loads(slots["intent_achieved"]) == "no"

    def test_unresolved_outcome_when_none_synthesis(self) -> None:
        decision = _make_decision()
        record_conversation_experience(self.store, decision, None)
        slots = self._set_slots()
        assert json.loads(slots["outcome"]) == "unresolved"
        assert json.loads(slots["intent_achieved"]) == "no"

    def test_label_contains_intent(self) -> None:
        decision = _make_decision(intent="weather")
        synthesis = _make_synthesis(applied=True)
        record_conversation_experience(self.store, decision, synthesis)
        entities = self.store.find_entities(kind="personal_experience")
        assert "weather" in entities[0].label

    def test_canonical_lemma_is_truncated_utterance(self) -> None:
        decision = _make_decision()
        decision = AssistantDecision(
            utterance="a" * 200,
            intent="weather",  # type:ignore[arg-type]
            route="local",
            answer="",
        )
        synthesis = _make_synthesis(applied=True)
        record_conversation_experience(self.store, decision, synthesis)
        entities = self.store.find_entities(kind="personal_experience")
        assert len(entities[0].canonical_lemma) == 80

    # --- multiple turns ---

    def test_multiple_turns_create_multiple_entities(self) -> None:
        for i in range(3):
            d = _make_decision(intent="weather")
            s = _make_synthesis(applied=True)
            record_conversation_experience(self.store, d, s)
        entities = self.store.find_entities(kind="personal_experience")
        assert len(entities) == 3
