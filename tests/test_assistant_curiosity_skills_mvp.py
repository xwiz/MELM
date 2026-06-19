"""Tests for curiosity/context/agreement skill modules.

Covers: deferred_tasks, research_deferral, novelty, epistemic,
greeting_context, commitments, background_runner.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def store():
    s = AssistantOSStore(":memory:")
    seed_class_schemas(s)
    return s


# ===================================================================
# assistant_skill_deferred_tasks
# ===================================================================

class TestDeferredTasks:
    def test_queue_deferred_task_returns_entity_id(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import queue_deferred_task
        eid = queue_deferred_task(store, topic="weather in Lagos", action="auto_research")
        assert eid is not None
        assert eid.startswith("dt_")

    def test_queue_deferred_task_with_all_kwargs(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import queue_deferred_task
        eid = queue_deferred_task(
            store, topic="igbo history", action="auto_research",
            due_at="2026-07-01T00:00:00", priority="high",
            engagement_prompt="Let me tell you about Igbo history",
            owner_session_id="sess_001",
        )
        assert eid is not None
        slot = store.get_entity_slot(eid, "priority")
        assert slot is not None and json.loads(slot.value_json) == "high"
        slot = store.get_entity_slot(eid, "engagement_prompt")
        assert slot is not None and "Igbo" in json.loads(slot.value_json)

    def test_queue_deferred_task_none_store(self):
        from melm.appliance.assistant_skill_deferred_tasks import queue_deferred_task
        assert queue_deferred_task(None, "topic", "action") is None

    def test_find_due_tasks_returns_queued(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import queue_deferred_task, find_due_tasks
        queue_deferred_task(store, topic="test topic", action="auto_research")
        due = find_due_tasks(store)
        assert len(due) >= 1
        assert due[0]["topic"] == "test topic"
        assert due[0]["action"] == "auto_research"

    def test_find_due_tasks_skips_completed(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, find_due_tasks, complete_deferred_task,
        )
        eid = queue_deferred_task(store, topic="completed topic", action="auto_research")
        complete_deferred_task(store, eid)
        due = find_due_tasks(store)
        topics = [t["topic"] for t in due]
        assert "completed topic" not in topics

    def test_find_due_tasks_none_store(self):
        from melm.appliance.assistant_skill_deferred_tasks import find_due_tasks
        assert find_due_tasks(None) == []

    def test_surface_task_context_auto_research(self):
        from melm.appliance.assistant_skill_deferred_tasks import surface_task_context
        result = surface_task_context(None, {"topic": "Yoruba art", "action": "auto_research"})
        assert "looking into" in result
        assert "Yoruba art" in result

    def test_surface_task_context_novelty(self):
        from melm.appliance.assistant_skill_deferred_tasks import surface_task_context
        result = surface_task_context(None, {"topic": "talking drums", "action": "novelty_review"})
        assert "interesting" in result
        assert "talking drums" in result

    def test_surface_task_context_default(self):
        from melm.appliance.assistant_skill_deferred_tasks import surface_task_context
        result = surface_task_context(None, {"topic": "groceries", "action": "shopping"})
        assert "pending task" in result

    def test_complete_deferred_task_sets_status(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, complete_deferred_task, find_due_tasks,
        )
        eid = queue_deferred_task(store, topic="finish me", action="auto_research")
        complete_deferred_task(store, eid, result_summary="All done")
        due = find_due_tasks(store)
        assert all(t["topic"] != "finish me" for t in due)
        slot = store.get_entity_slot(eid, "status")
        assert slot is not None and json.loads(slot.value_json) == "completed"
        slot = store.get_entity_slot(eid, "result_summary")
        assert slot is not None and json.loads(slot.value_json) == "All done"

    def test_cancel_deferred_task_sets_cancelled(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, cancel_deferred_task, find_due_tasks,
        )
        eid = queue_deferred_task(store, topic="cancel me", action="auto_research")
        cancel_deferred_task(store, eid)
        due = find_due_tasks(store)
        assert all(t["topic"] != "cancel me" for t in due)
        slot = store.get_entity_slot(eid, "status")
        assert slot is not None and json.loads(slot.value_json) == "cancelled"

    def test_complete_cancel_safe_on_none_store(self):
        from melm.appliance.assistant_skill_deferred_tasks import (
            complete_deferred_task, cancel_deferred_task,
        )
        complete_deferred_task(None, "eid")  # should not raise
        cancel_deferred_task(None, "eid")  # should not raise


# ===================================================================
# assistant_skill_research_deferral
# ===================================================================

class TestResearchDeferral:
    def test_should_defer_default_true(self):
        from melm.appliance.assistant_skill_research_deferral import should_defer_research
        decision = type("FakeDecision", (), {"utterance": "what is the weather"})()
        assert should_defer_research(decision, None, None) is True

    def test_queue_research_task_creates_deferred(self, store):
        from melm.appliance.assistant_skill_research_deferral import queue_research_task
        from melm.appliance.assistant_skill_deferred_tasks import find_due_tasks
        eid = queue_research_task(store, topic="igbo architecture", session_id="sess_001")
        assert eid is not None
        tasks = find_due_tasks(store)
        topics = [t["topic"] for t in tasks]
        assert "igbo architecture" in topics

    def test_run_deferred_research_no_provider(self, store):
        from melm.appliance.assistant_skill_research_deferral import (
            queue_research_task, run_deferred_research,
        )
        queue_research_task(store, topic="test research", session_id="sess_001")
        results = run_deferred_research(store, provider=None)
        assert len(results) >= 1
        assert results[0]["status"] == "completed"


# ===================================================================
# assistant_background_runner
# ===================================================================

class TestBackgroundRunner:
    def test_start_stop_is_running(self):
        from melm.appliance.assistant_background_runner import BackgroundTaskRunner
        runner = BackgroundTaskRunner(store=None)
        assert runner.is_running is False
        runner.start()
        assert runner.is_running is True
        runner.stop()
        assert runner.is_running is False

    def test_tick_returns_empty_when_not_running(self):
        from melm.appliance.assistant_background_runner import BackgroundTaskRunner
        runner = BackgroundTaskRunner(store=None)
        assert runner.tick() == []

    def test_tick_returns_empty_when_no_store(self):
        from melm.appliance.assistant_background_runner import BackgroundTaskRunner
        runner = BackgroundTaskRunner(store=None)
        runner.start()
        assert runner.tick() == []

    def test_execute_job_unknown_kind_raises(self):
        from melm.appliance.assistant_background_runner import BackgroundTaskRunner
        runner = BackgroundTaskRunner(store=None)
        import pytest
        with pytest.raises(ValueError, match="Unknown job kind"):
            runner._execute_job({"kind": "unknown", "payload_json": "{}"})

    def test_execute_job_deferred_research(self):
        from melm.appliance.assistant_background_runner import BackgroundTaskRunner
        runner = BackgroundTaskRunner(store=None)
        result = runner._execute_job({
            "kind": "deferred_research",
            "payload_json": json.dumps({"topic": "Yoruba sculpture"}),
        })
        assert result["status"] == "completed"
        assert "Yoruba sculpture" in result["summary"]


# ===================================================================
# assistant_skill_novelty
# ===================================================================

class TestNovelty:
    def test_novelty_candidate_dataclass(self):
        from melm.appliance.assistant_skill_novelty import NoveltyCandidate
        nc = NoveltyCandidate(
            surface_form="kayak",
            utterance_context="the word kayak",
            detection_reason="palindrome",
        )
        assert nc.surface_form == "kayak"
        assert nc.confidence == 0.5
        assert nc.decomposition == ""

    def test_is_palindrome_true(self):
        from melm.appliance.assistant_skill_novelty import _is_palindrome
        assert _is_palindrome("kayak") is True
        assert _is_palindrome("racecar") is True
        assert _is_palindrome("Aba") is True  # case-insensitive

    def test_is_palindrome_false(self):
        from melm.appliance.assistant_skill_novelty import _is_palindrome
        assert _is_palindrome("hello") is False
        assert _is_palindrome("ab") is False  # too short

    def test_has_morpheme_boundary_suffix(self):
        from melm.appliance.assistant_skill_novelty import _has_morpheme_boundary
        assert _has_morpheme_boundary("running", ["-ing", "un-", "-ed"]) is True
        assert _has_morpheme_boundary("talking", ["-ing"]) is True

    def test_has_morpheme_boundary_prefix(self):
        from melm.appliance.assistant_skill_novelty import _has_morpheme_boundary
        assert _has_morpheme_boundary("unfair", ["un-", "-ing"]) is True
        assert _has_morpheme_boundary("redo", ["re-", "-ed"]) is True

    def test_has_morpheme_boundary_false(self):
        from melm.appliance.assistant_skill_novelty import _has_morpheme_boundary
        assert _has_morpheme_boundary("cat", ["-ing", "un-"]) is False

    def test_check_cultural_symbol_found(self):
        from melm.appliance.assistant_skill_novelty import _check_cultural_symbol
        symbols = {
            "african_diaspora": {
                "patterns": ["adinkra", "kente", "ankara"],
            },
        }
        assert _check_cultural_symbol("kente", symbols) == "african_diaspora"
        assert _check_cultural_symbol("adinkra symbols", symbols) == "african_diaspora"

    def test_check_cultural_symbol_not_found(self):
        from melm.appliance.assistant_skill_novelty import _check_cultural_symbol
        symbols = {
            "african_diaspora": {
                "patterns": ["adinkra", "kente"],
            },
        }
        assert _check_cultural_symbol("python", symbols) is None

    def test_detect_novelty_returns_candidates(self):
        from melm.appliance.assistant_skill_novelty import detect_novelty
        bundle = type("FakeBundle", (), {
            "semantic_unknown_tokens": ["kayak", "unhappiness", "hello"],
            "text": "test utterance",
        })()
        candidates = detect_novelty(bundle, lexicon=None, store=None)
        assert len(candidates) >= 1
        reasons = {c.surface_form: c.detection_reason for c in candidates}
        assert "kayak" in reasons  # palindrome
        assert "unhappiness" in reasons  # morpheme cluster

    def test_detect_novelty_none_bundle(self):
        from melm.appliance.assistant_skill_novelty import detect_novelty
        candidates = detect_novelty(None, lexicon=None, store=None)
        assert candidates == []

    def test_record_novelty_candidates_creates_entities(self, store):
        from melm.appliance.assistant_skill_novelty import (
            NoveltyCandidate, record_novelty_candidates,
        )
        nc = NoveltyCandidate(
            surface_form="kayak",
            utterance_context="the word kayak",
            detection_reason="palindrome",
            decomposition="palindrome(5 chars)",
            confidence=0.65,
        )
        ids = record_novelty_candidates(store, [nc])
        assert len(ids) == 1
        slot = store.get_entity_slot(ids[0], "surface_form")
        assert slot is not None and json.loads(slot.value_json) == "kayak"
        slot = store.get_entity_slot(ids[0], "review_status")
        assert slot is not None and json.loads(slot.value_json) == "flagged"

    def test_record_novelty_candidates_none_store(self):
        from melm.appliance.assistant_skill_novelty import (
            NoveltyCandidate, record_novelty_candidates,
        )
        nc = NoveltyCandidate(surface_form="test", utterance_context="", detection_reason="test")
        assert record_novelty_candidates(None, [nc]) == []

    def test_build_novelty_digest(self, store):
        from melm.appliance.assistant_skill_novelty import (
            NoveltyCandidate, record_novelty_candidates, build_novelty_digest,
        )
        nc = NoveltyCandidate(
            surface_form="kayak",
            utterance_context="test",
            detection_reason="palindrome",
        )
        record_novelty_candidates(store, [nc])
        digest = build_novelty_digest(store, "sess_001")
        assert "kayak" in digest
        assert "palindrome" in digest

    def test_build_novelty_digest_empty(self, store):
        from melm.appliance.assistant_skill_novelty import build_novelty_digest
        assert build_novelty_digest(store, "sess_001") == ""

    def test_build_novelty_digest_none_store(self):
        from melm.appliance.assistant_skill_novelty import build_novelty_digest
        assert build_novelty_digest(None, "sess_001") == ""


# ===================================================================
# assistant_skill_epistemic
# ===================================================================

class TestEpistemic:
    def test_record_epistemic_state_returns_id(self, store):
        from melm.appliance.assistant_skill_epistemic import record_epistemic_state
        eid = record_epistemic_state(store, "curiosity", "Yoruba mythology")
        assert eid is not None
        assert eid.startswith("es_")

    def test_record_epistemic_state_with_source(self, store):
        from melm.appliance.assistant_skill_epistemic import record_epistemic_state
        eid = record_epistemic_state(store, "confusion", "igbo vowels",
                                      valence=-0.3, source_event_id="pe_abc123")
        assert eid is not None
        slot = store.get_entity_slot(eid, "source_event_id")
        assert slot is not None and json.loads(slot.value_json) == "pe_abc123"

    def test_record_epistemic_state_none_store(self):
        from melm.appliance.assistant_skill_epistemic import record_epistemic_state
        assert record_epistemic_state(None, "curiosity", "topic") is None

    def test_load_open_epistemic_states(self, store):
        from melm.appliance.assistant_skill_epistemic import (
            record_epistemic_state, load_open_epistemic_states,
        )
        record_epistemic_state(store, "curiosity", "Yoruba art")
        record_epistemic_state(store, "confusion", "tone marks")
        states = load_open_epistemic_states(store)
        assert len(states) == 2
        types = [s["state_type"] for s in states]
        assert "curiosity" in types
        assert "confusion" in types

    def test_load_open_excludes_resolved(self, store):
        from melm.appliance.assistant_skill_epistemic import (
            record_epistemic_state, load_open_epistemic_states,
        )
        eid = record_epistemic_state(store, "curiosity", "resolved topic")
        store.set_entity_slot(eid, "resolved_at", "2026-01-01T00:00:00")
        states = load_open_epistemic_states(store)
        assert all(s["topic"] != "resolved topic" for s in states)

    def test_load_open_epistemic_states_none_store(self):
        from melm.appliance.assistant_skill_epistemic import load_open_epistemic_states
        assert load_open_epistemic_states(None) == []

    def test_surface_open_states_returns_string(self, store):
        from melm.appliance.assistant_skill_epistemic import (
            record_epistemic_state, surface_open_states,
        )
        record_epistemic_state(store, "curiosity", "Yoruba art")
        text = surface_open_states(store)
        assert text is not None
        assert "curious" in text
        assert "Yoruba art" in text

    def test_surface_open_states_returns_none_when_empty(self, store):
        from melm.appliance.assistant_skill_epistemic import surface_open_states
        assert surface_open_states(store) is None


# ===================================================================
# assistant_skill_greeting_context
# ===================================================================

class TestGreetingContext:
    def test_build_greeting_context_returns_none_when_empty(self, store):
        from melm.appliance.assistant_skill_greeting_context import build_greeting_context
        result = build_greeting_context(store, "sess_001", None)
        assert result is None

    def test_build_greeting_context_with_completed_task(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, complete_deferred_task,
        )
        from melm.appliance.assistant_skill_greeting_context import build_greeting_context
        eid = queue_deferred_task(
            store, topic="Yoruba history", action="auto_research",
            owner_session_id="sess_old",
        )
        complete_deferred_task(store, eid, result_summary="Found interesting facts")
        result = build_greeting_context(store, "sess_001", None)
        assert result is not None
        assert "Yoruba history" in result
        assert "Found interesting facts" in result

    def test_build_greeting_context_skips_same_session(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, complete_deferred_task,
        )
        from melm.appliance.assistant_skill_greeting_context import build_greeting_context
        eid = queue_deferred_task(
            store, topic="current task", action="auto_research",
            owner_session_id="sess_001",
        )
        complete_deferred_task(store, eid)
        result = build_greeting_context(store, "sess_001", None)
        assert result is None

    def test_build_greeting_context_with_epistemic_states(self, store):
        from melm.appliance.assistant_skill_epistemic import record_epistemic_state
        from melm.appliance.assistant_skill_greeting_context import build_greeting_context
        record_epistemic_state(store, "curiosity", "igbo drums")
        result = build_greeting_context(store, "sess_001", None)
        assert result is not None
        assert "curious" in result

    def test_inject_greeting_prepends(self):
        from melm.appliance.assistant_skill_greeting_context import inject_greeting
        from melm.appliance.local_assistant_router import AssistantDecision
        decision = AssistantDecision(
            utterance="hello", intent="social_greeting", route="local", answer="Hello!",
        )
        new_dec = inject_greeting(decision, "I found something. ")
        assert "Hello!" in new_dec.answer
        assert "I found something." in new_dec.answer

    def test_inject_greeting_handles_empty_answer(self):
        from melm.appliance.assistant_skill_greeting_context import inject_greeting
        from melm.appliance.local_assistant_router import AssistantDecision
        decision = AssistantDecision(
            utterance="hello", intent="social_greeting", route="local", answer="",
        )
        new_dec = inject_greeting(decision, "I found something.")
        assert new_dec.answer == "I found something."


# ===================================================================
# assistant_skill_commitments
# ===================================================================

class TestCommitments:
    def test_extract_commitment_reminder_request(self):
        from melm.appliance.assistant_skill_commitments import extract_commitment
        result = extract_commitment("remind me to check the weather", None)
        assert result is not None
        assert result.commitment_type == "reminder_request"
        assert "check the weather" in result.topic

    def test_extract_commitment_remind_me(self):
        from melm.appliance.assistant_skill_commitments import extract_commitment
        result = extract_commitment("remind me to buy groceries tomorrow", None)
        assert result is not None
        assert "buy groceries" in result.topic

    def test_extract_commitment_remind_you(self):
        from melm.appliance.assistant_skill_commitments import extract_commitment
        result = extract_commitment("remind you to call mom", None)
        assert result is not None

    def test_extract_commitment_remind_us(self):
        from melm.appliance.assistant_skill_commitments import extract_commitment
        result = extract_commitment("remind us to leave at 5", None)
        assert result is not None

    def test_extract_commitment_dont_let_me_forget(self):
        from melm.appliance.assistant_skill_commitments import extract_commitment
        result = extract_commitment("do not let me forget to water the plants", None)
        assert result is not None
        assert result.commitment_type == "reminder_request"
        assert "water the plants" in result.topic

    def test_extract_commitment_no_match(self):
        from melm.appliance.assistant_skill_commitments import extract_commitment
        result = extract_commitment("hello how are you", None)
        assert result is None

    def test_extract_commitment_remind_me_no_topic(self):
        from melm.appliance.assistant_skill_commitments import extract_commitment
        result = extract_commitment("remind me", None)
        assert result is not None

    def test_record_commitment_creates_entity(self, store):
        from melm.appliance.assistant_skill_commitments import (
            UserCommitment, record_commitment,
        )
        c = UserCommitment(
            commitment_type="reminder_request",
            topic="buy groceries",
            user_utterance="remind me to buy groceries",
            session_id="sess_001",
        )
        eid = record_commitment(store, c)
        assert eid is not None
        assert eid.startswith("uc_")
        slot = store.get_entity_slot(eid, "topic")
        assert slot is not None and json.loads(slot.value_json) == "buy groceries"
        slot = store.get_entity_slot(eid, "status")
        assert slot is not None and json.loads(slot.value_json) == "pending"

    def test_record_commitment_with_times(self, store):
        from melm.appliance.assistant_skill_commitments import (
            UserCommitment, record_commitment,
        )
        c = UserCommitment(
            commitment_type="reminder_request",
            topic="call mom",
            promised_time="tomorrow",
            parsed_time="2026-06-20T09:00:00",
            user_utterance="remind me to call mom tomorrow",
            session_id="sess_001",
        )
        eid = record_commitment(store, c)
        assert eid is not None
        slot = store.get_entity_slot(eid, "promised_time")
        assert slot is not None and json.loads(slot.value_json) == "tomorrow"
        slot = store.get_entity_slot(eid, "parsed_time")
        assert slot is not None and "2026" in json.loads(slot.value_json)

    def test_record_commitment_none_store(self):
        from melm.appliance.assistant_skill_commitments import (
            UserCommitment, record_commitment,
        )
        c = UserCommitment(topic="test")
        assert record_commitment(None, c) is None

    def test_check_commitment_status(self):
        from melm.appliance.assistant_skill_commitments import check_commitment_status
        assert check_commitment_status(None, {"status": "pending"}) == "pending"
        assert check_commitment_status(None, {"status": "fulfilled"}) == "fulfilled"
        assert check_commitment_status(None, {"status": "broken"}) == "broken"

    def test_build_commitment_greeting_pending(self):
        from melm.appliance.assistant_skill_commitments import build_commitment_greeting
        result = build_commitment_greeting(None, {"topic": "buy groceries", "status": "pending"}, None)
        assert result is not None
        assert "buy groceries" in result

    def test_build_commitment_greeting_fulfilled(self):
        from melm.appliance.assistant_skill_commitments import build_commitment_greeting
        result = build_commitment_greeting(None, {"topic": "call mom", "status": "fulfilled"}, None)
        assert result is not None

    def test_build_commitment_greeting_broken(self):
        from melm.appliance.assistant_skill_commitments import build_commitment_greeting
        result = build_commitment_greeting(None, {"topic": "water plants", "status": "broken"}, None)
        assert result is not None


# ===================================================================
# Integration: commitment extraction + recording
# ===================================================================

class TestCommitmentIntegration:
    def test_extract_and_record(self, store):
        from melm.appliance.assistant_skill_commitments import (
            extract_commitment, record_commitment, UserCommitment,
        )
        utterance = "remind me to check the weather tomorrow"
        commitment = extract_commitment(utterance, None)
        assert commitment is not None
        eid = record_commitment(store, commitment)
        assert eid is not None
        slot = store.get_entity_slot(eid, "topic")
        assert slot is not None
        assert "check the weather" in json.loads(slot.value_json)
        slot = store.get_entity_slot(eid, "status")
        assert slot is not None and json.loads(slot.value_json) == "pending"


# ===================================================================
# Integration: greeting context pipeline
# ===================================================================

class TestGreetingContextIntegration:
    """Exercises the full pipeline: queue → complete → build_greeting_context → inject_greeting."""

    def test_completed_task_from_other_session_appears_in_context(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, complete_deferred_task,
        )
        from melm.appliance.assistant_skill_greeting_context import (
            build_greeting_context, inject_greeting,
        )
        from melm.appliance.local_assistant_router import AssistantDecision

        eid = queue_deferred_task(
            store, topic="Yoruba folktales", action="auto_research",
            owner_session_id="sess_old",
        )
        complete_deferred_task(store, eid, result_summary="Found 3 interesting stories")
        context = build_greeting_context(store, "sess_new", None)
        assert context is not None
        assert "Yoruba folktales" in context
        assert "Found 3 interesting stories" in context

        decision = AssistantDecision(
            utterance="hello", intent="social_greeting", route="local", answer="Hi there!",
        )
        injected = inject_greeting(decision, context)
        assert "Yoruba folktales" in injected.answer
        assert "Hi there!" in injected.answer

    def test_multiple_completed_tasks(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, complete_deferred_task,
        )
        from melm.appliance.assistant_skill_greeting_context import build_greeting_context

        for i, topic in enumerate(["igbo drums", "Yoruba art"]):
            eid = queue_deferred_task(
                store, topic=topic, action="auto_research",
                owner_session_id=f"sess_old_{i}",
            )
            complete_deferred_task(store, eid, result_summary=f"Results for {topic}")
        context = build_greeting_context(store, "sess_new", None)
        assert context is not None
        assert "igbo drums" in context
        assert "Yoruba art" in context

    def test_epistemic_states_combined_with_tasks(self, store):
        from melm.appliance.assistant_skill_deferred_tasks import (
            queue_deferred_task, complete_deferred_task,
        )
        from melm.appliance.assistant_skill_epistemic import record_epistemic_state
        from melm.appliance.assistant_skill_greeting_context import build_greeting_context

        eid = queue_deferred_task(
            store, topic="Hausa architecture", action="auto_research",
            owner_session_id="sess_old",
        )
        complete_deferred_task(store, eid, result_summary="Found references")
        record_epistemic_state(store, "curiosity", "igbo vowels")
        context = build_greeting_context(store, "sess_new", None)
        assert context is not None
        assert "Hausa architecture" in context
        assert "curious" in context

    def test_no_context_when_nothing_due(self, store):
        from melm.appliance.assistant_skill_greeting_context import build_greeting_context
        context = build_greeting_context(store, "sess_new", None)
        assert context is None


# ===================================================================
# Integration: novelty detection at router level
# ===================================================================

class TestNoveltyRouterIntegration:
    """Novelty should create novelty_candidate entities when unknown tokens exist."""

    def test_novelty_side_effect_creates_entities(self, store):
        from melm.appliance.local_assistant_router import OnDeviceAssistantRouter
        from melm.appliance import LocalAssistantProfile
        from melm.appliance.assistant_skill_novelty import _is_palindrome

        router = OnDeviceAssistantRouter(LocalAssistantProfile(), store=store)
        bundle = type("FakeBundle", (), {
            "semantic_unknown_tokens": ["kayak", "racecar"],
            "text": "kayak and racecar are palindromes",
            "language": "en",
            "tokens": ("kayak", "and", "racecar", "are", "palindromes"),
            "syntax_graph": None,
            "functional_parse": None,
            "uol_act": None,
            "session_id": "",
        })()
        decision = router._route_impl("kayak and racecar", parse_bundle=bundle)
        assert decision is not None
        entities = store.find_entities(kind="novelty_candidate")
        assert len(entities) >= 1
        surface_forms = []
        for ent in entities:
            slot = store.get_entity_slot(ent.entity_id, "surface_form")
            if slot and slot.value_json:
                surface_forms.append(json.loads(slot.value_json))
        assert "kayak" in surface_forms or "racecar" in surface_forms

    def test_novelty_skipped_when_no_unknown_tokens(self, store):
        from melm.appliance.local_assistant_router import OnDeviceAssistantRouter
        from melm.appliance import LocalAssistantProfile

        router = OnDeviceAssistantRouter(LocalAssistantProfile(), store=store)
        bundle = type("FakeBundle", (), {
            "semantic_unknown_tokens": [],
            "text": "hello world",
            "language": "en",
            "tokens": ("hello", "world"),
            "syntax_graph": None,
            "functional_parse": None,
            "uol_act": None,
            "session_id": "",
        })()
        decision = router._route_impl("hello world", parse_bundle=bundle)
        assert decision is not None
        entities = store.find_entities(kind="novelty_candidate")
        assert len(entities) == 0

    def test_novelty_skipped_when_no_store(self):
        from melm.appliance.local_assistant_router import OnDeviceAssistantRouter
        from melm.appliance import LocalAssistantProfile

        router = OnDeviceAssistantRouter(LocalAssistantProfile(), store=None)
        bundle = type("FakeBundle", (), {
            "semantic_unknown_tokens": ["kayak"],
            "text": "kayak",
            "language": "en",
            "tokens": ("kayak",),
            "syntax_graph": None,
            "functional_parse": None,
            "uol_act": None,
            "session_id": "",
        })()
        decision = router._route_impl("kayak", parse_bundle=bundle)
        assert decision is not None  # just should not crash


# ===================================================================
# Integration: commitment extraction at router level
# ===================================================================

class TestCommitmentRouterIntegration:
    """Commitment extraction should create user_commitment entities."""

    def test_remind_me_creates_commitment(self, store):
        from melm.appliance.local_assistant_router import OnDeviceAssistantRouter
        from melm.appliance import LocalAssistantProfile

        router = OnDeviceAssistantRouter(LocalAssistantProfile(), store=store)
        bundle = type("FakeBundle", (), {
            "semantic_unknown_tokens": [],
            "text": "remind me to check the weather",
            "language": "en",
            "tokens": ("remind", "me", "to", "check", "the", "weather"),
            "syntax_graph": None,
            "functional_parse": None,
            "uol_act": None,
            "session_id": "sess_001",
        })()
        decision = router._route_impl("remind me to check the weather", parse_bundle=bundle)
        assert decision is not None
        entities = store.find_entities(kind="user_commitment")
        assert len(entities) >= 1
        slot = store.get_entity_slot(entities[0].entity_id, "topic")
        assert slot is not None
        assert "check the weather" in json.loads(slot.value_json)
        slot = store.get_entity_slot(entities[0].entity_id, "status")
        assert slot is not None and json.loads(slot.value_json) == "pending"

    def test_no_commitment_for_normal_utterance(self, store):
        from melm.appliance.local_assistant_router import OnDeviceAssistantRouter
        from melm.appliance import LocalAssistantProfile

        router = OnDeviceAssistantRouter(LocalAssistantProfile(), store=store)
        bundle = type("FakeBundle", (), {
            "semantic_unknown_tokens": [],
            "text": "what is the weather like",
            "language": "en",
            "tokens": ("what", "is", "the", "weather", "like"),
            "syntax_graph": None,
            "functional_parse": None,
            "uol_act": None,
            "session_id": "",
        })()
        decision = router._route_impl("what is the weather like", parse_bundle=bundle)
        assert decision is not None
        entities = store.find_entities(kind="user_commitment")
        assert len(entities) == 0

    def test_commitment_skipped_when_no_store(self):
        from melm.appliance.local_assistant_router import OnDeviceAssistantRouter
        from melm.appliance import LocalAssistantProfile

        router = OnDeviceAssistantRouter(LocalAssistantProfile(), store=None)
        bundle = type("FakeBundle", (), {
            "semantic_unknown_tokens": [],
            "text": "remind me to call mom",
            "language": "en",
            "tokens": ("remind", "me", "to", "call", "mom"),
            "syntax_graph": None,
            "functional_parse": None,
            "uol_act": None,
            "session_id": "",
        })()
        decision = router._route_impl("remind me to call mom", parse_bundle=bundle)
        assert decision is not None  # just should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
