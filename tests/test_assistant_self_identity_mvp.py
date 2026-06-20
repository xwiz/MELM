"""Tests for the self-identity derivation system.

Covers:
- assistant_skill_self_identity.py (pure analysis/narrative/explanation)
- Router's name-related patterns in _identity_composition()
- Synthesis _handle_identity() dispatch with identity-aware fallback
- Full kernel integration (store-backed personal_experience → identity)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from melm.appliance.assistant_os_store import AssistantOSStore
from melm.appliance.assistant_os_store import seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile
from melm.appliance.local_assistant_router import (
    AssistantDecision,
    OnDeviceAssistantRouter,
    parse_assistant_debug_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> AssistantOSStore:
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    return store


def _add_personal_experience(
    store: AssistantOSStore,
    entity_id: str,
    intent: str,
    user_id: str,
    polarity: float,
) -> None:
    store.add_entity(
        entity_id=entity_id,
        kind="personal_experience",
        label=f"turn: {intent}",
        semantic_class_id="personal_experience",
        canonical_lemma=f"test utterance about {intent}",
    )
    store.set_entity_slot(entity_id, "outcome", "resolved", provenance="experience_writer")
    store.set_entity_slot(entity_id, "polarity", polarity, provenance="experience_writer")
    store.set_entity_slot(entity_id, "user_id", user_id, provenance="experience_writer")


# ---------------------------------------------------------------------------
# Module-level tests: assistant_skill_self_identity
# ---------------------------------------------------------------------------

class SelfIdentitySkillTests(unittest.TestCase):
    """Pure function tests for the self-identity derivation module."""

    def test_analyze_no_data_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            result = analyze_user_identity(store, "user_test")
            self.assertIsNone(result)
        finally:
            store.close()

    def test_analyze_single_intent(self) -> None:
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            _add_personal_experience(store, "pe_001", "story", "user_test", 0.5)
            _add_personal_experience(store, "pe_002", "story", "user_test", 0.3)
            _add_personal_experience(store, "pe_003", "story", "user_test", 0.4)
            result = analyze_user_identity(store, "user_test")
            self.assertIsNotNone(result)
            self.assertEqual(result.highest_meaning_intent, "story")
            self.assertEqual(result.top_intent, "story")
            self.assertEqual(result.total_turns, 3)
            self.assertAlmostEqual(result.highest_meaning_polarity, 0.4)
        finally:
            store.close()

    def test_analyze_highest_polarity_wins(self) -> None:
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            _add_personal_experience(store, "pe_001", "story", "user_test", 0.1)
            _add_personal_experience(store, "pe_002", "story", "user_test", 0.2)
            _add_personal_experience(store, "pe_003", "weather", "user_test", 0.9)
            result = analyze_user_identity(store, "user_test")
            self.assertIsNotNone(result)
            self.assertEqual(result.highest_meaning_intent, "weather")
            self.assertEqual(result.top_intent, "story")
        finally:
            store.close()

    def test_analyze_top_intent_wins_when_ties(self) -> None:
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            _add_personal_experience(store, "pe_001", "story", "user_test", 0.5)
            _add_personal_experience(store, "pe_002", "weather", "user_test", 0.5)
            _add_personal_experience(store, "pe_003", "weather", "user_test", 0.5)
            result = analyze_user_identity(store, "user_test")
            self.assertIsNotNone(result)
            self.assertEqual(result.highest_meaning_intent, "weather")
            self.assertEqual(result.top_intent, "weather")
        finally:
            store.close()

    def test_min_data_points_gate(self) -> None:
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            _add_personal_experience(store, "pe_001", "story", "user_test", 0.5)
            _add_personal_experience(store, "pe_002", "story", "user_test", 0.3)
            result = analyze_user_identity(store, "user_test")
            self.assertIsNone(result)
        finally:
            store.close()

    def test_analyze_per_user_isolation(self) -> None:
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            _add_personal_experience(store, "pe_001", "story", "user_alice", 0.5)
            _add_personal_experience(store, "pe_002", "weather", "user_alice", 0.3)
            _add_personal_experience(store, "pe_003", "weather", "user_alice", 0.4)
            _add_personal_experience(store, "pe_004", "meal_suggestion", "user_bob", 0.9)
            _add_personal_experience(store, "pe_005", "meal_suggestion", "user_bob", 0.8)
            _add_personal_experience(store, "pe_006", "meal_suggestion", "user_bob", 0.7)
            alice = analyze_user_identity(store, "user_alice")
            bob = analyze_user_identity(store, "user_bob")
            self.assertIsNotNone(alice)
            self.assertIsNotNone(bob)
            self.assertEqual(alice.highest_meaning_intent, "story")
            self.assertEqual(bob.highest_meaning_intent, "meal_suggestion")
        finally:
            store.close()

    def test_identity_narrative_selects_by_mood(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_narrative,
        )
        identity = DerivedIdentity(
            user_id="test", highest_meaning_intent="story",
            highest_meaning_polarity=0.5, top_intent="story",
            top_intent_count=3, top_intent_mean_polarity=0.5,
            total_turns=3, per_intent_counts={"story": 3},
            per_intent_mean_polarities={"story": 0.5},
        )
        narrative = derive_identity_narrative(identity, "happy")
        self.assertIsNotNone(narrative)
        self.assertIn("storyteller", narrative)

    def test_identity_narrative_falls_back_to_neutral(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_narrative,
        )
        identity = DerivedIdentity(
            user_id="test", highest_meaning_intent="weather",
            highest_meaning_polarity=0.5, top_intent="weather",
            top_intent_count=3, top_intent_mean_polarity=0.5,
            total_turns=3, per_intent_counts={"weather": 3},
            per_intent_mean_polarities={"weather": 0.5},
        )
        narrative = derive_identity_narrative(identity, "nonexistent_mood")
        self.assertIsNotNone(narrative)
        self.assertIn("weather watcher", narrative)

    def test_identity_explanation_format(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_explanation,
        )
        identity = DerivedIdentity(
            user_id="test", highest_meaning_intent="story",
            highest_meaning_polarity=0.6, top_intent="story",
            top_intent_count=12, top_intent_mean_polarity=0.6,
            total_turns=15, per_intent_counts={"story": 12, "weather": 3},
            per_intent_mean_polarities={"story": 0.6, "weather": 0.3},
        )
        explanation = derive_identity_explanation(identity)
        self.assertIsNotNone(explanation)
        self.assertIn("sharing stories", explanation)
        self.assertIn("12", explanation)
        self.assertIn("+0.6", explanation)

    def test_get_name_awareness_template(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            get_name_awareness_template,
        )
        identity = DerivedIdentity(
            user_id="test", highest_meaning_intent="story",
            highest_meaning_polarity=0.5, top_intent="story",
            top_intent_count=3, top_intent_mean_polarity=0.5,
            total_turns=3, per_intent_counts={"story": 3},
            per_intent_mean_polarities={"story": 0.5},
        )
        result = get_name_awareness_template(identity, "no_name")
        self.assertIsNotNone(result)
        self.assertIn("storyteller", result)


# ---------------------------------------------------------------------------
# Router tests: name-related identity patterns
# ---------------------------------------------------------------------------

class SelfIdentityRouterTests(unittest.TestCase):
    """Router tests for name-related identity patterns."""

    def test_router_detects_who_are_you(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Who are you?")
        self.assertEqual(decision.intent, "assistant_identity")

    def test_router_detects_suggest_name(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What should I call you?")
        self.assertEqual(decision.intent, "assistant_identity")
        self.assertIn("identity_action:suggest_name", decision.evidence_keys)

    def test_router_detects_name_awareness(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Do you have a name?")
        self.assertEqual(decision.intent, "assistant_identity")
        self.assertIn("identity_action:name_awareness", decision.evidence_keys)

    def test_router_detects_name_origin(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Did you name yourself?")
        self.assertEqual(decision.intent, "assistant_identity")
        self.assertIn("identity_action:name_origin", decision.evidence_keys)

    def test_router_detects_real_name_question(self) -> None:
        """'Real name' uses the name frame (action='name'), not the name_awareness path."""
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("What is your real name?")
        self.assertEqual(decision.intent, "assistant_identity")
        self.assertIn("self_model.name", decision.evidence_keys)

    def test_standard_identity_no_action_marker(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("Who are you?")
        self.assertEqual(decision.intent, "assistant_identity")
        identity_actions = [k for k in decision.evidence_keys if k.startswith("identity_action:")]
        self.assertEqual(len(identity_actions), 0)


# ---------------------------------------------------------------------------
# Kernel integration tests: full pipeline
# ---------------------------------------------------------------------------

class SelfIdentityKernelTests(unittest.TestCase):
    """Full pipeline integration: kernel → router → synthesis with identity."""

    def test_kernel_identity_answer_with_empty_store(self) -> None:
        """With no personal_experience data, should fall back to identity summary."""
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                from melm.appliance.assistant_os_kernel import AssistantOSKernel
                kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=store)
                decision = kernel.handle("Who are you?")
                self.assertEqual(decision.intent, "assistant_identity")
                self.assertIn("MELM", decision.answer)
            finally:
                store.close()

    def test_kernel_identity_answer_with_store_data(self) -> None:
        """With personal_experience data, should derive identity and use narrative."""
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                from melm.appliance.assistant_os_kernel import AssistantOSKernel
                kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=store)
                kernel.handle("Tell me a story.")
                kernel.handle("Tell me another story.")
                kernel.handle("Tell me a third story.")
                decision = kernel.handle("Who are you?")
                self.assertEqual(decision.intent, "assistant_identity")
                self.assertIn("storyteller", decision.answer)
            finally:
                store.close()

    def test_kernel_suggest_name(self) -> None:
        """'What should I call you?' should suggest a name based on usage."""
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                from melm.appliance.assistant_os_kernel import AssistantOSKernel
                kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=store)
                kernel.handle("Tell me a story.")
                kernel.handle("Tell me another story.")
                kernel.handle("Tell me a third story.")
                decision = kernel.handle("What should I call you?")
                self.assertEqual(decision.intent, "assistant_identity")
                self.assertIn("storyteller", decision.answer)
            finally:
                store.close()


# ---------------------------------------------------------------------------
# Store-backed self-state tests
# ---------------------------------------------------------------------------

class SelfIdentityStoreTests(unittest.TestCase):
    """Tests for store's self_identity persistence methods."""

    def test_save_and_load_self_identity(self) -> None:
        store = _make_store()
        try:
            payload = {
                "user_id": "test",
                "highest_meaning_intent": "story",
                "highest_meaning_polarity": 0.6,
                "top_intent": "story",
                "top_intent_count": 5,
            }
            store.save_self_identity(payload)
            loaded = store.load_self_identity()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["highest_meaning_intent"], "story")
            self.assertEqual(loaded["top_intent_count"], 5)
        finally:
            store.close()

    def test_load_self_identity_returns_none_when_not_set(self) -> None:
        store = _make_store()
        try:
            loaded = store.load_self_identity()
            self.assertIsNone(loaded)
        finally:
            store.close()

    def test_set_given_name(self) -> None:
        store = _make_store()
        try:
            store.set_given_name("Alice")
            state = store.load_self_state()
            self.assertEqual(state["given_name"], "Alice")
            self.assertTrue(state["has_name"])
        finally:
            store.close()

    def test_has_name_from_self_state(self) -> None:
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            _add_personal_experience(store, "pe_001", "story", "test", 0.5)
            _add_personal_experience(store, "pe_002", "story", "test", 0.6)
            _add_personal_experience(store, "pe_003", "story", "test", 0.4)
            store.set_given_name("Buddy")
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertTrue(result.has_name)
            self.assertEqual(result.given_name, "Buddy")
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class SelfIdentityEdgeCaseTests(unittest.TestCase):
    """Edge cases for identity derivation."""

    def test_no_polarity_data_still_counts(self) -> None:
        """Entities without polarity slots should still count toward frequency."""
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            store.add_entity(
                "pe_001", "personal_experience", "turn: story",
                "personal_experience", "test about story",
            )
            store.set_entity_slot("pe_001", "user_id", "test", provenance="experience_writer")
            store.add_entity(
                "pe_002", "personal_experience", "turn: story",
                "personal_experience", "test about story",
            )
            store.set_entity_slot("pe_002", "user_id", "test", provenance="experience_writer")
            store.add_entity(
                "pe_003", "personal_experience", "turn: story",
                "personal_experience", "test about story",
            )
            store.set_entity_slot("pe_003", "user_id", "test", provenance="experience_writer")
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertEqual(result.total_turns, 3)
            self.assertEqual(result.top_intent, "story")
            self.assertAlmostEqual(result.highest_meaning_polarity, 0.0)
        finally:
            store.close()

    def test_analyze_ignores_other_entity_kinds(self) -> None:
        """Non personal_experience entities should be ignored."""
        from melm.appliance.assistant_skill_self_identity import analyze_user_identity
        store = _make_store()
        try:
            store.add_entity("person_001", "person", "Alice", "person")
            store.add_entity("place_001", "place", "Home", "place")
            result = analyze_user_identity(store, "test")
            self.assertIsNone(result)
        finally:
            store.close()

    def test_identity_narrative_unknown_intent_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_narrative,
        )
        identity = DerivedIdentity(
            user_id="test", highest_meaning_intent="nonexistent_intent",
            highest_meaning_polarity=0.0, top_intent="nonexistent_intent",
            top_intent_count=0, top_intent_mean_polarity=0.0,
            total_turns=0,
        )
        result = derive_identity_narrative(identity, "neutral")
        self.assertIsNone(result)

    def test_identity_explanation_unknown_intent_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_explanation,
        )
        identity = DerivedIdentity(
            user_id="test", highest_meaning_intent="nonexistent_intent",
            highest_meaning_polarity=0.0, top_intent="nonexistent_intent",
            top_intent_count=0, top_intent_mean_polarity=0.0,
            total_turns=0,
        )
        result = derive_identity_explanation(identity)
        self.assertIsNone(result)

    def test_get_name_awareness_unknown_intent_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            get_name_awareness_template,
        )
        identity = DerivedIdentity(
            user_id="test", highest_meaning_intent="nonexistent_intent",
            highest_meaning_polarity=0.0, top_intent="nonexistent_intent",
            top_intent_count=0, top_intent_mean_polarity=0.0,
            total_turns=0,
        )
        result = get_name_awareness_template(identity, "no_name")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
