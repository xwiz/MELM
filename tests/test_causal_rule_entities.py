"""Tests for causal_rule entity storage and solver merge layer (V4B)."""

from __future__ import annotations

import unittest

from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.reasoning.solvers import solve, _load_entity_causal_rules_for_merge


class CausalRuleEntityTests(unittest.TestCase):
    def _make_store(self) -> AssistantOSStore:
        store = AssistantOSStore(":memory:")
        seed_class_schemas(store)
        return store

    def test_set_causal_rule_stores_slots(self) -> None:
        store = self._make_store()
        store.set_causal_rule(
            "cr_1",
            "rain",
            "wet",
            confidence=0.95,
            provenance="manual_curated",
            review_status="approved",
            scope="global",
        )
        rules = store.query_causal_rules("rain", "wet")
        self.assertEqual(len(rules), 1)
        rule = rules[0]
        self.assertEqual(rule["cause_lemma"], "rain")
        self.assertEqual(rule["effect_state"], "wet")
        self.assertEqual(rule["confidence"], 0.95)
        self.assertEqual(rule["review_status"], "approved")
        self.assertEqual(rule["scope"], "global")

    def test_query_filters_by_review_status(self) -> None:
        store = self._make_store()
        store.set_causal_rule("cr_1", "rain", "wet", review_status="approved")
        store.set_causal_rule("cr_2", "sun", "dry", review_status="pending")
        approved = store.query_causal_rules(review_status="approved")
        self.assertEqual({r["effect_state"] for r in approved}, {"wet"})
        pending = store.query_causal_rules(review_status="pending")
        self.assertEqual({r["effect_state"] for r in pending}, {"dry"})
        all_rules = store.query_causal_rules(review_status=None)
        self.assertEqual(len(all_rules), 2)

    def test_entity_rules_merge_for_cause_already_in_contract(self) -> None:
        """An approved local rule for a cause already in the contract should add new effects."""
        store = self._make_store()
        store.set_causal_rule(
            "cr_1",
            "rain",
            "slippery",
            confidence=0.9,
            review_status="approved",
            scope="user_local",
        )
        result, answer, refusal = solve(
            {"task": "causal_prediction", "cause": "rain"}, store=store
        )
        self.assertIsNone(refusal)
        states = {e["state"] for e in result.get("effects", [])}
        self.assertIn("slippery", states)
        self.assertIn("slippery", answer)

    def test_entity_rules_merge_for_explanation(self) -> None:
        """An approved entity rule should surface as a candidate cause for a new effect."""
        store = self._make_store()
        store.set_causal_rule(
            "cr_1",
            "rain",
            "slippery",
            confidence=0.9,
            review_status="approved",
            scope="user_local",
        )
        result, answer, refusal = solve(
            {"task": "causal_explanation", "effect": "slippery"}, store=store
        )
        self.assertIsNone(refusal)
        candidates = {c["predicate_id"] for c in result.get("candidate_causes", [])}
        self.assertIn("rain", candidates)
        self.assertIn("rain", answer)

    def test_approved_entity_rule_boosts_existing_effect_confidence(self) -> None:
        """If the entity rule duplicates a contract effect, the higher confidence wins."""
        store = self._make_store()
        store.set_causal_rule(
            "cr_1",
            "rain",
            "wet",
            confidence=0.99,
            review_status="approved",
            scope="user_local",
        )
        result, answer, refusal = solve(
            {"task": "causal_prediction", "cause": "rain"}, store=store
        )
        self.assertIsNone(refusal)
        wet_effect = next(e for e in result.get("effects", []) if e["state"] == "wet")
        self.assertEqual(wet_effect["confidence"], 0.99)

    def test_approved_entity_rule_includes_causal_contrast(self) -> None:
        """An approved local rule for a cause should appear in a causal contrast."""
        store = self._make_store()
        store.set_causal_rule(
            "cr_1",
            "rain",
            "slippery",
            confidence=0.9,
            review_status="approved",
            scope="user_local",
        )
        result, answer, refusal = solve(
            {"task": "causal_contrast", "cause_a": "rain", "cause_b": "snow"},
            store=store,
        )
        self.assertIsNone(refusal)
        self.assertIn("slippery", result.get("effects_a", []))
        self.assertIn("slippery", answer)

    def test_solver_ignores_pending_entity_rule(self) -> None:
        store = self._make_store()
        store.set_causal_rule(
            "cr_1",
            "ghost",
            "haunt",
            confidence=0.8,
            review_status="pending",
            scope="user_local",
        )
        result, answer, refusal = solve(
            {"task": "causal_explanation", "effect": "haunt"}, store=store
        )
        self.assertEqual(refusal, "no_cause_found")


if __name__ == "__main__":
    unittest.main()
