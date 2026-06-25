"""Tests for atomic causality architecture (V0.4).

Covers: causal frame loading, task extraction fixes, state definitions,
multi-cause explanation, compound resolution, entity rule augmentation,
and atom role scoping.
"""

from __future__ import annotations

import unittest
from typing import Any

from melm.appliance.reasoning.task_router import detect_reasoning_task
from melm.appliance.reasoning.solvers import solve
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.contracts import load_causal_frames


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_causal_frames_loads_and_has_atomic_sections():
    data = load_causal_frames()
    expected = {"predicate_frames", "state_definitions", "active_entity_affordances", "surface_aliases"}
    assert expected <= set(data)
    assert "wet" in data["state_definitions"]
    assert "rain" in data["predicate_frames"]
    assert "gunshot" in data["surface_aliases"]


# ---------------------------------------------------------------------------
# Task extraction tests
# ---------------------------------------------------------------------------


def test_what_happens_if_glass_breaks_keeps_break_predicate():
    result = detect_reasoning_task("What happens if glass breaks?")
    assert result == {"task": "causal_prediction", "cause": "break", "patient": "glass"}


def test_what_happens_if_the_sun_shines_keeps_shine_predicate_and_sun_actor():
    result = detect_reasoning_task("What happens if the sun shines?")
    assert result == {"task": "causal_prediction", "cause": "shine", "actor": "sun"}


def test_inflected_causal_cue_routes_through_lemma():
    result = detect_reasoning_task("What causes the ground to be wet?")
    assert result == {"task": "causal_explanation", "effect": "wet", "theme": "ground"}


def test_plain_what_if_fallback():
    result = detect_reasoning_task("What happens if it rains?")
    assert result == {"task": "causal_prediction", "cause": "rain"}


def test_why_question_returns_effect():
    result = detect_reasoning_task("Why is the ground wet?")
    assert result == {"task": "causal_explanation", "effect": "wet", "theme": "ground"}


def test_false_positive_meal_not_causal():
    assert detect_reasoning_task("What should I eat?") is None


def test_false_positive_identity_not_causal():
    assert detect_reasoning_task("What is your name?") is None


def test_false_positive_status_not_causal():
    assert detect_reasoning_task("How are you?") is None


def test_false_positive_weather_not_causal():
    assert detect_reasoning_task("What is the weather?") is None


def test_self_query_not_causal():
    # Self_query runs before causal detection in detect_reasoning_task.
    # "Are you conscious?" correctly routes to self_query, not causal.
    result = detect_reasoning_task("Are you conscious?")
    assert result is not None
    assert result["task"] == "self_query"


def test_why_ground_wet_effect_only():
    """Why is the ground wet? → causal_explanation, effect=wet (no theme check)."""
    t = detect_reasoning_task("Why is the ground wet?")
    assert t is not None
    assert t["task"] == "causal_explanation"
    assert t["effect"] == "wet"


def test_why_she_sad():
    """"Why is she sad?" → causal_explanation, effect=sad."""
    t = detect_reasoning_task("Why is she sad?")
    assert t is not None
    assert t["task"] == "causal_explanation"
    assert t["effect"] == "sad"


def test_what_happens_if_i_eat():
    """"What happens if I eat?" → causal_prediction, cause=eat."""
    t = detect_reasoning_task("What happens if I eat?")
    assert t is not None
    assert t["task"] == "causal_prediction"
    assert t["cause"] == "eat"


def test_what_should_i_eat_today_not_causal():
    """What should I eat today? is not a causal task."""
    assert detect_reasoning_task("What should I eat today?") is None


def test_metalinguistic_count_detected():
    """How many r's in strawberry? → metalinguistic_count task."""
    t = detect_reasoning_task("How many r's in strawberry?")
    assert t == {"task": "metalinguistic_count", "char": "r", "word": "strawberry"}


def test_letter_phrasing_detected():
    """How many letter a in banana? → char=a, word=banana."""
    t = detect_reasoning_task("How many letter a in banana?")
    assert t["char"] == "a"
    assert t["word"] == "banana"


def test_arithmetic_detected():
    """I have 3 apples and eat one → quantity_arithmetic."""
    t = detect_reasoning_task("I have 3 apples and eat one, how many are left?")
    assert t["task"] == "quantity_arithmetic"
    assert t["start"] == 3.0
    assert t["delta"] == 1.0
    assert t["sign"] == -1
    assert t["noun"] == "apples"


def test_non_count_question_not_a_task():
    """Tell me a story. is not a reasoning task."""
    assert detect_reasoning_task("Tell me a story.") is None


# ---------------------------------------------------------------------------
# Solver tests (structured results)
# ---------------------------------------------------------------------------


class CausalSolverTests(unittest.TestCase):
    def test_wet_explanation_returns_state_definition_and_cause_kinds(self):
        result, answer, refusal = solve({"task": "causal_explanation", "effect": "wet", "theme": "ground"})
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        self.assertIn("state_definition", result)
        self.assertTrue(result["state_definition"]["definition"].startswith("has moisture"))
        kinds = {c["cause_kind"] for c in result.get("candidate_causes", [])}
        self.assertTrue({"natural_process", "intentional_action", "accidental_process"} <= kinds)

    def test_wet_multi_cause_includes_kinds(self):
        result, answer, refusal = solve({"task": "causal_explanation", "effect": "wet"})
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        labels = {
            c["predicate_id"]: c["cause_kind"]
            for c in result.get("candidate_causes", [])
        }
        self.assertEqual(labels.get("rain"), "natural_process")
        self.assertEqual(labels.get("water"), "intentional_action")
        self.assertEqual(labels.get("leak"), "accidental_process")

    def test_shine_prediction_returns_effects(self):
        result, answer, refusal = solve({"task": "causal_prediction", "cause": "shine", "actor": "sun"})
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        self.assertEqual(result.get("cause"), "shine")
        self.assertEqual(result.get("actor"), "sun")
        states = {e["state"] for e in result.get("effects", [])}
        self.assertTrue({"bright", "warmer", "dry"} <= states)

    def test_unknown_causal_explanation_abstains(self):
        result, answer, refusal = solve({"task": "causal_explanation", "effect": "xyzzy"})
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal, "no_cause_found")

    def test_unknown_causal_prediction_abstains(self):
        result, answer, refusal = solve({"task": "causal_prediction", "cause": "xyzzy"})
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal, "no_effect_found")


# ---------------------------------------------------------------------------
# Compound / surface alias resolution
# ---------------------------------------------------------------------------


class CausalCompoundTests(unittest.TestCase):
    def test_gun_shot_resolves_to_gunshot_nominalized_event(self):
        result, answer, refusal = solve({"task": "causal_explanation", "effect": "gun shot"})
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        self.assertIn("surface_resolution", result)
        self.assertEqual(result["surface_resolution"]["canonical"], "gunshot")
        causes = result.get("candidate_causes", [])
        self.assertTrue(any(c.get("instrument") == "gun" for c in causes))

    def test_gunshot_canonical_also_resolves(self):
        result, answer, refusal = solve({"task": "causal_explanation", "effect": "gunshot"})
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        self.assertIn("surface_resolution", result)
        self.assertEqual(result["surface_resolution"]["canonical"], "gunshot")


# ---------------------------------------------------------------------------
# Entity rule augmentation
# ---------------------------------------------------------------------------


class CausalEntityRuleTests(unittest.TestCase):
    def _make_store(self) -> AssistantOSStore:
        store = AssistantOSStore(":memory:")
        seed_class_schemas(store)
        return store

    def test_entity_rule_augments_existing_contract_cause(self):
        store = self._make_store()
        store.set_causal_rule(
            "cr_local_rain_slippery",
            "rain",
            "slippery",
            confidence=0.99,
            review_status="approved",
            scope="user_local",
        )
        result, answer, refusal = solve({"task": "causal_prediction", "cause": "rain"}, store=store)
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        states = {e["state"] for e in result.get("effects", [])}
        self.assertIn("slippery", states)

    def test_entity_rule_new_predicate_appended(self):
        store = self._make_store()
        store.set_causal_rule(
            "cr_magic",
            "abracadabra",
            "vanished",
            confidence=0.8,
            review_status="approved",
            scope="user_local",
        )
        result, answer, refusal = solve({"task": "causal_prediction", "cause": "abracadabra"}, store=store)
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        states = {e["state"] for e in result.get("effects", [])}
        self.assertIn("vanished", states)

    def test_pending_entity_rule_not_used(self):
        store = self._make_store()
        store.set_causal_rule(
            "cr_pending",
            "rain",
            "purple",
            confidence=0.9,
            review_status="pending",
            scope="user_local",
        )
        result, answer, refusal = solve({"task": "causal_prediction", "cause": "rain"}, store=store)
        self.assertIsNone(refusal)
        self.assertIsNotNone(result)
        if result is None:
            return
        states = {e["state"] for e in result.get("effects", [])}
        self.assertNotIn("purple", states)


# ---------------------------------------------------------------------------
# Atom role scoping
# ---------------------------------------------------------------------------


class CausalAtomRoleScopingTests(unittest.TestCase):
    def test_causal_atoms_do_not_copy_whole_sentence_roles(self):
        from melm.appliance.language_adapters import get_adapter
        from melm.appliance.uol_atomizer import atomize_syntax_graph

        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        act = atomize_syntax_graph(adapter.tag(("the", "ground", "be", "wet", "because", "it", "rain")))
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(len(act.content), 2)
        main_atom = act.content[0]
        sub_atom = act.content[1]

        main_values = {role.value for role in main_atom.roles}
        sub_values = {role.value for role in sub_atom.roles}
        self.assertNotIn("because", main_values)
        self.assertNotIn("because", sub_values)
        self.assertIn("ground", main_values)
        self.assertNotIn("ground", sub_values)
