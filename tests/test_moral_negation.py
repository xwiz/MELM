"""Regression tests for the verb-causality negation bug.

The Tier 3c "verb causality contribution" block in
``melm.appliance.assistant_mood_engine.compute_utterance_affect`` derives a
moral context for the utterance's main verb and, when an implication is found,
appends a NEGATIVE ``source="verb_causality"`` AffectSignal (a harm signal).

Bug: it ignored clause polarity. A negated harm assertion ("do not hurt
her") is NOT a harm, yet the harmful causal signal was still emitted. The fix
adds a clause-polarity guard (``_clause_is_negated``) that suppresses the
causal signal when the main atom's context marks the clause as negated
(``polarity == "negative"`` or ``negation_scope is True``) or counterfactual
(``modality == "counterfactual"``).

Run with::

    cd <repo> && PYTHONPATH=. MELM_BULK_MAX_ENTRIES=200 \
        python -m pytest tests/test_moral_negation.py -q
"""

from __future__ import annotations

import copy
import unittest
import pytest

from melm.appliance.assistant_mood_engine import (
    _clause_is_negated,
    _patient_type_for_affect,
    compute_utterance_affect,
)
from melm.appliance.local_assistant_router import _build_parse_bundle


def _affect_for(text: str):
    """Compute the utterance affect for a surface string via the real parser."""
    bundle = _build_parse_bundle(text)
    lemmas = list(getattr(bundle, "lemmas", ()) or ())
    return bundle, compute_utterance_affect(lemmas, bundle.uol_act, None)


def _harmful_uol_act(verb: str = "hurt", patient_type: str = "person") -> dict:
    """A minimal UOL act dict shaped like the atomizer's output for a harm
    verb, with an asserted (positive) main-atom context."""
    return {
        "content": [
            {
                "id": "a0",
                "kind": "act",
                "predicate": {
                    "id": verb,
                    "semantic_class": "action",
                    "lemma": verb,
                    "language": "en",
                },
                "roles": {"patient": {"type": patient_type}},
                "context": {
                    "polarity": "positive",
                    "modality": "assertive",
                    "negation_scope": False,
                    "tense": "present",
                    "aspect": "simple",
                    "certainty": 1.0,
                    "time": None,
                },
                "links": [],
            }
        ]
    }


# ---- Parametrized test helpers ----


def _negated_uol_act(verb="hurt", patient_type="person"):
    act = _harmful_uol_act(verb, patient_type)
    act["content"][0]["context"]["polarity"] = "negative"
    act["content"][0]["context"]["negation_scope"] = True
    return act


def _counterfactual_uol_act(verb="hurt", patient_type="person"):
    act = _harmful_uol_act(verb, patient_type)
    act["content"][0]["context"]["modality"] = "counterfactual"
    return act


def _negated_realparser_sig():
    _, sig = _affect_for("do not hurt her")
    return sig


def _negated_synthetic_sig():
    return compute_utterance_affect(["hurt", "her"], _negated_uol_act(), None)


def _counterfactual_synthetic_sig():
    return compute_utterance_affect(["hurt", "her"], _counterfactual_uol_act(), None)


def _person_dict_shape_act():
    act = _harmful_uol_act("hurt")
    act["content"][0]["roles"] = {"theme": {"value": "him"}}
    return act


def _non_person_object_act():
    act = _harmful_uol_act("hurt")
    act["content"][0]["roles"] = [{"role": "theme", "value": "cup"}]
    return act


class TestRealParserShapes(unittest.TestCase):
    """Confirm the assumed context shape: polarity/negation differ for the
    positive vs negated harm clause produced by the real atomizer."""

    def test_positive_clause_context(self):
        bundle, _ = _affect_for("hurt her")
        ctx = bundle.uol_act["content"][0]["context"]
        self.assertEqual(ctx.get("polarity"), "positive")
        self.assertFalse(ctx.get("negation_scope"))

    def test_negated_clause_context(self):
        bundle, _ = _affect_for("do not hurt her")
        ctx = bundle.uol_act["content"][0]["context"]
        self.assertTrue(
            ctx.get("polarity") == "negative" or ctx.get("negation_scope") is True,
            f"expected negated context, got {ctx!r}",
        )


class TestClausePolarityGuard:
    """Direct unit tests of the guard that the fix introduces."""

    @pytest.mark.parametrize("mutate_fn,expected_negated", [
        pytest.param(lambda a: None, False, id="positive"),
        pytest.param(lambda a: a["content"][0]["context"].update(polarity="negative"), True, id="polarity_negative"),
        pytest.param(lambda a: a["content"][0]["context"].update(negation_scope=True), True, id="negation_scope"),
        pytest.param(lambda a: a["content"][0]["context"].update(modality="counterfactual"), True, id="counterfactual"),
    ])
    def test_clause_is_negated(self, mutate_fn, expected_negated):
        act = _harmful_uol_act()
        mutate_fn(act)
        assert _clause_is_negated(act) == expected_negated

    def test_real_parser_clauses(self):
        pos, _ = _affect_for("hurt her")
        neg, _ = _affect_for("do not hurt her")
        assert not _clause_is_negated(pos.uol_act)
        assert _clause_is_negated(neg.uol_act)

    def test_defensive_on_malformed_input(self):
        for bad in (None, {}, {"content": []}, {"content": [None]}, "nope"):
            assert not _clause_is_negated(bad)


class TestVerbCausalityNegation:
    """The fix's effect on ``compute_utterance_affect``."""

    # --- Test A: positive harm clause -------------------------------------
    def test_positive_harm_clause_is_not_treated_as_negated(self):
        """A positive (asserted) harm clause yields a valid AffectSignal and
        the negation guard does NOT fire for it, so the verb-causality harm
        contribution remains permitted (the guard passes the clause through)."""
        bundle, sig = _affect_for("hurt her")
        assert sig is not None
        assert hasattr(sig, "source")
        assert not _clause_is_negated(bundle.uol_act), (
            "asserted harm clause must not be gated out as negated"
        )
        assert "_negated" not in sig.source
        assert getattr(sig, "verb_causal_valence", 0.0) < 0.0

    # --- Tests B + C: negated / counterfactual harm clause -----------------
    @pytest.mark.parametrize("build_sig", [
        pytest.param(_negated_realparser_sig, id="realparser"),
        pytest.param(_negated_synthetic_sig, id="synthetic"),
        pytest.param(_counterfactual_synthetic_sig, id="counterfactual"),
    ])
    def test_negated_harm_suppresses_causal_signal(self, build_sig):
        """A negated or counterfactual harm clause must not emit a dominant
        verb_causality harm signal."""
        sig = build_sig()
        assert sig is not None
        assert sig.source != "verb_causality", (
            "negated/counterfactual harm must not produce a "
            "dominant verb_causality signal"
        )
        assert getattr(sig, "verb_causal_valence", 0.0) >= 0.0

    def test_negation_guard_distinguishes_otherwise_identical_acts(self):
        """A/B: two acts identical except for polarity must differ only in
        whether the negation guard fires — proving the guard is the cause."""
        pos = _harmful_uol_act("hurt", "person")
        neg = copy.deepcopy(pos)
        neg["content"][0]["context"]["polarity"] = "negative"
        neg["content"][0]["context"]["negation_scope"] = True

        assert not _clause_is_negated(pos)
        assert _clause_is_negated(neg)

        neg_sig = compute_utterance_affect(["hurt", "her"], neg, None)
        assert neg_sig.source != "verb_causality"
        assert getattr(neg_sig, "verb_causal_valence", 0.0) >= 0.0


class TestPatientTypeForAffect:
    """The follow-up fix: the Tier 3c block hardcoded ``"biological_body"``,
    which no verb in ``verb_states.v1.json`` lists as a ``patient_type`` — so
    ``derive_moral_context`` always returned an empty context and the harm
    path was dead. The fix maps the theme/patient surface to a real sentient
    label ("person") that harm verbs DO list."""

    @pytest.mark.parametrize("build_act", [
        pytest.param(lambda: _affect_for("hurt her")[0].uol_act, id="list_shape"),
        pytest.param(_person_dict_shape_act, id="dict_shape"),
    ])
    def test_person_marker_maps_to_person(self, build_act):
        assert _patient_type_for_affect(build_act()) == "person"

    @pytest.mark.parametrize("build_act", [
        pytest.param(_non_person_object_act, id="non_person_object"),
        pytest.param(lambda: None, id="null"),
        pytest.param(lambda: {}, id="empty_dict"),
        pytest.param(lambda: {"content": []}, id="empty_content"),
        pytest.param(lambda: {"content": [None]}, id="none_content"),
        pytest.param(lambda: "nope", id="string"),
    ])
    def test_default_is_person(self, build_act):
        assert _patient_type_for_affect(build_act()) == "person"

    def test_chosen_label_is_a_real_harm_patient_type(self):
        """The label the fix uses must actually be a ``patient_type`` for harm
        verbs in the contract (otherwise the path stays dead), and must be
        sentient so the harm valences register."""
        from melm.contracts import load_contract_json
        from melm.appliance.reasoning.implications import derive_moral_context

        vs = load_contract_json("verb_states.v1.json")
        val = load_contract_json("state_valences.v1.json").get("valences", {})
        for verb in ("hurt", "kill", "harm"):
            assert "person" in vs["verbs"][verb]["patient_types"], (
                f"{verb!r} must list 'person' as a patient_type"
            )
            mc = derive_moral_context(verb, "person", vs, val)
            assert mc.has_implication, (
                f"derive_moral_context({verb!r}, 'person') must yield an implication"
            )
        dead = derive_moral_context("hurt", "biological_body", vs, val)
        assert not dead.has_implication


class TestVerbCausalityHarmPathReachable(unittest.TestCase):
    """End-to-end: with the fix, a positive person-harm verb actually emits a
    reachable ``verb_causality`` harm signal (it was dead before)."""

    def test_positive_harm_emits_verb_causality_realparser(self):
        """"hurt her" (positive harm to a person) now produces a dominant
        ``verb_causality`` signal with negative valence."""
        bundle, sig = _affect_for("hurt her")
        self.assertEqual(
            sig.source,
            "verb_causality",
            "positive person-harm must surface a verb_causality harm signal",
        )
        self.assertLess(sig.valence, 0.0, "harm signal must be negative valence")
        self.assertLess(getattr(sig, "verb_causal_valence", 0.0), 0.0)

    def test_positive_harm_emits_verb_causality_synthetic(self):
        act = _harmful_uol_act("hurt", "person")
        act["content"][0]["roles"] = [{"role": "theme", "value": "her"}]
        sig = compute_utterance_affect(["hurt", "her"], act, None)
        self.assertEqual(sig.source, "verb_causality")
        self.assertLess(sig.valence, 0.0)

    def test_negated_harm_has_no_verb_causality(self):
        """The negation guard still holds: "do not hurt her" yields no
        dominant verb_causality harm signal."""
        _, sig = _affect_for("do not hurt her")
        self.assertNotEqual(sig.source, "verb_causality")
        self.assertGreaterEqual(getattr(sig, "verb_causal_valence", 0.0), 0.0)

    def test_benign_clause_has_no_verb_causality_harm(self):
        """A benign/non-harm clause ("help her") produces no verb_causality
        harm signal — derive_moral_context yields no harm implication for a
        benevolent verb, so the harm tier stays silent."""
        _, sig = _affect_for("help her")
        self.assertNotEqual(
            sig.source,
            "verb_causality",
            "benign clause must not emit a verb_causality harm signal",
        )
        self.assertGreaterEqual(getattr(sig, "verb_causal_valence", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
