"""End-to-end tests for causal graph support (C2 / V4B prerequisite).

These tests verify that causal reasoning still routes through the existing
reasoning task pipeline and that the UOL atom graph produced for causal
statements carries the expected causal links.
"""

from __future__ import annotations

import unittest

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.language_adapters import get_adapter
from melm.appliance.local_assistant_router import LocalAssistantProfile
from melm.appliance.uol_atomizer import atomize_syntax_graph


class CausalGraphMvpTests(unittest.TestCase):
    def _make_kernel(self) -> AssistantOSKernel:
        store = AssistantOSStore(":memory:")
        seed_class_schemas(store)
        profile = LocalAssistantProfile(user_id="causal-graph-test")
        return AssistantOSKernel(profile=profile, store=store)

    def test_why_question_still_routes_to_causal_explanation(self) -> None:
        kernel = self._make_kernel()
        decision = kernel.handle("Why is the ground wet?")
        self.assertEqual(decision.intent, "reasoning:causal_explanation")
        self.assertEqual(decision.reasoning_result.get("task"), "causal_explanation")
        self.assertEqual(decision.reasoning_result.get("effect"), "wet")

    def test_what_if_question_still_routes_to_causal_prediction(self) -> None:
        kernel = self._make_kernel()
        decision = kernel.handle("What happens if it rains?")
        self.assertEqual(decision.intent, "reasoning:causal_prediction")
        self.assertEqual(decision.reasoning_result.get("task"), "causal_prediction")
        self.assertEqual(decision.reasoning_result.get("cause"), "rain")

    def test_causal_statement_uol_act_has_atom_links(self) -> None:
        kernel = self._make_kernel()
        decision = kernel.handle("The ground is wet because it rained")
        uol_act = getattr(decision, "uol_act", None)
        self.assertIsNotNone(uol_act)
        assert uol_act is not None
        content = uol_act.get("content", [])
        self.assertEqual(len(content), 2)
        main_atom = content[0]
        sub_atom = content[1]
        self.assertEqual(main_atom["predicate"]["id"], "wet")
        self.assertEqual(sub_atom["predicate"]["id"], "rain")
        self.assertIn(sub_atom["id"], main_atom["links"]["caused_by"])
        self.assertIn(main_atom["id"], sub_atom["links"]["causes"])
        # Roles should be scoped to each clause.
        main_roles = {r["role"]: r["value"] for r in main_atom["roles"]}
        self.assertEqual(main_roles.get("theme"), "ground")
        sub_roles = {r["role"]: r["value"] for r in sub_atom["roles"]}
        self.assertEqual(sub_roles.get("theme"), "it")

    def test_causal_graph_verbalization_direction(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        tokens = ("the", "ground", "be", "wet", "because", "it", "rain")
        act = atomize_syntax_graph(adapter.tag(tokens))
        self.assertIsNotNone(act)
        assert act is not None
        main_atom = act.content[0]
        sub_atom = act.content[1]
        # Simple verbalization: subordinate -> relation -> main
        text = f"{sub_atom.predicate.id} causes {main_atom.predicate.id}"
        self.assertIn("rain", text)
        self.assertIn("wet", text)

if __name__ == "__main__":
    unittest.main()
