"""Tests for UOL causal atom links (C2 / V4B prerequisite)."""

from __future__ import annotations

import unittest

from melm.appliance.language_adapters import get_adapter
from melm.appliance.uol_atomizer import atomize_syntax_graph
from melm.appliance.uol_types import AtomLinks, UolAtom, PredicateRef, AtomContext


class UolAtomLinkTests(unittest.TestCase):
    def test_atom_links_default_causal_fields_empty(self) -> None:
        links = AtomLinks()
        self.assertEqual(links.causes, ())
        self.assertEqual(links.caused_by, ())
        self.assertEqual(links.enables, ())
        self.assertEqual(links.prevents, ())

    def test_atom_links_serialize_causal_fields(self) -> None:
        atom = UolAtom(
            id="a1",
            kind="state",
            predicate=PredicateRef(id="be", semantic_class="verb.stative"),
            links=AtomLinks(
                causes=("a2",),
                caused_by=("a3",),
                enables=("a4",),
                prevents=("a5",),
            ),
        )
        d = atom.to_dict()
        self.assertEqual(d["links"]["causes"], ["a2"])
        self.assertEqual(d["links"]["caused_by"], ["a3"])
        self.assertEqual(d["links"]["enables"], ["a4"])
        self.assertEqual(d["links"]["prevents"], ["a5"])

    def test_because_links_subordinate_causes_main(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        tokens = ("the", "ground", "be", "wet", "because", "it", "rain")
        act = atomize_syntax_graph(adapter.tag(tokens))
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(len(act.content), 2)
        main_atom = act.content[0]
        sub_atom = act.content[1]
        self.assertEqual(main_atom.predicate.id, "wet")
        self.assertEqual(sub_atom.predicate.id, "rain")
        self.assertIn(sub_atom.id, main_atom.links.caused_by)
        self.assertIn(main_atom.id, sub_atom.links.causes)
        # Roles should be scoped to each clause.
        main_roles = {r.role: r.value for r in main_atom.roles}
        self.assertEqual(main_roles.get("theme"), "ground")
        sub_roles = {r.role: r.value for r in sub_atom.roles}
        self.assertEqual(sub_roles.get("theme"), "it")

    def test_so_links_main_causes_subordinate(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        tokens = ("it", "rain", "so", "the", "ground", "be", "wet")
        act = atomize_syntax_graph(adapter.tag(tokens))
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(len(act.content), 2)
        main_atom = act.content[0]
        sub_atom = act.content[1]
        self.assertEqual(main_atom.predicate.id, "rain")
        self.assertEqual(sub_atom.predicate.id, "be")
        self.assertIn(sub_atom.id, main_atom.links.causes)
        self.assertIn(main_atom.id, sub_atom.links.caused_by)

    def test_if_links_subordinate_causes_main(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        tokens = ("if", "it", "rain", "the", "ground", "be", "wet")
        act = atomize_syntax_graph(adapter.tag(tokens))
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(len(act.content), 2)
        main_atom = act.content[0]
        sub_atom = act.content[1]
        self.assertEqual(main_atom.predicate.id, "be")
        self.assertEqual(sub_atom.predicate.id, "rain")
        self.assertIn(sub_atom.id, main_atom.links.caused_by)
        self.assertIn(main_atom.id, sub_atom.links.causes)

    def test_non_causal_sentence_emits_single_atom(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        tokens = ("what", "will", "i", "eat")
        act = atomize_syntax_graph(adapter.tag(tokens))
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(len(act.content), 1)
        self.assertEqual(act.content[0].links.causes, ())
        self.assertEqual(act.content[0].links.caused_by, ())


if __name__ == "__main__":
    unittest.main()
