"""Tests for the UOL atomizer: FunctionalParse → UolAct → atoms."""

import unittest

from melm.appliance.functional_grammar import parse_functional_relations
from melm.appliance.language_adapters import get_adapter
from melm.appliance.uol_atomizer import (
    atomize,
    atomize_syntax_graph,
    atoms_to_functional_parse,
)
from melm.appliance.uol_types import UolAct, UolAtom


class UolAtomizerTests(unittest.TestCase):
    def test_english_wh_question_atoms(self) -> None:
        tokens = ("what", "will", "i", "eat")
        fp = parse_functional_relations(tokens, question_mark=True)
        self.assertIsNotNone(fp)
        act = atomize(fp, language="en")
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.act, "question")
        self.assertEqual(act.speaker, "user")
        self.assertEqual(act.addressee, "assistant")
        self.assertEqual(act.expected_answer_type, "entity")
        self.assertEqual(len(act.content), 1)
        atom = act.content[0]
        self.assertEqual(atom.predicate.id, "eat")
        self.assertEqual(atom.kind, "event")
        roles = {r.role: r.value for r in atom.roles}
        self.assertIn("agent", roles)
        self.assertEqual(roles["agent"], "user")
        self.assertIn("predicate", roles)
        self.assertEqual(roles["predicate"], "eat")

    def test_expected_answer_type_comes_from_function_word_contract(self) -> None:
        tokens = ("why", "should", "i", "eat")
        fp = parse_functional_relations(tokens, question_mark=True)
        self.assertIsNotNone(fp)
        act = atomize(fp, language="en")
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.expected_answer_type, "reason")

    def test_modal_context_comes_from_function_word_contract(self) -> None:
        tokens = ("should", "i", "eat")
        fp = parse_functional_relations(tokens, question_mark=True)
        self.assertIsNotNone(fp)
        act = atomize(fp, language="en")
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.content[0].context.modality, "obligation")

    def test_igbo_question_atoms(self) -> None:
        tokens = ("gini", "m", "ga", "eri")
        fp = parse_functional_relations(tokens, question_mark=True, language="ig")
        self.assertIsNotNone(fp)
        act = atomize(fp, language="ig")
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.act, "question")
        self.assertEqual(act.expected_answer_type, "entity")
        atom = act.content[0]
        self.assertEqual(atom.predicate.id, "eat")
        roles = {r.role: r.value for r in atom.roles}
        self.assertIn("agent", roles)
        self.assertEqual(roles["agent"], "user")

    def test_backward_projection_roundtrip(self) -> None:
        tokens = ("what", "will", "i", "eat")
        fp = parse_functional_relations(tokens, question_mark=True)
        self.assertIsNotNone(fp)
        act = atomize(fp, language="en")
        self.assertIsNotNone(act)
        fp2 = atoms_to_functional_parse(act)
        self.assertIsNotNone(fp2)
        assert fp2 is not None
        self.assertEqual(fp2.speech_act, "wh_question")
        self.assertEqual(fp2.subject, "user")
        self.assertEqual(fp2.action, "eat")
        self.assertEqual(fp2.target, "assistant")

    def test_none_parse_returns_none(self) -> None:
        act = atomize(None)
        self.assertIsNone(act)

    def test_greeting_atoms(self) -> None:
        tokens = ("hello",)
        fp = parse_functional_relations(tokens)
        self.assertIsNotNone(fp)
        act = atomize(fp, language="en")
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.act, "greeting")
        self.assertEqual(len(act.content), 1)

    def test_syntax_graph_atomizer_matches_parse_bridge_for_english_question(self) -> None:
        tokens = ("what", "will", "i", "eat")
        fp = parse_functional_relations(tokens, question_mark=True)
        self.assertIsNotNone(fp)
        parse_act = atomize(fp, language="en")
        self.assertIsNotNone(parse_act)
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        graph = adapter.tag(tokens)
        graph_act = atomize_syntax_graph(graph)
        self.assertIsNotNone(graph_act)
        assert parse_act is not None and graph_act is not None
        self.assertEqual(graph_act.act, parse_act.act)
        self.assertEqual(graph_act.expected_answer_type, parse_act.expected_answer_type)
        self.assertEqual(graph_act.content[0].predicate.id, parse_act.content[0].predicate.id)
        self.assertEqual(
            {r.role: r.value for r in graph_act.content[0].roles},
            {r.role: r.value for r in parse_act.content[0].roles},
        )

    def test_syntax_graph_atomizer_matches_parse_bridge_for_igbo_question(self) -> None:
        tokens = ("gini", "m", "ga", "eri")
        fp = parse_functional_relations(tokens, question_mark=True, language="ig")
        self.assertIsNotNone(fp)
        parse_act = atomize(fp, language="ig")
        self.assertIsNotNone(parse_act)
        adapter = get_adapter("ig")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        graph = adapter.tag(tokens)
        graph_act = atomize_syntax_graph(graph)
        self.assertIsNotNone(graph_act)
        assert parse_act is not None and graph_act is not None
        self.assertEqual(graph_act.act, parse_act.act)
        self.assertEqual(graph_act.expected_answer_type, parse_act.expected_answer_type)
        self.assertEqual(graph_act.content[0].predicate.id, parse_act.content[0].predicate.id)
        self.assertEqual(
            {r.role: r.value for r in graph_act.content[0].roles},
            {r.role: r.value for r in parse_act.content[0].roles},
        )

    def test_syntax_graph_atomizer_supports_story_request_and_contact_command(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        for tokens, expected_predicate in (
            (("tell", "me", "a", "story"), "tell"),
            (("call", "mom"), "call"),
        ):
            act = atomize_syntax_graph(adapter.tag(tokens))
            self.assertIsNotNone(act)
            assert act is not None
            self.assertEqual(act.content[0].predicate.id, expected_predicate)

    def test_syntax_graph_atomizer_marks_polite_imperative_as_request(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None

        act = atomize_syntax_graph(adapter.tag(("please", "give", "me", "a", "fable")))

        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.act, "request")
        self.assertEqual(act.content[0].predicate.id, "give")

    def test_syntax_graph_atomizer_keeps_declarative_story_statement_as_claim(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None

        act = atomize_syntax_graph(adapter.tag(("the", "same", "people", "tell", "stories")))

        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.act, "claim")
        self.assertEqual(act.content[0].predicate.id, "tell")

    def test_syntax_graph_atomizer_marks_aux_led_yes_no_question_as_question(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None

        act = atomize_syntax_graph(adapter.tag(("are", "you", "using", "cloud")))

        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.act, "question")
        self.assertEqual(act.content[0].predicate.id, "be")

    def test_syntax_graph_atomizer_marks_unknown_words_unresolved(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        act = atomize_syntax_graph(adapter.tag(("zorp", "pasta")))
        self.assertIsNotNone(act)
        assert act is not None
        self.assertEqual(act.content[0].predicate.semantic_class, "unknown")
        self.assertTrue(
            any(role.status == "unresolved" for role in act.content[0].roles),
            "unknown-word graph atomization should preserve unresolved roles",
        )


if __name__ == "__main__":
    unittest.main()
