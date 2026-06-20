"""Regression tests for language-agnostic UOL parsing via contracts.

These tests verify that the functional_grammar parser can handle
non-English utterances (Igbo) by querying function_words.v1.json and
predicate_inventory.v1.json instead of relying on English hardcoded constants.
"""

import ast
from pathlib import Path
import unittest

from melm.appliance import functional_grammar
from melm.appliance.functional_grammar import parse_functional_relations
from melm.contracts import load_function_words, load_predicate_inventory


class UolLanguageAgnosticTests(unittest.TestCase):
    def test_contracts_contain_igbo_entries(self) -> None:
        fw = load_function_words()
        pi = load_predicate_inventory()
        igbo_fw = [e for e in fw["entries"] if e.get("language") == "ig"]
        igbo_pred = [p for p in pi["predicates"] if p.get("language") == "ig"]
        self.assertGreater(len(igbo_fw), 10, "function_words should have Igbo entries")
        self.assertGreater(len(igbo_pred), 5, "predicate_inventory should have Igbo predicates")

    def test_igbo_what_will_i_eat(self) -> None:
        """Gịnị m ga-eri? → wh_question with user subject."""
        tokens = ("gini", "m", "ga", "eri")
        result = parse_functional_relations(tokens, question_mark=True, language="ig")
        self.assertIsNotNone(result)
        self.assertEqual(result.speech_act, "wh_question")
        self.assertEqual(result.subject, "user")
        self.assertEqual(result.action, "eat")
        self.assertEqual(result.target, "assistant")

    def test_igbo_who_are_you(self) -> None:
        """Onye ị bụ? → wh_question about assistant identity."""
        tokens = ("onye", "i", "bu")
        result = parse_functional_relations(tokens, question_mark=True, language="ig")
        self.assertIsNotNone(result)
        self.assertEqual(result.speech_act, "wh_question")
        # subject should be the wh-word referent, not a pronoun

    def test_igbo_hello(self) -> None:
        """Ndeewo → greeting."""
        tokens = ("ndeewo",)
        result = parse_functional_relations(tokens, language="ig")
        self.assertIsNotNone(result)
        self.assertEqual(result.speech_act, "greeting")
        self.assertEqual(result.subject, "user")
        self.assertEqual(result.action, "greet")

    def test_english_still_works_with_default_language(self) -> None:
        """English utterances must still parse correctly with default language="en"."""
        tokens = ("what", "will", "i", "eat")
        result = parse_functional_relations(tokens, question_mark=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.speech_act, "wh_question")
        self.assertEqual(result.subject, "user")
        self.assertEqual(result.action, "eat")

    def test_igbo_tokens_not_translated_to_english(self) -> None:
        """Igbo lemmas must remain Igbo in the parse output, not be English."""
        tokens = ("gini", "m", "ga", "eri")
        result = parse_functional_relations(tokens, question_mark=True, language="ig")
        self.assertIsNotNone(result)
        lemmas = [r["lemma"] for r in result.token_roles]
        self.assertIn("gini", lemmas)
        self.assertIn("m", lemmas)
        self.assertIn("ga", lemmas)
        self.assertIn("eri", lemmas)
        # None of the lemmas should be their English equivalents
        self.assertNotIn("what", lemmas)
        self.assertNotIn("i", lemmas)
        self.assertNotIn("will", lemmas)
        self.assertNotIn("eat", lemmas)


class FunctionalGrammarContractProjectionTests(unittest.TestCase):
    def test_legacy_verb_export_is_projected_from_predicate_inventory(self) -> None:
        payload = load_predicate_inventory()
        expected = {
            entry["lemma"]: (entry["predicate_id"], entry["semantic_class"])
            for entry in payload["predicates"]
            if entry.get("language", "en") == "en"
        }
        self.assertEqual(functional_grammar._VERBS, expected)

    def test_legacy_closed_class_exports_are_projected_from_function_words(self) -> None:
        payload = load_function_words()
        by_role: dict[str, set[str]] = {}
        for entry in payload["entries"]:
            if entry.get("language", "en") != "en":
                continue
            by_role.setdefault(entry["role"], set()).add(entry["lemma"])

        self.assertEqual(functional_grammar._GREETINGS, by_role["greeting"])
        self.assertEqual(functional_grammar._WH_WORDS, by_role["wh_word"])
        self.assertEqual(functional_grammar._MODALS, by_role["modal"])
        self.assertEqual(functional_grammar._AUXILIARIES, by_role["auxiliary"])
        self.assertEqual(functional_grammar._NEGATIONS, by_role["negation"])
        self.assertEqual(functional_grammar._DETERMINERS, by_role["determiner"])
        self.assertEqual(functional_grammar._PREPOSITIONS, by_role["preposition"])
        self.assertEqual(functional_grammar._CONJUNCTIONS, by_role["conjunction"])
        self.assertEqual(functional_grammar._FREQUENCY, by_role["frequency"])
        self.assertEqual(functional_grammar._EQUIVALENCE, by_role["equivalence"])
        self.assertEqual(functional_grammar._POLITENESS, by_role["politeness"])
        self.assertEqual(functional_grammar._DISCOURSE_PARTICLES, by_role["discourse_particle"])

    def test_legacy_exports_are_not_reintroduced_as_literal_vocab_tables(self) -> None:
        source = Path(functional_grammar.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        guarded_names = {
            "_GREETINGS",
            "_WH_WORDS",
            "_MODALS",
            "_AUXILIARIES",
            "_NEGATIONS",
            "_DETERMINERS",
            "_PREPOSITIONS",
            "_CONJUNCTIONS",
            "_FREQUENCY",
            "_EQUIVALENCE",
            "_POLITENESS",
            "_DISCOURSE_PARTICLES",
            "_PRONOUNS",
            "_VERBS",
            "_KNOWN_NOMINAL_DOMAINS",
        }
        literal_nodes = (ast.Dict, ast.Set, ast.List, ast.Tuple)
        offenders: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in guarded_names:
                    if isinstance(node.value, literal_nodes):
                        offenders.append(target.id)
        self.assertEqual(
            offenders,
            [],
            f"legacy vocabulary exports must be projections, not literals: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
