"""Tests for UOL lexicon integration — acquired verbs lemmatize and get semantic classes."""

import unittest

from melm.appliance.functional_grammar import (
    _lemma,
    _semantic_class,
    _VERBS,
    set_uol_lexicon,
)


class UolLexiconLemmaTests(unittest.TestCase):
    """_lemma() resolves inflected forms for lexicon-only verbs."""

    def setUp(self) -> None:
        # Inject synthetic verbs into the UOL lexicon
        self._test_lexicon = {
            "dance": frozenset(["movement_action"]),
            "climb": frozenset(["movement_action"]),
            "bake": frozenset(["creation_action"]),
        }
        set_uol_lexicon(self._test_lexicon)

    def test_ing_resolves_to_stem(self) -> None:
        self.assertEqual(_lemma("dancing"), "dance")

    def test_ed_resolves_to_stem(self) -> None:
        self.assertEqual(_lemma("danced"), "dance")

    def test_s_resolves_to_stem(self) -> None:
        self.assertEqual(_lemma("dances"), "dance")

    def test_double_consonant_ing(self) -> None:
        self.assertEqual(_lemma("climbing"), "climb")
        self.assertEqual(_lemma("climbed"), "climb")

    def test_e_verb_ing(self) -> None:
        self.assertEqual(_lemma("baking"), "bake")

    def test_e_verb_ed(self) -> None:
        self.assertEqual(_lemma("baked"), "bake")

    def test_base_form_unchanged(self) -> None:
        self.assertEqual(_lemma("dance"), "dance")


class UolLexiconSemanticClassTests(unittest.TestCase):
    """_semantic_class() returns lexicon class for verbs not in _VERBS."""

    def setUp(self) -> None:
        self._test_lexicon = {
            "dance": frozenset(["movement_action"]),
            "bake": frozenset(["creation_action"]),
        }
        set_uol_lexicon(self._test_lexicon)

    def test_lexicon_verb_gets_class(self) -> None:
        self.assertEqual(_semantic_class("dance"), "movement_action")

    def test_lexicon_verb_gets_first_class_when_multiple(self) -> None:
        lex = {"jump": frozenset(["movement_action", "sports_action"])}
        set_uol_lexicon(lex)
        self.assertIn(_semantic_class("jump"), {"movement_action", "sports_action"})

    def test_verb_not_in_lexicon_or_verbs_returns_empty(self) -> None:
        set_uol_lexicon({})
        self.assertEqual(_semantic_class("xyzzy"), "")

    def test_empty_action_returns_empty(self) -> None:
        self.assertEqual(_semantic_class(""), "")


class UolLexiconVerbsDictPriorityTests(unittest.TestCase):
    """_VERBS dict takes priority over lexicon for known verbs."""

    def setUp(self) -> None:
        # Lexicon has a different class for "walk", but _VERBS should win
        self._test_lexicon = {
            "walk": frozenset(["movement_action"]),
        }
        set_uol_lexicon(self._test_lexicon)

    def test_verbs_dict_takes_priority(self) -> None:
        # "walk" is in _VERBS with class "verb.move"
        self.assertIn("walk", _VERBS)
        self.assertEqual(_semantic_class("walk"), "verb.move")

    def test_verbs_dict_lemmatization_still_works(self) -> None:
        self.assertEqual(_lemma("walked"), "walk")
        self.assertEqual(_lemma("walking"), "walk")
        self.assertEqual(_lemma("walks"), "walk")


class UolLexiconEmptyFallbackTests(unittest.TestCase):
    """When no lexicon is set, existing _VERBS behavior is preserved."""

    def setUp(self) -> None:
        set_uol_lexicon({})

    def test_verbs_dict_verbs_still_return_class(self) -> None:
        for verb in ("eat", "play", "tell", "walk"):
            expected = _VERBS[verb][1]
            self.assertEqual(_semantic_class(verb), expected)

    def test_verbs_dict_lemmatization_still_works(self) -> None:
        self.assertEqual(_lemma("eats"), "eat")
        self.assertEqual(_lemma("playing"), "play")
        self.assertEqual(_lemma("told"), "tell")

    def test_unknown_verb_returns_empty(self) -> None:
        self.assertEqual(_semantic_class("xyzzy"), "")
        self.assertEqual(_lemma("xyzzy"), "xyzzy")


if __name__ == "__main__":
    unittest.main()
