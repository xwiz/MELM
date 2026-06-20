"""Tests for the Layer-0 input normalization pipeline.

Covers the normalization_expansions.v1 contract, EnglishAdapter.correct(), and
its integration into router parsing (slang/abbreviation/typo inputs route like
their clean equivalents). See docs/human-friendly-NLG-pipeline.md.
"""

import unittest

from melm.contracts.validation import (
    ContractValidationError,
    load_normalization_expansions,
    validate_normalization_expansions,
)
from melm.appliance.language_adapters.english import EnglishAdapter
from melm.appliance.local_assistant_router import (
    OnDeviceAssistantRouter,
    LocalAssistantProfile,
)


class NormalizationExpansionsContractTests(unittest.TestCase):
    def test_contract_loads(self):
        data = load_normalization_expansions()
        self.assertIn("entries", data)
        self.assertGreater(len(data["entries"]), 0)

    def test_gimme_maps_to_give_me(self):
        entries = {e["raw"]: e for e in load_normalization_expansions()["entries"]}
        self.assertEqual(entries["gimme"]["standard"], "give me")

    def test_no_real_word_collisions(self):
        # Guard against ambiguous raws that collide with valid words.
        raws = {e["raw"] for e in load_normalization_expansions()["entries"]}
        for ambiguous in ("cause", "cos", "r", "y", "bc"):
            self.assertNotIn(ambiguous, raws)

    def test_invalid_entry_missing_raw_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            validate_normalization_expansions(
                {"schema_id": "melm.normalization_expansions.v1",
                 "entries": [{"standard": "hello", "category": "typo"}]}
            )

    def test_invalid_entry_missing_standard_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            validate_normalization_expansions(
                {"schema_id": "melm.normalization_expansions.v1",
                 "entries": [{"raw": "foo", "category": "typo"}]}
            )

    def test_wrong_schema_id_rejected(self):
        with self.assertRaises(ContractValidationError):
            validate_normalization_expansions({"schema_id": "wrong", "entries": []})


class EnglishAdapterCorrectTests(unittest.TestCase):
    def setUp(self):
        self.adapter = EnglishAdapter()

    def test_expands_gimme(self):
        self.assertEqual(self.adapter.correct("gimme pasta"), "give me pasta")

    def test_expands_u(self):
        self.assertEqual(self.adapter.correct("u there"), "you there")

    def test_expands_pls(self):
        self.assertEqual(self.adapter.correct("pls help"), "please help")

    def test_expands_gonna(self):
        self.assertEqual(self.adapter.correct("gonna eat"), "going to eat")

    def test_expands_wanna(self):
        self.assertEqual(self.adapter.correct("wanna sleep"), "want to sleep")

    def test_expands_idk_multiword(self):
        self.assertEqual(self.adapter.correct("idk really"), "i do not know really")

    def test_fixes_teh_typo(self):
        self.assertEqual(self.adapter.correct("teh weather"), "the weather")

    def test_clean_text_unchanged(self):
        clean = "what is the weather today"
        self.assertEqual(self.adapter.correct(clean), clean)

    def test_preserves_trailing_punctuation(self):
        self.assertEqual(self.adapter.correct("pls!"), "please!")

    def test_proper_noun_untouched(self):
        # Names not in the contract must pass through unchanged (no mangling).
        self.assertEqual(self.adapter.correct("call Nneka now"), "call Nneka now")

    def test_no_op_does_not_crash_on_empty(self):
        self.assertEqual(self.adapter.correct(""), "")


class NormalizedRoutingTests(unittest.TestCase):
    """A slang/typo input must route to the same intent as its clean form."""

    def _intent(self, utterance):
        router = OnDeviceAssistantRouter(profile=LocalAssistantProfile())
        return router.handle(utterance).intent

    def test_slang_matches_clean_equivalent(self):
        for slang, clean in [
            ("gimme a story", "give me a story"),
            ("pls tell me a story", "please tell me a story"),
            ("u there", "you there"),
            ("teh weather today", "the weather today"),
        ]:
            with self.subTest(slang=slang):
                self.assertEqual(self._intent(slang), self._intent(clean))

    def test_clean_canonical_still_routes(self):
        # Regression anchor: correction must not break known clean routing.
        self.assertEqual(self._intent("tell me a story"), "story")

    def test_raw_utterance_preserved_in_decision(self):
        router = OnDeviceAssistantRouter(profile=LocalAssistantProfile())
        decision = router.handle("gimme a story")
        self.assertEqual(decision.utterance, "gimme a story")


class NerMaskTests(unittest.TestCase):
    """Deterministic proper-noun / number protection (Tier 1.5b)."""

    def setUp(self):
        from melm.appliance.normalization.ner_mask import protected_indices
        self.protected = protected_indices

    def test_capitalized_non_initial_is_protected(self):
        self.assertEqual(self.protected(("call", "Nneka", "now")), {1})

    def test_sentence_initial_capital_not_protected(self):
        self.assertEqual(self.protected(("Tell", "me", "a", "story")), set())

    def test_number_token_protected(self):
        self.assertEqual(self.protected(("set", "alarm", "7am")), {2})

    def test_all_caps_acronym_protected(self):
        self.assertEqual(self.protected(("call", "NASA")), {1})

    def test_known_name_protected_case_insensitive(self):
        self.assertEqual(
            self.protected(("ring", "onitsha"), known_names=frozenset({"onitsha"})),
            {1},
        )


class SymSpellCorrectorTests(unittest.TestCase):
    """Tier 1 lexicon-backed SymSpell typo correction (needs symspellpy)."""

    def setUp(self):
        from melm.appliance.normalization.symspell import available
        if not available():
            self.skipTest("symspellpy not installed (optional dependency)")
        self.adapter = EnglishAdapter()

    def test_fixes_oov_typo_not_in_contract(self):
        # Unambiguous typo (not in the Tier-0 contract) -> single valid word.
        # NB: ambiguous near-typos (e.g. "wheather" -> whether/weather) are
        # left to the context tier (T3); SymSpell alone ranks by frequency.
        self.assertEqual(self.adapter.correct("messsage me"), "message me")

    def test_fixes_remmind(self):
        self.assertEqual(self.adapter.correct("remmind me"), "remind me")

    def test_capitalized_proper_noun_protected(self):
        self.assertEqual(self.adapter.correct("call Nneka now"), "call Nneka now")

    def test_number_preserved(self):
        self.assertIn("7am", self.adapter.correct("set an alarm 7am"))

    def test_clean_sentence_unchanged(self):
        clean = "what is the weather today"
        self.assertEqual(self.adapter.correct(clean), clean)

    def test_short_real_words_unchanged(self):
        self.assertEqual(self.adapter.correct("go to the cat"), "go to the cat")


class AgreementTier15aTests(unittest.TestCase):
    """Deterministic subject-verb agreement / tense fix (Tier 1.5a).

    No ML: rewrites a 1st/2nd-person or plural subject pronoun followed by a
    3rd-person-singular verb form to the agreeing form. See
    docs/human-friendly-NLG-pipeline.md (§4 T1.5a).
    """

    def setUp(self):
        from melm.appliance.normalization.agreement import correct_agreement
        self.fix = correct_agreement
        self.adapter = EnglishAdapter()

    # --- positives (must be corrected) ---------------------------------
    def test_i_goes_to_go(self):
        self.assertEqual(self.adapter.correct("i goes to school"), "i go to school")

    def test_they_was_to_were(self):
        self.assertEqual(self.adapter.correct("they was here"), "they were here")

    def test_we_has_to_have(self):
        self.assertEqual(self.adapter.correct("we has it"), "we have it")

    def test_function_direct_i_does(self):
        self.assertEqual(self.fix("i does the work"), "i do the work")

    def test_intervening_adverb(self):
        self.assertEqual(self.fix("they always was late"), "they always were late")

    def test_regular_s_verb_known(self):
        # "want" is in MELM's verb inventory -> regular "-s" de-inflection.
        self.assertEqual(self.fix("you wants tea"), "you want tea")

    def test_preserves_trailing_punctuation(self):
        self.assertEqual(self.fix("we has it!"), "we have it!")

    # --- negatives (must pass through unchanged) -----------------------
    def test_she_goes_unchanged(self):
        self.assertEqual(self.adapter.correct("she goes home"), "she goes home")

    def test_you_go_unchanged(self):
        self.assertEqual(self.adapter.correct("you go now"), "you go now")

    def test_they_were_unchanged(self):
        self.assertEqual(self.adapter.correct("they were ready"), "they were ready")

    def test_he_runs_unchanged(self):
        self.assertEqual(self.adapter.correct("he runs fast"), "he runs fast")

    def test_third_singular_noun_subject_unchanged(self):
        self.assertEqual(self.fix("the dog runs fast"), "the dog runs fast")

    def test_protected_name_subject_unchanged(self):
        # A capitalized non-initial token is protected; not a pronoun anyway.
        self.assertEqual(self.fix("call Nneka now"), "call Nneka now")

    def test_empty_and_single_token(self):
        self.assertEqual(self.fix(""), "")
        self.assertEqual(self.fix("goes"), "goes")


class CorrectNeverRaisesTests(unittest.TestCase):
    def test_correct_returns_str_and_never_raises(self):
        adapter = EnglishAdapter()
        for text in ("", "gimme", "wheather", "CALL NNEKA NOW", "12345", "a"):
            self.assertIsInstance(adapter.correct(text), str)


if __name__ == "__main__":
    unittest.main()
