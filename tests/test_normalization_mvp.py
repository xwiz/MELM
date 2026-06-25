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

    def test_invalid_entry_rejected(self):
        for missing_field, entries in [
            ("raw", [{"standard": "hello", "category": "typo"}]),
            ("standard", [{"raw": "foo", "category": "typo"}]),
        ]:
            with self.subTest(missing_field=missing_field):
                with self.assertRaises(ContractValidationError):
                    validate_normalization_expansions(
                        {"schema_id": "melm.normalization_expansions.v1",
                         "entries": entries}
                    )

    def test_wrong_schema_id_rejected(self):
        with self.assertRaises(ContractValidationError):
            validate_normalization_expansions({"schema_id": "wrong", "entries": []})


class EnglishAdapterCorrectTests(unittest.TestCase):
    def setUp(self):
        self.adapter = EnglishAdapter()

    def test_correct_expansions(self):
        for inp, expected in [
            ("gimme pasta", "give me pasta"),
            ("u there", "you there"),
            ("pls help", "please help"),
            ("gonna eat", "going to eat"),
            ("wanna sleep", "want to sleep"),
            ("idk really", "i do not know really"),
            ("teh weather", "the weather"),
            ("what is the weather today", "what is the weather today"),
            ("pls!", "please!"),
            ("call Nneka now", "call Nneka now"),
            ("Jollof is the best food", "Jollof is the best food"),
            ("lmao that was great", "lmao that was great"),
            ("", ""),
        ]:
            with self.subTest(input=inp):
                self.assertEqual(self.adapter.correct(inp), expected)

    def test_acquired_lexicon_word_untouched(self):
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            inject_lexicon_entry,
            replace_in_memory_lexicon,
        )

        saved = dict(_IN_MEMORY_LEXICON)
        try:
            inject_lexicon_entry("zindle", "physical_object.instrument")
            self.assertEqual(self.adapter.correct("play zindle"), "play zindle")
        finally:
            replace_in_memory_lexicon(saved)


class ContactsNerTests(unittest.TestCase):
    """Profile contacts feed the NER mask — contact names never mangled."""

    def test_contact_name_protected_by_correct(self):
        from melm.appliance.language_adapters.english import EnglishAdapter

        adapter = EnglishAdapter()
        # "leo" is a common short name; without known_names SymSpell might mangle it.
        result = adapter.correct("call leo now", known_names=frozenset({"leo"}))
        self.assertIn("leo", result)

    def test_contact_name_protected_via_router_profile(self):
        profile = LocalAssistantProfile()
        # Default profile has "leo" in contacts; it must survive NER.
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router.handle("call leo")
        # Utterance preserved, contact name not mangled.
        self.assertIn("leo", decision.utterance.lower())

    def test_correct_accepts_known_names_kwarg(self):
        from melm.appliance.language_adapters.english import EnglishAdapter

        adapter = EnglishAdapter()
        # Verify the API accepts the kwarg without TypeError.
        result = adapter.correct("gimme a story", known_names=frozenset({"zork"}))
        self.assertIsInstance(result, str)


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

    def test_meal_skill_uses_adapter_correction_for_scope(self):
        from melm.appliance.assistant_skill_meal import _meal_scope

        self.assertEqual(_meal_scope("brekky ideas please"), "breakfast")


class NerMaskTests(unittest.TestCase):
    """Deterministic proper-noun / number protection (Tier 1.5b)."""

    def setUp(self):
        from melm.appliance.normalization.ner_mask import (
            protected_indices,
            syntactic_entity_indices,
        )
        self.protected = protected_indices
        self.syntactic_protected = syntactic_entity_indices

    def test_protected_indices(self):
        for tokens, kwargs, expected in [
            (("call", "Nneka", "now"), {}, {1}),
            (("Tell", "me", "a", "story"), {}, set()),
            (("set", "alarm", "7am"), {}, {2}),
            (("call", "NASA"), {}, {1}),
            (("ring", "onitsha"), {"known_names": frozenset({"onitsha"})}, {1}),
        ]:
            with self.subTest(tokens=" ".join(tokens)):
                self.assertEqual(self.protected(tokens, **kwargs), expected)

    def test_syntactic_entity_indices(self):
        for tokens, expected in [
            (("Jollof", "is", "the", "best", "food"), {0}),
            (("explain", "quasar", "algebra", "to", "my", "zorbulator"), {5}),
        ]:
            with self.subTest(tokens=" ".join(tokens)):
                self.assertEqual(self.syntactic_protected(tokens), expected)


class SymSpellCorrectorTests(unittest.TestCase):
    """Tier 1 lexicon-backed SymSpell typo correction (needs symspellpy)."""

    def setUp(self):
        from melm.appliance.normalization.symspell import available
        if not available():
            self.skipTest("symspellpy not installed (optional dependency)")
        self.adapter = EnglishAdapter()

    def test_correct_cases(self):
        for inp, expected in [
            ("messsage me", "message me"),
            ("remmind me", "remind me"),
            ("call Nneka now", "call Nneka now"),
            ("Jollof is the best food", "Jollof is the best food"),
            ("lmao that was great", "lmao that was great"),
            ("explain quasar algebra to my zorbulator", "explain quasar algebra to my zorbulator"),
            ("Messsage me", "message me"),
            ("what is the weather today", "what is the weather today"),
            ("go to the cat", "go to the cat"),
        ]:
            with self.subTest(input=inp):
                self.assertEqual(self.adapter.correct(inp), expected)

    def test_number_preserved(self):
        self.assertIn("7am", self.adapter.correct("set an alarm 7am"))


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
    def test_positive_agreement(self):
        for fn, inp, expected in [
            (self.adapter.correct, "i goes to school", "i go to school"),
            (self.adapter.correct, "they was here", "they were here"),
            (self.adapter.correct, "we has it", "we have it"),
            (self.fix, "i does the work", "i do the work"),
            (self.fix, "they always was late", "they always were late"),
            (self.fix, "you wants tea", "you want tea"),
            (self.fix, "we has it!", "we have it!"),
        ]:
            with self.subTest(input=inp):
                self.assertEqual(fn(inp), expected)

    # --- negatives (must pass through unchanged) -----------------------
    def test_negative_agreement_unchanged(self):
        for fn, inp, expected in [
            (self.adapter.correct, "she goes home", "she goes home"),
            (self.adapter.correct, "you go now", "you go now"),
            (self.adapter.correct, "they were ready", "they were ready"),
            (self.adapter.correct, "he runs fast", "he runs fast"),
            (self.fix, "the dog runs fast", "the dog runs fast"),
            (self.fix, "call Nneka now", "call Nneka now"),
            (self.fix, "", ""),
            (self.fix, "goes", "goes"),
        ]:
            with self.subTest(input=inp):
                self.assertEqual(fn(inp), expected)


class CorrectNeverRaisesTests(unittest.TestCase):
    def test_correct_returns_str_and_never_raises(self):
        adapter = EnglishAdapter()
        for text in ("", "gimme", "wheather", "CALL NNEKA NOW", "12345", "a"):
            self.assertIsInstance(adapter.correct(text), str)


if __name__ == "__main__":
    unittest.main()
