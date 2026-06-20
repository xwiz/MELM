"""Tests for Igbo normalisation, tokenisation, and UOL bridging.

These tests exercise the Igbo compatibility module without requiring the full
MELM kernel. Igbo surface text should normalize to Igbo lemmas and feed the
contract-backed parser with ``language="ig"``. It must not translate the token
stream into English first.
"""

from __future__ import annotations

from melm.appliance.language_adapters.igbo import (
    normalise_igbo_for_uol,
    strip_igbo_diacritics,
    tokenize_igbo,
    translate_igbo_tokens,
)


class TestStripDiacritics:
    def test_dotted_vowels(self) -> None:
        assert strip_igbo_diacritics("g\u1ecbn\u1ecb") == "gini"
        assert strip_igbo_diacritics("ch\u1ecdo") == "choo"
        assert strip_igbo_diacritics("b\u1ecba") == "bia"
        assert strip_igbo_diacritics("\u1e45\u1ee5") == "nu"

    def test_tone_marks(self) -> None:
        assert strip_igbo_diacritics("ndeewo") == "ndeewo"
        assert strip_igbo_diacritics("nn\u1ecd\u1ecd") == "nnoo"
        assert strip_igbo_diacritics("\u1ee5t\u1ee5t\u1ee5") == "ututu"

    def test_no_diacritics_unchanged(self) -> None:
        assert strip_igbo_diacritics("m") == "m"
        assert strip_igbo_diacritics("lagos") == "lagos"


class TestTokenizeIgbo:
    def test_simple_sentence(self) -> None:
        assert tokenize_igbo("Gini m ga-eri?") == ("gini", "m", "ga", "eri")

    def test_hyphenated_future(self) -> None:
        assert tokenize_igbo("Anyi ga-aga") == ("anyi", "ga", "aga")

    def test_punctuation_stripped(self) -> None:
        assert tokenize_igbo("Ndeewo!") == ("ndeewo",)
        assert tokenize_igbo("Gini m ga-eri.") == ("gini", "m", "ga", "eri")

    def test_empty(self) -> None:
        assert tokenize_igbo("") == ()
        assert tokenize_igbo("   ") == ()


class TestTranslateTokens:
    def test_keeps_question_tokens_as_igbo_lemmas(self) -> None:
        tokens = ("gini", "m", "ga", "eri")
        assert translate_igbo_tokens(tokens) == ("gini", "m", "ga", "eri")

    def test_keeps_greeting_as_igbo_lemma(self) -> None:
        tokens = ("ndeewo",)
        assert translate_igbo_tokens(tokens) == ("ndeewo",)

    def test_unknown_left_as_is(self) -> None:
        tokens = ("xyz", "m")
        assert translate_igbo_tokens(tokens) == ("xyz", "m")


class TestNormaliseIgboForUol:
    def test_what_will_i_eat(self) -> None:
        assert normalise_igbo_for_uol("Gini m ga-eri?") == ("gini", "m", "ga", "eri")

    def test_who_are_you(self) -> None:
        assert normalise_igbo_for_uol("Onye bu gi?") == ("onye", "bu", "gi")

    def test_i_want_water(self) -> None:
        assert normalise_igbo_for_uol("M choo mmiri.") == ("m", "choo", "mmiri")

    def test_hello(self) -> None:
        assert normalise_igbo_for_uol("Ndeewo") == ("ndeewo",)


class TestIgboUolIntegration:
    """End-to-end: Igbo text -> normalise -> parse with functional_grammar."""

    def test_parse_what_will_i_eat(self) -> None:
        from melm.appliance.functional_grammar import parse_functional_relations

        tokens = normalise_igbo_for_uol("Gini m ga-eri?")
        parse = parse_functional_relations(tokens, question_mark=True, language="ig")
        assert parse is not None
        assert parse.speech_act == "wh_question"
        assert parse.subject == "user"
        assert parse.action == "eat"

    def test_parse_i_want_food(self) -> None:
        from melm.appliance.functional_grammar import parse_functional_relations

        tokens = normalise_igbo_for_uol("M choo nri")
        parse = parse_functional_relations(tokens, question_mark=False, language="ig")
        assert parse is not None
        assert parse.speech_act == "statement"
        assert parse.subject == "user"
        assert parse.action == "want"
        assert parse.object == "nri"

    def test_parse_greeting(self) -> None:
        from melm.appliance.functional_grammar import parse_functional_relations

        tokens = normalise_igbo_for_uol("Ndeewo")
        parse = parse_functional_relations(tokens, question_mark=False, language="ig")
        assert parse is not None
        assert parse.speech_act == "greeting"
        assert parse.action == "greet"
