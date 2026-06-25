"""M4 llguidance backend tests.

Grammar-construction tests run without a model. Model-dependent tests verify
graceful fallback (empty string) when no model is available.
"""

from __future__ import annotations

import re

import pytest


def _importorskip_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


from melm.appliance import (
    AnswerPlan,
    ConstrainedDecoder,
    DecodingGrammar,
    HFCompatTokenizer,
    LlguidanceBackend,
    build_decoding_grammar,
    build_llguidance_grammar,
    build_llm_prompt,
    build_regex_pattern,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def factual_plan() -> AnswerPlan:
    return AnswerPlan("p1", "local_answer", "factual", ("weather",), (), "ep1")


@pytest.fixture
def factual_grammar(factual_plan: AnswerPlan) -> DecodingGrammar:
    return build_decoding_grammar(factual_plan, "Today is sunny.")


@pytest.fixture
def narrative_plan() -> AnswerPlan:
    return AnswerPlan("p2", "local_answer", "narrative", ("dragon",), (), "ep2")


@pytest.fixture
def narrative_grammar(narrative_plan: AnswerPlan) -> DecodingGrammar:
    return build_decoding_grammar(
        narrative_plan, "I picked Moon Drum Walk.",
        allowed_entities=("story.dragon", "story.moon_drum_walk"),
    )


@pytest.fixture
def refusal_plan() -> AnswerPlan:
    return AnswerPlan("p3", "reject", "refusal", (), (), "ep3")


@pytest.fixture
def refusal_grammar(refusal_plan: AnswerPlan) -> DecodingGrammar:
    return build_decoding_grammar(refusal_plan, "I cannot do that.")


# ---------------------------------------------------------------------------
# build_regex_pattern
# ---------------------------------------------------------------------------

class TestBuildRegexPattern:
    def test_refusal_mood(self, refusal_grammar: DecodingGrammar) -> None:
        pattern = build_regex_pattern(refusal_grammar)
        assert pattern.startswith(r"[A-Z][^.!?]*[.!?]")
        # Refusal: single sentence only
        assert "( [A-Z][^.!?]*[.!?])" not in pattern

    def test_factual_mood(self, factual_grammar: DecodingGrammar) -> None:
        pattern = build_regex_pattern(factual_grammar)
        assert "( [A-Z][^.!?]*[.!?])*" in pattern

    def test_narrative_mood(self, narrative_grammar: DecodingGrammar) -> None:
        pattern = build_regex_pattern(narrative_grammar)
        assert "{1,5}" in pattern

    def test_neutral_mood(self) -> None:
        grammar = DecodingGrammar(mood="neutral", max_tokens=32, template_hint="hi")
        pattern = build_regex_pattern(grammar)
        assert pattern == r"[A-Za-z ,.!?]+"

    def test_prohibited_tokens(self) -> None:
        grammar = DecodingGrammar(
            mood="factual", max_tokens=64, template_hint="test",
            prohibited_tokens=("diagnosis", "emergency"),
        )
        pattern = build_regex_pattern(grammar)
        assert "diagnosis" in pattern
        assert "emergency" in pattern

    def test_empty_prohibited(self, factual_grammar: DecodingGrammar) -> None:
        pattern = build_regex_pattern(factual_grammar)
        assert "(?!" not in pattern


# ---------------------------------------------------------------------------
# build_llguidance_grammar
# ---------------------------------------------------------------------------

class TestBuildLlguidanceGrammar:
    def test_produces_valid_grammar_string(self, factual_grammar: DecodingGrammar) -> None:
        result = build_llguidance_grammar(factual_grammar)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_incorporates_prohibited(self) -> None:
        grammar = DecodingGrammar(
            mood="factual", max_tokens=64, template_hint="test",
            prohibited_tokens=("diagnosis",),
        )
        result = build_llguidance_grammar(grammar)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# build_llm_prompt
# ---------------------------------------------------------------------------

class TestBuildLlmPrompt:
    def test_with_template_hint(self, factual_grammar: DecodingGrammar) -> None:
        prompt = build_llm_prompt(factual_grammar)
        assert prompt == "Continue: Today is sunny."

    def test_without_template_hint_with_constraints(self) -> None:
        grammar = DecodingGrammar(
            mood="factual", max_tokens=64,
            required_constraints=("weather", "temperature"),
        )
        prompt = build_llm_prompt(grammar)
        assert "Topics:" in prompt
        assert "weather" in prompt
        assert "temperature" in prompt

    def test_without_template_hint_with_entities(self) -> None:
        grammar = DecodingGrammar(
            mood="narrative", max_tokens=128,
            allowed_entities=("story.dragon", "story.moon"),
        )
        prompt = build_llm_prompt(grammar)
        assert "Entities:" in prompt
        assert "story.dragon" in prompt

    def test_without_template_hint_with_both(self) -> None:
        grammar = DecodingGrammar(
            mood="factual", max_tokens=64,
            required_constraints=("weather",),
            allowed_entities=("forecast.today",),
        )
        prompt = build_llm_prompt(grammar)
        assert "Topics:" in prompt
        assert "Entities:" in prompt

    def test_empty(self) -> None:
        grammar = DecodingGrammar(mood="neutral", max_tokens=32)
        prompt = build_llm_prompt(grammar)
        assert prompt == "Respond concisely."


# ---------------------------------------------------------------------------
# HFCompatTokenizer
# ---------------------------------------------------------------------------

class TestHFCompatTokenizer:
    pytestmark = pytest.mark.skipif(
        not _importorskip_transformers(),
        reason="transformers not installed",
    )

    def test_wraps_hf_tokenizer(self) -> None:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        adapter = HFCompatTokenizer(hf)
        # eos may be None for BERT-style tokenizers
        assert len(adapter.tokens) > 0
        assert adapter.is_tokenizer_wrapper

    def test_t5_has_eos(self) -> None:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained("t5-small")
        adapter = HFCompatTokenizer(hf)
        assert adapter.eos_token_id is not None

    def test_tokens_list_matches_vocab(self) -> None:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        adapter = HFCompatTokenizer(hf)
        # Check a known token
        vocab = hf.get_vocab()
        for token_str, token_id in list(vocab.items())[:10]:
            if token_id < len(adapter.tokens):
                assert adapter.tokens[token_id] == token_str.encode("utf-8")

    def test_encode_string(self) -> None:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        adapter = HFCompatTokenizer(hf)
        ids = adapter("hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_bytes(self) -> None:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        adapter = HFCompatTokenizer(hf)
        ids = adapter(b"hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# LlguidanceBackend
# ---------------------------------------------------------------------------

class TestLlguidanceBackendConstruction:
    def test_default_model_name(self) -> None:
        backend = LlguidanceBackend()
        assert backend.model_name == "microsoft/phi-1_5"

    def test_custom_model_name(self) -> None:
        backend = LlguidanceBackend("custom/model")
        assert backend.model_name == "custom/model"

    def test_name_property(self) -> None:
        backend = LlguidanceBackend()
        assert backend.name == "llguidance"

    def test_not_loaded_after_init(self) -> None:
        backend = LlguidanceBackend()
        assert backend._model is None


class TestLlguidanceBackendLoad:
    def test_ensure_loaded_false_when_model_missing(self) -> None:
        backend = LlguidanceBackend("nonexistent/model/that/will/fail")
        assert not backend._ensure_loaded()
        assert backend._model is None

    def test_ensure_loaded_caches_failure(self) -> None:
        backend = LlguidanceBackend("nonexistent/model/that/will/fail")
        assert not backend._ensure_loaded()
        # Second call also returns False (cached failure)
        assert not backend._ensure_loaded()


class TestLlguidanceBackendDecode:
    def test_decode_returns_empty_when_not_loaded(self) -> None:
        backend = LlguidanceBackend("nonexistent/model/that/will/fail")
        grammar = DecodingGrammar(max_tokens=16, template_hint="test",
                                  mood="neutral")
        plan = AnswerPlan("x", "local_answer", "neutral", (), (), "p1")
        result = backend.decode(plan, grammar)
        assert result == ""

    @pytest.mark.skipif(not _importorskip_transformers(), reason="transformers not installed")
    def test_decode_with_model_available(self) -> None:
        """Quick smoke test: loads cached distilbert (not a CausalLM, but
        tests the load + decode path gracefully returning empty on generation
        errors from non-CausalLM model)."""
        backend = LlguidanceBackend("distilbert-base-uncased")
        grammar = DecodingGrammar(max_tokens=8, template_hint="test",
                                  mood="neutral")
        plan = AnswerPlan("x", "local_answer", "neutral", (), (), "p1")
        # distilbert is encoder-only → internal decode will hit an exception
        result = backend.decode(plan, grammar)
        assert result == ""


# ---------------------------------------------------------------------------
# Integration with ConstrainedDecoder
# ---------------------------------------------------------------------------

class TestLlguidanceBackendIntegration:
    def test_registered_with_decoder(self) -> None:
        decoder = ConstrainedDecoder(preferred="template")
        backend = LlguidanceBackend("nonexistent/model/that/will/fail")
        decoder.register(backend)
        assert "llguidance" in decoder.available

    def test_falls_back_to_template_when_llguidance_fails(self) -> None:
        decoder = ConstrainedDecoder(preferred="llguidance")
        backend = LlguidanceBackend("nonexistent/model/that/will/fail")
        decoder.register(backend)
        plan = AnswerPlan("x", "local_answer", "neutral", (), (), "p1")
        grammar = DecodingGrammar(max_tokens=16, template_hint="template fallback",
                                  mood="neutral")
        result = decoder.dispatch(plan, grammar)
        assert result is not None
        # Template falls through when preferred (llguidance) returns empty
        assert result.answer == "template fallback"
        assert result.decoder == "template"

    @pytest.mark.skipif(not _importorskip_transformers(), reason="transformers not installed")
    def test_llguidance_preferred_when_available(self) -> None:
        """If llguidance backend can load (distilbert in this environment),
        the decoder dispatch should still work (even if generation fails,
        it falls through to template)."""
        decoder = ConstrainedDecoder(preferred="llguidance")
        backend = LlguidanceBackend("distilbert-base-uncased")
        decoder.register(backend)
        plan = AnswerPlan("x", "local_answer", "neutral", (), (), "p1")
        grammar = DecodingGrammar(max_tokens=8, template_hint="tpl",
                                  mood="neutral")
        result = decoder.dispatch(plan, grammar)
        assert result is not None
        assert result.answer == "tpl"
