"""Tests for the LlamaCppBackend pluggable decoder.

These tests verify registration, dispatch integration, and graceful fallback
when llama-cpp-python is not installed or no model path is configured.
They do **not** require a ``.gguf`` file at runtime.
"""

from __future__ import annotations

import pytest

from melm.appliance.assistant_decoder import ConstrainedDecoder, DecodingGrammar
from melm.appliance.assistant_decoder_llama_cpp import LlamaCppBackend
from melm.appliance.assistant_authority import AnswerPlan


@pytest.fixture
def empty_plan() -> AnswerPlan:
    return AnswerPlan(
        plan_id="test",
        route="open_domain",
        mode="neutral",
        requires=(),
        forbids=(),
        evidence_packet_id="p1",
    )


class TestLlamaCppBackendRegistration:
    def test_backend_has_name(self) -> None:
        backend = LlamaCppBackend()
        assert backend.name == "llamacpp"

    def test_registered_with_decoder(self) -> None:
        decoder = ConstrainedDecoder(preferred="template")
        assert "llamacpp" in decoder.available

    def test_preferred_can_be_set(self) -> None:
        decoder = ConstrainedDecoder(preferred="template")
        decoder.preferred("llamacpp")
        assert decoder.preferred() == "llamacpp"


class TestLlamaCppBackendGracefulFallback:
    def test_decode_returns_empty_when_no_model_path(self, empty_plan: AnswerPlan) -> None:
        backend = LlamaCppBackend(model_path="")
        grammar = DecodingGrammar(template_hint="hello", max_tokens=8, mood="neutral")
        assert backend.decode(empty_plan, grammar) == ""

    def test_decode_returns_empty_when_package_missing(self, empty_plan: AnswerPlan) -> None:
        backend = LlamaCppBackend(model_path="/fake/model.gguf")
        grammar = DecodingGrammar(template_hint="hello", max_tokens=8, mood="neutral")
        # The _ensure_loaded call will fail because llama_cpp is not installed
        # or the model path is invalid; decode should return ""
        assert backend.decode(empty_plan, grammar) == ""

    def test_decoder_falls_through_to_template(self, empty_plan: AnswerPlan) -> None:
        decoder = ConstrainedDecoder(preferred="llamacpp")
        grammar = DecodingGrammar(
            template_hint="template fallback",
            max_tokens=16,
            mood="neutral",
        )
        result = decoder.dispatch(empty_plan, grammar)
        assert result is not None
        assert result.answer == "template fallback"
        assert result.decoder == "template"

    def test_llamacpp_not_preferred_by_default(self) -> None:
        decoder = ConstrainedDecoder()
        assert decoder.preferred() == "template"


class TestLlamaCppBackendPromptBuilder:
    def test_build_prompt_uses_template_hint(self) -> None:
        backend = LlamaCppBackend()
        plan = AnswerPlan(
            plan_id="weather_001",
            route="local_answer",
            mode="factual",
            requires=(),
            forbids=(),
            evidence_packet_id="p1",
        )
        grammar = DecodingGrammar(
            template_hint="You are a weather assistant. What is the forecast?",
            max_tokens=32,
            mood="factual",
        )
        messages = backend._build_messages(plan, grammar)
        assert messages[0]["role"] == "system"
        assert "You are a weather assistant." in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_temperature_mapping(self) -> None:
        backend = LlamaCppBackend()
        assert backend._temperature(DecodingGrammar(mood="factual")) == pytest.approx(0.30)
        assert backend._temperature(DecodingGrammar(mood="narrative")) == pytest.approx(0.70)
        assert backend._temperature(DecodingGrammar(mood="neutral")) == pytest.approx(0.50)
        assert backend._temperature(DecodingGrammar(mood="refusal")) == pytest.approx(0.20)
        assert backend._temperature(DecodingGrammar(mood="unknown")) == pytest.approx(0.50)
