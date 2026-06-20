"""M4 constrained-decoder scaffold tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from melm.appliance import (
    AnswerPlan,
    AuthorityEvidencePacket,
    BoundedLocalSynthesizer,
    ConstrainedDecoder,
    DecoderBackend,
    DecoderResult,
    DecodingGrammar,
    LocalAssistantProfile,
    TemplateBackend,
    build_decoding_grammar,
)
from melm.appliance.assistant_synthesis import SynthesisEvidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_plan() -> AnswerPlan:
    return AnswerPlan(
        plan_id="test",
        route="local_answer",
        mode="factual",
        requires=(),
        forbids=(),
        evidence_packet_id="p1",
    )


@pytest.fixture
def stub_grammar() -> DecodingGrammar:
    return DecodingGrammar(
        allowed_entities=(),
        required_constraints=(),
        prohibited_tokens=(),
        max_tokens=64,
        template_hint="hello from template",
        mood="neutral",
    )


@pytest.fixture
def decoder() -> ConstrainedDecoder:
    return ConstrainedDecoder()


# ---------------------------------------------------------------------------
# TemplateBackend
# ---------------------------------------------------------------------------

class TestTemplateBackend:
    def test_returns_template_hint(self, empty_plan: AnswerPlan, stub_grammar: DecodingGrammar) -> None:
        backend = TemplateBackend()
        assert backend.name == "template"
        result = backend.decode(empty_plan, stub_grammar)
        assert result == "hello from template"

    def test_empty_template_hint(self, empty_plan: AnswerPlan) -> None:
        backend = TemplateBackend()
        grammar = DecodingGrammar(template_hint="", mood="neutral")
        result = backend.decode(empty_plan, grammar)
        assert result == ""


# ---------------------------------------------------------------------------
# ConstrainedDecoder — registry & dispatch
# ---------------------------------------------------------------------------

class TestConstrainedDecoderRegistry:
    def test_default_backend_registered(self, decoder: ConstrainedDecoder) -> None:
        assert "template" in decoder.available

    def test_register_custom_backend(self, decoder: ConstrainedDecoder) -> None:
        @dataclass
        class EchoBackend:
            name: str = "echo"
            def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
                return grammar.template_hint.upper()
        decoder.register(EchoBackend())
        assert "echo" in decoder.available

    def test_preferred_get_set(self, decoder: ConstrainedDecoder) -> None:
        assert decoder.preferred() == "template"
        decoder.preferred("nonexistent")
        assert decoder.preferred() == "template"
        @dataclass
        class EchoBackend:
            name: str = "echo"
            def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
                return "echo"
        decoder.register(EchoBackend())
        decoder.preferred("echo")
        assert decoder.preferred() == "echo"

    def test_preferred_none_when_empty(self) -> None:
        d = ConstrainedDecoder()
        d._backends.clear()
        assert d.preferred() is None


class TestConstrainedDecoderDispatch:
    def test_dispatches_default_template(self, decoder: ConstrainedDecoder, empty_plan: AnswerPlan, stub_grammar: DecodingGrammar) -> None:
        result = decoder.dispatch(empty_plan, stub_grammar)
        assert result is not None
        assert result.answer == "hello from template"
        assert result.decoder == "template"

    def test_preferred_backend_wins(self, decoder: ConstrainedDecoder, empty_plan: AnswerPlan) -> None:
        @dataclass
        class EchoBackend:
            name: str = "echo"
            def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
                return f"echo:{grammar.mood}"
        decoder.register(EchoBackend())
        decoder.preferred("echo")
        grammar = DecodingGrammar(template_hint="t", mood="factual", max_tokens=16)
        result = decoder.dispatch(empty_plan, grammar)
        assert result is not None
        assert result.answer == "echo:factual"
        assert result.decoder == "echo"

    def test_fallback_when_preferred_fails(self, decoder: ConstrainedDecoder, empty_plan: AnswerPlan, stub_grammar: DecodingGrammar) -> None:
        @dataclass
        class FailingBackend:
            name: str = "failing"
            def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
                raise RuntimeError("backend failure")
        decoder._backends = {"failing": FailingBackend()}
        decoder._preferred = "failing"
        result = decoder.dispatch(empty_plan, stub_grammar)
        assert result is None

    def test_fallback_skips_empty(self, decoder: ConstrainedDecoder, empty_plan: AnswerPlan) -> None:
        @dataclass
        class EmptyBackend:
            name: str = "empty"
            def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
                return ""
        decoder._backends = {"empty": EmptyBackend()}
        decoder._preferred = "empty"
        grammar = DecodingGrammar(template_hint="", max_tokens=16, mood="neutral")
        result = decoder.dispatch(empty_plan, grammar)
        assert result is None

    def test_returns_none_when_all_backends_empty(self) -> None:
        d = ConstrainedDecoder()
        d._backends.clear()
        plan = AnswerPlan("x", "local_answer", "factual", (), (), "p1")
        grammar = DecodingGrammar(template_hint="t", max_tokens=16, mood="neutral")
        assert d.dispatch(plan, grammar) is None

    def test_tokens_generated_count(self, decoder: ConstrainedDecoder, empty_plan: AnswerPlan) -> None:
        grammar = DecodingGrammar(template_hint="one two three", max_tokens=32, mood="neutral")
        result = decoder.dispatch(empty_plan, grammar)
        assert result is not None
        assert result.tokens_generated == 3


# ---------------------------------------------------------------------------
# build_decoding_grammar
# ---------------------------------------------------------------------------

class TestBuildDecodingGrammar:
    def test_builds_from_plan(self) -> None:
        plan = AnswerPlan(
            plan_id="p1",
            route="local_answer",
            mode="factual",
            requires=("weather",),
            forbids=("diagnosis",),
            evidence_packet_id="ep1",
        )
        grammar = build_decoding_grammar(plan, "today it is sunny", ("weather.today",))
        assert grammar.required_constraints == ("weather",)
        assert grammar.prohibited_tokens == ("diagnosis",)
        assert grammar.template_hint == "today it is sunny"
        assert grammar.allowed_entities == ("weather.today",)
        assert grammar.mood == "factual"

    def test_empty_allowed_entities(self) -> None:
        plan = AnswerPlan("x", "local_answer", "narrative", (), (), "p1")
        grammar = build_decoding_grammar(plan, "story time")
        assert grammar.allowed_entities == ()


# ---------------------------------------------------------------------------
# DecoderBackend protocol conformance
# ---------------------------------------------------------------------------

class TestDecoderBackendProtocol:
    def test_template_backend_conforms(self) -> None:
        backend: DecoderBackend = TemplateBackend()
        assert isinstance(backend, TemplateBackend)


# ---------------------------------------------------------------------------
# Integration: synthesis + decoder (template-only fallback)
# ---------------------------------------------------------------------------

_PROFILE = LocalAssistantProfile(
    user_name="Test",
    location="test",
    culture="",
    age="",
    facts={"pet": "dog"},
    preferences={},
    health_goals=[],
    food_inventory=(),
    media_library=(),
    contacts={},
    weekly_weather={},
    story_models={},
)


@pytest.fixture
def synth() -> BoundedLocalSynthesizer:
    return BoundedLocalSynthesizer(_PROFILE)


def _decision(**overrides: Any) -> Any:
    from melm.appliance.local_assistant_router import AssistantDecision
    kwargs = dict(
        utterance="hello",
        intent="open_domain",
        route="local_answer",
        reason="test",
        evidence_keys=("facts.pet",),
        answer="template fallback",
        slot_states={},
    )
    kwargs.update(overrides)
    return AssistantDecision(**kwargs)


def _evidence_items() -> tuple[SynthesisEvidence, ...]:
    return (
        SynthesisEvidence(
            key="facts.pet", kind="user_fact", value="dog",
            source="user_profile", license="private_local", local_only=True,
        ),
    )


class TestSynthesisWithDecoder:
    def test_synthesizer_accepts_decoder(self) -> None:
        d = ConstrainedDecoder()
        s = BoundedLocalSynthesizer(_PROFILE, decoder=d)
        assert s.decoder is d

    def test_decoder_none_by_default(self) -> None:
        s = BoundedLocalSynthesizer(_PROFILE)
        assert s.decoder is None

    def test_synthesis_with_decoder_template_fallback(self) -> None:
        d = ConstrainedDecoder()
        s = BoundedLocalSynthesizer(_PROFILE, decoder=d)
        result = s.synthesize(
            _decision(intent="personal_memory", evidence_keys=("facts.pet",)),
            boundary_crossed="local",
            membrane_allowed=True,
        )
        assert result.applied
        assert "dog" in result.answer
        assert result.authority is not None
        assert result.authority.verification.passed

    def test_decoder_forbidden_output_falls_back_to_template(self) -> None:
        """Mock decoder returns a forbidden term; verifier rejects it and fallback is used."""
        from melm.appliance.assistant_authority import (
            AnswerPlan, AuthorityEvidencePacket, verify_answer,
        )
        from melm.appliance.assistant_synthesis import SynthesisEvidence

        @dataclass
        class BadBackend:
            name: str = "bad"

            def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
                return "This diagnosis shows you need treatment immediately."

        d = ConstrainedDecoder()
        d.register(BadBackend())
        d.preferred("bad")
        s = BoundedLocalSynthesizer(_PROFILE, decoder=d)

        plan = AnswerPlan(
            plan_id="p_adv",
            route="local_answer",
            mode="factual",
            requires=(),
            forbids=("diagnosis",),
            evidence_packet_id="ep_adv",
        )
        packet = AuthorityEvidencePacket(
            packet_id="ep_adv",
            items=(),
            admitted_count=0,
            blocked_keys=(),
            boundary="local",
        )
        evidence = (
            SynthesisEvidence(
                key="facts.pet", kind="user_fact", value="dog",
                source="user_profile", license="private_local", local_only=True,
            ),
        )
        decision = _decision(intent="personal_memory", evidence_keys=("facts.pet",))
        template_answer = "Your pet is doing well."

        answer, decoder_used = s._decode_verified(plan, evidence, decision, template_answer, packet)
        # The forbidden decoder output was rejected; template fallback used
        assert answer == template_answer
        assert decoder_used == "template"
        # Verify the decoder output itself would have failed
        v = verify_answer(plan, "This diagnosis shows you need treatment immediately.", packet)
        assert not v.passed
        assert "constraint_violation" in v.failure_codes
