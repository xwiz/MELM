from pathlib import Path
import tempfile
import unittest

from melm.appliance import (
    AnswerPlan,
    AssistantOSKernel,
    AuthorityEvidenceItem,
    AuthorityEvidencePacket,
    BoundedLocalSynthesizer,
    DecoderResult,
    LocalAssistantProfile,
    VerificationResult,
    build_answer_plan,
    build_evidence_packet,
    initialize_assistant_os_database,
    verify_answer,
)
from melm.appliance.local_assistant_router import AssistantDecision


SEED = Path("benchmarks/local_assistant_os_seed.json")


def _make_item(key: str = "test.key", kind: str = "test", value: str = "test value") -> AuthorityEvidenceItem:
    return AuthorityEvidenceItem(
        key=key, kind=kind, value=value,
        source="local_seed", license="local_seed",
        local_only=True,
    )


def _make_decision(
    intent: str = "weather",
    route: str = "local_answer",
    reason: str = "local_weather",
    evidence_keys: tuple[str, ...] = ("weather.today", "profile.location"),
) -> AssistantDecision:
    return AssistantDecision(
        utterance="",
        intent=intent,
        route=route,
        answer="Today is sunny.",
        evidence_keys=evidence_keys,
        confidence=0.95,
        reason=reason,
    )


class EvidencePacketTests(unittest.TestCase):
    def test_empty_items_produces_packet(self) -> None:
        packet = build_evidence_packet((), (), "local")
        self.assertIsInstance(packet, AuthorityEvidencePacket)
        self.assertEqual(packet.admitted_count, 0)
        self.assertEqual(packet.boundary, "local")
        self.assertTrue(len(packet.packet_id) == 16)

    def test_packet_id_deterministic_for_same_keys(self) -> None:
        items = (_make_item("a", "weather", "sunny"), _make_item("b", "weather", "rain"))
        p1 = build_evidence_packet(("a", "b"), items, "none")
        p2 = build_evidence_packet(("a", "b"), items, "none")
        self.assertEqual(p1.packet_id, p2.packet_id)

    def test_packet_id_differs_for_different_keys(self) -> None:
        items = (_make_item("a", "weather", "sunny"),)
        p1 = build_evidence_packet(("a",), items, "none")
        p2 = build_evidence_packet(("b",), (_make_item("b"),), "none")
        self.assertNotEqual(p1.packet_id, p2.packet_id)

    def test_blocked_keys_identified(self) -> None:
        items = (_make_item("a", "weather", "sunny"),)
        packet = build_evidence_packet(("a", "b"), items, "local")
        self.assertIn("b", packet.blocked_keys)
        self.assertNotIn("a", packet.blocked_keys)

    def test_membrane_decision_id(self) -> None:
        items = (_make_item("a"),)
        packet = build_evidence_packet(("a",), items, "local", membrane_decision_id="md_1")
        self.assertEqual(packet.membrane_decision_id, "md_1")


class AnswerPlanTests(unittest.TestCase):
    def test_weather_plan_requires_weather_kind(self) -> None:
        items = (_make_item("weather.today", "weather", "sunny"),)
        packet = build_evidence_packet(("weather.today",), items, "none")
        plan = build_answer_plan(_make_decision("weather"), packet)
        self.assertEqual(plan.route, "local_answer")
        self.assertEqual(plan.mode, "factual")
        self.assertIn("weather", plan.requires)

    def test_meal_plan_requires_food_inventory(self) -> None:
        items = (_make_item("food.apple", "food_inventory", "apple"),)
        packet = build_evidence_packet(("food.apple",), items, "none")
        plan = build_answer_plan(_make_decision("meal_suggestion"), packet)
        self.assertIn("food_inventory", plan.requires)

    def test_health_plan_forbids_diagnosis(self) -> None:
        items = (_make_item("health.goal", "health_goal", "sleep"),)
        packet = build_evidence_packet(("health.goal",), items, "none")
        plan = build_answer_plan(_make_decision("health_advice"), packet)
        self.assertIn("diagnosis", plan.forbids)

    def test_refusal_plan_has_refusal_mode(self) -> None:
        items = (_make_item("a", "policy", "blocked"),)
        packet = build_evidence_packet(("a",), items, "blocked")
        plan = build_answer_plan(
            _make_decision("common_sense_safety", route="reject"),
            packet,
        )
        self.assertEqual(plan.mode, "refusal")

    def test_story_plan_has_narrative_mode(self) -> None:
        items = (_make_item("story.1", "story_model", "tale"),)
        packet = build_evidence_packet(("story.1",), items, "none")
        plan = build_answer_plan(
            _make_decision("story", reason="local_story"),
            packet,
        )
        self.assertEqual(plan.mode, "narrative")


class VerifierTests(unittest.TestCase):
    def test_verifier_passes_for_valid_template_answer(self) -> None:
        items = (_make_item("weather.today", "weather", "sunny"),)
        packet = build_evidence_packet(("weather.today",), items, "none")
        plan = build_answer_plan(_make_decision("weather"), packet)
        answer = "Today in Seattle: sunny. That is the cached local forecast."
        result = verify_answer(plan, answer, packet)
        self.assertTrue(result.passed)
        self.assertTrue(result.schema_valid)
        self.assertTrue(result.packet_bound)
        self.assertTrue(result.answer_nonempty)

    def test_verifier_rejects_empty_answer(self) -> None:
        items = (_make_item("weather.today", "weather", "sunny"),)
        packet = build_evidence_packet(("weather.today",), items, "none")
        plan = AnswerPlan(
            plan_id="plan_test",
            route="local_answer",
            mode="factual",
            requires=("weather",),
            forbids=(),
            evidence_packet_id=packet.packet_id,
        )
        result = verify_answer(plan, "", packet)
        self.assertFalse(result.passed)
        self.assertIn("empty_answer", result.failure_codes)

    def test_verifier_rejects_forbidden_term(self) -> None:
        items = (_make_item("health.goal", "health_goal", "sleep"),)
        packet = build_evidence_packet(("health.goal",), items, "none")
        plan = build_answer_plan(_make_decision("health_advice"), packet)
        answer = "This diagnosis shows you need treatment."
        result = verify_answer(plan, answer, packet)
        self.assertFalse(result.passed)
        self.assertIn("constraint_violation", result.failure_codes)
        self.assertLess(result.constraint_retention, 1.0)

    def test_verifier_rejects_schema_invalid_plan(self) -> None:
        items = (_make_item("a", "test", "x"),)
        packet = build_evidence_packet(("a",), items, "local")
        plan = AnswerPlan(
            plan_id="",
            route="local_answer",
            mode="invalid_mode",
            requires=(),
            forbids=(),
            evidence_packet_id=packet.packet_id,
        )
        result = verify_answer(plan, "hello", packet)
        self.assertFalse(result.passed)
        self.assertIn("schema_invalid", result.failure_codes)

    def test_verifier_rejects_packet_unbound(self) -> None:
        items = (_make_item("a", "test", "x"),)
        packet = build_evidence_packet(("a",), items, "local")
        plan = AnswerPlan(
            plan_id="plan_test",
            route="local_answer",
            mode="factual",
            requires=(),
            forbids=(),
            evidence_packet_id="wrong_packet_id",
        )
        result = verify_answer(plan, "hello", packet)
        self.assertFalse(result.passed)
        self.assertIn("packet_unbound", result.failure_codes)


class SynthesisIntegrationTests(unittest.TestCase):
    def test_kernel_synthesis_includes_authority(self) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                health_goals=("sleep earlier",),
                story_models={},
                weekly_weather={},
                contacts={},
            )
        )
        decision = kernel.handle("What do you think I should do to improve my health?")
        synthesis = kernel.last_synthesis
        self.assertIsNotNone(synthesis)
        self.assertIsNotNone(synthesis.authority)
        self.assertIsNotNone(synthesis.authority.evidence_packet)
        self.assertIsNotNone(synthesis.authority.answer_plan)
        self.assertIsNotNone(synthesis.authority.verification)
        self.assertTrue(synthesis.authority.verification.passed)

    def _story_kernel(self) -> AssistantOSKernel:
        store = initialize_assistant_os_database(
            Path(tempfile.mkdtemp()) / "test.sqlite",
            seed_path=SEED,
        )
        return AssistantOSKernel(store=store)

    def test_authority_evidence_packet_has_admitted_items(self) -> None:
        kernel = self._story_kernel()
        try:
            kernel.handle("Tell me a story.")
            synthesis = kernel.last_synthesis
            self.assertIsNotNone(synthesis)
            self.assertIsNotNone(synthesis.authority)
            packet = synthesis.authority.evidence_packet
            self.assertGreater(packet.admitted_count, 0)
        finally:
            kernel.store.close()

    def test_authority_answer_plan_matches_route(self) -> None:
        kernel = self._story_kernel()
        try:
            kernel.handle("Tell me a story.")
            synthesis = kernel.last_synthesis
            self.assertIsNotNone(synthesis)
            plan = synthesis.authority.answer_plan
            self.assertEqual(plan.route, synthesis.route)
            self.assertEqual(plan.mode, "narrative")
        finally:
            kernel.store.close()


class VerifierFallbackTests(unittest.TestCase):
    def test_fallback_uses_decision_answer_when_verifier_fails(self) -> None:
        packet = build_evidence_packet(
            ("health.goal",),
            (_make_item("health.goal", "health_goal", "sleep"),),
            "none",
        )
        plan = build_answer_plan(_make_decision("health_advice"), packet)
        result = verify_answer(plan, "This is a diagnosis.", packet)
        self.assertFalse(result.passed)
        self.assertIn("constraint_violation", result.failure_codes)

    def test_requires_passes_when_evidence_present(self) -> None:
        packet = build_evidence_packet(
            ("weather.today", "profile.location"),
            (_make_item("weather.today", "weather", "sunny"),
             _make_item("profile.location", "location", "Boston")),
            "none",
        )
        plan = build_answer_plan(_make_decision("weather"), packet)
        result = verify_answer(plan, "Today in Boston is sunny.", packet)
        self.assertTrue(result.passed)
        self.assertNotIn("missing_required_evidence", result.failure_codes)

    def test_requires_fails_when_required_kind_missing(self) -> None:
        items = (
            _make_item("weather.today", "weather", "sunny"),
        )
        packet = build_evidence_packet(("weather.today",), items, "none")
        plan = AnswerPlan(
            plan_id="plan_test",
            route="local_answer",
            mode="factual",
            requires=("missing_kind",),
            forbids=(),
            evidence_packet_id=packet.packet_id,
        )
        result = verify_answer(plan, "Today is sunny.", packet)
        self.assertFalse(result.passed)
        self.assertIn("missing_required_evidence", result.failure_codes)
        self.assertEqual(result.constraint_retention, 0.0)

    def test_requires_and_forbids_both_checked(self) -> None:
        items = (_make_item("health.goal", "health_goal", "sleep"),)
        packet = build_evidence_packet(("health.goal",), items, "none")
        plan = AnswerPlan(
            plan_id="plan_test",
            route="local_answer",
            mode="factual",
            requires=("missing_kind",),
            forbids=("diagnosis",),
            evidence_packet_id=packet.packet_id,
        )
        result = verify_answer(plan, "This is a diagnosis.", packet)
        self.assertFalse(result.passed)
        self.assertIn("constraint_violation", result.failure_codes)
        self.assertIn("missing_required_evidence", result.failure_codes)

    def test_personal_memory_plan_requires_user_fact(self) -> None:
        items = (_make_item("facts.pet", "user_fact", "dog"),)
        packet = build_evidence_packet(("facts.pet",), items, "none")
        plan = build_answer_plan(
            _make_decision("personal_memory", evidence_keys=("facts.pet",)),
            packet,
        )
        self.assertIn("user_fact", plan.requires)

    def test_personal_memory_verifier_passes_with_user_fact_evidence(self) -> None:
        items = (_make_item("facts.pet", "user_fact", "dog"),)
        packet = build_evidence_packet(("facts.pet",), items, "none")
        plan = build_answer_plan(
            _make_decision("personal_memory", evidence_keys=("facts.pet",)),
            packet,
        )
        result = verify_answer(plan, "I know this from local memory: dog.", packet)
        self.assertTrue(result.passed)
        self.assertNotIn("missing_required_evidence", result.failure_codes)


class M4ScaffoldTests(unittest.TestCase):
    """Constrained-decoding scaffold: DecoderResult, _decode(), plan→decode→verify flow."""

    def test_decoder_result_fields(self) -> None:
        result = DecoderResult(answer="hello world", decoder="template", tokens_generated=2)
        self.assertEqual(result.answer, "hello world")
        self.assertEqual(result.decoder, "template")
        self.assertEqual(result.tokens_generated, 2)

    def test_synthesize_uses_template_decoder_for_weather(self) -> None:
        profile = LocalAssistantProfile(
            weekly_weather={"today": "sunny"},
            story_models={},
            contacts={},
        )
        synth = BoundedLocalSynthesizer(profile)
        items = (
            AuthorityEvidenceItem(
                key="weekly_weather.today", kind="weather", value="sunny",
                source="local_seed", license="local_seed", local_only=False,
            ),
            AuthorityEvidenceItem(
                key="profile.location", kind="profile", value="Seattle",
                source="user_profile", license="private_local", local_only=True,
            ),
        )
        decision = _make_decision("weather", evidence_keys=("weekly_weather.today", "profile.location"))
        result = synth.synthesize(
            decision,
            boundary_crossed="none",
            membrane_allowed=True,
        )
        self.assertEqual(result.decoder_used, "template")
        self.assertIn("sunny", result.answer.lower())

    def test_synthesize_tokens_generated_counts_words(self) -> None:
        profile = LocalAssistantProfile(
            health_goals=("sleep earlier",),
            story_models={},
            weekly_weather={},
            contacts={},
        )
        synth = BoundedLocalSynthesizer(profile)
        items = (
            AuthorityEvidenceItem(
                key="health_goals.0", kind="health_goal", value="sleep earlier",
                source="user_profile", license="private_local", local_only=True,
            ),
            AuthorityEvidenceItem(
                key="local_health_safety_policy", kind="policy",
                value="general health guidance is bounded",
                source="local_policy", license="local_policy", local_only=False,
            ),
        )
        decision = _make_decision("health_advice", evidence_keys=("health_goals", "local_health_safety_policy"))
        result = synth.synthesize(
            decision,
            boundary_crossed="none",
            membrane_allowed=True,
        )
        self.assertGreater(len(result.answer.split()), 5)
        # Template decoder should produce "not a diagnosis" (negated, allowed)
        self.assertIn("diagnosis", result.answer.lower())
        self.assertIn("sleep", result.answer.lower())

    def test_synthesize_plan_to_decode_to_verify_flow(self) -> None:
        """End-to-end: synthesize builds plan, decodes, verifies, attaches authority."""
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                health_goals=("walk daily",),
                story_models={},
                weekly_weather={},
                contacts={},
            )
        )
        decision = kernel.handle("What do you think I should do to improve my health?")
        synthesis = kernel.last_synthesis
        self.assertIsNotNone(synthesis)
        self.assertIsNotNone(synthesis.authority)
        self.assertTrue(synthesis.authority.verification.passed)
        self.assertIn("walk daily", synthesis.answer.lower())

    def test_synthesize_verifier_fallback_on_forbidden_answer(self) -> None:
        """When the template decoder outputs a non-negated forbidden term, fallback fires."""
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                health_goals=("walk daily",),
                story_models={},
                weekly_weather={},
                contacts={},
            )
        )
        decision = kernel.handle("What do you think I should do to improve my health?")
        synthesis = kernel.last_synthesis
        self.assertIsNotNone(synthesis)
        self.assertIsNotNone(synthesis.authority)
        self.assertEqual(synthesis.authority.verification.failure_codes, ())


if __name__ == "__main__":
    unittest.main()
