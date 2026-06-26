"""Tests for semantic attention packet + NLG renderer.

Covers: contract validation, packet topic selection, constraint separation,
response-artifact suppression, technical-token alerts, reasoning-protected
renderer, learned-fact enrichment, and production probe pass-through.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from melm.contracts import (
    load_semantic_attention_rules,
    load_nlg_atomic_renderers,
    validate_semantic_attention_rules,
    validate_nlg_atomic_renderers,
)
from melm.appliance.assistant_semantic_attention import (
    ConstraintBinding,
    CapabilityBinding,
    SemanticAttentionPacket,
    build_attention_packet,
    resolve_task_topic,
    extract_constraints,
)
from melm.appliance.assistant_nlg_renderer import (
    render_from_packet,
    score_renderer_family,
    render_template,
)
from melm.appliance.assistant_os_store import AssistantOSStore
from melm.appliance.assistant_skill_research import (
    StubResearchProvider,
    learn_topic,
)


_SUMMARIES = {
    "poem": (
        "A poem is a piece of writing that uses rhythm, imagery, and sound "
        "to express emotion or imagination."
    ),
    "rich": (
        "Getting rich usually means increasing assets through earning, "
        "saving, investing, and reducing avoidable losses."
    ),
    "div": (
        "A div is an HTML element used as a generic container; centering it "
        "is usually a CSS layout task."
    ),
    "riddle": (
        "A riddle is a puzzle stated as a question or description with a "
        "clever hidden answer."
    ),
    "fish": (
        "A fish is an aquatic animal that lives in water and usually has "
        "gills and fins."
    ),
}


def _store_with_learned_facts() -> AssistantOSStore:
    store = AssistantOSStore(":memory:")
    provider = StubResearchProvider(_SUMMARIES)
    for topic in _SUMMARIES:
        learn_topic(store, topic, provider)
    return store


class TestContractValidation(unittest.TestCase):
    """New contracts load and validate correctly."""

    def test_semantic_attention_rules_loads(self) -> None:
        rules = load_semantic_attention_rules()
        self.assertIn("response_artifact_terms", rules)
        self.assertIn("technical_token_terms", rules)
        self.assertIn("output_type_terms", rules)
        self.assertIn("reasoning_cues", rules)
        self.assertIn("stopwords", rules)

    def test_semantic_attention_rules_artifact_terms_not_empty(self) -> None:
        rules = load_semantic_attention_rules()
        self.assertGreater(len(rules["response_artifact_terms"]), 0)

    def test_semantic_attention_rules_rejects_wrong_schema_id(self) -> None:
        with self.assertRaises(ValueError):
            validate_semantic_attention_rules({"schema_id": "melm.wrong", "response_artifact_terms": ["a"], "technical_token_terms": ["b"], "output_type_terms": {"c": "d"}, "reasoning_cues": ["e"], "stopwords": ["f"]})

    def test_nlg_atomic_renderers_loads(self) -> None:
        renderers = load_nlg_atomic_renderers()
        families = renderers.get("renderer_families", {})
        self.assertIn("definition", families)
        self.assertIn("how_to", families)
        self.assertIn("creative_riddle", families)
        self.assertIn("entity_property_advice", families)
        self.assertIn("learned_unknown_skill", families)

    def test_nlg_atomic_renderers_rejects_wrong_schema_id(self) -> None:
        with self.assertRaises(ValueError):
            validate_nlg_atomic_renderers({"schema_id": "melm.wrong", "renderer_families": {"x": {"description": "d", "priority": 1, "required_conditions": {}, "forbidden_conditions": {}, "templates": ["t"]}}})

    def test_nlg_atomic_renderers_each_family_has_required_fields(self) -> None:
        renderers = load_nlg_atomic_renderers()
        for fname, family in renderers["renderer_families"].items():
            self.assertIn("description", family, fname)
            self.assertIn("priority", family, fname)
            self.assertIn("required_conditions", family, fname)
            self.assertIn("templates", family, fname)
            self.assertIsInstance(family["templates"], list, fname)
            self.assertGreater(len(family["templates"]), 0, fname)

    def test_semantic_attention_rules_artifact_terms_excludes_riddle(self) -> None:
        rules = load_semantic_attention_rules()
        self.assertNotIn("riddle", rules["response_artifact_terms"])

    def test_semantic_attention_rules_technical_terms_include_html_css(self) -> None:
        rules = load_semantic_attention_rules()
        self.assertIn("html", rules["technical_token_terms"])
        self.assertIn("css", rules["technical_token_terms"])


class TestPacketBuilder(unittest.TestCase):
    """SemanticAttentionPacket is built correctly from utterance text."""

    def setUp(self) -> None:
        self.store = _store_with_learned_facts()

    def test_task_topic(self) -> None:
        cases = [
            ("How to write a poem?", "poem"),
            ("How do I get rich?", "rich"),
            ("How do I center a div in html?", "div"),
            ("Tell me a brief riddle about a calm fish", "riddle"),
            ("How do I safely carry a fragile vase?", "vase"),
            ("What is a fish?", "fish"),
            ("What is a poem?", "poem"),
            ("Make a concise calm puzzle riddle about a fish", "riddle"),
            ("Give me a concise professional answer about writing a beautiful poem quickly", "poem"),
            ("How do I center an html div quickly?", "div"),
        ]
        for utterance, expected in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                self.assertEqual(packet.task_topic, expected)

    def test_task_topic_not_answer(self) -> None:
        packet = build_attention_packet(
            "Give me a concise professional answer about writing a beautiful poem quickly",
            store=self.store,
        )
        self.assertNotEqual(packet.task_topic, "answer")

    def test_learned_summary_contains_topic(self) -> None:
        cases = [
            "How to write a poem?",
            "How do I get rich?",
        ]
        for utterance in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                topic = packet.task_topic
                self.assertIn(topic, packet.learned_summary.lower())
                self.assertGreater(len(packet.learned_summary), 0)

    def test_constraints(self) -> None:
        cases: list[tuple[str, set[str]]] = [
            ("How to write a poem?", set()),
            ("How to write a beautiful poem?", {"beautiful"}),
            ("How to quickly write a poem?", {"quickly"}),
            ("Tell me a brief riddle about a calm fish", {"brief", "calm"}),
            ("How can I quickly write a beautiful poem?", {"beautiful", "quickly"}),
            ("Give me a concise professional answer about writing a beautiful poem quickly",
             {"concise", "professional", "beautiful", "quickly"}),
        ]
        for utterance, expected in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                constraint_lemmas = {c.lemma for c in packet.constraints}
                if expected:
                    for lemma in expected:
                        self.assertIn(lemma, constraint_lemmas)
                else:
                    self.assertEqual(len(packet.constraints), 0)

    def test_normalization_alerts(self) -> None:
        cases = [
            ("How do I center a div in html?",),
            ("How do I center an html div quickly?",),
        ]
        for (utterance,) in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                self.assertGreater(len(packet.normalization_alerts), 0)
                self.assertTrue(any("html" in a for a in packet.normalization_alerts),
                                msg=f"No html alert in {packet.normalization_alerts}")

    def test_content_entities(self) -> None:
        cases = [
            ("Tell me a brief riddle about a calm fish", {"fish"}),
            ("Make a concise calm puzzle riddle about a fish", {"fish"}),
        ]
        for utterance, expected in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                entity_lemmas = {e["lemma"] for e in packet.content_entities}
                for lemma in expected:
                    self.assertIn(lemma, entity_lemmas)

    def test_output_type(self) -> None:
        cases = [
            ("Tell me a brief riddle about a calm fish", "riddle"),
        ]
        for utterance, expected in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                self.assertEqual(packet.output_type, expected)

    def test_reasoning_protected(self) -> None:
        cases = [
            ("Why is a fish usually wet?",),
            ("Why does a fragile vase break easily?",),
        ]
        for (utterance,) in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                self.assertTrue(packet.reasoning_protected)

    def test_entity_fragile(self) -> None:
        cases = [
            ("How do I safely carry a fragile vase?",),
            ("Why does a fragile vase break easily?",),
        ]
        for (utterance,) in cases:
            with self.subTest(utterance=utterance):
                packet = build_attention_packet(utterance, store=self.store)
                self.assertTrue(packet.entity_fragile)


class TestPacketNoStore(unittest.TestCase):
    """Packet still builds without a store (graceful degradation)."""

    def test_build_packet_no_store_no_crash(self) -> None:
        packet = build_attention_packet("How to write a poem?")
        self.assertEqual(packet.task_topic, "poem")

    def test_build_packet_no_store_learned_summary_empty(self) -> None:
        packet = build_attention_packet("What is a poem?")
        self.assertEqual(packet.learned_summary, "")

    def test_build_packet_no_store_html_alert(self) -> None:
        packet = build_attention_packet("How do I center a div in html?")
        self.assertGreater(len(packet.normalization_alerts), 0)


class TestConstraintExtraction(unittest.TestCase):
    """Constraints are extracted correctly from tokens."""

    def test_output_type_constraint(self) -> None:
        from melm.contracts import load_semantic_attention_rules
        rules = load_semantic_attention_rules()
        constraints = extract_constraints(
            ["brief", "poem"], {}, rules, {}, ""
        )
        lemmas = {c.lemma for c in constraints}
        self.assertIn("brief", lemmas)

    def test_adverb_constraint(self) -> None:
        from melm.contracts import load_semantic_attention_rules
        rules = load_semantic_attention_rules()
        constraints = extract_constraints(
            ["quickly", "write"], {}, rules, {}, ""
        )
        lemmas = {c.lemma for c in constraints}
        self.assertIn("quickly", lemmas)

    def test_short_adverb_not_constrained(self) -> None:
        from melm.contracts import load_semantic_attention_rules
        rules = load_semantic_attention_rules()
        constraints = extract_constraints(
            ["fly", "write"], {}, rules, {}, ""
        )
        lemmas = {c.lemma for c in constraints}
        self.assertNotIn("fly", lemmas)


class TestTopicResolution(unittest.TestCase):
    """Task topic is resolved correctly."""

    def test_topic_from_object(self) -> None:
        from melm.contracts import load_semantic_attention_rules
        rules = load_semantic_attention_rules()
        class FakeParse:
            object = "poem"
        topic = resolve_task_topic(
            ["how", "to", "write", "a", "poem"],
            FakeParse(),
            [],
            set(),
            rules,
        )
        self.assertEqual(topic, "poem")

    def test_topic_falls_back_to_candidates(self) -> None:
        from melm.contracts import load_semantic_attention_rules
        rules = load_semantic_attention_rules()
        topic = resolve_task_topic(
            ["what", "is", "a", "poem"],
            None,
            [{"lemma": "poem"}],
            set(),
            rules,
        )
        self.assertEqual(topic, "poem")

    def test_riddle_wins_over_object(self) -> None:
        from melm.contracts import load_semantic_attention_rules
        rules = load_semantic_attention_rules()
        class FakeParse:
            object = "fish"
        topic = resolve_task_topic(
            ["tell", "me", "a", "riddle", "about", "a", "fish"],
            FakeParse(),
            [{"lemma": "fish"}],
            set(),
            rules,
        )
        self.assertEqual(topic, "riddle")


class TestRendererScoring(unittest.TestCase):
    """Renderer family scoring selects correctly."""

    def test_definition_matches_when_topic_and_class_present(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="What is a fish?",
            task_topic="fish",
            task_topic_class="living_thing.animal",
            task_topic_source="noun_atoms",
        )
        renderers = load_nlg_atomic_renderers()
        family = renderers["renderer_families"]["definition"]
        score = score_renderer_family("definition", family, packet)
        self.assertGreater(score, 0)

    def test_definition_blocked_when_riddle_output(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="Tell me a riddle",
            task_topic="riddle",
            output_type="riddle",
        )
        renderers = load_nlg_atomic_renderers()
        family = renderers["renderer_families"]["definition"]
        score = score_renderer_family("definition", family, packet)
        self.assertEqual(score, 0.0)

    def test_creative_riddle_matches_when_riddle_output(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="Tell me a riddle",
            task_topic="riddle",
            output_type="riddle",
        )
        renderers = load_nlg_atomic_renderers()
        family = renderers["renderer_families"]["creative_riddle"]
        score = score_renderer_family("creative_riddle", family, packet)
        self.assertGreater(score, 0)

    def test_entity_property_advice_matches_fragile(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="How do I carry a fragile vase?",
            task_topic="vase",
            entity_fragile=True,
            predicate="carry",
            capability=CapabilityBinding(),
        )
        renderers = load_nlg_atomic_renderers()
        family = renderers["renderer_families"]["entity_property_advice"]
        score = score_renderer_family("entity_property_advice", family, packet)
        self.assertGreater(score, 0)

    def test_entity_property_explanation_matches_fragile_reasoning(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="Why is a vase fragile?",
            task_topic="vase",
            entity_fragile=True,
            reasoning_protected=True,
            capability=CapabilityBinding(),
        )
        renderers = load_nlg_atomic_renderers()
        family = renderers["renderer_families"]["entity_property_explanation"]
        score = score_renderer_family("entity_property_explanation", family, packet)
        self.assertGreater(score, 0)

    def test_how_to_matches_write_poem(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="How to write a poem?",
            task_topic="poem",
            predicate="write",
            capability=CapabilityBinding(),
        )
        renderers = load_nlg_atomic_renderers()
        family = renderers["renderer_families"]["how_to"]
        score = score_renderer_family("how_to", family, packet)
        self.assertGreater(score, 0)

    def test_how_to_blocked_by_reasoning_protected(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="Why does a vase break?",
            task_topic="vase",
            reasoning_protected=True,
            capability=CapabilityBinding(),
        )
        renderers = load_nlg_atomic_renderers()
        family = renderers["renderer_families"]["how_to"]
        score = score_renderer_family("how_to", family, packet)
        self.assertEqual(score, 0.0)


class TestRendererOutput(unittest.TestCase):
    """End-to-end renderer output checks."""

    def test_render_definition_contains_topic_and_class(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="What is a fish?",
            task_topic="fish",
            task_topic_class="living_thing.animal",
            task_topic_source="noun_atoms",
            capability=CapabilityBinding(),
        )
        result = render_from_packet(packet)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("fish", result.lower())
        self.assertIn("living_thing.animal", result.lower())

    def test_render_creative_riddle(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="Tell me a riddle",
            output_type="riddle",
            task_topic="riddle",
            capability=CapabilityBinding(),
        )
        result = render_from_packet(packet)
        self.assertIsNotNone(result)

    def test_render_how_to_contains_predicate_and_topic(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="How to write a poem?",
            task_topic="poem",
            predicate="write",
            capability=CapabilityBinding(),
        )
        result = render_from_packet(packet)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("write", result.lower())
        self.assertIn("poem", result.lower())

    def test_render_learned_unknown_skill_contains_fallback(self) -> None:
        """When the packet has a topic but no installation, renders fallback."""
        packet = SemanticAttentionPacket(
            input_text="Tell me about poems",
            task_topic="poem",
            capability=CapabilityBinding(installed=False, fallback="open_domain"),
        )
        result = render_from_packet(packet)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("poem", result)

    def test_render_entity_fragile_advice(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="How do I carry a fragile vase?",
            task_topic="vase",
            entity_fragile=True,
            capability=CapabilityBinding(),
        )
        result = render_from_packet(packet)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("fragile", result.lower())

    def test_render_with_technical_alert(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="How do I center a div in html?",
            task_topic="div",
            predicate="center",
            normalization_alerts=("html_normalized_alert",),
            capability=CapabilityBinding(),
        )
        result = render_from_packet(packet)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("technical", result.lower())

    def test_render_no_match_returns_none(self) -> None:
        packet = SemanticAttentionPacket(
            input_text="xyzzy",
            capability=CapabilityBinding(),
        )
        result = render_from_packet(packet)
        self.assertIsNone(result)


class TestDataclassFrozen(unittest.TestCase):
    """Packet and constraint dataclasses are correctly frozen."""

    def test_packet_is_frozen(self) -> None:
        packet = SemanticAttentionPacket(input_text="test")
        with self.assertRaises(AttributeError):
            packet.input_text = "changed"  # type: ignore[misc]

    def test_constraint_is_frozen(self) -> None:
        c = ConstraintBinding(lemma="test")
        with self.assertRaises(AttributeError):
            c.lemma = "changed"  # type: ignore[misc]

    def test_capability_is_frozen(self) -> None:
        c = CapabilityBinding(installed=True)
        with self.assertRaises(AttributeError):
            c.installed = False  # type: ignore[misc]


class TestProbePassThrough(unittest.TestCase):
    """The experiment probe still runs and the winner passes 18/18."""

    def test_probe_runs_and_winner_passes(self) -> None:
        from pathlib import Path
        probe_path = Path("experiments/nlg_attention_packet_probe.py")
        if not probe_path.is_file():
            raise unittest.SkipTest("experiments/nlg_attention_packet_probe.py not found")
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(probe_path), "--json"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr[:500])
        import json
        report = json.loads(result.stdout)
        self.assertEqual(report["winner"], "role_separated_packet")
        winner_data = report["ranked"][0]
        self.assertEqual(winner_data["passed"], winner_data["total"],
                         msg=f"Winner {winner_data['approach']} failed some cases")


class TestFallbackPhrasesContract(unittest.TestCase):
    """nlg_fallback_phrases.v1.json loads correctly."""

    def test_fallback_phrases_loads(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases, validate_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        self.assertIn("entity_nlg_templates", data)
        self.assertIn("refusal_templates", data)
        self.assertIn("social_status_templates", data)
        self.assertIn("safety_school_templates", data)
        self.assertIn("music_templates", data)
        self.assertIn("story_verb_tenses", data)
        self.assertIn("story_sentence_patterns", data)

    def test_fallback_phrases_rejects_wrong_schema_id(self) -> None:
        from melm.contracts import validate_nlg_fallback_phrases
        with self.assertRaises(ValueError):
            validate_nlg_fallback_phrases({"schema_id": "melm.wrong", "entity_nlg_templates": {"fragile": "x"}, "refusal_templates": {"default": "x"}, "social_status_templates": {"default": "x"}, "safety_school_templates": {"weather_policy": "x"}, "music_templates": {"success": "x", "failure": "x"}, "story_verb_tenses": {"walk": "walked"}, "story_sentence_patterns": ["a", "b", "c", "d"]})

    def test_entity_nlg_templates_has_all_keys(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        templates = data["entity_nlg_templates"]
        for key in ("fragile", "materials", "functional_uses", "color", "place_context", "high_harm", "low_harm"):
            self.assertIn(key, templates, f"missing key: {key}")

    def test_refusal_template_is_string(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        self.assertIsInstance(data["refusal_templates"]["default"], str)

    def test_social_status_templates_have_mood_and_default(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        self.assertIn("with_mood", data["social_status_templates"])
        self.assertIn("default", data["social_status_templates"])

    def test_music_templates_have_success_and_failure(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        self.assertIn("success", data["music_templates"])
        self.assertIn("failure", data["music_templates"])

    def test_story_verb_tenses_is_nonempty_dict(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        tenses = data["story_verb_tenses"]
        self.assertIsInstance(tenses, dict)
        self.assertGreater(len(tenses), 0)

    def test_story_sentence_patterns_has_minimum_four(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        self.assertGreaterEqual(len(data["story_sentence_patterns"]), 4)

    def test_safety_school_template_is_string(self) -> None:
        from melm.contracts import load_nlg_fallback_phrases
        data = load_nlg_fallback_phrases()
        self.assertIsInstance(data["safety_school_templates"]["weather_policy"], str)


class TestFallbackPhrasesWiring(unittest.TestCase):
    """Synthesis wire points read from contract instead of hardcoded strings."""

    def test_get_fallback_phrases_returns_cache(self) -> None:
        from melm.appliance.assistant_synthesis import _get_fallback_phrases
        data = _get_fallback_phrases()
        self.assertIn("entity_nlg_templates", data)

    def test_fallback_phrases_cache_is_memoized(self) -> None:
        from melm.appliance.assistant_synthesis import _get_fallback_phrases
        a = _get_fallback_phrases()
        b = _get_fallback_phrases()
        self.assertIs(a, b)

    def test_refusal_template_format(self) -> None:
        from melm.appliance.assistant_synthesis import _get_fallback_phrases
        tpl = _get_fallback_phrases().get("refusal_templates", {}).get("default", "")
        self.assertIsInstance(tpl, str)
        self.assertGreater(len(tpl), 0)

    def test_story_follow_up_module_level(self) -> None:
        from melm.appliance.assistant_synthesis import _get_story_follow_up
        data = _get_story_follow_up()
        self.assertIsInstance(data, dict)

    def test_story_follow_up_cache_is_memoized(self) -> None:
        from melm.appliance.assistant_synthesis import _get_story_follow_up
        a = _get_story_follow_up()
        b = _get_story_follow_up()
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
