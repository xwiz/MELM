"""Rule-backed language adapter tests for the SyntaxGraph bridge."""

from __future__ import annotations

import unittest

from melm.appliance.language_adapters import SyntaxGraph, detect_language, get_adapter


class LanguageAdapterMvpTests(unittest.TestCase):
    def test_english_adapter_exposes_lemmatize_and_tag(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None

        tokens = adapter.tokenize("What will I eat?")
        lemmas = adapter.lemmatize(tokens)
        graph = adapter.tag(tokens)

        self.assertEqual(tokens, ("what", "will", "i", "eat"))
        self.assertEqual(lemmas, ("what", "will", "i", "eat"))
        self.assertIsInstance(graph, SyntaxGraph)
        self.assertEqual(graph.tokens, tokens)
        self.assertEqual(graph.lemmas, lemmas)
        self.assertEqual(graph.language, "en")
        self.assertEqual(len(graph.pos_tags), len(tokens))
        self.assertEqual(len(graph.morph_features), len(tokens))
        self.assertTrue(any(edge.relation == "root" for edge in graph.dependencies))

    def test_igbo_adapter_exposes_lemmatize_and_tag(self) -> None:
        adapter = get_adapter("ig")
        self.assertIsNotNone(adapter)
        assert adapter is not None

        tokens = adapter.tokenize("Gini m ga-eri?")
        lemmas = adapter.lemmatize(tokens)
        graph = adapter.tag(tokens)

        self.assertEqual(tokens, ("gini", "m", "ga", "eri"))
        self.assertEqual(lemmas, ("gini", "m", "ga", "eri"))
        self.assertEqual(graph.tokens, tokens)
        self.assertEqual(graph.lemmas, lemmas)
        self.assertEqual(graph.language, "ig")
        self.assertEqual(len(graph.pos_tags), len(tokens))
        self.assertTrue(any(edge.relation == "root" for edge in graph.dependencies))

    def test_tag_marks_expected_root_and_subject_roles(self) -> None:
        adapter = get_adapter("en")
        self.assertIsNotNone(adapter)
        assert adapter is not None

        graph = adapter.tag(("call", "mom"))
        root_edges = [edge for edge in graph.dependencies if edge.relation == "root"]
        obj_edges = [edge for edge in graph.dependencies if edge.relation == "obj"]

        self.assertEqual(len(root_edges), 1)
        self.assertEqual(root_edges[0].dependent, 0)
        self.assertGreaterEqual(len(obj_edges), 1)
        self.assertEqual(obj_edges[0].head, 0)

    def test_detect_language_keeps_plain_english_story_request_in_english(self) -> None:
        language, confidence = detect_language("Tell me a story about a dragon.")

        self.assertEqual(language, "en")
        self.assertGreaterEqual(confidence, 0.1)
