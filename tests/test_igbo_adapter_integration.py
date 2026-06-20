"""Integration tests for Igbo adapter seeding ownership."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from melm.appliance.language_adapters.igbo import seed_igbo_lexicon
from melm.appliance import local_assistant_router


class IgboAdapterIntegrationTests(unittest.TestCase):
    def test_adapter_exposes_igbo_lexicon_seeder(self) -> None:
        lexicon: dict[str, frozenset[str]] = {}
        entries = [
            {"lemma": "eri", "semantic_class": "verb.consume"},
            {"lemma": "mara", "semantic_class": "verb.cognition"},
        ]
        seed_igbo_lexicon(lexicon, entries)
        self.assertEqual(lexicon["eri"], frozenset({"verb.consume"}))
        self.assertEqual(lexicon["mara"], frozenset({"verb.cognition"}))

    def test_router_no_longer_imports_seed_from_compatibility_normalizer(self) -> None:
        source = Path(local_assistant_router.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offending_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "melm.appliance.igbo_normalizer":
                offending_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module == ".igbo_normalizer":
                offending_imports.extend(alias.name for alias in node.names)
        self.assertEqual(
            offending_imports,
            [],
            f"router should seed Igbo lexicon from adapter layer, not compatibility shim: {offending_imports}",
        )
