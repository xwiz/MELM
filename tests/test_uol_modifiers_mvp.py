"""Tests for the UOL Modifiers slot (adjectives/descriptors).

Covers contract loading, POS detection, functional grammar role assignment,
atomizer modifier extraction (both FP and SyntaxGraph paths), and enrichment
integration. See docs/assistant_os_architecture_v0_4_1.md § "UOL Modifiers
Slot" for the design.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest

from melm.contracts import load_modifier_atoms, validate_modifier_atoms
from melm.appliance.language_adapters import (
    SyntaxGraph,
    DepEdge,
    pos_tag_for,
    simple_dependencies,
)
from melm.appliance.functional_grammar import (
    parse_functional_relations,
    _is_adjective_candidate,
)
from melm.appliance.uol_atomizer import (
    atomize,
    atomize_syntax_graph,
    _modifier_semantic_class,
    _load_modifier_semclass_map,
    _extract_modifiers_from_parse,
    _extract_modifiers_from_graph,
    _attach_modifiers,
)
from melm.appliance.uol_types import UolAtom, UolAct
from melm.appliance.local_assistant_router import OnDeviceAssistantRouter


# =========================================================================
# Phase 1: Contract
# =========================================================================


class TestModifierAtomContract(unittest.TestCase):
    def test_loads_30_entries(self) -> None:
        payload = load_modifier_atoms()
        assert len(payload.get("entries", [])) == 30

    def test_validate_passes(self) -> None:
        payload = load_modifier_atoms()
        validate_modifier_atoms(payload)

    def test_entry_fields(self) -> None:
        """Parametrized: every entry has canonical_lemma, valid modifier_type,
        and semantic_class_id (if present) exists in spine."""
        payload = load_modifier_atoms()
        from melm.contracts import load_semantic_class_ids

        spine = load_semantic_class_ids()
        for i, e in enumerate(payload["entries"]):
            with self.subTest(entry=i, lemma=e.get("canonical_lemma", "?")):
                assert e.get("canonical_lemma"), f"missing lemma in {e}"
                assert e.get("modifier_type") in {"adjective"}, f"bad type in {e}"
                sc = e.get("semantic_class_id", "")
                if sc:
                    assert sc in spine, f"{e['canonical_lemma']}: {sc} not in spine"

    def test_semantic_class_lookup_returns_expected(self) -> None:
        assert _modifier_semantic_class("calm") == "media_descriptor.music_mood"
        assert _modifier_semantic_class("sad") == "emotion"
        assert _modifier_semantic_class("beautiful") == "personal_attribute"
        assert _modifier_semantic_class("unknown_word") == "attribute"


# =========================================================================
# Phase 2: POS Detection
# =========================================================================


class TestAdjectivePosDetection(unittest.TestCase):
    def test_pos_tag_for(self) -> None:
        cases = [
            ("calm", "ADJ"),
            ("music", "NOUN"),
            ("beautiful", "ADJ"),
            ("table", "NOUN"),
            ("jazz", "NOUN"),
            ("play", "VERB"),
            ("the", "DET"),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                assert pos_tag_for("en", word) == expected

    def test_caching_does_not_raise(self) -> None:
        assert pos_tag_for("en", "calm") == "ADJ"
        assert pos_tag_for("en", "calm") == "ADJ"


class TestAdjectiveCandidate(unittest.TestCase):
    def test_is_adjective_candidate(self) -> None:
        cases = [
            ("calm", True),
            ("music", False),
            ("beautiful", True),
            ("table", False),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                assert _is_adjective_candidate(word) == expected


class TestAmodDependency(unittest.TestCase):
    def test_amod_relation(self) -> None:
        cases = [
            (("play", "calm", "music"), 1, "calm"),
            (("the", "people", "tell", "stories"), 0, None),
        ]
        for lemmas, expected_count, expected_adj in cases:
            with self.subTest(lemmas=lemmas):
                language = "en"
                tags = tuple(pos_tag_for(language, l) for l in lemmas)
                deps = simple_dependencies(language, lemmas, tags)
                amod_edges = [d for d in deps if d.relation == "amod"]
                assert len(amod_edges) == expected_count
                if expected_adj is not None and expected_count > 0:
                    assert lemmas[amod_edges[0].dependent] == expected_adj


# =========================================================================
# Phase 2b: Functional Grammar Adjective Role
# =========================================================================


class TestFunctionalGrammarAdjectiveRole(unittest.TestCase):
    def test_adjectival_modifier_role(self) -> None:
        cases = [
            (("play", "calm", "music"), 1, "calm"),
            (("play", "music"), 0, None),
            (("tell", "a", "story"), 0, None),
        ]
        for lemmas, expected_count, expected_token in cases:
            with self.subTest(lemmas=lemmas):
                parse = parse_functional_relations(lemmas, language="en")
                adj_roles = [
                    tr
                    for tr in parse.token_roles
                    if tr.get("role") == "adjectival_modifier"
                ]
                assert len(adj_roles) == expected_count
                if expected_token is not None:
                    assert adj_roles[0]["token"] == expected_token
                    assert adj_roles[0]["meaning"] == "descriptor"


# =========================================================================
# Phase 3: Atomizer Modifier Extraction (FunctionalParse Path)
# =========================================================================


class TestModifierExtractionFunctionalParse(unittest.TestCase):
    def test_extract_modifiers_from_parse(self) -> None:
        cases = [
            (("play", "calm", "music"), 1, 1, "calm", "media_descriptor.music_mood"),
            (("play", "music"), 0, None, None, None),
        ]
        for lemmas, expected_count, expected_idx, expected_lemma, expected_sc in cases:
            with self.subTest(lemmas=lemmas):
                parse = parse_functional_relations(lemmas, language="en")
                index_to_role = {tr["index"]: dict(tr) for tr in parse.token_roles}
                modifiers = _extract_modifiers_from_parse(parse, index_to_role)
                assert len(modifiers) == expected_count
                if expected_count > 0:
                    idx, lemma, sem_class = modifiers[0]
                    assert idx == expected_idx
                    assert lemma == expected_lemma
                    assert sem_class == expected_sc

    def test_atomize_attaches_modifiers(self) -> None:
        cases = [
            (("play", "calm", "music"), {"calm"}),
            (("play", "beautiful", "calm", "music"), {"calm", "beautiful"}),
        ]
        for lemmas, expected_lemmas in cases:
            with self.subTest(lemmas=lemmas):
                parse = parse_functional_relations(lemmas, language="en")
                act = atomize(parse, language="en")
                assert act is not None
                assert len(act.content) == 1
                modifier_lemmas = {m.lemma for m in act.content[0].modifiers}
                assert modifier_lemmas == expected_lemmas
                for m in act.content[0].modifiers:
                    if m.lemma == "calm":
                        assert m.semantic_class == "media_descriptor.music_mood"
                        assert m.modifier_type == "adjective"

    def test_atom_dict_includes_modifiers(self) -> None:
        parse = parse_functional_relations(("play", "calm", "music"), language="en")
        act = atomize(parse, language="en")
        assert act is not None
        d = act.to_dict()
        for atom_dict in d.get("content", []):
            mods = atom_dict.get("modifiers", [])
            if mods:
                assert any(m.get("lemma") == "calm" for m in mods)
                assert any(
                    m.get("semantic_class") == "media_descriptor.music_mood"
                    for m in mods
                )


# =========================================================================
# Phase 3: Atomizer Modifier Extraction (SyntaxGraph Path)
# =========================================================================


def _make_adapter() -> Any:
    """Create a language adapter for SyntaxGraph construction."""
    from melm.appliance.language_adapters import get_adapter

    return get_adapter("en")


class TestModifierExtractionSyntaxGraph(unittest.TestCase):
    def test_extract_modifiers_from_graph(self) -> None:
        adapter = _make_adapter()
        graph = adapter.tag(("play", "calm", "music"))
        assert graph is not None
        all_indices = set(range(len(graph.lemmas)))
        modifiers = _extract_modifiers_from_graph(graph, all_indices)
        assert len(modifiers) >= 1
        lemmas_found = {m[1] for m in modifiers}
        assert "calm" in lemmas_found

    def test_syntax_graph_atomize_attaches_modifiers(self) -> None:
        adapter = _make_adapter()
        graph = adapter.tag(("play", "calm", "music"))
        act = atomize_syntax_graph(graph)
        assert act is not None
        assert len(act.content) >= 1
        all_mod_lemmas: set[str] = set()
        for atom in act.content:
            for m in atom.modifiers:
                all_mod_lemmas.add(m.lemma)
                assert m.semantic_class, f"modifier {m.lemma} has empty semantic_class"
        assert "calm" in all_mod_lemmas

    def test_no_modifiers_when_no_adj_graph(self) -> None:
        cases = [
            ("no_adj", ("tell", "stories")),
            ("noun_adj_candidate", ("play", "table", "tennis")),
        ]
        for desc, lemmas in cases:
            with self.subTest(case=desc):
                adapter = _make_adapter()
                graph = adapter.tag(lemmas)
                act = atomize_syntax_graph(graph)
                assert act is not None
                for atom in act.content:
                    assert len(atom.modifiers) == 0, f"unexpected modifiers on {atom}"

    def test_subordinate_clause_modifiers(self) -> None:
        adapter = _make_adapter()
        graph = adapter.tag(("play", "music", "because", "i", "feel", "happy"))
        act = atomize_syntax_graph(graph)
        assert act is not None
        all_mod_lemmas: set[str] = set()
        for atom in act.content:
            for m in atom.modifiers:
                all_mod_lemmas.add(m.lemma)
        assert "happy" in all_mod_lemmas


# =========================================================================
# Phase 3: _attach_modifiers
# =========================================================================


class TestAttachModifiers:
    def test_empty_modifiers_unchanged(self) -> None:
        from melm.appliance.uol_atomizer import _new_id
        from melm.appliance.uol_types import PredicateRef, AtomContext

        atom = UolAtom(
            id=_new_id(),
            kind="event",
            predicate=PredicateRef(id="test", semantic_class="verb.change"),
        )
        result = _attach_modifiers(atom, [])
        assert result is atom  # same object when no modifiers

    def test_attaches_single_modifier(self) -> None:
        from melm.appliance.uol_atomizer import _new_id
        from melm.appliance.uol_types import PredicateRef

        atom = UolAtom(
            id=_new_id(),
            kind="event",
            predicate=PredicateRef(id="test", semantic_class="verb.change"),
        )
        result = _attach_modifiers(
            atom, [(1, "calm", "media_descriptor.music_mood")]
        )
        assert len(result.modifiers) == 1
        assert result.modifiers[0].lemma == "calm"
        assert result.modifiers[0].semantic_class == "media_descriptor.music_mood"

    def test_preserves_existing_modifiers(self) -> None:
        from melm.appliance.uol_atomizer import _new_id
        from melm.appliance.uol_types import PredicateRef, Modifier

        existing = (Modifier(lemma="existing", semantic_class="test"),)
        atom = UolAtom(
            id=_new_id(),
            kind="event",
            predicate=PredicateRef(id="test", semantic_class="verb.change"),
            modifiers=existing,
        )
        result = _attach_modifiers(
            atom, [(1, "calm", "media_descriptor.music_mood")]
        )
        assert len(result.modifiers) == 2
        assert result.modifiers[0].lemma == "existing"
        assert result.modifiers[1].lemma == "calm"


# =========================================================================
# Phase 4: Enrichment Integration
# =========================================================================


class TestEnrichmentIntegration:
    def test_music_mood_modifier_enriches_ambient(self) -> None:
        from melm.appliance.language_adapters import get_adapter

        adapter = get_adapter("en")
        graph = adapter.tag(("play", "calm", "music"))
        act = atomize_syntax_graph(graph)
        assert act is not None
        enriched = OnDeviceAssistantRouter._enrich_media_playback_answer(
            "Playing music from your library.", act.to_dict()
        )
        assert enriched is not None
        assert "ambient" in enriched.lower()

    def test_no_modifier_falls_back_to_role_class(self) -> None:
        adapter = _make_adapter()
        graph = adapter.tag(("play", "jazz"))
        act = atomize_syntax_graph(graph)
        assert act is not None
        enriched = OnDeviceAssistantRouter._enrich_media_playback_answer(
            "Playing jazz.", act.to_dict()
        )
        assert enriched is not None

    def test_enrichment_falls_back_ambient_without_uol_act(self) -> None:
        result = OnDeviceAssistantRouter._enrich_media_playback_answer("Hello.", None)
        assert result is not None
        assert "ambient" in result.lower()


# =========================================================================
# Phase 4: Music Style Templates Contract
# =========================================================================


class TestMusicStyleTemplates:
    def test_has_music_mood_mapping(self) -> None:
        from melm.contracts import load_contract_json

        payload = load_contract_json("music_style_templates.v1.json")
        sc_map = payload.get("semantic_class_mapping", [])
        mood_entries = [
            e
            for e in sc_map
            if e.get("semantic_class") == "media_descriptor.music_mood"
        ]
        assert len(mood_entries) >= 1
        assert mood_entries[0].get("style") == "ambient"

    def test_version_is_1_1_0(self) -> None:
        from melm.contracts import load_contract_json

        payload = load_contract_json("music_style_templates.v1.json")
        assert payload.get("version") == "1.1.0"
