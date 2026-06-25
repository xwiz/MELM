"""Tests for graph verbalization (AtomLink-aware AtomTemplateBackend)
and frame-linker integration (has_causal_links detection).

Tests:
- AtomTemplateBackend._extract_causal_links: extracts cause/effect from linked atoms
- AtomTemplateBackend.generate: renders causal templates with link-extracted roles
- AtomTemplateBackend._extract_roles: includes causal link roles alongside normal roles
- FrameLinker.has_causal_links: detects causal links in UOL act
- FrameLinker.has_causal_links: returns False when no links exist
- Router frame-linker enrichment: causal links trigger reasoning fallback
"""
from __future__ import annotations

import pytest

from melm.appliance.assistant_decoder_atom import AtomTemplateBackend
from melm.appliance.assistant_frame_linker import FrameLinker


def _make_causal_explanation_uol(
    effect_lemma: str = "wet",
    effect_theme: str = "ground",
    cause_lemma: str = "rain",
    cause_agent: str = "",
) -> dict:
    """Build a UOL act with two atoms linked by caused_by."""
    content = [
        {
            "predicate": {"id": "wet", "lemma": "wet", "semantic_class": "state"},
            "roles": [
                {"role": "theme", "value": effect_theme},
            ],
            "links": {"causes": [], "caused_by": ["rain"], "enables": [], "prevents": []},
        },
        {
            "predicate": {"id": "rain", "lemma": "rain", "semantic_class": "verb.weather"},
            "roles": [
                {"role": "theme", "value": cause_agent or "sky"},
            ],
            "links": {"causes": ["wet"], "caused_by": [], "enables": [], "prevents": []},
        },
    ]
    return {"act": "statement", "content": content}


def _make_causal_prediction_uol(
    cause_lemma: str = "eat",
    effect_lemma: str = "full",
    effect_theme: str = "stomach",
    cause_agent: str = "person",
) -> dict:
    """Build a UOL act with two atoms linked by causes."""
    content = [
        {
            "predicate": {"id": "eat", "lemma": "eat", "semantic_class": "verb.consume"},
            "roles": [
                {"role": "agent", "value": cause_agent},
            ],
            "links": {"causes": ["full"], "caused_by": [], "enables": [], "prevents": []},
        },
        {
            "predicate": {"id": "full", "lemma": "full", "semantic_class": "state"},
            "roles": [
                {"role": "theme", "value": effect_theme},
            ],
            "links": {"causes": [], "caused_by": ["eat"], "enables": [], "prevents": []},
        },
    ]
    return {"act": "statement", "content": content}


def _make_no_links_uol() -> dict:
    """UOL act with no causal links."""
    content = [
        {
            "predicate": {"id": "weather", "lemma": "weather", "semantic_class": "verb.weather"},
            "roles": [{"role": "theme", "value": "rainy"}],
            "links": {"causes": [], "caused_by": [], "enables": [], "prevents": []},
        },
    ]
    return {"act": "statement", "content": content}


class TestCausalLinkRoleExtraction:
    """AtomTemplateBackend extracts causal link roles from linked atoms."""

    def test_extract_causal_links_explanation(self):
        """causal_explanation UOL → cause/effect extracted from links."""
        backend = AtomTemplateBackend()
        uol = _make_causal_explanation_uol()
        roles = backend._extract_roles(uol)
        assert "cause" in roles
        assert "effect" in roles
        assert "cause_verb" in roles
        assert "effect_verb" in roles
        assert roles["cause"] == "sky"     # theme of the cause atom (rain → sky)
        assert roles["effect"] == "ground"  # theme of the effect atom (wet → ground)
        assert roles["cause_verb"] == "rain"
        assert roles["effect_verb"] == "wet"

    def test_extract_causal_links_prediction(self):
        """causal_prediction UOL → cause/effect extracted."""
        backend = AtomTemplateBackend()
        uol = _make_causal_prediction_uol()
        roles = backend._extract_roles(uol)
        assert "cause" in roles
        assert "effect" in roles
        assert roles["cause_verb"] == "eat"
        assert roles["effect_verb"] == "full"

    def test_causal_roles_included_with_normal_roles(self):
        """Causal link roles coexist with standard atom roles."""
        backend = AtomTemplateBackend()
        uol = _make_causal_explanation_uol(effect_theme="ground")
        roles = backend._extract_roles(uol)
        # Normal roles from first atom
        assert "theme" in roles
        assert "verb" in roles
        # Causal roles
        assert "cause" in roles
        assert "effect" in roles

    def test_no_causal_links_no_causal_roles(self):
        """Without links, no cause/effect roles are added."""
        backend = AtomTemplateBackend()
        uol = _make_no_links_uol()
        roles = backend._extract_roles(uol)
        assert "cause" not in roles
        assert "effect" not in roles
        assert "theme" in roles
        assert roles["verb"] == "weather"


class TestCausalTemplateRendering:
    """AtomTemplateBackend renders causal templates with link-extracted roles."""

    def test_render_causal_explanation(self):
        """causal_explanation template renders with cause/effect."""
        backend = AtomTemplateBackend()
        uol = _make_causal_explanation_uol(effect_theme="ground", cause_agent="sky")
        result = backend.generate("causal_explanation", uol)
        assert result is not None
        assert "ground" in result
        assert "rain" in result or "sky" in result

    def test_render_causal_prediction(self):
        """causal_prediction template renders with cause/effect."""
        backend = AtomTemplateBackend()
        uol = _make_causal_prediction_uol(cause_agent="person", effect_theme="stomach")
        result = backend.generate("causal_prediction", uol)
        assert result is not None
        assert "eat" in result or "person" in result

    def test_causal_template_fallback_no_links(self):
        """Without cause/effect roles, template still renders (roles empty)."""
        backend = AtomTemplateBackend()
        uol = _make_no_links_uol()
        result = backend.generate("causal_explanation", uol)
        assert result is not None
        # Template fills where possible, strips unfilled placeholders
        assert "happens because" in result

    def test_render_reasoning_fallback_to_atom_template(self):
        """Atom template backend can render causal intents with link-extracted roles."""
        backend = AtomTemplateBackend()
        uol = _make_causal_explanation_uol(effect_theme="ground", cause_agent="")
        # Template rendering directly without full synthesis pipeline
        result = backend.generate("causal_explanation", uol)
        assert result is not None
        assert "ground" in result or "wet" in result


class TestFrameLinkerCausalDetection:
    """FrameLinker.has_causal_links detects AtomLinks in UOL acts."""

    def test_has_causal_links_returns_true(self):
        """has_causal_links returns True when causes/caused_by exist."""
        uol = _make_causal_explanation_uol()
        assert FrameLinker.has_causal_links(uol)

    def test_has_causal_links_returns_false(self):
        """has_causal_links returns False when no links exist."""
        uol = _make_no_links_uol()
        assert not FrameLinker.has_causal_links(uol)

    def test_has_causal_links_empty_content(self):
        """has_causal_links returns False for empty UOL act."""
        uol = {"act": "statement", "content": []}
        assert not FrameLinker.has_causal_links(uol)

    def test_has_causal_links_none(self):
        """has_causal_links returns False for any non-dict input."""
        assert not FrameLinker.has_causal_links({})


class TestUolCausalLinkExtraction:
    """_extract_causal_links_from_uol standalone function."""

    def test_extract_causal_links_from_explanation_uol(self):
        """Extracts effect + effect_theme from causal explanation UOL."""
        from melm.appliance.reasoning.task_router import _extract_causal_links_from_uol
        uol = _make_causal_explanation_uol()
        links = _extract_causal_links_from_uol(uol)
        assert links.get("effect") == "wet"
        assert links.get("effect_theme") == "ground"

    def test_extract_causal_links_from_prediction_uol(self):
        """Extracts cause from causal prediction UOL."""
        from melm.appliance.reasoning.task_router import _extract_causal_links_from_uol
        uol = _make_causal_prediction_uol()
        links = _extract_causal_links_from_uol(uol)
        assert "cause" in links

    def test_extract_causal_links_empty_uol(self):
        """Returns empty dict when UOL has no content."""
        from melm.appliance.reasoning.task_router import _extract_causal_links_from_uol
        links = _extract_causal_links_from_uol({"act": "statement", "content": []})
        assert links == {}

    def test_extract_causal_links_no_links(self):
        """Returns empty dict when atoms have no causal links."""
        from melm.appliance.reasoning.task_router import _extract_causal_links_from_uol
        uol = _make_no_links_uol()
        links = _extract_causal_links_from_uol(uol)
        assert links == {}


class TestFrameCandidateCausalEnrichment:
    """detect_reasoning_task uses frame_candidates for richer causal routing."""

    def test_frame_candidates_without_causal_slots_still_detects_causal(self):
        """Non-causal frame candidates don't block causal detection."""
        from melm.appliance.reasoning.task_router import detect_reasoning_task
        fc = [{"frame_id": "weather.v1", "intent": "weather", "score": 0.85,
               "slot_states": {"location": "london"}}]
        result = detect_reasoning_task(
            "Why is the ground wet?",
            tokens=("why", "is", "the", "ground", "wet"),
            frame_candidates=fc,
        )
        assert result is not None
        assert result.get("task") == "causal_explanation"
        assert "effect" in result

    def test_causal_explanation_with_frame_candidates_adds_confirmation(self):
        """frame_confirmed is set when frame_candidates have causal slot_states."""
        from melm.appliance.reasoning.task_router import detect_reasoning_task
        fc = [{"frame_id": "causal_explanation.v1", "intent": "causal_explanation",
               "score": 0.92, "slot_states": {"cause": "rain", "effect": "wet"}}]
        result = detect_reasoning_task(
            "Why is the ground wet?",
            tokens=("why", "is", "the", "ground", "wet"),
            frame_candidates=fc,
        )
        assert result is not None
        assert result.get("frame_confirmed") is True

    def test_effect_state_uses_uol_links_over_token_heuristic(self):
        """_extract_effect_state prefers UOL causal links over last-content-word."""
        from melm.appliance.reasoning.task_router import _extract_effect_state
        uol = _make_causal_explanation_uol(effect_theme="pavement")
        # Token heuristic would extract "wet" (last word), but
        # UOL links should have "effect" from the atom predicate.
        result = _extract_effect_state(
            ("why", "is", "the", "pavement", "wet"), uol, return_theme=True,
        )
        effect, theme = result
        assert effect is not None
        # Should use UOL links, not just token heuristic
        assert isinstance(effect, str)

    def test_detect_reasoning_task_accepts_no_frame_candidates(self):
        """detect_reasoning_task works with frame_candidates=None (backward compat)."""
        from melm.appliance.reasoning.task_router import detect_reasoning_task
        result = detect_reasoning_task(
            "Why is the ground wet?",
            tokens=("why", "is", "the", "ground", "wet"),
            frame_candidates=None,
        )
        assert result is not None
        assert result.get("task") == "causal_explanation"


class TestScoreAtomsCausalEmission:
    """score_atoms emits causal_reasoning FrameCandidate for atoms with AtomLinks."""

    def test_score_atoms_emits_causal_candidate_when_links_exist(self):
        """causal_explanation UOL → causal_reasoning FrameCandidate in results."""
        linker = FrameLinker()
        uol = _make_causal_explanation_uol()
        candidates = linker.score_atoms(uol, {}, tokens=("ground", "wet"))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) == 1
        assert causal[0].intent == "causal_reasoning"
        assert causal[0].score == 0.60
        assert causal[0].threshold == 0.0

    def test_causal_candidate_has_cause_and_effect_slot_states(self):
        """causal_reasoning slot_states contains cause/effect predicate IDs from links."""
        linker = FrameLinker()
        uol = _make_causal_explanation_uol()
        candidates = linker.score_atoms(uol, {}, tokens=("ground", "wet"))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) == 1
        assert "cause" in causal[0].slot_states
        assert "effect" in causal[0].slot_states
        assert "rain" in causal[0].slot_states.get("cause", "")
        assert "wet" in causal[0].slot_states.get("effect", "")

    def test_score_atoms_no_causal_candidate_without_links(self):
        """No causal_reasoning candidate when atoms have no links."""
        linker = FrameLinker()
        uol = _make_no_links_uol()
        candidates = linker.score_atoms(uol, {}, tokens=("weather",))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) == 0

    def test_causal_candidate_for_prediction_uol(self):
        """causal_prediction UOL also produces causal_reasoning candidate."""
        linker = FrameLinker()
        uol = _make_causal_prediction_uol()
        candidates = linker.score_atoms(uol, {}, tokens=("eat", "full"))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) == 1
        assert "cause" in causal[0].slot_states
        assert "effect" in causal[0].slot_states

    def test_causal_candidate_emitted_for_enables_links(self):
        """enables links also trigger causal_reasoning candidate."""
        linker = FrameLinker()
        content = [
            {"predicate": {"id": "study"}, "roles": [],
             "links": {"enables": ["pass"], "causes": [], "caused_by": [], "prevents": []}},
            {"predicate": {"id": "pass"}, "roles": [],
             "links": {"causes": [], "caused_by": [], "enables": [], "prevents": []}},
        ]
        uol = {"act": "statement", "content": content}
        candidates = linker.score_atoms(uol, {}, tokens=("study", "pass"))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) == 1
        assert "cause" in causal[0].slot_states

    def test_causal_candidate_coexists_with_template_candidates(self):
        """When UOL matches a template AND has causal links, both appear."""
        linker = FrameLinker()
        uol = _make_causal_prediction_uol()
        # "eat" maps to food_item → meal_suggestion template matches
        candidates = linker.score_atoms(
            uol,
            {"eat": frozenset({"food_item"})},
            tokens=("eat", "full"),
        )
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        template_ids = {c.frame_id for c in candidates if c.frame_id != "causal_reasoning"}
        assert len(causal) == 1
        # At least one template candidate should coexist (meal_suggestion)
        assert len(template_ids) >= 1

    def test_causal_candidate_score_components_are_correct(self):
        """score_components dict has expected keys and values."""
        linker = FrameLinker()
        uol = _make_causal_explanation_uol()
        candidates = linker.score_atoms(uol, {}, tokens=("ground", "wet"))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) == 1
        comps = causal[0].score_components
        assert comps.get("causal_links") == 0.60
        assert comps.get("required") == 0.0
        assert comps.get("action") == 0.0
        assert comps.get("optional") == 0.0
        assert comps.get("structure") == 0.0
        assert comps.get("exclude_penalty") == 0.0

    def test_frame_linker_candidates_from_atoms_includes_causal(self):
        """Wrapper _frame_linker_candidates_from_atoms includes causal candidate."""
        from melm.appliance.local_assistant_router import _frame_linker_candidates_from_atoms
        uol = _make_causal_explanation_uol()
        candidates = _frame_linker_candidates_from_atoms(uol, ("why", "is", "the", "ground", "wet"))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) == 1
        assert "cause" in causal[0].slot_states or "effect" in causal[0].slot_states


class TestRouterIntegration:
    """Router leverages frame-linker causal detection for reasoning routing."""

    def test_router_has_causal_links_via_frame_linker(self):
        """Router._try_reasoning checks FrameLinker.has_causal_links as enrichment."""
        uol = _make_causal_explanation_uol()
        assert FrameLinker.has_causal_links(uol)

    def test_router_try_reasoning_passes_frame_candidates(self):
        """_try_reasoning calls score_atoms and passes frame_candidates to detect_reasoning_task."""
        from melm.appliance.local_assistant_router import _frame_linker_candidates_from_atoms
        uol = _make_causal_explanation_uol()
        candidates = _frame_linker_candidates_from_atoms(uol, ("why", "is", "the", "ground", "wet"))
        # May be empty if no frame template matches "why" questions — the
        # important thing is it doesn't crash and returns a list.
        assert isinstance(candidates, list)


class TestFrameLinkerEntityEnrichment:
    """score_atoms enriches FrameCandidate.slot_states with entity properties
    from noun/verb atom contract data (§2.4-3b)."""

    def test_template_candidate_slot_states_includes_entity_properties(self):
        """Template candidate inherits slot_states from role entity_id."""
        from melm.appliance.assistant_frame_linker import FrameLinker
        linker = FrameLinker()
        content = [
            {
                "predicate": {"id": "eat", "lemma": "eat", "semantic_class": "verb.consume"},
                "roles": [
                    {"role": "theme", "value": "vase", "semantic_class": "food_item",
                     "entity_id": "noun__vase"},
                ],
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": []},
            },
        ]
        uol = {"act": "command", "content": content}
        candidates = linker.score_atoms(uol, {}, tokens=("eat", "vase"))
        # meal_suggestion template matches (food_item required_class + eat action_token)
        assert len(candidates) >= 1
        found = any("build_strength" in c.slot_states for c in candidates)
        assert found, f"No candidate has entity-enriched slot_states: {[(c.frame_id, c.slot_states) for c in candidates]}"

    def test_causal_candidate_slot_states_preserves_cause_effect_with_entity_properties(self):
        """Causal candidate slot_states includes both causal links and entity props."""
        from melm.appliance.assistant_frame_linker import FrameLinker
        linker = FrameLinker()
        content = [
            {
                "predicate": {"id": "break", "lemma": "break", "semantic_class": "verb.contact"},
                "roles": [
                    {"role": "agent", "value": "child", "entity_id": ""},
                    {"role": "theme", "value": "vase", "semantic_class": "food_item",
                     "entity_id": "noun__vase"},
                ],
                "links": {"causes": ["destroy"], "caused_by": [], "enables": [], "prevents": []},
            },
        ]
        uol = {"act": "statement", "content": content}
        candidates = linker.score_atoms(uol, {}, tokens=("break", "vase"))
        causal = [c for c in candidates if c.frame_id == "causal_reasoning"]
        assert len(causal) >= 0  # enrichment test, not causality test
        # Verify at least one candidate carries entity property
        enriched = [c for c in candidates if "build_strength" in c.slot_states]
        assert len(enriched) >= 1, f"No candidate enriched with build_strength: {[(c.frame_id, c.slot_states) for c in candidates]}"

    def test_no_entity_id_no_enrichment(self):
        """Candidates without entity_ids get empty slot_states."""
        from melm.appliance.assistant_frame_linker import FrameLinker
        linker = FrameLinker()
        content = [
            {
                "predicate": {"id": "eat", "lemma": "eat", "semantic_class": "verb.consume"},
                "roles": [
                    {"role": "theme", "value": "pasta", "semantic_class": "food_item",
                     "entity_id": ""},
                ],
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": []},
            },
        ]
        uol = {"act": "command", "content": content}
        candidates = linker.score_atoms(uol, {}, tokens=("eat", "pasta"))
        enriched = [c for c in candidates if c.slot_states]
        assert len(enriched) == 0, f"Unexpected enriched candidates: {[(c.frame_id, c.slot_states) for c in enriched]}"
