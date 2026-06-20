"""Tests for FrameLinker.score_atoms — atom-aware frame matching."""
import pytest
from melm.appliance.assistant_frame_linker import FrameLinker


def _make_act(act_type: str, pred_id: str, sem_class: str = "unknown", themes: list = None) -> dict:
    """Build a minimal serialized UolAct dict for testing."""
    roles = [{"role": "predicate", "value": pred_id, "status": "asserted", "confidence": 1.0}]
    for t in (themes or []):
        roles.append({"role": "theme", "value": t, "status": "asserted", "confidence": 0.88})
    return {
        "id": "uol_test",
        "act": act_type,
        "speaker": "user",
        "addressee": "assistant",
        "content": [
            {
                "id": "uol_atom",
                "kind": "event",
                "predicate": {"id": pred_id, "semantic_class": sem_class, "lemma": pred_id, "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "assertive", "negation_scope": False, "tense": "present"},
            }
        ],
        "expected_answer_type": None,
    }


class TestScoreAtomsBasic:
    def setup_method(self):
        self.linker = FrameLinker()
        self.lexicon = {}

    def test_empty_content_returns_empty(self):
        act = {"id": "x", "act": "question", "speaker": "user", "addressee": "assistant", "content": [], "expected_answer_type": None}
        assert self.linker.score_atoms(act, self.lexicon) == []

    def test_score_atoms_returns_list(self):
        act = _make_act("question", "eat", sem_class="verb.consume")
        result = self.linker.score_atoms(act, self.lexicon)
        assert isinstance(result, list)

    def test_meal_suggestion_via_semantic_class(self):
        # food_item class in lexicon matched through theme value
        lexicon = {"rice": frozenset({"food_item"})}
        act = _make_act("question", "eat", sem_class="unknown", themes=["rice"])
        candidates = self.linker.score_atoms(act, lexicon)
        frame_ids = [c.frame_id for c in candidates]
        assert any("meal" in fid for fid in frame_ids), f"Expected meal frame, got: {frame_ids}"

    def test_candidates_sorted_by_score_descending(self):
        lexicon = {"rice": frozenset({"food_item"})}
        act = _make_act("request", "eat", themes=["rice"])
        candidates = self.linker.score_atoms(act, lexicon)
        if len(candidates) >= 2:
            scores = [c.score for c in candidates]
            assert scores == sorted(scores, reverse=True)

    def test_igbo_predicate_id_matches_frame(self):
        """An Igbo predicate ID 'eri' (eat) should match meal_suggestion frame when action_tokens includes 'eat'.

        The Igbo predicate ID ('eri') may differ from English surface token ('eat').
        This test verifies the score_atoms path handles non-English pred IDs via lexicon class lookup
        rather than requiring English surface tokens.
        """
        # 'eri' is Igbo for eat; theme 'nri' is Igbo for food
        lexicon = {"nri": frozenset({"food_item"})}
        act = _make_act("question", "eri", sem_class="unknown", themes=["nri"])
        candidates = self.linker.score_atoms(act, lexicon)
        # Should find meal frame via food_item class on theme, even though pred_id is 'eri' not 'eat'
        meal_candidates = [c for c in candidates if "meal" in c.frame_id]
        assert len(meal_candidates) >= 1, (
            f"Expected meal_suggestion candidate for Igbo pred 'eri'/theme 'nri', got: {candidates}"
        )

    def test_score_atoms_uses_pred_id_for_action_tokens(self):
        """Predicate ID matches action_tokens in frame template."""
        # Most frames use English predicate IDs as action_tokens, e.g. "play" for media_playback
        lexicon = {"song": frozenset({"media_content"})}
        act = _make_act("command", "play", sem_class="unknown", themes=["song"])
        candidates = self.linker.score_atoms(act, lexicon)
        media_candidates = [c for c in candidates if "media" in c.frame_id]
        assert len(media_candidates) >= 1, f"Expected media_playback candidate, got: {candidates}"

    def test_score_atoms_respects_weather_concept_gate(self):
        """Atom scoring should preserve the weather definition blocker from the token linker."""
        lexicon = {"weather": frozenset({"weather_phenomenon"})}
        act = _make_act("question", "be", sem_class="verb.stative", themes=["weather"])
        candidates = self.linker.score_atoms(act, lexicon, tokens=("what", "is", "weather"))
        weather_candidates = [c for c in candidates if c.frame_id == "weather"]
        assert weather_candidates == []

    def test_score_atoms_applies_context_bonus(self):
        """Atom scoring should preserve context_score bonuses from the token linker."""
        lexicon = {"rice": frozenset({"food_item"})}
        act = _make_act("question", "eat", sem_class="verb.consume", themes=["rice"])
        candidates = self.linker.score_atoms(act, lexicon, tokens=("what", "should", "i", "eat"))
        meal_candidates = [c for c in candidates if c.frame_id == "meal_suggestion"]
        assert meal_candidates, f"Expected meal_suggestion candidate, got: {candidates}"
        assert meal_candidates[0].score > 0.55


class TestScoreAtomsScoreComponents:
    def setup_method(self):
        self.linker = FrameLinker()

    def test_score_components_present(self):
        lexicon = {"song": frozenset({"media_content"})}
        act = _make_act("command", "play", themes=["song"])
        candidates = self.linker.score_atoms(act, lexicon)
        if candidates:
            c = candidates[0]
            assert "required" in c.score_components
            assert "action" in c.score_components
            assert "optional" in c.score_components
            assert "structure" in c.score_components
