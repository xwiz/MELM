"""Tests for the data-driven frame linker (E3/M5)."""

import unittest

from melm.appliance.assistant_frame_linker import FrameLinker


_LEXICON = {
    "rain": frozenset({"weather_phenomenon"}),
    "snow": frozenset({"weather_phenomenon"}),
    "today": frozenset({"temporal_descriptor"}),
    "call": frozenset({"contact_action", "communication_action"}),
    "phone": frozenset({"contact_action", "communication_action"}),
    "mom": frozenset({"social_relation"}),
    "tell": frozenset({"communication_action", "narrative_content"}),
    "story": frozenset({"narrative_content"}),
    "play": frozenset({"action", "media_content"}),
    "song": frozenset({"media_content"}),
    "hurt": frozenset({"health_condition"}),
    "eat": frozenset({"action", "food_item"}),
    "dinner": frozenset({"food_item"}),
    "remember": frozenset({"memory_recall"}),
    "forget": frozenset({"memory_recall"}),
    "define": frozenset({"definition_request"}),
    "weather": frozenset({"weather_phenomenon", "abstract_concept"}),
    "shirt": frozenset({"clothing_item"}),
    "park": frozenset({"public_place"}),
}


class FrameLinkerInitMvpTests(unittest.TestCase):
    """FrameLinker loads templates from contract at init."""

    def test_loads_all_templates(self) -> None:
        linker = FrameLinker()
        self.assertGreaterEqual(len(linker._templates), 9)

    def test_each_template_has_required_keys(self) -> None:
        linker = FrameLinker()
        for fid, tmpl in linker._templates.items():
            with self.subTest(template=fid):
                self.assertIn("frame_id", tmpl)
                self.assertIn("intent", tmpl)
                self.assertIn("family", tmpl)
                self.assertIn("activation", tmpl)
                self.assertIn("threshold", tmpl)
                self.assertIn("priority", tmpl)


class MatchRequiredClassesMvpTests(unittest.TestCase):
    """_match_required_classes returns full weight on any match, 0 otherwise."""

    def setUp(self) -> None:
        self.linker = FrameLinker()

    def test_matches_class_from_token(self) -> None:
        score = self.linker._match_required_classes({"rain"}, ["weather_phenomenon"], _LEXICON)
        self.assertAlmostEqual(score, 0.40)

    def test_no_match_returns_zero(self) -> None:
        score = self.linker._match_required_classes({"rain"}, ["narrative_content"], _LEXICON)
        self.assertAlmostEqual(score, 0.0)

    def test_empty_required_classes_returns_zero(self) -> None:
        score = self.linker._match_required_classes({"rain"}, [], _LEXICON)
        self.assertAlmostEqual(score, 0.0)

    def test_any_token_can_satisfy_any_of_required(self) -> None:
        score = self.linker._match_required_classes(
            {"mom", "rain"}, ["contact_action", "social_relation"], _LEXICON
        )
        self.assertAlmostEqual(score, 0.40)

    def test_missing_token_unknown_to_lexicon(self) -> None:
        score = self.linker._match_required_classes({"unknownword"}, ["weather_phenomenon"], _LEXICON)
        self.assertAlmostEqual(score, 0.0)

    def test_token_in_lexicon_but_not_required_class(self) -> None:
        score = self.linker._match_required_classes({"today"}, ["weather_phenomenon"], _LEXICON)
        self.assertAlmostEqual(score, 0.0)


class ExcludePenaltyMvpTests(unittest.TestCase):
    """_compute_exclude_penalty applies full penalty on ANY exclude-class match."""

    def setUp(self) -> None:
        self.linker = FrameLinker()

    def test_exclude_class_matched_applies_penalty(self) -> None:
        penalty = self.linker._compute_exclude_penalty({"define"}, ["definition_request"], _LEXICON)
        self.assertAlmostEqual(penalty, 0.25)

    def test_no_exclude_class_match_returns_zero(self) -> None:
        penalty = self.linker._compute_exclude_penalty({"rain"}, ["definition_request"], _LEXICON)
        self.assertAlmostEqual(penalty, 0.0)

    def test_empty_exclude_classes_returns_zero(self) -> None:
        penalty = self.linker._compute_exclude_penalty({"define"}, [], _LEXICON)
        self.assertAlmostEqual(penalty, 0.0)


class MatchActionTokensMvpTests(unittest.TestCase):
    """_match_action_tokens returns full weight if any token matches."""

    def setUp(self) -> None:
        self.linker = FrameLinker()

    def test_match_action_token(self) -> None:
        score = self.linker._match_action_tokens({"call", "mom"}, ["call", "phone"])
        self.assertAlmostEqual(score, 0.15)

    def test_no_match_returns_zero(self) -> None:
        score = self.linker._match_action_tokens({"mom"}, ["call"])
        self.assertAlmostEqual(score, 0.0)

    def test_empty_action_tokens_returns_zero(self) -> None:
        score = self.linker._match_action_tokens({"call"}, [])
        self.assertAlmostEqual(score, 0.0)


class MatchStructureMvpTests(unittest.TestCase):
    """_match_structure awards weight for questions and requests."""

    def setUp(self) -> None:
        self.linker = FrameLinker()

    def test_question_awards_structure(self) -> None:
        score = self.linker._match_structure(is_question_like=True, is_request_like=False)
        self.assertAlmostEqual(score, 0.15)

    def test_request_awards_structure(self) -> None:
        score = self.linker._match_structure(is_question_like=False, is_request_like=True)
        self.assertAlmostEqual(score, 0.15)

    def test_neither_returns_zero(self) -> None:
        score = self.linker._match_structure(is_question_like=False, is_request_like=False)
        self.assertAlmostEqual(score, 0.0)


class ScoreTemplateMvpTests(unittest.TestCase):
    """_score_template combines all sub-scores correctly."""

    def setUp(self) -> None:
        self.linker = FrameLinker()

    def test_weather_template_score(self) -> None:
        act = {
            "required_classes": ["weather_phenomenon"],
            "optional_classes": ["temporal_descriptor"],
            "exclude_classes": ["definition_request"],
            "action_tokens": [],
        }
        score, components = self.linker._score_template(
            {"rain", "today"}, act, _LEXICON,
            is_question_like=True, is_request_like=False,
        )
        # required(0.40) + optional(0.15) + action(0.00) + structure(0.15) = 0.70
        self.assertAlmostEqual(score, 0.70, places=4)
        self.assertIn("required", components)
        self.assertIn("optional", components)
        self.assertIn("action", components)
        self.assertIn("structure", components)

    def test_exclude_reduces_score(self) -> None:
        act = {
            "required_classes": ["weather_phenomenon"],
            "optional_classes": [],
            "exclude_classes": ["abstract_concept"],
            "action_tokens": [],
        }
        score, components = self.linker._score_template(
            {"weather"}, act, _LEXICON,
            is_question_like=False, is_request_like=True,
        )
        # required(0.40, "weather" → weather_phenomenon) + optional(0.00) + action(0.00) + structure(0.15, request) - exclude(0.25, "weather" → abstract_concept) = 0.30
        self.assertAlmostEqual(score, 0.30, places=4)

    def test_concept_question_lowered_by_exclude(self) -> None:
        act = {
            "required_classes": ["weather_phenomenon"],
            "optional_classes": [],
            "exclude_classes": ["definition_request"],
            "action_tokens": [],
        }
        score, components = self.linker._score_template(
            {"define", "weather"}, act, _LEXICON,
            is_question_like=True, is_request_like=False,
        )
        # required(0.40, "weather" → weather_phenomenon) + optional(0.00) + action(0.00) + structure(0.15) - exclude(0.25, "define" → definition_request) = 0.30
        self.assertAlmostEqual(score, 0.30, places=4)


class ScoreProducesCandidatesMvpTests(unittest.TestCase):
    """score() returns sorted candidates that meet threshold."""

    def setUp(self) -> None:
        self.linker = FrameLinker()

    def test_weather_tokens_score(self) -> None:
        candidates = self.linker.score(
            ("rain", "today"), _LEXICON,
            is_question_like=True, is_request_like=False,
        )
        self.assertTrue(any(c.frame_id == "weather" for c in candidates))

    def test_candidates_sorted_by_score_desc(self) -> None:
        candidates = self.linker.score(
            ("tell", "me", "a", "story"), _LEXICON,
            is_question_like=False, is_request_like=True,
        )
        for i in range(len(candidates) - 1):
            self.assertGreaterEqual(candidates[i].score, candidates[i + 1].score)

    def test_no_candidates_for_non_matching_tokens(self) -> None:
        candidates = self.linker.score(
            ("foo", "bar", "baz"), _LEXICON,
            is_question_like=False, is_request_like=False,
        )
        self.assertEqual(len(candidates), 0)

    def test_weather_concept_excluded_via_exclude_classes(self) -> None:
        candidates = self.linker.score(
            ("define", "weather"), _LEXICON,
            is_question_like=True, is_request_like=False,
        )
        weather = [c for c in candidates if c.frame_id == "weather"]
        self.assertEqual(len(weather), 0)

    def test_all_candidates_have_valid_frame_ids(self) -> None:
        candidates = self.linker.score(
            ("call", "mom"), _LEXICON,
            is_question_like=False, is_request_like=True,
        )
        self.assertGreater(len(candidates), 0)
        for c in candidates:
            self.assertIsInstance(c.frame_id, str)
            self.assertIsInstance(c.intent, str)
            self.assertGreaterEqual(c.score, 0.0)
            self.assertLessEqual(c.score, 1.0)
            self.assertIsInstance(c.score_components, dict)

    def test_action_tokens_boost_score(self) -> None:
        no_action = self.linker.score(
            ("song",), _LEXICON,
            is_question_like=False, is_request_like=False,
        )
        with_action = self.linker.score(
            ("play", "song"), _LEXICON,
            is_question_like=False, is_request_like=True,
        )
        media_no_action = [c for c in no_action if c.frame_id == "media_playback"]
        media_with_action = [c for c in with_action if c.frame_id == "media_playback"]
        self.assertGreater(len(media_no_action), 0, "media_playback must appear without action tokens")
        self.assertGreater(len(media_with_action), 0, "media_playback must appear with action tokens")
        self.assertGreater(media_with_action[0].score, media_no_action[0].score)


class FrameCandidateDataclassMvpTests(unittest.TestCase):
    """FrameCandidate dataclass stores all scoring fields."""

    def test_dataclass_attributes(self) -> None:
        from melm.appliance.assistant_frame_linker import FrameCandidate
        c = FrameCandidate(
            frame_id="test",
            intent="weather",
            score=0.65,
            score_components={"required": 0.40, "action": 0.15, "structure": 0.10},
            threshold=0.40,
        )
        self.assertEqual(c.frame_id, "test")
        self.assertEqual(c.intent, "weather")
        self.assertEqual(c.score, 0.65)
        self.assertEqual(c.threshold, 0.40)
        self.assertIn("required", c.score_components)
        with self.assertRaises(AttributeError):
            _ = c.nonexistent


class FrameTemplateSeedRowDeletionMvpTests(unittest.TestCase):
    """Deleting a frame template or semantic class changes behavior predictably."""

    def setUp(self) -> None:
        self.linker = FrameLinker()

    def test_deleting_weather_template_removes_weather_candidate(self) -> None:
        """Removing the 'weather' frame template stops 'rain' from routing to weather."""
        lexicon = {"rain": frozenset({"weather_phenomenon"})}
        tokens = ("will", "it", "rain", "tomorrow")
        before = self.linker.score(tokens, lexicon, is_question_like=True)
        self.assertTrue(any(c.frame_id == "weather" for c in before))

        # Delete the weather template
        del self.linker._templates["weather"]
        after = self.linker.score(tokens, lexicon, is_question_like=True)
        self.assertFalse(any(c.frame_id == "weather" for c in after))

    def test_deleting_story_template_removes_story_candidate(self) -> None:
        """Removing the 'story' frame template stops 'tell me a story' from matching story."""
        lexicon = {"story": frozenset({"narrative_content"})}
        tokens = ("tell", "me", "a", "story")
        before = self.linker.score(tokens, lexicon, is_request_like=True)
        self.assertTrue(any(c.frame_id == "story" for c in before))

        del self.linker._templates["story"]
        after = self.linker.score(tokens, lexicon, is_request_like=True)
        self.assertFalse(any(c.frame_id == "story" for c in after))

    def test_deleting_semantic_class_from_lexicon_drops_template_score(self) -> None:
        """Removing a required semantic class from the lexicon drops the matching template's score."""
        lexicon = {"rain": frozenset({"weather_phenomenon"})}
        tokens = ("rain", "today")
        before = self.linker.score(tokens, lexicon, is_question_like=True)
        weather_before = [c for c in before if c.frame_id == "weather"]
        self.assertEqual(len(weather_before), 1)
        before_score = weather_before[0].score

        # Remove weather_phenomenon from lexicon
        lexicon_no_weather = {"rain": frozenset()}
        after = self.linker.score(tokens, lexicon_no_weather, is_question_like=True)
        weather_after = [c for c in after if c.frame_id == "weather"]
        if weather_after:
            self.assertLess(weather_after[0].score, before_score)
        else:
            # If the score drops below threshold, the candidate is filtered out entirely
            self.assertEqual(len(weather_after), 0)


if __name__ == "__main__":
    unittest.main()
