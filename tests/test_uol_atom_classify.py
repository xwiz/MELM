"""Parity tests: _classify_from_atoms (atom-based) matches _classify_from_functional_parse."""
import os
import pytest
from melm.appliance.local_assistant_router import (
    _classify_from_atoms,
    _classify_from_functional_parse,
    is_question_like_act,
    is_request_like_act,
)


class TestAtomClassifyParity:
    """_classify_from_atoms returns same intent as _classify_from_functional_parse for migrated intents."""

    def _make_act(self, act_type, pred_id, sem_class="unknown", themes=None, agents=None):
        roles = [{"role": "predicate", "value": pred_id, "status": "asserted", "confidence": 1.0}]
        for a in (agents or []):
            roles.append({"role": "agent", "value": a, "status": "asserted", "confidence": 0.9})
        for t in (themes or []):
            roles.append({"role": "theme", "value": t, "status": "asserted", "confidence": 0.88})
        return {
            "id": "uol_test",
            "act": act_type,
            "speaker": "user",
            "addressee": "assistant",
            "content": [{
                "id": "uol_atom_test",
                "kind": "event",
                "predicate": {"id": pred_id, "semantic_class": sem_class, "lemma": pred_id, "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "assertive", "negation_scope": False, "tense": "present"},
            }],
            "expected_answer_type": None,
        }

    def test_meal_suggestion(self):
        act = self._make_act("question", "eat", agents=["user"])
        tokens = ("what", "should", "i", "eat")
        assert _classify_from_atoms(act, tokens, "what should i eat?") == "meal_suggestion"

    def test_story_request(self):
        act = self._make_act("request", "tell", themes=["story"])
        tokens = ("tell", "me", "a", "story")
        assert _classify_from_atoms(act, tokens, "tell me a story") == "story"

    def test_weather_question(self):
        act = self._make_act("question", "rain")
        tokens = ("will", "it", "rain", "today")
        assert _classify_from_atoms(act, tokens, "will it rain today?") == "weather"

    def test_media_playback(self):
        act = self._make_act("command", "play", themes=["song"])
        tokens = ("play", "a", "song")
        assert _classify_from_atoms(act, tokens, "play a song") == "media_playback"

    def test_health_urgent_overrides(self):
        act = self._make_act("claim", "hurt")
        tokens = ("chest", "pain")
        assert _classify_from_atoms(act, tokens, "chest pain") == "health_advice"

    def test_none_act_returns_none(self):
        assert _classify_from_atoms(None, ("hello",), "hello") is None

    def test_empty_content_returns_none(self):
        act = {"id": "x", "act": "claim", "speaker": "user", "addressee": "assistant", "content": [], "expected_answer_type": None}
        assert _classify_from_atoms(act, ("hmm",), "hmm") is None

    def test_health_semantic_class(self):
        act = self._make_act("question", "treat", sem_class="health_domain")
        tokens = ("how", "to", "treat")
        assert _classify_from_atoms(act, tokens, "how to treat?") == "health_advice"

    def test_meal_suggestion_cook_verb(self):
        act = self._make_act("request", "cook", agents=["user"])
        tokens = ("what", "should", "i", "cook")
        assert _classify_from_atoms(act, tokens, "what should i cook?") == "meal_suggestion"

    def test_weather_forecast_predicate(self):
        act = self._make_act("question", "forecast")
        tokens = ("what", "is", "the", "forecast")
        assert _classify_from_atoms(act, tokens, "what is the forecast?") == "weather"

    def test_media_playback_music_theme(self):
        act = self._make_act("request", "play", themes=["music"])
        tokens = ("play", "music")
        assert _classify_from_atoms(act, tokens, "play music") == "media_playback"

    def test_story_tale_theme(self):
        act = self._make_act("request", "tell", themes=["tale"])
        tokens = ("tell", "me", "a", "tale")
        assert _classify_from_atoms(act, tokens, "tell me a tale") == "story"

    def test_negated_media_command_does_not_trigger_media_playback(self):
        act = self._make_act("command", "play", themes=["song"])
        act["content"][0]["context"]["polarity"] = "negative"
        act["content"][0]["context"]["negation_scope"] = True
        tokens = ("do", "not", "play", "a", "song")
        assert _classify_from_atoms(act, tokens, "do not play a song") is None

    def test_counterfactual_story_request_does_not_trigger_story(self):
        act = self._make_act("request", "tell", themes=["story"])
        act["content"][0]["context"]["modality"] = "counterfactual"
        tokens = ("if", "you", "could", "tell", "me", "a", "story")
        assert _classify_from_atoms(act, tokens, "if you could tell me a story") is None

    def test_classify_from_functional_parse_is_callable(self):
        """Renamed function is still importable and returns None for None parse."""
        result = _classify_from_functional_parse(None, ("hello",), "hello")
        # With None functional_parse and no urgent tokens, should return None
        assert result is None

    def test_classify_from_functional_parse_urgent_health(self):
        """Renamed function still handles urgent health tokens."""
        result = _classify_from_functional_parse(None, ("chest", "pain"), "chest pain")
        assert result == "health_advice"


class TestActHelpers:
    def test_is_question_like_act_question(self):
        assert is_question_like_act({"act": "question"}) is True

    def test_is_question_like_act_request(self):
        assert is_question_like_act({"act": "request"}) is False

    def test_is_question_like_act_none(self):
        assert is_question_like_act(None) is False

    def test_is_question_like_act_claim(self):
        assert is_question_like_act({"act": "claim"}) is False

    def test_is_question_like_act_command(self):
        assert is_question_like_act({"act": "command"}) is False

    def test_is_request_like_act_request(self):
        assert is_request_like_act({"act": "request"}) is True

    def test_is_request_like_act_command(self):
        assert is_request_like_act({"act": "command"}) is True

    def test_is_request_like_act_question(self):
        assert is_request_like_act({"act": "question"}) is False

    def test_is_request_like_act_none(self):
        assert is_request_like_act(None) is False

    def test_is_request_like_act_claim(self):
        assert is_request_like_act({"act": "claim"}) is False

    def test_is_question_like_act_empty_dict(self):
        assert is_question_like_act({}) is False

    def test_is_request_like_act_empty_dict(self):
        assert is_request_like_act({}) is False
