"""Tests for story planning module and contract."""

import json
import pytest
from melm.contracts.validation import validate_story_plan_schema, load_story_plan_schema


def test_validate_story_plan_schema_valid():
    payload = {
        "schema_id": "melm.story_plan_schema.v1",
        "version": "1.0.0",
        "story_plan_fields": {
            "lesson": "string",
            "themes": "array:string",
        },
        "required": ["lesson", "themes"],
    }
    validate_story_plan_schema(payload)


def test_validate_story_plan_schema_wrong_schema_id():
    payload = {
        "schema_id": "melm.wrong_id",
        "version": "1.0.0",
        "story_plan_fields": {},
        "required": [],
    }
    with pytest.raises(Exception, match="schema_id"):
        validate_story_plan_schema(payload)


def test_validate_story_plan_schema_no_fields():
    payload = {
        "schema_id": "melm.story_plan_schema.v1",
        "version": "1.0.0",
    }
    with pytest.raises(Exception, match="story_plan_fields"):
        validate_story_plan_schema(payload)


def test_load_story_plan_schema():
    result = load_story_plan_schema()
    assert isinstance(result, dict)
    assert result["schema_id"] == "melm.story_plan_schema.v1"
    assert "story_plan_fields" in result


def test_validate_story_plan_schema_registered():
    from melm.contracts.validation import get_contract_info
    info = get_contract_info("story_plan_schema.v1")
    assert info is not None
    assert info["path"] == "story_plan_schema.v1.json"


# ---------------------------------------------------------------------------
# StoryPlan dataclass tests (Step 7)
# ---------------------------------------------------------------------------

from melm.appliance.assistant_skill_story_planning import StoryPlan, plan_story, MANIFEST


def test_story_plan_dataclass_defaults():
    plan = StoryPlan()
    assert plan.lesson == ""
    assert plan.scene_suggestion == 3
    assert plan.length_guide == "medium"
    assert plan.literary_devices == ()


def test_story_plan_to_dict_roundtrip():
    plan = StoryPlan(lesson="patience", themes=("rain",), literary_devices=("proverb",))
    plan.plan_signature = plan.compute_signature()
    data = plan.to_dict()
    assert data["lesson"] == "patience"
    restored = StoryPlan.from_dict(data)
    assert restored.lesson == "patience"
    assert restored.plan_signature == plan.plan_signature


def test_story_plan_signature_deterministic():
    a = StoryPlan(lesson="kindness", themes=("sharing",), cultural_texture="yoruba", literary_devices=("proverb",))
    b = StoryPlan(lesson="kindness", themes=("sharing",), cultural_texture="yoruba", literary_devices=("proverb",))
    assert a.compute_signature() == b.compute_signature()


def test_story_plan_signature_changes_with_lesson():
    a = StoryPlan(lesson="kindness", themes=("sharing",), cultural_texture="yoruba")
    b = StoryPlan(lesson="patience", themes=("sharing",), cultural_texture="yoruba")
    assert a.compute_signature() != b.compute_signature()


def test_skill_manifest():
    assert MANIFEST.family == "story"
    assert "story" in MANIFEST.frames
    assert "story_plan_schema.v1" in MANIFEST.knowledge_refs


# ---------------------------------------------------------------------------
# Planner heuristic tests (Step 11)
# ---------------------------------------------------------------------------


def test_plan_story_lesson_extraction_from_utterance():
    plan = plan_story(utterance="Tell me a story about rain that teaches patience")
    assert plan.lesson == "patience"


def test_plan_story_scene_count_short():
    plan = plan_story(utterance="Tell me a short story")
    assert plan.scene_suggestion == 2


def test_plan_story_devices_for_patience():
    plan = plan_story(utterance="Tell me a story about patience")
    assert "proverb" in plan.literary_devices


def test_plan_story_devices_for_curiosity():
    plan = plan_story(utterance="Tell me a story about curiosity")
    assert "riddle" in plan.literary_devices


def test_plan_story_mood_tone_warm():
    plan = plan_story(utterance="Tell me a story", valence=0.5, arousal=0.6)
    assert "warm" in plan.mood_tone


def test_plan_story_mood_tone_gentle():
    plan = plan_story(utterance="Tell me a story", valence=-0.5, arousal=0.3)
    assert "gentle" in plan.mood_tone


def test_plan_story_cultural_texture():
    plan = plan_story(utterance="Tell me a story", culture="yoruba")
    assert plan.cultural_texture == "yoruba"
    assert "proverb" in plan.literary_devices


def test_plan_story_personal_facts():
    plan = plan_story(utterance="Tell me a story", personal_facts=("loves drums",))
    assert "loves drums" in plan.personal_facts


def test_plan_story_signature_computed():
    plan = plan_story(utterance="Tell me a story about patience", culture="yoruba")
    assert len(plan.plan_signature) == 16
    assert plan.plan_signature == plan.compute_signature()


def test_plan_story_deterministic():
    a = plan_story(utterance="Tell me a story about rain", culture="yoruba")
    b = plan_story(utterance="Tell me a story about rain", culture="yoruba")
    assert a.plan_signature == b.plan_signature


def test_plan_story_fallback_to_kindness():
    plan = plan_story(utterance="Tell me something")
    assert plan.lesson == "kindness"


def test_plan_story_length_short_for_young_child():
    plan = plan_story(utterance="Tell me a story", age=5)
    assert plan.length_guide == "short"


def test_extract_lesson_with_functional_parse():
    from melm.appliance.assistant_skill_story_planning import plan_story
    plan = plan_story(utterance="Tell me a story", functional_parse={"object": "bravery"})
    assert plan.lesson == "bravery"


def test_mood_tone_neutral():
    from melm.appliance.assistant_skill_story_planning import _compute_mood_tone
    assert _compute_mood_tone(0.0, 0.4) == "neutral"


def test_mood_tone_warm_curious():
    from melm.appliance.assistant_skill_story_planning import _compute_mood_tone
    assert _compute_mood_tone(0.5, 0.6) == "warm_curious"


def test_mood_tone_gentle_calm():
    from melm.appliance.assistant_skill_story_planning import _compute_mood_tone
    assert _compute_mood_tone(-0.5, 0.2) == "gentle_calm"
