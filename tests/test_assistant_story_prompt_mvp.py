"""Tests for story prompt pipeline."""

import pytest
from melm.appliance.assistant_skill_story_planning import StoryPlan
from melm.appliance.assistant_story_prompt_pipeline import StoryPromptPipeline


def test_pipeline_returns_two_messages():
    plan = StoryPlan(lesson="patience", themes=("rain",), protagonist_name="Maya",
                     scene_suggestion=3, length_guide="medium", cultural_texture="yoruba",
                     setting_location="Lagos", personal_facts=("loves drums",),
                     recent_context=("asked about igbo vowels",), literary_devices=("proverb",),
                     mood_tone="warm_curious")
    messages = StoryPromptPipeline().build(plan)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_pipeline_system_contains_culture():
    messages = StoryPromptPipeline().build(StoryPlan(cultural_texture="yoruba"))
    assert "yoruba" in messages[0]["content"].lower()


def test_pipeline_system_contains_devices():
    messages = StoryPromptPipeline().build(StoryPlan(literary_devices=("proverb", "riddle")))
    assert "proverb" in messages[0]["content"]
    assert "riddle" in messages[0]["content"]


def test_pipeline_user_contains_lesson():
    messages = StoryPromptPipeline().build(StoryPlan(lesson="patience"))
    assert "patience" in messages[1]["content"].lower()


def test_pipeline_user_contains_location():
    messages = StoryPromptPipeline().build(StoryPlan(setting_location="Lagos"))
    assert "Lagos" in messages[1]["content"]


def test_pipeline_user_contains_personal_facts():
    messages = StoryPromptPipeline().build(StoryPlan(personal_facts=("loves drums",)))
    assert "drums" in messages[1]["content"]


def test_pipeline_user_contains_mood_tone():
    messages = StoryPromptPipeline().build(StoryPlan(mood_tone="warm_curious"))
    assert "warm" in messages[1]["content"].lower()


def test_pipeline_minimal_plan():
    messages = StoryPromptPipeline().build(StoryPlan())
    assert len(messages) == 2


def test_pipeline_system_includes_scene_count():
    messages = StoryPromptPipeline().build(StoryPlan(scene_suggestion=3))
    assert "3" in messages[0]["content"]


def test_pipeline_build_string():
    plan = StoryPlan(lesson="patience", literary_devices=("proverb",))
    messages = StoryPromptPipeline().build(plan)
    hint = StoryPromptPipeline().build_string(plan)
    assert isinstance(hint, str)
    assert len(hint) > 20
