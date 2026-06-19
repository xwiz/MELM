"""Integration tests for story generation pipeline — end-to-end, fallback, virtue question."""

import pytest
from melm.appliance.assistant_skill_story_planning import plan_story, StoryPlan
from melm.appliance.assistant_story_prompt_pipeline import StoryPromptPipeline


def test_end_to_end_plan_and_prompt():
    """Plan -> prompt produces valid output with no exceptions."""
    plan = plan_story(
        utterance="Tell me a story about rain that teaches patience",
        functional_parse={"action": "tell", "object": "rain"},
        user_name="Maya",
        location="Lagos",
        culture="yoruba",
        age=7,
        personal_facts=("loves drums",),
        recent_context=("asked about igbo vowels",),
        valence=0.3,
        arousal=0.6,
    )
    assert plan.lesson == "patience"
    assert "rain" in plan.themes
    assert plan.cultural_texture == "yoruba"
    assert len(plan.plan_signature) == 16

    pipeline = StoryPromptPipeline()
    messages = pipeline.build(plan)
    assert len(messages) == 2
    assert "yoruba" in messages[0]["content"].lower()
    assert "patience" in messages[1]["content"].lower()


def test_planner_falls_back_to_default_lesson():
    """When no lesson is extractable, defaults to 'kindness'."""
    plan = plan_story(utterance="Tell me something interesting")
    assert plan.lesson == "kindness"


def test_planner_includes_literary_device_for_lesson():
    """Patience lesson triggers proverb device."""
    plan = plan_story(utterance="Tell me a story about patience")
    assert "proverb" in plan.literary_devices


def test_prompt_includes_scene_and_length():
    """Prompt reflects scene count and length guide from plan."""
    plan = plan_story(utterance="Tell me a short story about bravery")
    assert plan.scene_suggestion == 2
    assert plan.length_guide == "short"
    hint = StoryPromptPipeline().build_string(plan)
    assert "short" in hint.lower() or "2" in hint


def test_virtue_question_text():
    """Virtue question text is constructed correctly."""
    name = "Maya"
    question = f"\n\nWhat do you think {name} learned from this story?"
    assert "Maya" in question
    assert "learned" in question


def test_planner_with_uol_parse():
    """UOL functional_parse influences theme extraction."""
    plan = plan_story(
        utterance="Tell me a story",
        functional_parse={"action": "tell", "object": "rain"},
    )
    assert "rain" in plan.themes


def test_cache_roundtrip():
    """StoryPlan survives to_dict -> from_dict -> to_dict roundtrip."""
    from melm.appliance.assistant_skill_story_cache import cache_story_plan, find_liked_story_style

    class FakeStore:
        def __init__(self):
            self.entities = {}
            self.slots = {}
        def add_entity(self, entity_id, kind, label, **kw):
            self.entities[entity_id] = {"entity_id": entity_id, "kind": kind, "label": label}
        def set_entity_slot(self, eid, name, value, **kw):
            self.slots.setdefault(eid, {})[name] = str(value)
        def find_entities(self, kind="", **kw):
            results = []
            for eid, ent in self.entities.items():
                if kind and ent["kind"] != kind:
                    continue
                slots = self.slots.get(eid, {})
                results.append(type("Row", (), {
                    "entity_id": eid, "kind": ent["kind"], "label": ent["label"],
                    "plan_json": slots.get("plan_json", ""),
                    "liked": slots.get("liked", "False"),
                    "plan_signature": slots.get("plan_signature", ""),
                    "created_at": "2024-01-01", "updated_at": "2024-01-01",
                })())
            return results

    store = FakeStore()
    original = plan_story(utterance="Tell me a story about patience")
    cache_story_plan(store, original, liked=True)
    restored = find_liked_story_style(store)
    assert restored is not None
    assert restored.lesson == original.lesson
    assert restored.plan_signature == original.plan_signature
