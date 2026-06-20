"""Tests for story cache — entity store CRUD for story_style kind."""

import pytest
from melm.appliance.assistant_skill_story_planning import StoryPlan
from melm.appliance.assistant_skill_story_cache import (
    cache_story_plan, find_liked_story_style, STORY_STYLE_KIND,
)


class FakeStore:
    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.slots: dict[str, dict[str, str]] = {}

    def add_entity(self, entity_id, kind, label, semantic_class_id="", canonical_lemma=""):
        self.entities[entity_id] = {
            "entity_id": entity_id, "kind": kind, "label": label,
        }

    def set_entity_slot(self, entity_id, slot_name, value, **kw):
        self.slots.setdefault(entity_id, {})[slot_name] = str(value)

    def _row(self, eid, ent, slots):
        return type("Row", (), {
            "entity_id": eid, "kind": ent["kind"], "label": ent["label"],
            "plan_json": slots.get("plan_json", ""),
            "liked": slots.get("liked", "False"),
            "plan_signature": slots.get("plan_signature", ""),
            "created_at": slots.get("created_at", "2024-01-01"),
            "updated_at": slots.get("updated_at", "2024-01-01"),
        })()

    def find_entities(self, kind="", semantic_class_id=""):
        results = []
        for eid, ent in self.entities.items():
            if kind and ent["kind"] != kind:
                continue
            slots = self.slots.get(eid, {})
            results.append(self._row(eid, ent, slots))
        return results


def test_cache_story_plan_saves_entity():
    store = FakeStore()
    eid = cache_story_plan(store, StoryPlan(lesson="patience"), liked=True)
    assert eid in store.entities
    assert store.entities[eid]["kind"] == STORY_STYLE_KIND


def test_cache_story_plan_saves_signature():
    store = FakeStore()
    plan = StoryPlan(lesson="patience", plan_signature="sig123")
    eid = cache_story_plan(store, plan, liked=True)
    assert store.slots[eid].get("plan_signature") == "sig123"


def test_cache_story_plan_saves_plan_json():
    store = FakeStore()
    plan = StoryPlan(lesson="patience", literary_devices=("proverb",))
    plan.plan_signature = plan.compute_signature()
    eid = cache_story_plan(store, plan, liked=True)
    assert "patience" in store.slots[eid].get("plan_json", "")
    assert "proverb" in store.slots[eid].get("plan_json", "")


def test_find_liked_story_style_returns_none_when_empty():
    assert find_liked_story_style(FakeStore()) is None


def test_find_liked_story_style_returns_liked():
    store = FakeStore()
    cache_story_plan(store, StoryPlan(lesson="patience", plan_signature="s1"), liked=True)
    result = find_liked_story_style(store)
    assert result is not None
    assert result.lesson == "patience"


def test_find_liked_story_style_ignores_disliked():
    store = FakeStore()
    cache_story_plan(store, StoryPlan(lesson="kindness", plan_signature="s1"), liked=False)
    cache_story_plan(store, StoryPlan(lesson="patience", plan_signature="s2"), liked=True)
    result = find_liked_story_style(store)
    assert result is not None
    assert result.lesson == "patience"


def test_find_liked_story_style_most_recent():
    store = FakeStore()
    a_id = cache_story_plan(store, StoryPlan(lesson="patience", plan_signature="s1"), liked=True)
    b_id = cache_story_plan(store, StoryPlan(lesson="kindness", plan_signature="s2"), liked=True)
    store.slots[a_id]["updated_at"] = "2024-01-01"
    store.slots[b_id]["updated_at"] = "2024-06-01"
    result = find_liked_story_style(store)
    assert result is not None
    assert result.lesson == "kindness"


def test_find_liked_story_style_chronological_order():
    store = FakeStore()
    a_id = cache_story_plan(store, StoryPlan(lesson="patience", plan_signature="s1"), liked=True)
    b_id = cache_story_plan(store, StoryPlan(lesson="kindness", plan_signature="s2"), liked=True)
    store.slots[a_id]["updated_at"] = "2024-01-01"
    store.slots[b_id]["updated_at"] = "2024-06-01"
    result = find_liked_story_style(store)
    assert result is not None
    assert result.lesson == "kindness"


def test_cache_fallback_to_memory():
    eid = cache_story_plan(None, StoryPlan(lesson="patience"), liked=True)
    assert eid == "memory"
    result = find_liked_story_style(None)
    assert result is not None
    assert result.lesson == "patience"
