"""Story plan caching — entity store CRUD for story_style kind."""

import json
import uuid
from typing import Any

from .assistant_skill_story_planning import StoryPlan

STORY_STYLE_KIND = "story_style"

_memory_cache: list[dict[str, Any]] = []


def cache_story_plan(store: Any | None, plan: StoryPlan, liked: bool) -> str:
    """Save a StoryPlan to the entity store (or memory fallback)."""
    if store is None:
        _memory_cache.append(plan.to_dict() | {"liked": liked})
        return "memory"
    entity_id = str(uuid.uuid4())
    store.add_entity(entity_id=entity_id, kind=STORY_STYLE_KIND,
                     label=f"story_{plan.lesson}_{plan.plan_signature}")
    store.set_entity_slot(entity_id, "plan_signature", plan.plan_signature)
    store.set_entity_slot(entity_id, "plan_json", json.dumps(plan.to_dict()))
    store.set_entity_slot(entity_id, "liked", str(liked))
    store.set_entity_slot(entity_id, "lesson", plan.lesson)
    return entity_id


def find_liked_story_style(store: Any | None) -> StoryPlan | None:
    """Find the most recent liked story plan."""
    if store is None:
        liked = [i for i in _memory_cache if i.get("liked")]
        if not liked:
            return None
        return StoryPlan.from_dict(liked[-1])
    entities = store.find_entities(kind=STORY_STYLE_KIND)
    liked_plans = []
    for e in entities:
        liked_val = getattr(e, "liked", "False")
        if liked_val == "True" and hasattr(e, "plan_json") and e.plan_json:
            liked_plans.append(e)
    if not liked_plans:
        return None
    liked_plans.sort(key=lambda e: getattr(e, "updated_at", ""), reverse=True)
    latest = liked_plans[0]
    try:
        return StoryPlan.from_dict(json.loads(latest.plan_json))
    except (json.JSONDecodeError, TypeError):
        return None
