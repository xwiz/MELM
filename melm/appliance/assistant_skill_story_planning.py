"""UOL-driven story planner — pure Python, zero ML deps."""

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any

from .assistant_skill_base import SkillManifest

MANIFEST = SkillManifest(
    family="story",
    frames=("story",),
    knowledge_refs=("story_plan_schema.v1", "story_components.v1", "lesson_keywords.v1", "literary_device_map.v1"),
)


def _get_lesson_keywords() -> frozenset[str]:
    if not hasattr(_get_lesson_keywords, "_cache"):
        try:
            from melm.contracts import load_lesson_keywords
            _get_lesson_keywords._cache = frozenset(load_lesson_keywords())
        except Exception:
            _get_lesson_keywords._cache = frozenset({
                "patience", "kindness", "honesty", "bravery", "courage",
                "friendship", "sharing", "gratitude", "respect", "curiosity",
                "perseverance", "forgiveness", "generosity", "humility", "wisdom",
            })
    return _get_lesson_keywords._cache


def _get_literary_device_map() -> dict[str, Any]:
    if not hasattr(_get_literary_device_map, "_cache"):
        try:
            from melm.contracts import load_literary_device_map
            _get_literary_device_map._cache = load_literary_device_map()
        except Exception:
            _get_literary_device_map._cache = {
                "lesson_device_groups": {
                    "proverb": ["patience", "kindness", "honesty", "gratitude", "humility", "wisdom"],
                    "riddle": ["curiosity", "bravery", "perseverance"],
                    "poem": ["courage"],
                },
                "theme_default_device": "riddle",
                "default_device": "proverb",
            }
    return _get_literary_device_map._cache


@dataclass
class StoryPlan:
    lesson: str = ""
    themes: tuple[str, ...] = ()
    protagonist_name: str = ""
    scene_suggestion: int = 3
    length_guide: str = "medium"
    cultural_texture: str = ""
    setting_location: str = ""
    personal_facts: tuple[str, ...] = ()
    recent_context: tuple[str, ...] = ()
    literary_devices: tuple[str, ...] = ()
    mood_tone: str = "neutral"
    plan_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryPlan":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def compute_signature(self) -> str:
        canonical = (
            self.lesson,
            tuple(sorted(self.themes)),
            self.cultural_texture,
            tuple(sorted(self.literary_devices)),
        )
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def plan_story(
    utterance: str = "",
    functional_parse: dict[str, Any] | None = None,
    user_name: str = "",
    location: str = "",
    culture: str = "",
    age: int = 0,
    personal_facts: tuple[str, ...] = (),
    recent_context: tuple[str, ...] = (),
    valence: float = 0.0,
    arousal: float = 0.0,
) -> StoryPlan:
    lesson = _extract_lesson(utterance, functional_parse)
    themes = _extract_themes(utterance, functional_parse)
    scene_suggestion = _determine_scene_count(utterance, age)
    length_guide = _determine_length(utterance, age)
    literary_devices = _select_literary_devices(lesson, themes, culture)
    mood_tone = _compute_mood_tone(valence, arousal)
    plan = StoryPlan(
        lesson=lesson,
        themes=themes,
        protagonist_name=user_name or "a young child",
        scene_suggestion=scene_suggestion,
        length_guide=length_guide,
        cultural_texture=culture or "general",
        setting_location=location or "a quiet village",
        personal_facts=personal_facts,
        recent_context=recent_context,
        literary_devices=literary_devices,
        mood_tone=mood_tone,
    )
    plan.plan_signature = plan.compute_signature()
    return plan


def _extract_lesson(utterance: str, functional_parse: dict[str, Any] | None) -> str:
    lower = utterance.lower()
    keywords = _get_lesson_keywords()
    for word in keywords:
        if word in lower:
            return word
    if functional_parse:
        obj = (functional_parse.get("object") or "").lower()
        for word in keywords:
            if word in obj:
                return word
    return "kindness"


def _extract_themes(utterance: str, functional_parse: dict[str, Any] | None) -> tuple[str, ...]:
    lower = utterance.lower()
    themes: set[str] = set()
    keywords = _get_lesson_keywords()
    for word in keywords:
        if word in lower:
            themes.add(word)
    if functional_parse:
        obj = (functional_parse.get("object") or "").lower()
        if obj and obj not in themes:
            themes.add(obj)
        action = (functional_parse.get("action") or "").lower()
        if action and action not in themes:
            themes.add(action)
    return tuple(sorted(themes)) or ("adventure",)


def _determine_scene_count(utterance: str, age: int) -> int:
    lower = utterance.lower()
    if "short" in lower or "quick" in lower:
        return 2
    if "long" in lower or "epic" in lower:
        return 5
    return 3 if age <= 12 or age == 0 else 4


def _determine_length(utterance: str, age: int) -> str:
    lower = utterance.lower()
    if "short" in lower or "quick" in lower:
        return "short"
    if "long" in lower or "epic" in lower:
        return "long"
    return "short" if age <= 7 else "medium"


def _select_literary_devices(lesson: str, themes: tuple[str, ...], culture: str) -> tuple[str, ...]:
    device_map = _get_literary_device_map()
    groups = device_map.get("lesson_device_groups", {})
    devices: list[str] = []
    for device, lessons in groups.items():
        if lesson in lessons:
            devices.append(device)
    if "adventure" in themes and not devices:
        devices.append(device_map.get("theme_default_device", "riddle"))
    if culture and culture != "general":
        devices.append("proverb")
    return tuple(sorted(set(devices))) or (device_map.get("default_device", "proverb"),)


def _compute_mood_tone(valence: float, arousal: float) -> str:
    if valence > 0.2:
        tone = "warm"
    elif valence < -0.2:
        tone = "gentle"
    else:
        tone = "neutral"
    if arousal > 0.5:
        tone += "_curious"
    elif arousal < 0.3:
        tone += "_calm"
    return tone
