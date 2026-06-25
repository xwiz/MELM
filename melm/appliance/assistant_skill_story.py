"""Story answer skill module — radial consumer of knowledge contracts."""

from __future__ import annotations

from typing import Any

from melm.contracts import load_story_components
from .assistant_skill_base import SkillManifest, register_skill

MANIFEST = SkillManifest(
    family="story",
    frames=("story",),
    knowledge_refs=("story_components.v1.json",),
    template_refs={},
)

register_skill(MANIFEST)


def _get_story_answer_templates() -> list[str]:
    if not hasattr(_get_story_answer_templates, "_cache"):
        components = load_story_components()
        _get_story_answer_templates._cache = list(
            components.get("answer_templates") or [
                "I picked {title} from the local story inventory{fit}. In {location}, {name} {image}. {challenge} By the end, {name} {lesson}",
                "Here is a story called {title}{fit}. {name} is in {location}, where {name} {image}. {challenge} At the end, {name} {lesson}",
                "Let me tell you {title}{fit}. Once, in {location}, {name} {image}. {challenge} In the end, {name} {lesson}",
            ]
        )
    return _get_story_answer_templates._cache


def format_story_answer(
    title: str,
    summary: str,
    topics: tuple[str, ...],
    cultures: tuple[str, ...],
    *,
    name: str,
    location: str,
    culture: str,
) -> str:
    image = _story_image(title, summary, topics)
    challenge = _story_challenge(topics, summary)
    lesson = _story_lesson(topics)
    fit = ""
    if culture and culture in cultures:
        fit = f" with a {culture} flavor"
    elif location and location in cultures:
        fit = f" in {location}"
    templates = _get_story_answer_templates()
    idx = (len(title) + len(summary)) % len(templates)
    return templates[idx].format(
        title=title, name=name, location=location,
        image=image, challenge=challenge, lesson=lesson, fit=fit,
    )


def format_story_frame(frame: str, *, name: str, location: str, culture: str) -> str:
    if not frame:
        return "a local story frame with enough metadata for a short safe adventure"
    try:
        return frame.format(name=name, location=location, culture=culture)
    except (KeyError, IndexError, ValueError):
        return frame


def _story_image(title: str, summary: str, topics: tuple[str, ...]) -> str:
    title_lower = title.lower()
    text = " ".join((title, summary, *topics)).lower()
    components = load_story_components()
    images = components.get("images", {})
    title_keywords = images.get("title_keywords", {})
    for keyword in title_keywords:
        if keyword in title_lower:
            return title_keywords[keyword]
    full_text_keywords = images.get("full_text_keywords", {})
    for keyword in full_text_keywords:
        if keyword in text:
            return full_text_keywords[keyword]
    return images.get("default", "")


def _story_challenge(topics: tuple[str, ...], summary: str) -> str:
    text = " ".join((*topics, summary)).lower()
    components = load_story_components()
    challenges = components.get("challenges", {})
    for keyword in challenges:
        if keyword in text:
            return challenges[keyword]
    return challenges.get("default", "")


def _story_lesson(topics: tuple[str, ...]) -> str:
    components = load_story_components()
    lessons = components.get("lessons", {})
    for topic in topics:
        if topic in lessons:
            return lessons[topic]
    return lessons.get("default", "")
