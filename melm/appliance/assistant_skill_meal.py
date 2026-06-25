"""Meal suggestion skill module — radial consumer of knowledge contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from melm.contracts import load_food_tags, load_meal_scopes
from melm.appliance._utils import tokenize as _tokenize
from .language_adapters import get_adapter
from .assistant_skill_base import SkillManifest, register_skill

MANIFEST = SkillManifest(
    family="meal_suggestion",
    frames=("meal_suggestion",),
    knowledge_refs=("food_tags.v1.json", "meal_scopes.v1.json"),
    template_refs={},
)

register_skill(MANIFEST)


@dataclass(frozen=True)
class MealSuggestion:
    items: tuple[str, ...]
    backups: tuple[str, ...]
    reason_tags: tuple[str, ...]
    meal_scope: str
    warm_note: bool

    @property
    def phrase(self) -> str:
        return _natural_list(self.items) or "a simple meal"


def suggest_meal(
    foods: tuple[str, ...] | list[str],
    preferences: dict[str, str] | None = None,
    weather: str = "",
    utterance: str = "",
) -> MealSuggestion:
    inventory = tuple(
        dict.fromkeys(
            _clean_food_name(item) for item in foods if _clean_food_name(item)
        )
    )
    scope = _meal_scope(utterance)
    warm_weather = _weather_suggests_warm_food(weather)
    if not inventory:
        return MealSuggestion(
            items=("a simple meal",),
            backups=(),
            reason_tags=("empty_inventory_fallback",),
            meal_scope=scope,
            warm_note=warm_weather,
        )
    preference_text = " ".join(
        str(value) for value in (preferences or {}).values()
    ).lower()
    scored = sorted(
        (
            (
                _food_inventory_score(
                    food,
                    index=index,
                    preference_text=preference_text,
                    scope=scope,
                    warm_weather=warm_weather,
                    utterance=utterance,
                ),
                index,
                food,
            )
            for index, food in enumerate(inventory)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    selected_count = min(3, max(1, 2 if len(scored) > 1 else 1))
    selected = tuple(food for _, _, food in scored[:selected_count])
    backups = tuple(food for _, _, food in scored[selected_count : selected_count + 2])
    reason_tags = _meal_reason_tags(
        selected,
        scope=scope,
        warm_weather=warm_weather,
        preference_text=preference_text,
    )
    return MealSuggestion(
        items=selected,
        backups=backups,
        reason_tags=reason_tags,
        meal_scope=scope,
        warm_note=warm_weather
        and any(_food_tags(food) & {"warm", "staple", "protein"} for food in selected),
    )


def format_meal_answer(
    foods: list[str],
    weather: str,
    utterance: str,
    preferences: dict[str, str] | None = None,
    answer: str = "",
) -> str:
    choice = suggest_meal(foods, preferences=preferences, weather=weather, utterance=utterance)
    base = choice.phrase or _meal_phrase(answer)
    side = (
        f" A backup from the same inventory is "
        f"{_join_short_list(choice.backups, fallback='nothing else saved yet')}."
        if choice.backups
        else ""
    )
    note = ""
    if choice.warm_note:
        note = " It may rain, so something warm is sensible."
    inventory_text = _meal_inventory_text(foods, choice.items)
    if note or side:
        return f"You could eat {base}.{note}{side} I chose it from local food inventory: {inventory_text}."
    return (
        f"You could eat {base}. I chose it from local food inventory: {inventory_text}. "
        "That keeps the suggestion grounded in what is already available on this device."
    )


# -- helpers --

def _clean_food_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _normalize(text: str) -> str:
    adapter = get_adapter("en")
    if adapter is not None:
        return adapter.normalize(adapter.correct(str(text)))
    return " ".join(text.lower().strip().split())


def _has_token_sequence(tokens: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    if not sequence:
        return False
    width = len(sequence)
    return any(
        tokens[index : index + width] == sequence
        for index in range(0, len(tokens) - width + 1)
    )


def _meal_scope(utterance: str) -> str:
    tokens = set(_tokenize(_normalize(utterance)))
    scope_pairs = load_meal_scopes()
    for token, scope in scope_pairs:
        if token in tokens:
            return scope
    return "meal"


def _weather_suggests_warm_food(weather: str) -> bool:
    text = weather.lower()
    tokens = _tokenize(text)
    return any(
        _has_token_sequence(tokens, _tokenize(term))
        for term in ("rain", "cold", "storm", "snow", "wind")
    )


def _food_tags(food: str) -> set[str]:
    tokens = _tokenize(food.lower())
    tags: set[str] = set()
    mapping = load_food_tags()
    for marker, marker_tags in mapping.items():
        if _has_token_sequence(tokens, _tokenize(marker)):
            tags.update(marker_tags)
    return tags or {"food"}


def _food_inventory_score(
    food: str,
    index: int,
    preference_text: str,
    scope: str,
    warm_weather: bool,
    utterance: str,
) -> float:
    tags = _food_tags(food)
    utterance_tokens = set(_tokenize(_normalize(utterance)))
    score = 1.0 - index * 0.01
    preference_tokens = _tokenize(preference_text)
    food_tokens = _tokenize(food)
    if food_tokens and _has_token_sequence(preference_tokens, food_tokens):
        score += 1.2
    if scope == "breakfast":
        score += 0.7 * len(tags & {"breakfast", "protein", "fruit", "grain"})
    elif scope in {"lunch", "dinner", "cooking"}:
        score += 0.55 * len(tags & {"staple", "protein", "vegetable", "warm"})
    else:
        score += 0.4 * len(tags & {"staple", "protein", "fruit", "vegetable"})
    if warm_weather:
        score += 0.45 * len(tags & {"warm", "staple", "protein"})
    if utterance_tokens & {"healthy", "healthier", "light", "energy"}:
        score += 0.45 * len(tags & {"fruit", "vegetable", "protein"})
    return round(score, 3)


def _meal_reason_tags(
    selected: tuple[str, ...],
    scope: str,
    warm_weather: bool,
    preference_text: str,
) -> tuple[str, ...]:
    tags: list[str] = [f"scope:{scope}"]
    combined_tags = (
        set().union(*(_food_tags(food) for food in selected)) if selected else set()
    )
    for tag in ("protein", "staple", "fruit", "vegetable", "warm", "light"):
        if tag in combined_tags:
            tags.append(tag)
    if warm_weather:
        tags.append("weather:warm_food_helpful")
    preference_tokens = _tokenize(preference_text)
    if any(
        _has_token_sequence(preference_tokens, _tokenize(food)) for food in selected
    ):
        tags.append("preference_match")
    return tuple(dict.fromkeys(tags))


def _natural_list(items: tuple[str, ...]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _meal_phrase(answer: str) -> str:
    cleaned = " ".join(answer.strip().split()).strip(".")
    lowered = cleaned.lower()
    for prefix in ("you could eat ", "eat "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned.strip(".") or "a simple meal"


def _meal_inventory_text(foods: list[str], selected_items: tuple[str, ...]) -> str:
    selected = [food for food in foods if food.lower() in set(selected_items)]
    if not selected:
        selected = foods[:3]
    return _join_short_list(tuple(selected[:4]), fallback="your saved food items")


def _join_short_list(items: tuple[str, ...], *, fallback: str) -> str:
    if not items:
        return fallback
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"
