"""Micro probe for realistic on-device assistant routing.

The grounded child-room MVP proves the inner state/evidence mechanics. This
module uses a broader but still tiny assistant surface to compare MVP
directions against realistic user asks:

- answer locally from policy or memory;
- answer from cached tool data;
- trigger a device action;
- fetch missing non-LLM data;
- hand off only the right cases to a large model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .functional_grammar import (
    FunctionalParse,
    functional_frame_kind,
    parse_functional_relations,
)

AssistantIntent = Literal[
    "assistant_identity",
    "assistant_status",
    "story",
    "weather",
    "common_sense_safety",
    "media_playback",
    "health_advice",
    "personal_memory",
    "autobiographical_memory",
    "meal_suggestion",
    "social_contact",
    "social_greeting",
    "assistant_behavior",
    "personal_goal_advice",
    "open_domain",
    "unknown",
]
AssistantRoute = Literal[
    "local_answer",
    "cached_tool",
    "device_action",
    "external_fetch",
    "cloud_handoff",
    "clarify",
    "reject",
]

LOCAL_STATE_ROUTER_BASELINE = "local_state_router_no_lifecycle"


@dataclass(frozen=True)
class LocalAssistantProfile:
    user_name: str = "Maya"
    age: int = 7
    location: str = "Lagos"
    culture: str = "Yoruba"
    facts: dict[str, str] = field(
        default_factory=lambda: {
            "favorite_color": "green",
            "school": "you usually go to school on weekdays",
            "friend": "Leo is one of your trusted contacts",
        }
    )
    preferences: dict[str, str] = field(
        default_factory=lambda: {
            "breakfast": "eggs and fruit",
            "music": "calm piano",
        }
    )
    health_goals: tuple[str, ...] = ("sleep earlier", "walk after school")
    contacts: dict[str, str] = field(
        default_factory=lambda: {
            "mom": "+234-000-MOM",
            "leo": "+234-000-LEO",
        }
    )
    weekly_weather: dict[str, str] = field(
        default_factory=lambda: {
            "today": "warm with afternoon rain",
            "tomorrow": "cloudy and humid",
        }
    )
    story_models: dict[str, str] = field(
        default_factory=lambda: {
            "local_folk_tale": (
                "{name} found a talking drum in {location}. The drum taught "
                "{name} to share, listen, and come home before the rain."
            )
        }
    )
    media_library: tuple[str, ...] = ("calm piano", "rain sounds")
    food_inventory: tuple[str, ...] = ("rice", "beans", "eggs", "plantain", "fruit")


@dataclass(frozen=True)
class AssistantDecision:
    utterance: str
    intent: AssistantIntent
    route: AssistantRoute
    answer: str
    evidence_keys: tuple[str, ...] = ()
    cloud_needed: bool = False
    external_fetch_needed: bool = False
    privacy_exposure: bool = False
    local_memory_used: bool = False
    device_action: bool = False
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class AssistantFrameMatch:
    """Primary UOL/ChatFrame match selected by the assistant frame registry."""

    registry: str
    frame_id: str
    composition: dict[str, Any]
    source_policy: str = "primary_uol_chatframe_only"
    secondary_hint_policy: str = "debug_only_never_primary_route"

    def to_composition(self) -> dict[str, Any]:
        enriched = dict(self.composition)
        enriched["frame_registry"] = self.registry
        enriched["frame_id"] = self.frame_id
        enriched["source_policy"] = self.source_policy
        enriched["secondary_hint_policy"] = self.secondary_hint_policy
        enriched["frame_match"] = {
            "registry": self.registry,
            "frame_id": self.frame_id,
            "source": str(enriched.get("source", "")),
            "pattern": str(enriched.get("pattern", "")),
            "source_policy": self.source_policy,
            "secondary_hint_policy": self.secondary_hint_policy,
        }
        return enriched


class AssistantFrameRegistry:
    """Typed boundary for primary UOL/ChatFrame matches.

    Phrase and vocabulary markers may appear later as secondary debug hints,
    but a local route must first be owned by one of these frame matches.
    The match is not meant to be utterance-only: a scaled frame must also cite
    user/self memory, event history, inventories, action state, or world-atlas
    support before local/tool/action behavior is considered accepted.
    """

    registry_id = "melm.assistant_frame_registry.v1"
    source_policy = "primary_uol_chatframe_only"
    secondary_hint_policy = "debug_only_never_primary_route"

    @classmethod
    def match(
        cls,
        text: str,
        tokens: tuple[str, ...],
        intent: AssistantIntent,
    ) -> AssistantFrameMatch | None:
        if intent == "assistant_identity":
            composition = _identity_composition(text, tokens)
        elif intent == "assistant_status":
            composition = _self_status_composition(text, tokens)
        elif intent in {
            "social_greeting",
            "assistant_behavior",
            "personal_goal_advice",
            "open_domain",
        }:
            composition = _functional_relation_composition(text, tokens, intent)
        else:
            composition = _semantic_slot_composition(text, tokens, intent)
        if composition is None:
            return None
        return AssistantFrameMatch(
            registry=cls.registry_id,
            frame_id=_assistant_frame_id(composition),
            composition=composition,
            source_policy=cls.source_policy,
            secondary_hint_policy=cls.secondary_hint_policy,
        )


@dataclass(frozen=True)
class LocalMealChoice:
    items: tuple[str, ...]
    backups: tuple[str, ...]
    reason_tags: tuple[str, ...]
    meal_scope: str
    warm_note: bool

    @property
    def phrase(self) -> str:
        return _natural_list(self.items) or "a simple meal"


@dataclass(frozen=True)
class AssistantDebugParse:
    """Machine-readable parse trace for local assistant routing debug."""

    utterance: str
    normalized: str
    tokens: tuple[str, ...]
    uol: dict[str, Any]
    chat_frame: dict[str, Any]
    secondary_meaning_hints: tuple[str, ...]
    nlp: dict[str, Any] = field(default_factory=dict)
    mapping: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    schema: str = "melm.assistant_debug_parse.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "utterance": self.utterance,
            "normalized": self.normalized,
            "tokens": list(self.tokens),
            "uol": self.uol,
            "chat_frame": self.chat_frame,
            "secondary_meaning_hints": list(self.secondary_meaning_hints),
            "nlp": self.nlp,
            "mapping": list(self.mapping),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AssistantStrategyReport:
    strategy: str
    cases: int
    local_or_device_resolved: int
    cloud_handoffs: int
    external_fetches: int
    clarifications: int
    privacy_exposures: int
    memory_uses: int
    decisions: tuple[AssistantDecision, ...]

    @property
    def local_resolution_rate(self) -> float:
        return (
            round(self.local_or_device_resolved / self.cases, 3) if self.cases else 0.0
        )


class OnDeviceAssistantRouter:
    """Tiny deterministic assistant router over local memory/tool/action state."""

    def __init__(
        self,
        profile: LocalAssistantProfile | None = None,
        *,
        lexical_class_lookup: Callable[[str], frozenset[str]] | None = None,
        lexicon_owned_families: frozenset[str] = frozenset(),
    ) -> None:
        self.profile = profile or LocalAssistantProfile()
        self.lexical_class_lookup = lexical_class_lookup
        self.lexicon_owned_families = lexicon_owned_families

    def handle(self, utterance: str) -> AssistantDecision:
        text = _normalize(utterance)
        tokens = _tokenize(text)
        intent = _classify_intent_from_uol_slots(
            text,
            tokens,
            trusted_contact_names=tuple(self.profile.contacts),
            lexical_class_lookup=self.lexical_class_lookup,
            lexicon_owned_families=self.lexicon_owned_families,
        )
        if intent == "personal_memory" and _is_private_cloud_export_request(
            text, tokens
        ):
            evidence_keys = _private_cloud_evidence_keys(
                text, trusted_contact_names=tuple(self.profile.contacts)
            )
            return AssistantDecision(
                utterance=utterance,
                intent="personal_memory",
                route="cloud_handoff",
                answer="Hand off to a larger model.",
                evidence_keys=evidence_keys,
                cloud_needed=True,
                privacy_exposure=True,
                confidence=0.52,
                reason="private_memory_cloud_request",
            )
        if intent == "story":
            return self._story(utterance)
        if intent == "assistant_identity":
            return self._assistant_identity(utterance)
        if intent == "assistant_status":
            return self._assistant_status(utterance)
        if intent == "weather":
            return self._weather(utterance)
        if intent == "common_sense_safety":
            return self._safety(utterance)
        if intent == "media_playback":
            return self._media(utterance)
        if intent == "health_advice":
            return self._health(utterance)
        if intent == "personal_memory":
            return self._personal_memory(utterance)
        if intent == "meal_suggestion":
            return self._meal(utterance)
        if intent == "social_contact":
            return self._contact(utterance)
        if intent == "social_greeting":
            return self._greeting(utterance)
        if intent == "assistant_behavior":
            return self._assistant_behavior(utterance)
        if intent == "personal_goal_advice":
            return _cloud(utterance, intent, reason="understood_personal_goal_advice")
        if intent == "open_domain":
            return _cloud(utterance, intent, reason="understood_open_domain")
        return AssistantDecision(
            utterance=utterance,
            intent="unknown",
            route="cloud_handoff",
            answer="I should ask the larger model to interpret that.",
            cloud_needed=True,
            confidence=0.2,
            reason="unknown_intent",
        )

    def _greeting(self, utterance: str) -> AssistantDecision:
        return AssistantDecision(
            utterance=utterance,
            intent="social_greeting",
            route="local_answer",
            answer="Hi. What would you like help with?",
            evidence_keys=("self_model.purpose",),
            confidence=0.98,
            reason="local_social_greeting",
        )

    def _assistant_behavior(self, utterance: str) -> AssistantDecision:
        return AssistantDecision(
            utterance=utterance,
            intent="assistant_behavior",
            route="local_answer",
            answer=(
                "I can repeat an answer when the same grounded evidence applies, "
                "but I should adapt when your meaning or context changes."
            ),
            evidence_keys=("self_model.purpose", "self_model.limits"),
            local_memory_used=True,
            confidence=0.9,
            reason="self_model_response_behavior",
        )

    def _assistant_identity(self, utterance: str) -> AssistantDecision:
        return AssistantDecision(
            utterance=utterance,
            intent="assistant_identity",
            route="local_answer",
            answer=(
                "I am MELM Local Assistant OS. I run local-first on this device, "
                "using local memory, cached tools, and confirmed actions before "
                "asking a larger model."
            ),
            evidence_keys=(
                "self_model.name",
                "self_model.purpose",
                "self_model.local_capabilities",
                "self_model.limits",
            ),
            local_memory_used=True,
            confidence=0.97,
            reason="self_model_identity",
        )

    def _assistant_status(self, utterance: str) -> AssistantDecision:
        return AssistantDecision(
            utterance=utterance,
            intent="assistant_status",
            route="local_answer",
            answer=(
                "I can report my runtime status from the local ledger when the "
                "assistant OS kernel has a store attached."
            ),
            evidence_keys=(
                "self_model.name",
                "self_model.local_capabilities",
                "self_status.no_store",
            ),
            local_memory_used=True,
            confidence=0.76,
            reason="self_status_no_ledger",
        )

    def _story(self, utterance: str) -> AssistantDecision:
        requested_constraints = _requested_story_constraints(utterance)
        if self.profile.story_models:
            matching_story = _matching_story_model(
                self.profile.story_models,
                requested_constraints,
            )
            if matching_story is None:
                available = _available_story_inventory_label(self.profile.story_models)
                constraint_text = ", ".join(sorted(requested_constraints))
                return AssistantDecision(
                    utterance=utterance,
                    intent="story",
                    route="clarify",
                    answer=(
                        "I do not have a local story that matches"
                        f" {constraint_text}. I can tell {available}, or I can ask for help."
                    ),
                    evidence_keys=tuple(
                        f"story_models.{story_key}"
                        for story_key in self.profile.story_models
                    ),
                    local_memory_used=True,
                    confidence=0.72,
                    reason="story_constraint_unmet",
                )
            story_key, frame = matching_story
            answer = _render_story_frame(
                frame,
                name=self.profile.user_name,
                location=self.profile.location,
                culture=self.profile.culture,
            )
            return AssistantDecision(
                utterance=utterance,
                intent="story",
                route="local_answer",
                answer=answer,
                evidence_keys=(f"story_models.{story_key}", "profile.location"),
                local_memory_used=True,
                confidence=0.86,
                reason="local_story_inventory",
            )
        return _cloud(utterance, "story", reason="missing_story_model")

    def _weather(self, utterance: str) -> AssistantDecision:
        cached = self.profile.weekly_weather.get("today")
        if cached:
            return AssistantDecision(
                utterance=utterance,
                intent="weather",
                route="cached_tool",
                answer=f"Today in {self.profile.location}: {cached}.",
                evidence_keys=("weekly_weather.today", "profile.location"),
                local_memory_used=True,
                confidence=0.94,
                reason="weather_cache_hit",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="weather",
            route="external_fetch",
            answer="I need to fetch today's weather first.",
            external_fetch_needed=True,
            confidence=0.88,
            reason="weather_cache_miss",
        )

    def _safety(self, utterance: str) -> AssistantDecision:
        text = _normalize(utterance)
        tokens = _tokenize(text)
        token_set = set(tokens)
        if (
            token_set & {"wear", "clothes", "coat", "raincoat"}
            and "school" in token_set
            and "naked" not in token_set
        ):
            weather = self.profile.weekly_weather.get("today")
            if not weather:
                return AssistantDecision(
                    utterance=utterance,
                    intent="common_sense_safety",
                    route="external_fetch",
                    answer="I should fetch the weather before giving school clothing advice.",
                    external_fetch_needed=True,
                    confidence=0.84,
                    reason="clothing_needs_weather_cache",
                )
            return AssistantDecision(
                utterance=utterance,
                intent="common_sense_safety",
                route="local_answer",
                answer="Wear school clothes and carry rain protection if the forecast mentions rain.",
                evidence_keys=("weekly_weather.today", "facts.school"),
                local_memory_used=True,
                confidence=0.91,
                reason="school_clothing_weather_policy",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="common_sense_safety",
            route="local_answer",
            answer="No. Wear proper clothes before going to school.",
            evidence_keys=("local_safety_policy.clothing_public_school",),
            confidence=0.99,
            reason="local_common_sense_policy",
        )

    def _media(self, utterance: str) -> AssistantDecision:
        text = _normalize(utterance)
        if not self.profile.media_library:
            return AssistantDecision(
                utterance=utterance,
                intent="media_playback",
                route="clarify",
                answer="I do not see any local songs yet. Which music app should I use?",
                confidence=0.58,
                reason="empty_media_library",
            )
        requested = _requested_media(text, self.profile.media_library)
        song = (
            requested
            or self.profile.preferences.get("music")
            or self.profile.media_library[0]
        )
        if song in self.profile.media_library:
            return AssistantDecision(
                utterance=utterance,
                intent="media_playback",
                route="device_action",
                answer=f"Playing {song}.",
                evidence_keys=("preferences.music", "media_library"),
                local_memory_used=True,
                device_action=True,
                confidence=0.9,
                reason="local_media_action",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="media_playback",
            route="clarify",
            answer="Which song should I play?",
            confidence=0.62,
            reason="missing_media_choice",
        )

    def _health(self, utterance: str) -> AssistantDecision:
        text = _normalize(utterance)
        tokens = _tokenize(text)
        if _has_urgent_health_frame(tokens):
            return AssistantDecision(
                utterance=utterance,
                intent="health_advice",
                route="local_answer",
                answer=(
                    "This sounds urgent. Call emergency services or a trusted adult now. "
                    "I cannot diagnose you, but I should not delay help."
                ),
                evidence_keys=("local_health_safety_policy",),
                confidence=0.95,
                reason="urgent_health_safety_escalation",
            )
        goals = ", ".join(self.profile.health_goals)
        return AssistantDecision(
            utterance=utterance,
            intent="health_advice",
            route="local_answer",
            answer=(
                "For general health, start with water, sleep, movement, and "
                f"your current goals: {goals}. For pain, danger, or illness, "
                "talk to a trusted adult or clinician."
            ),
            evidence_keys=("health_goals", "local_health_safety_policy"),
            local_memory_used=True,
            confidence=0.78,
            reason="bounded_general_health_guidance",
        )

    def _personal_memory(self, utterance: str) -> AssistantDecision:
        text = _normalize(utterance)
        tokens = _tokenize(text)
        if _is_routine_memory_request(tokens) and not _has_routine_fact(self.profile):
            return AssistantDecision(
                utterance=utterance,
                intent="personal_memory",
                route="clarify",
                answer="I do not have a routine memory for you yet.",
                confidence=0.72,
                reason="personal_memory_empty",
            )
        if _is_child_memory_request(tokens) and not _has_child_fact(self.profile):
            return AssistantDecision(
                utterance=utterance,
                intent="personal_memory",
                route="clarify",
                answer="I do not have a child memory fact for you yet.",
                confidence=0.72,
                reason="personal_memory_empty",
            )
        if _is_household_memory_request(tokens) and not _has_household_fact(
            self.profile
        ):
            return AssistantDecision(
                utterance=utterance,
                intent="personal_memory",
                route="clarify",
                answer="I do not have household memory ownership set up yet.",
                confidence=0.72,
                reason="personal_memory_empty",
            )
        if _is_broad_personal_memory_request(tokens) and _has_personal_summary_memory(
            self.profile
        ):
            evidence_keys = _personal_summary_evidence_keys(self.profile)
            return AssistantDecision(
                utterance=utterance,
                intent="personal_memory",
                route="local_answer",
                answer="I can summarize a few local memories about you.",
                evidence_keys=evidence_keys,
                local_memory_used=True,
                confidence=0.88,
                reason="personal_memory_summary",
            )
        if self.profile.facts:
            if _is_routine_memory_request(tokens):
                fact_key, fact = _first_matching_fact(
                    self.profile, ("routine", "schedule")
                )
                if fact_key:
                    return AssistantDecision(
                        utterance=utterance,
                        intent="personal_memory",
                        route="local_answer",
                        answer=f"I remember that your {fact_key.replace('_', ' ')} is {fact}.",
                        evidence_keys=(f"facts.{fact_key}",),
                        local_memory_used=True,
                        confidence=0.91,
                        reason="personal_memory_recall",
                    )
            if _is_child_memory_request(tokens):
                child_markers = ("child", "son", "daughter")
                if "school" in set(tokens):
                    child_markers = ("child_school", "son_school", "daughter_school")
                elif set(tokens) & {"age", "old"}:
                    child_markers = ("child_age", "son_age", "daughter_age")
                elif set(tokens) & {"name", "called"}:
                    child_markers = ("child_name", "son_name", "daughter_name")
                fact_key, fact = _first_matching_fact(self.profile, child_markers)
                if fact_key:
                    return AssistantDecision(
                        utterance=utterance,
                        intent="personal_memory",
                        route="local_answer",
                        answer=f"I remember that your {fact_key.replace('_', ' ')} is {fact}.",
                        evidence_keys=(f"facts.{fact_key}",),
                        local_memory_used=True,
                        confidence=0.91,
                        reason="personal_memory_recall",
                    )
                return AssistantDecision(
                    utterance=utterance,
                    intent="personal_memory",
                    route="clarify",
                    answer="I do not have that child memory fact for you yet.",
                    confidence=0.72,
                    reason="personal_memory_empty",
                )
            if _is_household_memory_request(tokens):
                fact_key, fact = _first_matching_fact(
                    self.profile, ("household", "family")
                )
                if fact_key:
                    return AssistantDecision(
                        utterance=utterance,
                        intent="personal_memory",
                        route="local_answer",
                        answer=f"I remember that your {fact_key.replace('_', ' ')} is {fact}.",
                        evidence_keys=(f"facts.{fact_key}",),
                        local_memory_used=True,
                        confidence=0.91,
                        reason="personal_memory_recall",
                    )
            if _is_broad_personal_memory_request(tokens):
                evidence_keys = _personal_summary_evidence_keys(self.profile)
                return AssistantDecision(
                    utterance=utterance,
                    intent="personal_memory",
                    route="local_answer",
                    answer="I can summarize a few local memories about you.",
                    evidence_keys=evidence_keys,
                    local_memory_used=True,
                    confidence=0.88,
                    reason="personal_memory_summary",
                )
            if "favorite_color" in self.profile.facts:
                fact_key = "favorite_color"
                answer = f"I remember that your favorite color is {self.profile.facts[fact_key]}."
            else:
                fact_key, fact = next(iter(self.profile.facts.items()))
                label = fact_key.replace("_", " ")
                answer = f"I remember that your {label} is {fact}."
            return AssistantDecision(
                utterance=utterance,
                intent="personal_memory",
                route="local_answer",
                answer=answer,
                evidence_keys=(f"facts.{fact_key}",),
                local_memory_used=True,
                confidence=0.91,
                reason="personal_memory_recall",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="personal_memory",
            route="clarify",
            answer="I do not have enough memory about you yet.",
            confidence=0.71,
            reason="personal_memory_empty",
        )

    def _meal(self, utterance: str) -> AssistantDecision:
        weather = self.profile.weekly_weather.get("today", "")
        choice = choose_local_meal(
            self.profile.food_inventory,
            preferences=self.profile.preferences,
            weather=weather,
            utterance=utterance,
        )
        weather_note = (
            " It may rain, so something warm is sensible." if choice.warm_note else ""
        )
        return AssistantDecision(
            utterance=utterance,
            intent="meal_suggestion",
            route="local_answer",
            answer=f"You could eat {choice.phrase}.{weather_note}",
            evidence_keys=("food_inventory", "weekly_weather.today"),
            local_memory_used=True,
            confidence=0.82,
            reason="memory_plus_weather_cache",
        )

    def _contact(self, utterance: str) -> AssistantDecision:
        text = _normalize(utterance)
        if not self.profile.contacts:
            return AssistantDecision(
                utterance=utterance,
                intent="social_contact",
                route="clarify",
                answer="Who should I call?",
                confidence=0.68,
                reason="missing_contact",
            )
        contact = _requested_contact(text, self.profile.contacts)
        number = self.profile.contacts.get(contact)
        if number:
            return AssistantDecision(
                utterance=utterance,
                intent="social_contact",
                route="device_action",
                answer=f"I can call {contact}.",
                evidence_keys=(f"contacts.{contact}",),
                local_memory_used=True,
                device_action=True,
                confidence=0.88,
                reason="trusted_contact_action",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="social_contact",
            route="clarify",
            answer="Who should I call?",
            confidence=0.68,
            reason="missing_contact",
        )


def assistant_direction_examples() -> tuple[str, ...]:
    return (
        "Tell me a story.",
        "What is the weather today?",
        "Should I go to school dressed naked?",
        "Play a song for me.",
        "What do you think I should do to improve my health?",
        "Tell me something about myself.",
        "What do you think I should eat today?",
        "I need to talk to someone.",
    )


def compare_assistant_mvp_directions(
    profile: LocalAssistantProfile | None = None,
) -> tuple[AssistantStrategyReport, ...]:
    profile = profile or LocalAssistantProfile()
    return compare_assistant_strategy_reports_for_utterances(
        assistant_direction_examples(),
        profile=profile,
        memory_strategy_name="memory_centric_local_triage",
    )


def compare_assistant_strategy_reports_for_utterances(
    utterances: tuple[str, ...] | list[str],
    *,
    profile: LocalAssistantProfile | None = None,
    memory_strategy_name: str = LOCAL_STATE_ROUTER_BASELINE,
) -> tuple[AssistantStrategyReport, ...]:
    profile = profile or LocalAssistantProfile()
    examples = tuple(str(item) for item in utterances)
    return (
        _report(
            memory_strategy_name,
            tuple(OnDeviceAssistantRouter(profile).handle(item) for item in examples),
        ),
        _report(
            "thin_tools_plus_cloud",
            tuple(_thin_tools_decision(item, profile) for item in examples),
        ),
        _report(
            "cloud_first_assistant",
            tuple(_cloud_first_decision(item, profile) for item in examples),
        ),
        _report(
            "secondary_lexical_baseline",
            tuple(_secondary_lexical_baseline_decision(item) for item in examples),
        ),
    )


def _thin_tools_decision(
    utterance: str, profile: LocalAssistantProfile
) -> AssistantDecision:
    text = _normalize(utterance)
    intent = _classify_intent_from_uol_slots(
        text,
        _tokenize(text),
        trusted_contact_names=tuple(profile.contacts),
    )
    if intent in {"weather", "common_sense_safety", "media_playback", "social_contact"}:
        return OnDeviceAssistantRouter(profile).handle(utterance)
    if intent == "personal_memory":
        return AssistantDecision(
            utterance=utterance,
            intent=intent,
            route="clarify",
            answer="I do not have a memory system for that.",
            confidence=0.62,
            reason="no_personal_memory_layer",
        )
    return _cloud(utterance, intent, reason="thin_local_tools_need_cloud")


def _cloud_first_decision(
    utterance: str, profile: LocalAssistantProfile
) -> AssistantDecision:
    text = _normalize(utterance)
    intent = _classify_intent_from_uol_slots(
        text,
        _tokenize(text),
        trusted_contact_names=tuple(profile.contacts),
    )
    if intent in {"media_playback", "social_contact"}:
        return OnDeviceAssistantRouter(profile).handle(utterance)
    if intent == "weather":
        return AssistantDecision(
            utterance=utterance,
            intent=intent,
            route="external_fetch",
            answer="Fetch weather from the network.",
            external_fetch_needed=True,
            confidence=0.86,
            reason="cloud_first_tool_fetch",
        )
    return _cloud(
        utterance,
        intent,
        reason="cloud_first_general_language",
        privacy_exposure=intent
        in {"health_advice", "personal_memory", "meal_suggestion"},
    )


def _secondary_lexical_baseline_decision(utterance: str) -> AssistantDecision:
    intent = _classify_intent_for_secondary_lexical_baseline(_normalize(utterance))
    if intent == "common_sense_safety":
        return AssistantDecision(
            utterance=utterance,
            intent=intent,
            route="local_answer",
            answer="No. Wear proper clothes before going to school.",
            confidence=0.95,
            reason="single_local_policy",
        )
    return AssistantDecision(
        utterance=utterance,
        intent=intent,
        route="cloud_handoff" if intent != "social_contact" else "clarify",
        answer="I can label the intent, but I lack the memory/tool/action layer to finish it.",
        cloud_needed=intent != "social_contact",
        confidence=0.5,
        reason="intent_without_grounded_runtime",
    )


def _classify_intent_for_secondary_lexical_baseline(text: str) -> AssistantIntent:
    for intent, markers in _secondary_meaning_hint_groups().items():
        if any(_has_marker(text, marker) for marker in markers):
            return intent
    return "unknown"


def _cloud(
    utterance: str,
    intent: AssistantIntent,
    *,
    reason: str,
    privacy_exposure: bool = True,
) -> AssistantDecision:
    return AssistantDecision(
        utterance=utterance,
        intent=intent,
        route="cloud_handoff",
        answer="Hand off to a larger model.",
        cloud_needed=True,
        privacy_exposure=privacy_exposure,
        confidence=0.74,
        reason=reason,
    )


def _render_story_frame(frame: str, *, name: str, location: str, culture: str) -> str:
    try:
        return frame.format(name=name, location=location, culture=culture)
    except (KeyError, IndexError, ValueError):
        return frame


def choose_local_meal(
    foods: tuple[str, ...] | list[str],
    *,
    preferences: dict[str, str] | None = None,
    weather: str = "",
    utterance: str = "",
) -> LocalMealChoice:
    inventory = tuple(
        dict.fromkeys(
            _clean_food_name(item) for item in foods if _clean_food_name(item)
        )
    )
    scope = _meal_scope(utterance)
    warm_weather = _weather_suggests_warm_food(weather)
    if not inventory:
        return LocalMealChoice(
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
    return LocalMealChoice(
        items=selected,
        backups=backups,
        reason_tags=reason_tags,
        meal_scope=scope,
        warm_note=warm_weather
        and any(_food_tags(food) & {"warm", "staple", "protein"} for food in selected),
    )


def _clean_food_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _meal_scope(utterance: str) -> str:
    tokens = set(_tokenize(_normalize(utterance)))
    for scope in ("breakfast", "lunch", "dinner"):
        if scope in tokens:
            return scope
    if tokens & {"cook", "cooking"}:
        return "cooking"
    return "meal"


def _weather_suggests_warm_food(weather: str) -> bool:
    text = weather.lower()
    tokens = _tokenize(text)
    return any(
        _has_token_sequence(tokens, _tokenize(term))
        for term in ("rain", "cold", "storm", "snow", "wind")
    )


def _food_inventory_score(
    food: str,
    *,
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


def _food_tags(food: str) -> set[str]:
    tokens = _tokenize(food.lower())
    tags: set[str] = set()
    mapping = {
        "rice": {"staple", "grain", "warm"},
        "beans": {"protein", "staple", "warm"},
        "plantain": {"staple", "fruit", "warm"},
        "egg": {"protein", "breakfast"},
        "oat": {"grain", "breakfast", "warm"},
        "fruit": {"fruit", "light"},
        "salad": {"vegetable", "light"},
        "soup": {"vegetable", "warm", "light"},
        "vegetable": {"vegetable"},
        "fish": {"protein"},
        "chicken": {"protein"},
        "bread": {"grain", "breakfast"},
    }
    for marker, marker_tags in mapping.items():
        if _has_token_sequence(tokens, _tokenize(marker)):
            tags.update(marker_tags)
    return tags or {"food"}


def _meal_reason_tags(
    selected: tuple[str, ...],
    *,
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


def _asks_to_send_user_context(text: str) -> bool:
    return _has_private_context_frame(_tokenize(text))


def _is_private_cloud_export_request(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    cloud_target = bool(token_set & {"cloud", "model", "llm"})
    export_action = bool(
        token_set & {"send", "share", "upload", "export", "give", "tell"}
    )
    if not (cloud_target and export_action):
        return False
    return _has_private_context_frame(tokens)


def _has_private_context_frame(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    owned_memory_subject = bool(
        token_set
        & {
            "mom",
            "contact",
            "job",
            "trip",
            "routine",
            "accessibility",
            "preference",
            "age",
            "child",
            "son",
            "daughter",
            "kid",
            "school",
            "household",
            "family",
            "profile",
            "location",
            "conversation",
        }
    )
    favorite_color = {"favorite", "color"} <= token_set
    health_goal = {"health", "goal"} <= token_set or {"health", "goals"} <= token_set
    public_profile = {"public", "profile"} <= token_set
    about_self = "about" in token_set and bool(token_set & {"me", "myself"})
    prior_conversation = bool(
        token_set & {"previous", "earlier", "recent"}
        and token_set & {"conversation", "conversations"}
    )
    where_i_live = {"where", "i", "live"} <= token_set or {
        "where",
        "i",
        "lived",
    } <= token_set
    return bool(
        owned_memory_subject
        or favorite_color
        or health_goal
        or public_profile
        or about_self
        or prior_conversation
        or where_i_live
    )


def _private_cloud_evidence_keys(
    text: str,
    *,
    trusted_contact_names: tuple[str, ...] = (),
) -> tuple[str, ...]:
    tokens = _tokenize(text)
    token_set = set(tokens)
    keys: list[str] = []
    if {"favorite", "color"} <= token_set:
        keys.append("facts.favorite_color")
    trusted_contact = _matched_trusted_contact_name(tokens, trusted_contact_names)
    if trusted_contact:
        keys.append(f"contacts.{trusted_contact}")
    elif token_set & {"mom", "dad", "caregiver", "contact"}:
        keys.append("contacts.local")
    if "job" in token_set:
        keys.append("facts.job")
    if "trip" in token_set:
        keys.append("facts.trip")
    if "routine" in token_set:
        keys.append("facts.morning_routine")
    if "accessibility" in token_set:
        keys.append("facts.accessibility")
    if "preference" in token_set:
        keys.append("preferences.local")
    if {"health", "goal"} <= token_set or {"health", "goals"} <= token_set:
        keys.append("health_goals")
    child_context = _is_child_memory_request(tokens) or bool(
        token_set & {"son", "daughter", "kid"}
    )
    if child_context:
        if token_set & {"age", "old", "child", "kid"}:
            keys.append("facts.child_age")
        if token_set & {"school", "son", "daughter"}:
            keys.append("facts.child_school")
        if token_set & {"location", "where", "lives", "live"}:
            keys.append("facts.child_location")
        if token_set & {"name", "called"}:
            keys.append("facts.child_name")
    elif "my" in token_set and "age" in token_set:
        keys.append("profile.age")
    if "school" in token_set and not child_context:
        keys.append("facts.school")
    if token_set & {"household", "family"} or {"shared", "device"} <= token_set:
        keys.append("facts.household_context")
    if {"public", "profile"} <= token_set:
        keys.append("facts.public_profile")
    if (
        token_set & {"location"} or {"where", "i", "live"} <= token_set
    ) and not child_context:
        keys.append("profile.location")
    if "about" in token_set and token_set & {"me", "myself"}:
        keys.append("facts.local_profile")
    if "conversation" in token_set or {"talked", "about"} <= token_set:
        keys.append("events.local_conversation")
    return tuple(dict.fromkeys(keys or ["profile.local_private_context"]))


def _report(
    strategy: str, decisions: tuple[AssistantDecision, ...]
) -> AssistantStrategyReport:
    resolved_routes = {"local_answer", "cached_tool", "device_action"}
    return AssistantStrategyReport(
        strategy=strategy,
        cases=len(decisions),
        local_or_device_resolved=sum(
            decision.route in resolved_routes for decision in decisions
        ),
        cloud_handoffs=sum(decision.cloud_needed for decision in decisions),
        external_fetches=sum(decision.external_fetch_needed for decision in decisions),
        clarifications=sum(decision.route == "clarify" for decision in decisions),
        privacy_exposures=sum(decision.privacy_exposure for decision in decisions),
        memory_uses=sum(decision.local_memory_used for decision in decisions),
        decisions=decisions,
    )


def parse_assistant_debug_frame(
    utterance: str,
    decision: AssistantDecision | None = None,
) -> AssistantDebugParse:
    """Map text into a small UOL/ChatFrame-style debug trace."""

    normalized = _normalize(utterance)
    tokens = _tokenize(normalized)
    intent = decision.intent if decision is not None else _classify_intent(normalized)
    composition = _assistant_compositional_parse(normalized, tokens, intent)
    route = decision.route if decision is not None else _route_hint(intent, composition)
    reason = (
        decision.reason
        if decision is not None
        else _route_reason_hint(intent, composition)
    )
    secondary_meaning_hints = _secondary_meaning_hints(normalized, intent)
    secondary_domain_hints = _secondary_domain_hints(normalized)
    domain_hints = _domain_hints(normalized, tokens, intent, composition)
    uol = _assistant_uol(normalized, tokens, intent, composition)
    slot_sources = _slot_sources(normalized, tokens, intent, uol, composition)
    uol["slot_sources"] = slot_sources
    frame_capabilities = _frame_capabilities(intent, route, decision)
    primary_routing_basis = _primary_routing_basis(
        intent, route, reason, uol, frame_capabilities, composition
    )
    secondary_debug_hints = _secondary_debug_hints(secondary_meaning_hints)
    chat_frame = {
        "schema": "melm.assistant_chat_frame_debug.v1",
        "intent": intent,
        "domain": _intent_domain(intent),
        "route": route,
        "reason": reason,
        "needs_tool": route in {"cached_tool", "external_fetch"},
        "needs_cloud": route == "cloud_handoff",
        "needs_confirmation": bool(decision.device_action)
        if decision is not None
        else intent in {"media_playback", "social_contact"},
        "can_answer_locally": route in {"local_answer", "cached_tool", "device_action"},
        "local_memory_candidate": intent
        in {
            "assistant_identity",
            "assistant_status",
            "story",
            "weather",
            "health_advice",
            "personal_memory",
            "autobiographical_memory",
            "meal_suggestion",
        },
        "slots": {
            "subject": uol["subject"],
            "action": uol["action"],
            "object": uol["object"],
            "source": uol["source"],
            "target": uol["target"],
        },
        "capabilities": frame_capabilities,
        "frame_registry": str((composition or {}).get("frame_registry", "")),
        "frame_id": str((composition or {}).get("frame_id", "")),
        "frame_source_policy": str(
            (composition or {}).get("source_policy", "no_local_composition")
        ),
        "primary_routing_basis": primary_routing_basis,
        "secondary_debug_hints": secondary_debug_hints,
        "secondary_hint_policy": "debug_only_never_primary_route",
        "complexity_score": _assistant_frame_complexity(uol, intent),
    }
    return AssistantDebugParse(
        utterance=utterance,
        normalized=normalized,
        tokens=tokens,
        uol=uol,
        chat_frame=chat_frame,
        secondary_meaning_hints=secondary_meaning_hints,
        nlp=_basic_nlp_debug(
            normalized,
            tokens,
            intent,
            secondary_meaning_hints,
            domain_hints,
            secondary_domain_hints,
            composition,
        ),
        mapping=_debug_mapping(
            normalized,
            tokens,
            intent,
            uol,
            chat_frame,
            secondary_meaning_hints,
            domain_hints,
            secondary_domain_hints,
            composition,
        ),
        notes=_debug_notes(normalized, intent, route, reason),
    )


def compose_assistant_status_frame(utterance: str) -> dict[str, Any] | None:
    """Return the primary self-status UOL composition used by router and kernel."""

    normalized = _normalize(utterance)
    return _self_status_composition(normalized, _tokenize(normalized))


def compose_autobiographical_memory_frame(utterance: str) -> dict[str, Any] | None:
    """Return the primary autobiographical-memory UOL composition used by router and kernel."""

    normalized = _normalize(utterance)
    tokens = _tokenize(normalized)
    if not _is_autobiographical_debug_request(normalized, tokens):
        return None
    return _semantic_slot_composition(normalized, tokens, "autobiographical_memory")


def classify_autobiographical_memory_scope(utterance: str) -> str:
    """Classify the structural scope of an autobiographical-memory request."""

    normalized = _normalize(utterance)
    tokens = _tokenize(normalized)
    if not _is_autobiographical_debug_request(normalized, tokens):
        return ""
    if _autobiographical_long_horizon_frame(normalized, tokens):
        return "long_horizon"
    if _autobiographical_session_summary_frame(normalized, tokens):
        return "session_summary"
    if _autobiographical_latest_event_frame(normalized, tokens):
        return "latest_event"
    return "event_query"


def _classify_intent(text: str) -> AssistantIntent:
    tokens = _tokenize(text)
    return _classify_intent_from_uol_slots(text, tokens)


def _classify_intent_from_uol_slots(
    text: str,
    tokens: tuple[str, ...],
    *,
    trusted_contact_names: tuple[str, ...] = (),
    lexical_class_lookup: Callable[[str], frozenset[str]] | None = None,
    lexicon_owned_families: frozenset[str] = frozenset(),
) -> AssistantIntent:
    functional_parse = parse_functional_relations(tokens, question_mark="?" in text)
    if _is_assistant_identity_request(text, tokens):
        return "assistant_identity"
    if _is_assistant_status_request(text, tokens):
        return "assistant_status"
    if _is_story_request(
        text,
        tokens,
        lexical_class_lookup=lexical_class_lookup,
        lexicon_owned="story" in lexicon_owned_families,
    ) and (
        functional_parse is None
        or functional_parse.speech_act in {"request", "yes_no_question", "wh_question"}
    ):
        return "story"
    if _is_weather_request(
        text,
        tokens,
        lexical_class_lookup=lexical_class_lookup,
        lexicon_owned="weather" in lexicon_owned_families,
    ):
        return "weather"
    if _is_common_sense_safety_request(text, tokens):
        return "common_sense_safety"
    if _is_media_request(
        text,
        tokens,
        lexical_class_lookup=lexical_class_lookup,
        lexicon_owned="media" in lexicon_owned_families,
    ):
        return "media_playback"
    if _is_health_advice_request(text, tokens):
        return "health_advice"
    if _is_social_contact_request(
        text, tokens, trusted_contact_names=trusted_contact_names
    ):
        return "social_contact"
    if _is_personal_memory_frame(text, tokens):
        return "personal_memory"
    if _is_autobiographical_debug_request(text, tokens):
        return "autobiographical_memory"
    if _is_meal_suggestion_request(text, tokens):
        return "meal_suggestion"
    functional_kind = functional_frame_kind(functional_parse)
    if functional_kind in {
        "social_greeting",
        "assistant_behavior",
        "personal_goal_advice",
        "open_domain",
    }:
        return functional_kind  # type: ignore[return-value]
    return "unknown"


def _is_assistant_identity_request(
    text: str, tokens: tuple[str, ...] | None = None
) -> bool:
    return _identity_composition(text, tokens or _tokenize(text)) is not None


def _is_assistant_status_request(
    text: str, tokens: tuple[str, ...] | None = None
) -> bool:
    return _self_status_composition(text, tokens or _tokenize(text)) is not None


def _is_story_request(
    text: str,
    tokens: tuple[str, ...],
    *,
    lexical_class_lookup: Callable[[str], frozenset[str]] | None = None,
    lexicon_owned: bool = False,
) -> bool:
    token_set = set(tokens)
    story_objects = _semantic_family_terms(
        tokens,
        fallback={"story", "stories", "tale", "tales", "fable", "fables"},
        semantic_classes={"narrative_content"},
        lexical_class_lookup=lexical_class_lookup,
        lexicon_owned=lexicon_owned,
    )
    if not token_set & story_objects:
        return False
    story_actions = {"tell", "read", "make", "give"}
    return bool(token_set & story_actions) or _story_request_question(text, tokens)


_STORY_CONSTRAINT_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "with",
    "featuring",
    "feature",
    "features",
    "for",
    "me",
    "please",
    "story",
    "stories",
    "tale",
    "tales",
    "fable",
    "fables",
    "tell",
    "read",
    "make",
    "give",
}


def _requested_story_constraints(utterance: str) -> frozenset[str]:
    tokens = _tokenize(_normalize(utterance))
    constraint_markers = {"about", "with", "featuring"}
    try:
        marker_index = min(
            index for index, token in enumerate(tokens) if token in constraint_markers
        )
    except ValueError:
        return frozenset()
    constraints = {
        token
        for token in tokens[marker_index + 1 :]
        if len(token) >= 3 and token not in _STORY_CONSTRAINT_STOPWORDS
    }
    return frozenset(constraints)


def _matching_story_model(
    story_models: dict[str, str],
    requested_constraints: frozenset[str],
) -> tuple[str, str] | None:
    if not story_models:
        return None
    if not requested_constraints:
        return next(iter(story_models.items()))
    for story_key, frame in story_models.items():
        searchable = _normalize(f"{story_key} {frame}")
        if all(constraint in searchable for constraint in requested_constraints):
            return story_key, frame
    return None


def _available_story_inventory_label(story_models: dict[str, str]) -> str:
    if not story_models:
        return "no local story yet"
    first_key, first_frame = next(iter(story_models.items()))
    words = [word for word in _tokenize(_normalize(first_frame)) if len(word) >= 4]
    if words:
        return f"the local {first_key.replace('_', ' ')} story"
    return f"the local {first_key.replace('_', ' ')} story"


def _is_weather_request(
    text: str,
    tokens: tuple[str, ...],
    *,
    lexical_class_lookup: Callable[[str], frozenset[str]] | None = None,
    lexicon_owned: bool = False,
) -> bool:
    token_set = set(tokens)
    weather_terms = _semantic_family_terms(
        tokens,
        fallback={"weather", "forecast", "temperature", "rain"},
        semantic_classes={"weather_phenomenon"},
        lexical_class_lookup=lexical_class_lookup,
        lexicon_owned=lexicon_owned,
    )
    if _is_weather_concept_question(tokens, weather_terms=weather_terms):
        return False
    weather_object = bool(
        token_set & weather_terms & {"weather", "forecast", "temperature"}
    )
    rain_time_query = bool(
        "rain" in weather_terms and token_set & {"today", "tomorrow", "outside"}
    )
    return (weather_object or rain_time_query) and (
        _is_question_like(text, tokens)
        or _is_request_like(tokens)
        or bool(token_set & {"today", "tomorrow", "outside"})
    )


def _is_weather_concept_question(
    tokens: tuple[str, ...],
    *,
    weather_terms: set[str] | None = None,
) -> bool:
    token_set = set(tokens)
    if _weather_observation_context(tokens):
        return False
    weather_terms = weather_terms or {"weather", "forecast", "temperature"}
    concept_terms = {
        "define",
        "explain",
        "mean",
        "means",
        "work",
        "works",
        "system",
        "systems",
    }
    return bool(
        token_set & weather_terms
        and (
            token_set & concept_terms
            or _is_bare_domain_definition_question(tokens, weather_terms)
            or tokens[:1] in {("how",), ("why",)}
        )
    )


def _semantic_family_terms(
    tokens: tuple[str, ...],
    *,
    fallback: set[str],
    semantic_classes: set[str],
    lexical_class_lookup: Callable[[str], frozenset[str]] | None,
    lexicon_owned: bool,
) -> set[str]:
    if not lexicon_owned:
        return set(tokens) & fallback
    if lexical_class_lookup is None:
        return set()
    return {token for token in tokens if lexical_class_lookup(token) & semantic_classes}


def _weather_observation_context(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if token_set & {"today", "tomorrow", "outside", "now", "tonight", "week", "weekly"}:
        return True
    live_weather_sequences = (
        ("the", "weather"),
        ("the", "temperature"),
        ("the", "forecast"),
    )
    return _has_any_token_sequence(tokens, live_weather_sequences)


def _is_bare_domain_definition_question(
    tokens: tuple[str, ...], domain_terms: set[str]
) -> bool:
    semantic_tokens = tuple(
        token for token in tokens if token not in {"please", "really", "now"}
    )
    if semantic_tokens[:2] == ("what", "is"):
        remainder = tuple(
            token for token in semantic_tokens[2:] if token not in {"a", "an"}
        )
        return len(remainder) == 1 and remainder[0] in domain_terms
    if semantic_tokens[:1] == ("what's",):
        remainder = tuple(
            token for token in semantic_tokens[1:] if token not in {"a", "an"}
        )
        return len(remainder) == 1 and remainder[0] in domain_terms
    return False


def _story_request_question(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if not _is_question_like(text, tokens):
        return False
    if token_set & {"tell", "read", "make", "give"}:
        return True
    return False


def _is_common_sense_safety_request(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    safety_frame = (
        _is_question_like(text, tokens)
        or _is_request_like(tokens)
        or bool(
            token_set
            & {
                "go",
                "going",
                "wear",
                "dress",
                "dressed",
                "school",
                "class",
                "outside",
                "public",
            }
        )
    )
    safety_subject_or_context = bool(
        token_set
        & {
            "i",
            "me",
            "my",
            "go",
            "going",
            "walk",
            "wear",
            "dress",
            "dressed",
            "school",
            "class",
            "outside",
            "public",
        }
    )
    if token_set & {"naked", "undressed"}:
        return safety_frame and safety_subject_or_context
    if {"without", "clothes"} <= token_set:
        return safety_frame and safety_subject_or_context
    if not safety_frame:
        return False
    clothing_terms = {"wear", "clothes", "coat", "raincoat", "dress", "dressed"}
    public_context = {"school", "class", "outside", "public"}
    return bool(token_set & clothing_terms and token_set & public_context)


def _is_health_advice_request(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    health_terms = {
        "health",
        "healthy",
        "healthier",
        "wellness",
    }
    care_terms = {
        "doctor",
        "diagnose",
        "rash",
        "medicine",
        "medication",
        "fever",
        "sick",
        "ill",
        "pain",
    }
    if _has_urgent_health_frame(tokens):
        return True
    advice_frame = (
        _is_question_like(text, tokens)
        or _is_request_like(tokens)
        or bool(token_set & {"advice", "advise", "help", "improve"})
    )
    if not advice_frame:
        return False
    personal_context = bool(token_set & {"i", "me", "my", "myself"})
    health_action_context = bool(
        token_set
        & {
            "advice",
            "advise",
            "better",
            "do",
            "goals",
            "goal",
            "help",
            "improve",
            "sleep",
            "take",
            "see",
        }
    )
    health_question_context = tokens[:1] in {("how",), ("should",)} or (
        tokens[:1] in {("what",), ("can",), ("could",)}
        and (personal_context or health_action_context)
    )
    if token_set & health_terms:
        return personal_context or health_action_context or health_question_context
    if token_set & care_terms:
        return personal_context or health_action_context
    return False


def _is_social_contact_request(
    text: str,
    tokens: tuple[str, ...] | None = None,
    *,
    trusted_contact_names: tuple[str, ...] = (),
) -> bool:
    token_tuple = tokens or _tokenize(text)
    token_set = set(token_tuple)
    if token_tuple[:1] in {("call",), ("phone",), ("ring",)}:
        return _has_contact_target(
            token_tuple, trusted_contact_names=trusted_contact_names
        )
    if token_set & {"call", "ring"} and _is_request_like(token_tuple):
        return _has_contact_target(
            token_tuple, trusted_contact_names=trusted_contact_names
        )
    if (
        "phone" in token_set
        and _phone_is_contact_action(token_tuple)
        and _is_request_like(token_tuple)
    ):
        return _has_contact_target(
            token_tuple, trusted_contact_names=trusted_contact_names
        )
    if "reach" in token_set and _has_contact_target(
        token_tuple,
        trusted_contact_names=trusted_contact_names,
    ):
        return True
    talk_terms = {"talk", "talking", "speak", "speaking"}
    target_terms = {"someone", "person", "caregiver", "contact", "adult", "mom", "dad"}
    return bool(
        token_set & talk_terms
        and (
            token_set & target_terms
            or _matched_trusted_contact_name(token_tuple, trusted_contact_names)
        )
        and (
            token_set & {"need", "help", "please"}
            or _is_question_like(text, token_tuple)
        )
    )


def _has_contact_target(
    tokens: tuple[str, ...],
    *,
    trusted_contact_names: tuple[str, ...] = (),
) -> bool:
    target_terms = {
        "someone",
        "person",
        "caregiver",
        "contact",
        "adult",
        "mom",
        "dad",
        "sister",
        "brother",
        "daughter",
        "son",
        "child",
    }
    return bool(
        set(tokens) & target_terms
        or _matched_trusted_contact_name(tokens, trusted_contact_names)
    )


def _phone_is_contact_action(tokens: tuple[str, ...]) -> bool:
    for index, token in enumerate(tokens):
        if token != "phone":
            continue
        previous = tokens[index - 1] if index > 0 else ""
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if previous in {"my", "the", "your", "this", "that"}:
            return False
        if next_token in {"number", "battery", "screen", "charger", "case"}:
            return False
        return True
    return False


def _is_personal_memory_frame(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if _is_private_cloud_export_request(text, tokens):
        return True
    if _is_child_memory_request(tokens):
        return True
    memory_frame = (
        _is_question_like(text, tokens)
        or _is_request_like(tokens)
        or bool(token_set & {"know", "remember", "recall", "forget"})
    )
    if _is_routine_memory_request(tokens):
        owned_or_recalled = bool(
            token_set & {"my", "our", "me", "i"}
            or token_set & {"know", "remember", "recall"}
            or _about_targets_self(tokens)
        )
        return memory_frame and owned_or_recalled
    if _is_household_memory_request(tokens):
        owned_or_recalled = bool(
            token_set & {"my", "our", "we", "us", "this"}
            or token_set & {"memory", "know", "remember", "recall"}
            or _is_device_user_memory_question(tokens)
        )
        return memory_frame and owned_or_recalled
    if {"who", "am", "i"} <= token_set:
        return True
    recall_terms = {"remember", "know", "recall"}
    first_person_targets = {"me", "my", "myself", "i"}
    if token_set & recall_terms and token_set & first_person_targets:
        return True
    return _about_targets_self(tokens)


def _about_targets_self(tokens: tuple[str, ...]) -> bool:
    self_targets = {"me", "myself"}
    for index, token in enumerate(tokens):
        if token != "about":
            continue
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if next_token in self_targets:
            return True
        if next_token == "my" and index + 2 < len(tokens):
            return True
    return False


def _is_autobiographical_debug_request(
    text: str, tokens: tuple[str, ...] | None = None
) -> bool:
    token_tuple = tokens or _tokenize(text)
    token_set = set(token_tuple)
    if not _autobiographical_question_or_command(text, token_tuple):
        return False
    if _autobiographical_long_horizon_frame(text, token_tuple):
        return True
    if _autobiographical_session_summary_frame(text, token_tuple):
        return True
    if _autobiographical_latest_event_frame(text, token_tuple):
        return True
    event_objects = {
        "conversation",
        "conversations",
        "session",
        "sessions",
        "question",
        "questions",
        "answer",
        "answers",
    }
    recall_actions = {
        "talk",
        "talked",
        "ask",
        "asked",
        "answer",
        "answered",
        "summarize",
        "recap",
        "happened",
    }
    temporal_scope = {"earlier", "previous", "recent", "last", "past"}
    shared_context = bool(
        token_set & {"we", "our"}
        and token_set & {"talk", "talked", "conversation", "conversations"}
    )
    return bool(
        (token_set & event_objects or shared_context)
        and (token_set & recall_actions or token_set & temporal_scope)
    )


def _autobiographical_question_or_command(text: str, tokens: tuple[str, ...]) -> bool:
    return _is_question_like(text, tokens) or tokens[:1] in {
        ("summarize",),
        ("recap",),
        ("show",),
        ("list",),
        ("tell",),
    }


def _autobiographical_long_horizon_frame(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if not _autobiographical_question_or_command(text, tokens):
        return False
    day_span = "days" in token_set and bool(token_set & {"last", "few", "past", "over"})
    all_history = bool(
        token_set & {"all", "everything", "whole"}
        and token_set & {"history", "sessions", "conversations"}
    )
    long_term_memory = bool(
        {"long", "term", "memory"} <= token_set or {"long", "horizon"} <= token_set
    )
    return day_span or all_history or long_term_memory


def _autobiographical_session_summary_frame(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if not _autobiographical_question_or_command(text, tokens):
        return False
    session_objects = {"session", "sessions", "conversation", "conversations"}
    summary_actions = {"summarize", "recap", "happened", "talk", "talked"}
    scoped_to_user_history = bool(
        token_set & {"our", "we"} or token_set & {"recent", "previous", "last"}
    )
    return bool(
        token_set & session_objects
        and token_set & summary_actions
        and scoped_to_user_history
    )


def _autobiographical_latest_event_frame(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if not _autobiographical_question_or_command(text, tokens):
        return False
    latest_scope = bool("last" in token_set or {"most", "recent"} <= token_set)
    event_object = bool(
        token_set
        & {
            "question",
            "questions",
            "thing",
            "things",
            "ask",
            "asked",
            "answer",
            "answered",
        }
    )
    user_or_assistant_context = bool(token_set & {"i", "my", "me", "we", "our", "you"})
    return latest_scope and event_object and user_or_assistant_context


def _is_media_request(
    text: str,
    tokens: tuple[str, ...],
    *,
    lexical_class_lookup: Callable[[str], frozenset[str]] | None = None,
    lexicon_owned: bool = False,
) -> bool:
    token_set = set(tokens)
    media_objects = _semantic_family_terms(
        tokens,
        fallback={
            "song",
            "music",
            "piano",
            "radio",
            "lofi",
            "audio",
            "track",
            "sound",
            "sounds",
        },
        semantic_classes={
            "media_content",
            "media_descriptor",
            "physical_object.instrument",
            "physical_object.media_source",
        },
        lexical_class_lookup=lexical_class_lookup,
        lexicon_owned=lexicon_owned,
    )
    media_action = bool(token_set & {"play", "start"})
    if media_action and token_set & media_objects:
        return True
    if media_action and "something" in token_set and token_set & {"sound", "sounds"}:
        return True
    return False


def _is_meal_suggestion_request(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    food_terms = {"eat", "food", "lunch", "dinner", "breakfast", "meal", "cook"}
    if not token_set & food_terms:
        return False
    if _meal_request_is_direct_suggestion(tokens):
        return True
    if not _meal_request_has_user_choice_frame(text, tokens):
        return False
    meal_action_context = bool(
        token_set & {"eat", "cook", "have", "suggest", "recommend"}
    )
    suggestion_question = _is_question_like(text, tokens) and bool(
        token_set
        & {"should", "could", "can", "eat", "cook", "have", "suggest", "recommend"}
    )
    return meal_action_context and suggestion_question


def _meal_request_is_direct_suggestion(tokens: tuple[str, ...]) -> bool:
    return tokens[:1] in {("suggest",), ("recommend",)}


def _meal_request_has_user_choice_frame(text: str, tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    user_context = bool(token_set & {"i", "me", "my", "we", "us", "our"})
    if not user_context:
        return False
    if token_set & {"should", "could"}:
        return True
    return bool(
        tokens[:1] in {("what",), ("what's",)}
        and "can" in token_set
        and _is_question_like(text, tokens)
    )


def _is_question_like(text: str, tokens: tuple[str, ...]) -> bool:
    return "?" in text or tokens[:1] in {
        ("who",),
        ("what",),
        ("why",),
        ("how",),
        ("should",),
        ("can",),
        ("could",),
        ("would",),
        ("do",),
        ("does",),
        ("did",),
        ("will",),
        ("is",),
        ("are",),
    }


def _is_request_like(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    return bool(
        tokens[:1]
        in {
            ("tell",),
            ("describe",),
            ("read",),
            ("make",),
            ("give",),
            ("show",),
            ("play",),
            ("start",),
            ("call",),
            ("phone",),
            ("ring",),
            ("reach",),
            ("remember",),
            ("forget",),
        }
        or (token_set & {"can", "could", "would"} and "you" in token_set)
        or "please" in token_set
    )


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9']+", text))


def _assistant_compositional_parse(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
) -> dict[str, Any] | None:
    frame_match = AssistantFrameRegistry.match(text, tokens, intent)
    return frame_match.to_composition() if frame_match is not None else None


def _assistant_frame_id(composition: dict[str, Any]) -> str:
    intent = str(composition.get("intent", "unknown"))
    pattern = str(composition.get("pattern", "unmatched"))
    return f"{intent}.{pattern}"


def _semantic_slot_composition(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
) -> dict[str, Any] | None:
    if intent == "unknown":
        return None
    projection = _semantic_projection(text, tokens, intent)
    if projection is None:
        return None
    pattern, action, object_value, source, target, basis = projection
    route_hint = (
        "cloud_handoff"
        if intent == "personal_memory" and target == "external_cloud_model"
        else _route_hint(intent)
    )
    speech_act = (
        "request"
        if pattern == "request_private_memory_cloud_boundary"
        else _speech_act_from_tokens(text, tokens)
    )
    return {
        "schema": "melm.intent_uol_composition.v1",
        "source": "slot_role_relation",
        "intent": intent,
        "pattern": pattern,
        "action": action,
        "focus": object_value,
        "basis": list(basis),
        "token_roles": _semantic_token_roles(tokens, intent, action, object_value),
        "uol_projection": {
            "speech_act": speech_act,
            "subject": "user" if intent not in {"assistant_status"} else "assistant",
            "action": action,
            "object": object_value,
            "source": source,
            "target": target,
        },
        "chat_frame_projection": {
            "domain": _intent_domain(intent),
            "route_hint": route_hint,
        },
        "notes": [],
    }


def _functional_relation_composition(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
) -> dict[str, Any] | None:
    parse = parse_functional_relations(tokens, question_mark="?" in text)
    if parse is None or functional_frame_kind(parse) != intent:
        return None
    payload = parse.to_dict()
    basis = [
        f"speech_act:{parse.speech_act}",
        f"subject:{parse.subject}",
        f"action:{parse.action}",
        f"object:{parse.object or 'none'}",
        f"parse_score:{parse.parse_score}",
    ]
    if parse.complement_action:
        basis.append(f"complement_action:{parse.complement_action}")
    if parse.indirect_object:
        basis.append(f"indirect_object:{parse.indirect_object}")
    return {
        "schema": "melm.weighted_functional_uol_composition.v1",
        "source": "weighted_functional_relation",
        "intent": intent,
        "pattern": parse.pattern,
        "action": parse.action,
        "focus": parse.object,
        "basis": basis,
        "token_roles": list(parse.token_roles),
        "semantic_unknown_tokens": list(parse.semantic_unknown_tokens),
        "functional_parse": payload,
        "candidate_parses": list(parse.candidates),
        "uol_projection": {
            "speech_act": parse.speech_act,
            "subject": parse.subject,
            "action": parse.action,
            "object": parse.object,
            "source": "functional_grammar",
            "target": parse.target,
            "complement_action": parse.complement_action,
            "indirect_object": parse.indirect_object,
            "modifiers": payload["modifiers"],
            "relations": payload["relations"],
            "parse_score": parse.parse_score,
        },
        "chat_frame_projection": {
            "domain": _intent_domain(intent),
            "route_hint": _route_hint(intent),
        },
        "notes": ["capability_route_selected_after_weighted_relation_parse"],
    }


def _semantic_projection(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
) -> tuple[str, str, str, str, str, tuple[str, ...]] | None:
    token_set = set(tokens)
    if intent == "assistant_status":
        action = (
            "plan" if token_set & {"need", "next", "missing", "build"} else "report"
        )
        object_value = "next_steps" if action == "plan" else "runtime_status"
        return (
            "self_status_question",
            action,
            object_value,
            "event_ledger",
            "user",
            (
                "subject:assistant_self_model",
                f"action:{action}",
                f"object:{object_value}",
                "source:event_ledger",
            ),
        )
    if intent == "story":
        return (
            "request_story_inventory",
            "tell",
            "story",
            "story_inventory",
            "assistant",
            (
                "speech_act:request",
                f"action:{_first_present(tokens, ('tell', 'read', 'make')) or 'tell'}",
                "object:story",
                "source:story_inventory",
            ),
        )
    if intent == "weather":
        return (
            "question_weather_cache",
            "answer",
            "weather",
            "weather_cache",
            "today" if "today" in token_set else "forecast",
            (
                "speech_act:question",
                "object:weather",
                "source:weather_cache",
                f"time:{'today' if 'today' in token_set else 'forecast'}",
            ),
        )
    if intent == "common_sense_safety":
        object_value = "school_clothing" if "school" in token_set else "public_safety"
        return (
            "judgement_safety_policy",
            "judge",
            object_value,
            "local_policy",
            "user",
            (
                "speech_act:judgement_request",
                "action:judge",
                f"object:{object_value}",
                "source:local_policy",
            ),
        )
    if intent == "media_playback":
        object_value = _media_object_from_request_tokens(tokens)
        return (
            "command_media_playback",
            "play",
            object_value,
            "media_library",
            "local_device",
            (
                "speech_act:command",
                f"action:{_first_present(tokens, ('play', 'start')) or 'play'}",
                f"object:{object_value}",
                "target:local_device",
            ),
        )
    if intent == "health_advice":
        return (
            "request_bounded_health_advice",
            "advise",
            "health",
            "local_health_policy",
            "user",
            (
                "speech_act:advice_request",
                "action:advise",
                "object:health",
                "source:local_health_policy",
            ),
        )
    if intent == "personal_memory":
        object_value = _personal_memory_object_from_text(text, tokens)
        if _is_private_cloud_export_request(text, tokens):
            action = (
                _first_present(
                    tokens, ("send", "share", "upload", "export", "give", "tell")
                )
                or "export"
            )
            return (
                "request_private_memory_cloud_boundary",
                action,
                object_value,
                "local_memory",
                "external_cloud_model",
                (
                    "speech_act:memory_export_request",
                    f"action:{action}",
                    f"object:{object_value}",
                    "source:local_memory",
                    "target:external_cloud_model",
                    "policy:private_memory_requires_boundary_gate",
                ),
            )
        return (
            "request_child_owned_memory"
            if object_value.startswith("facts.child_")
            else "request_personal_memory",
            "recall",
            object_value,
            "local_memory",
            "user",
            (
                "speech_act:memory_request",
                "action:recall",
                f"object:{object_value}",
                "source:local_memory",
            ),
        )
    if intent == "autobiographical_memory":
        return (
            "request_conversation_memory",
            "recall",
            "conversation_events",
            "event_ledger",
            "user",
            (
                "speech_act:memory_request",
                "action:recall",
                "object:conversation_events",
                "source:event_ledger",
            ),
        )
    if intent == "meal_suggestion":
        return (
            "request_meal_suggestion",
            "suggest",
            "meal",
            "food_inventory",
            "user",
            (
                "speech_act:suggestion_request",
                "action:suggest",
                "object:meal",
                "source:food_inventory",
            ),
        )
    if intent == "social_contact":
        object_value = _contact_object_from_tokens(tokens)
        return (
            "command_trusted_contact",
            "call",
            object_value,
            "trusted_contacts",
            "trusted_contact",
            (
                "speech_act:action_request",
                f"action:{_first_present(tokens, ('call', 'phone', 'ring', 'reach', 'talk')) or 'call'}",
                f"object:{object_value}",
                "target:trusted_contact",
            ),
        )
    return None


def _semantic_token_roles(
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    action: str,
    object_value: str,
) -> list[dict[str, Any]]:
    object_terms = _semantic_object_role_tokens(intent, object_value)
    source_terms = set(_tokenize(_evidence_source_for_intent(intent).replace("_", " ")))
    structural_terms = _structural_debug_tokens() | {
        "am",
        "can",
        "could",
        "does",
        "have",
        "has",
        "in",
        "of",
        "on",
        "please",
        "would",
    }
    roles: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        role = "unresolved_token"
        meaning = "not_used_by_slot_composition"
        if token in {"what", "who", "why", "how", "should"}:
            role = "interrogative"
            meaning = "question_operator"
        elif token in structural_terms:
            role = "syntax"
            meaning = "structural_or_function_token"
        elif token in {
            "tell",
            "play",
            "start",
            "call",
            "phone",
            "ring",
            "reach",
            "talk",
            "eat",
            "cook",
            "remember",
            "send",
            "share",
            "upload",
            "export",
            "give",
        }:
            role = "request_action"
            meaning = "candidate_user_intent_action"
        elif token == action or token in _tokenize(action):
            role = "uol_action"
            meaning = f"{intent}_action"
        elif token in object_terms or token in source_terms:
            role = "uol_object"
            meaning = object_value
        elif intent == "personal_memory" and token in {"cloud", "model", "llm"}:
            role = "policy_target"
            meaning = "external_cloud_model"
        elif intent == "personal_memory" and token in {
            "child",
            "child's",
            "kid",
            "kid's",
            "son",
            "son's",
            "daughter",
            "daughter's",
        }:
            role = "owned_memory_subject"
            meaning = "child_local_memory_owner"
        elif token in {
            "me",
            "my",
            "myself",
            "you",
            "your",
            "someone",
            "mom",
            "dad",
            "caregiver",
        }:
            role = "participant_or_target"
            meaning = "conversation_deixis_or_contact_target"
        elif token in {
            "today",
            "tomorrow",
            "breakfast",
            "lunch",
            "dinner",
            "school",
            "local",
        }:
            role = "scope_modifier"
            meaning = "bounded_context"
        roles.append({"index": index, "token": token, "role": role, "meaning": meaning})
    return roles


def _semantic_object_role_tokens(
    intent: AssistantIntent, object_value: str
) -> set[str]:
    tokens = set(_tokenize(object_value.replace("_", " ")))
    aliases = {
        "story": {"story", "stories", "tale", "tales", "fable", "fables", "bedtime"},
        "weather": {
            "weather",
            "forecast",
            "temperature",
            "rain",
            "outside",
            "today",
            "tomorrow",
        },
        "common_sense_safety": {
            "naked",
            "undressed",
            "clothes",
            "wear",
            "dress",
            "dressed",
            "school",
            "class",
            "outside",
            "public",
        },
        "media_playback": {
            "song",
            "music",
            "piano",
            "radio",
            "lofi",
            "audio",
            "track",
            "sound",
            "sounds",
        },
        "health_advice": {
            "health",
            "healthy",
            "healthier",
            "wellness",
            "doctor",
            "medicine",
            "medication",
            "fever",
            "sick",
            "pain",
            "sleep",
        },
        "personal_memory": {
            "about",
            "memory",
            "remember",
            "recall",
            "routine",
            "morning",
            "household",
            "family",
            "child",
            "kid",
            "son",
            "daughter",
            "school",
            "age",
            "name",
            "location",
            "favorite",
            "health",
            "goal",
            "goals",
            "contact",
        },
        "autobiographical_memory": {
            "conversation",
            "conversations",
            "sessions",
            "earlier",
            "previous",
            "recent",
            "last",
            "question",
            "days",
        },
        "meal_suggestion": {
            "eat",
            "food",
            "meal",
            "breakfast",
            "lunch",
            "dinner",
            "cook",
        },
        "social_contact": {
            "someone",
            "person",
            "caregiver",
            "contact",
            "adult",
            "mom",
            "dad",
            "sister",
            "brother",
            "daughter",
            "son",
            "child",
        },
    }
    tokens.update(aliases.get(intent, set()))
    return tokens


def _speech_act_from_tokens(text: str, tokens: tuple[str, ...]) -> str:
    if "?" in text or tokens[:1] in {
        ("who",),
        ("what",),
        ("why",),
        ("how",),
        ("should",),
    }:
        return "question"
    if tokens[:1] in {
        ("tell",),
        ("describe",),
        ("play",),
        ("start",),
        ("call",),
        ("phone",),
        ("remember",),
        ("send",),
        ("share",),
        ("upload",),
        ("export",),
    }:
        return "request"
    if "need" in tokens and any(
        token in tokens for token in ("talk", "call", "phone", "reach")
    ):
        return "request"
    return "statement"


def _first_present(tokens: tuple[str, ...], candidates: tuple[str, ...]) -> str:
    candidate_set = set(candidates)
    return next((token for token in tokens if token in candidate_set), "")


def _self_status_composition(
    text: str, tokens: tuple[str, ...]
) -> dict[str, Any] | None:
    token_set = set(tokens)
    if not tokens:
        return None
    if "think" in token_set and (
        token_set & {"i", "me", "my"} or token_set & {"health", "eat", "food"}
    ):
        return None
    command_like = tokens[:1] in {("show",), ("report",), ("summarize",), ("list",)}
    question_or_command = _is_question_like(text, tokens) or command_like
    if not question_or_command:
        return None
    self_reference = bool(token_set & {"you", "your"})
    status_objects = {"status", "ledger", "events", "memory"}
    boundary_objects = {"cloud", "local"}
    progress_actions = {"done", "learn", "learned", "stored", "using", "use"}
    planning_terms = {"need", "needs", "next", "missing", "build"}
    count_question = bool(
        token_set & {"how"}
        and token_set & {"many", "much"}
        and token_set & {"events", "memory"}
    )
    status_object_query = bool(
        token_set & status_objects
        and (self_reference or command_like or count_question)
    )
    boundary_object_query = bool(
        token_set & boundary_objects
        and (self_reference or command_like or count_question)
        and (
            command_like
            or tokens[:1] in {("are",), ("is",)}
            or bool(token_set & {"using", "use", "status", "ledger", "memory"})
        )
    )
    object_query = status_object_query or boundary_object_query
    progress_query = bool(
        self_reference and (token_set & progress_actions or {"did", "do"} <= token_set)
    )
    planning_query = bool(self_reference and token_set & planning_terms)
    if not (object_query or progress_query or planning_query or count_question):
        return None
    action = "plan" if planning_query else "report"
    focus = "next_steps" if action == "plan" else "runtime_status"
    if planning_query:
        pattern = "self_status_planning_question"
    elif count_question:
        pattern = "self_status_count_question"
    elif token_set & {"cloud", "local"}:
        pattern = "self_status_boundary_question"
    elif progress_query:
        pattern = "self_status_progress_question"
    else:
        pattern = "self_status_ledger_question"
    basis = _self_status_basis(tokens, action, focus, pattern)
    return {
        "schema": "melm.self_status_uol_composition.v1",
        "source": "slot_role_relation",
        "intent": "assistant_status",
        "pattern": pattern,
        "action": action,
        "focus": focus,
        "basis": list(basis),
        "token_roles": _self_status_token_roles(tokens, action, focus),
        "uol_projection": {
            "speech_act": "request" if command_like else "question",
            "subject": "assistant",
            "action": action,
            "object": focus,
            "source": "event_ledger",
            "target": "user",
        },
        "chat_frame_projection": {
            "domain": "runtime_self_observation",
            "route_hint": "local_answer",
        },
        "notes": [],
    }


def _self_status_basis(
    tokens: tuple[str, ...],
    action: str,
    focus: str,
    pattern: str,
) -> tuple[str, ...]:
    token_set = set(tokens)
    basis = [
        f"pattern:{pattern}",
        "subject:assistant_self_model",
        f"action:{action}",
        f"object:{focus}",
        "source:event_ledger",
    ]
    if token_set & {"you", "your"}:
        basis.append("you:assistant_deixis")
    if token_set & {"status", "ledger", "events", "memory"}:
        basis.append("status_object:ledger_or_memory")
    if token_set & {"cloud", "local"}:
        basis.append("boundary_object:local_cloud_state")
    if (
        token_set & {"done", "learn", "learned", "stored", "using", "use"}
        or {"did", "do"} <= token_set
    ):
        basis.append("progress_action:runtime_history")
    if token_set & {"need", "needs", "next", "missing", "build"}:
        basis.append("planning_signal:self_observed_gap")
    if token_set & {"how", "many", "much"}:
        basis.append("quantity_question:ledger_counts")
    return tuple(dict.fromkeys(basis))


def _self_status_token_roles(
    tokens: tuple[str, ...],
    action: str,
    focus: str,
) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    structural_terms = _structural_debug_tokens() | {
        "have",
        "has",
        "so",
        "far",
        "can",
        "could",
        "would",
    }
    for index, token in enumerate(tokens):
        role = "unresolved_token"
        meaning = "not_used_by_status_composition"
        if token in {
            "what",
            "how",
            "are",
            "did",
            "do",
            "is",
            "should",
            "can",
            "could",
            "would",
        }:
            role = "interrogative_or_auxiliary"
            meaning = "status_question_operator"
        elif token in structural_terms:
            role = "syntax_or_time_scope"
            meaning = "status_context_token"
        elif token in {"show", "report", "summarize", "list"}:
            role = "request_action"
            meaning = "ask_for_status_report"
        elif token in {"you", "your"}:
            role = "deictic_pronoun"
            meaning = "second_person_resolves_to_assistant"
        elif token in {"status", "ledger", "events", "memory"}:
            role = "uol_object"
            meaning = focus
        elif token in {"cloud", "local"}:
            role = "boundary_state"
            meaning = "local_cloud_runtime_state"
        elif token in {"done", "learn", "learned", "stored", "using", "use"}:
            role = "progress_action"
            meaning = "runtime_history_query"
        elif token in {"need", "needs", "next", "missing", "build"}:
            role = "planning_signal"
            meaning = "self_observed_gap_or_next_step"
        elif token in {"many", "much"}:
            role = "quantity_modifier"
            meaning = "ledger_count_request"
        elif token == action or token in _tokenize(action):
            role = "uol_action"
            meaning = "assistant_status_action"
        roles.append({"index": index, "token": token, "role": role, "meaning": meaning})
    return roles


def _identity_composition(text: str, tokens: tuple[str, ...]) -> dict[str, Any] | None:
    pattern = ""
    action = "identify"
    focus = "identity"
    basis: tuple[str, ...] = ()
    purpose_frame = _purpose_identity_frame(tokens)
    if _matches_who_identity_frame(tokens):
        pattern = "who_copula_second_person"
        basis = (
            "who:interrogative_identity",
            "are:state_relation",
            "you:assistant_deixis",
        )
    elif _matches_name_identity_frame(text, tokens):
        pattern = "what_copula_possessive_name"
        action = "name"
        focus = "name"
        basis = (
            "what:attribute_question",
            "is:state_relation",
            "your:assistant_possessive",
            "name:self_model_attribute",
        )
    elif _matches_kind_identity_frame(tokens):
        pattern = "what_copula_second_person"
        basis = (
            "what:kind_question",
            "are:state_relation",
            "you:assistant_deixis",
        )
    elif _matches_capability_identity_frame(tokens):
        pattern = (
            "what_modal_second_person_do"
            if "do" in tokens
            else "modal_second_person_capability"
        )
        action = "describe_capabilities"
        focus = "local_capabilities"
        basis = _capability_identity_basis(tokens)
    elif purpose_frame is not None:
        pattern, basis = purpose_frame
        action = "describe_purpose"
        focus = "purpose"
    elif _matches_self_description_frame(tokens):
        pattern = "request_reflexive_second_person_description"
        action = "describe_self"
        focus = "self_description"
        request_action = "describe" if "describe" in tokens else "tell"
        topic_basis = (
            "about:topic_relation" if "about" in tokens else "yourself:topic_relation"
        )
        basis = (
            f"{request_action}:request",
            topic_basis,
            "yourself:assistant_reflexive",
        )
    if not pattern:
        return None
    notes: list[str] = []
    if any(token in tokens for token in ("don't", "dont", "not")) and "know" in tokens:
        notes.append("identity_challenge_detected")
    projection_speech_act = (
        "challenge"
        if notes
        else ("request" if action == "describe_self" else "question")
    )
    return {
        "schema": "melm.identity_uol_composition.v1",
        "source": "token_role_relation",
        "intent": "assistant_identity",
        "pattern": pattern,
        "action": action,
        "focus": focus,
        "basis": list(basis),
        "token_roles": _identity_token_roles(tokens),
        "uol_projection": {
            "speech_act": projection_speech_act,
            "subject": "assistant",
            "action": action,
            "object": "self_model",
            "target": "user",
        },
        "chat_frame_projection": {
            "domain": "self_model",
            "route_hint": "local_answer",
        },
        "notes": notes,
    }


def _matches_who_identity_frame(tokens: tuple[str, ...]) -> bool:
    for index, token in enumerate(tokens):
        if token != "who":
            continue
        if _identity_deixis_relation_frame(
            _identity_frame_segment(tokens[index:]), interrogative="who"
        ):
            return True
    return False


def _matches_name_identity_frame(text: str, tokens: tuple[str, ...]) -> bool:
    if "name" not in tokens or "your" not in tokens:
        return False
    token_set = set(tokens)
    attribute_question = bool(token_set & {"what", "what's"})
    explicit_request = "tell" in token_set
    question_fragment = "?" in text and _is_possessive_name_question_fragment(tokens)
    return attribute_question or explicit_request or question_fragment


def _is_possessive_name_question_fragment(tokens: tuple[str, ...]) -> bool:
    semantic_tokens = tuple(
        token for token in tokens if token not in {"please", "now", "really"}
    )
    return (
        len(semantic_tokens) == 2
        and "your" in semantic_tokens
        and "name" in semantic_tokens
    )


def _matches_kind_identity_frame(tokens: tuple[str, ...]) -> bool:
    for index, token in enumerate(tokens):
        if token != "what":
            continue
        if _identity_deixis_relation_frame(
            _identity_frame_segment(tokens[index:]), interrogative="what"
        ):
            return True
    return False


def _matches_capability_identity_frame(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    question = bool(token_set & {"what", "how"})
    capability_action = bool(token_set & {"do", "help", "support"})
    if not (question and {"can", "you"} <= token_set and capability_action):
        return False
    task_domain_terms = {
        "health",
        "healthy",
        "wellness",
        "eat",
        "food",
        "meal",
        "cook",
        "breakfast",
        "lunch",
        "dinner",
        "weather",
        "forecast",
        "rain",
        "story",
        "stories",
        "tale",
        "song",
        "music",
        "call",
        "phone",
        "school",
        "clothes",
        "naked",
        "routine",
        "family",
        "household",
        "child",
    }
    return not bool(token_set & task_domain_terms)


def _capability_identity_basis(tokens: tuple[str, ...]) -> tuple[str, ...]:
    basis = [
        "can:ability_modal",
        "you:assistant_deixis",
    ]
    if "what" in tokens:
        basis.insert(0, "what:capability_question")
    elif "how" in tokens:
        basis.insert(0, "how:capability_question")
    if "do" in tokens:
        basis.append("do:capability_action")
    if "help" in tokens:
        basis.append("help:capability_support_action")
    if "support" in tokens:
        basis.append("support:capability_support_action")
    return tuple(dict.fromkeys(basis))


def _purpose_identity_frame(
    tokens: tuple[str, ...],
) -> tuple[str, tuple[str, ...]] | None:
    if _second_person_runtime_purpose_question(tokens):
        return (
            "why_copula_second_person_here",
            (
                "why:purpose_question",
                "are:state_relation",
                "you:assistant_deixis",
                "here:runtime_purpose_context",
            ),
        )
    if _assistant_possessive_attribute_question(tokens, "purpose"):
        return (
            "what_copula_possessive_purpose",
            (
                "what:attribute_question",
                "is:state_relation",
                "your:assistant_possessive",
                "purpose:self_model_attribute",
            ),
        )
    return None


def _matches_self_description_frame(tokens: tuple[str, ...]) -> bool:
    return _assistant_reflexive_description_request(tokens)


def _second_person_runtime_purpose_question(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    return bool(
        token_set & {"why"}
        and token_set & {"you"}
        and token_set & {"are", "is"}
        and token_set & {"here"}
    )


def _assistant_possessive_attribute_question(
    tokens: tuple[str, ...], attribute: str
) -> bool:
    token_set = set(tokens)
    return bool(
        token_set & {"what", "what's"}
        and token_set & {"your"}
        and token_set & {"is", "are"}
        and attribute in token_set
    )


def _assistant_reflexive_description_request(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    request_action = bool(token_set & {"tell", "describe"})
    topic_relation = bool(token_set & {"about"} or token_set & {"yourself"})
    assistant_reflexive = bool(token_set & {"yourself"})
    return request_action and topic_relation and assistant_reflexive


def _identity_frame_segment(tokens: tuple[str, ...]) -> tuple[str, ...]:
    boundary = next(
        (
            index
            for index, token in enumerate(tokens)
            if index > 0 and token in {"and", "but", "or"}
        ),
        None,
    )
    return tokens[:boundary] if boundary is not None else tokens


def _identity_deixis_relation_frame(
    tokens: tuple[str, ...], *, interrogative: str
) -> bool:
    token_set = set(tokens)
    if not ({interrogative, "you"} <= token_set and token_set & {"are", "is"}):
        return False
    extra = (
        token_set
        - {interrogative, "you", "are", "is"}
        - _identity_relation_scope_tokens()
    )
    return not extra


def _identity_relation_scope_tokens() -> set[str]:
    return {
        "actually",
        "app",
        "assistant",
        "bot",
        "device",
        "even",
        "exactly",
        "for",
        "in",
        "inside",
        "kind",
        "local",
        "me",
        "now",
        "of",
        "on",
        "please",
        "really",
        "service",
        "sort",
        "still",
        "system",
        "that",
        "thing",
        "this",
        "to",
        "type",
    }


def _identity_token_roles(tokens: tuple[str, ...]) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    role_map = {
        "who": ("interrogative", "agent_or_identity_question"),
        "what": ("interrogative", "kind_attribute_or_capability_question"),
        "what's": ("interrogative", "attribute_question_with_copula"),
        "why": ("interrogative", "purpose_question"),
        "are": ("relation", "copula_state_relation"),
        "is": ("relation", "copula_state_relation"),
        "you": ("deictic_pronoun", "second_person_resolves_to_assistant"),
        "your": ("deictic_possessive", "assistant_owned_attribute"),
        "yourself": ("deictic_reflexive", "assistant_self_reference"),
        "name": ("identity_attribute", "self_model_name"),
        "purpose": ("identity_attribute", "self_model_purpose"),
        "can": ("modal", "capability"),
        "do": ("capability_action", "available_action_space"),
        "help": ("capability_action", "available_support_space"),
        "support": ("capability_action", "available_support_space"),
        "here": ("context", "runtime_presence_or_purpose"),
        "device": ("context", "runtime_scope"),
        "app": ("context", "runtime_scope"),
        "system": ("context", "runtime_scope"),
        "service": ("context", "runtime_scope"),
        "local": ("context", "runtime_scope"),
        "this": ("context", "runtime_scope_deictic"),
        "that": ("context", "runtime_scope_deictic"),
        "kind": ("identity_kind", "assistant_category_scope"),
        "type": ("identity_kind", "assistant_category_scope"),
        "sort": ("identity_kind", "assistant_category_scope"),
        "thing": ("identity_kind", "assistant_category_scope"),
        "assistant": ("identity_kind", "assistant_category_scope"),
        "bot": ("identity_kind", "assistant_category_scope"),
        "tell": ("request_action", "ask_assistant_to_describe"),
        "describe": ("request_action", "ask_assistant_to_describe"),
        "about": ("topic_relation", "topic_link"),
        "of": ("topic_relation", "kind_relation"),
        "on": ("scope_relation", "runtime_scope_relation"),
        "in": ("scope_relation", "runtime_scope_relation"),
        "inside": ("scope_relation", "runtime_scope_relation"),
        "to": ("scope_relation", "relation_to_user"),
        "for": ("scope_relation", "relation_to_user"),
        "with": ("scope_relation", "capability_scope_relation"),
        "me": ("response_target", "user"),
        "know": ("cognition_probe", "self_knowledge_challenge"),
        "don't": ("negation", "challenge_or_doubt"),
        "dont": ("negation", "challenge_or_doubt"),
        "not": ("negation", "challenge_or_doubt"),
        "wow": ("discourse_marker", "emotional_preface"),
        "exactly": ("emphasis", "identity_focus"),
        "even": ("emphasis", "identity_focus"),
        "still": ("emphasis", "identity_focus"),
        "really": ("emphasis", "identity_focus"),
        "actually": ("emphasis", "identity_focus"),
        "now": ("time_modifier", "current_identity_focus"),
        "please": ("politeness", "request_softener"),
        "and": ("clause_boundary", "additional_request_boundary"),
    }
    for index, token in enumerate(tokens):
        role, meaning = role_map.get(
            token, ("unresolved_token", "not_used_by_identity_composition")
        )
        roles.append({"index": index, "token": token, "role": role, "meaning": meaning})
    return roles


def _assistant_uol(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    composition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    speech_act = (
        "question"
        if "?" in text or tokens[:1] in {("who",), ("what",), ("why",), ("how",)}
        else "request"
    )
    subject = "user"
    action = _first_action_token(tokens)
    object_value = _object_hint(text, tokens, intent)
    source = ""
    target = "assistant"
    modifiers: list[str] = []
    notes: list[str] = []
    projection = (composition or {}).get("uol_projection", {}) if composition else {}
    if intent == "assistant_identity":
        subject = "assistant"
        action = str(
            projection.get("action") or ("name" if "name" in tokens else "identify")
        )
        object_value = "self_model"
        target = "user"
        notes.extend(str(item) for item in (composition or {}).get("notes", []))
        if (not notes) and ("don't" in tokens or "dont" in tokens):
            speech_act = "challenge"
            notes.append("identity_challenge_detected")
        elif notes:
            speech_act = "challenge"
    elif intent == "assistant_status":
        subject = "assistant"
        action = str(
            projection.get("action")
            or (
                "plan"
                if any(
                    token in tokens for token in ("need", "next", "missing", "build")
                )
                else "report"
            )
        )
        object_value = str(
            projection.get("object")
            or ("next_steps" if action == "plan" else "runtime_status")
        )
        source = str(projection.get("source") or "event_ledger")
        target = str(projection.get("target") or "user")
    elif intent == "weather":
        action = "answer"
        object_value = "weather"
        target = "today" if "today" in tokens else "forecast"
    elif intent == "story":
        action = "tell"
        object_value = "story"
    elif intent == "common_sense_safety":
        action = "judge"
        object_value = "school_clothing" if "school" in tokens else "public_safety"
    elif intent == "media_playback":
        action = "play"
        object_value = _media_object_from_request_tokens(tokens)
        target = "local_device"
    elif intent == "health_advice":
        action = "advise"
        object_value = "health"
    elif intent == "personal_memory":
        action = str(projection.get("action") or "recall")
        object_value = str(
            projection.get("object") or _personal_memory_object_from_text(text, tokens)
        )
        source = str(projection.get("source") or "local_memory")
        target = str(projection.get("target") or "user")
    elif intent == "autobiographical_memory":
        action = "recall"
        object_value = "conversation_events"
        source = "event_ledger"
    elif intent == "meal_suggestion":
        action = "suggest"
        object_value = "meal"
        source = "food_inventory"
    elif intent == "social_contact":
        action = "call"
        object_value = _contact_object_from_tokens(tokens)
        target = "trusted_contact"
    elif intent in {
        "social_greeting",
        "assistant_behavior",
        "personal_goal_advice",
        "open_domain",
    }:
        speech_act = str(projection.get("speech_act", speech_act))
        subject = str(projection.get("subject", subject))
        action = str(projection.get("action", action))
        object_value = str(projection.get("object", ""))
        source = str(projection.get("source", "functional_grammar"))
        target = str(projection.get("target", target))
    if projection and intent != "assistant_identity":
        source = source or str(projection.get("source", "") or "")
    if "today" in tokens:
        modifiers.append("today")
    if "local" in tokens:
        modifiers.append("local")
    return {
        "schema": "melm.assistant_uol_debug.v1",
        "speech_act": speech_act,
        "subject": subject,
        "action": action,
        "object": object_value,
        "source": source,
        "target": target,
        "modifiers": modifiers,
        "parse_score": round(
            float(
                projection.get(
                    "parse_score", _assistant_parse_score(intent, object_value, action)
                )
            ),
            3,
        ),
        "notes": notes,
        "decomposition": composition or {},
        "complement_action": str(projection.get("complement_action", "") or ""),
        "indirect_object": str(projection.get("indirect_object", "") or ""),
        "relations": list(projection.get("relations", []) or []),
    }


def _route_hint(
    intent: AssistantIntent, composition: dict[str, Any] | None = None
) -> AssistantRoute:
    projection = (composition or {}).get("uol_projection", {})
    if (
        intent == "personal_memory"
        and projection.get("target") == "external_cloud_model"
    ):
        return "cloud_handoff"
    if intent in {"weather"}:
        return "cached_tool"
    if intent in {"media_playback", "social_contact"}:
        return "device_action"
    if intent == "unknown":
        return "cloud_handoff"
    if intent in {"personal_goal_advice", "open_domain"}:
        return "cloud_handoff"
    return "local_answer"


def _route_reason_hint(
    intent: AssistantIntent, composition: dict[str, Any] | None = None
) -> str:
    projection = (composition or {}).get("uol_projection", {})
    if (
        intent == "personal_memory"
        and projection.get("target") == "external_cloud_model"
    ):
        return "private_memory_cloud_request"
    return "pre_route_parse"


def _first_action_token(tokens: tuple[str, ...]) -> str:
    action_words = (
        "tell",
        "what",
        "who",
        "should",
        "play",
        "call",
        "eat",
        "improve",
        "remember",
        "know",
        "send",
        "share",
        "upload",
        "export",
        "explain",
        "write",
    )
    return next(
        (token for token in tokens if token in action_words),
        tokens[0] if tokens else "",
    )


def _object_hint(text: str, tokens: tuple[str, ...], intent: AssistantIntent) -> str:
    if intent == "unknown":
        return " ".join(tokens[1:4]) if len(tokens) > 1 else ""
    if "story" in tokens:
        return "story"
    if "weather" in tokens:
        return "weather"
    if "school" in tokens:
        return "school"
    if "health" in tokens:
        return "health"
    if "me" in tokens or "myself" in tokens:
        return "user_profile"
    return text


def _media_object_from_request_tokens(tokens: tuple[str, ...]) -> str:
    token_set = set(tokens)
    for token in ("song", "music", "radio", "lofi", "audio", "track"):
        if token in token_set:
            return token
    if token_set & {"piano", "sound", "sounds"}:
        return "music"
    return "media"


def _personal_memory_object_from_text(text: str, tokens: tuple[str, ...]) -> str:
    token_set = set(tokens)
    if _is_child_memory_request(tokens):
        if "school" in token_set:
            return "facts.child_school"
        if token_set & {"age", "old"}:
            return "facts.child_age"
        if token_set & {"name", "called"}:
            return "facts.child_name"
        return "child_memory"
    if {"favorite", "color"} <= token_set:
        return "facts.favorite_color"
    if {"where", "i", "live"} <= token_set or {"where", "i", "lived"} <= token_set:
        return "profile.location"
    if "my" in token_set and "age" in token_set:
        return "profile.age"
    if {"health", "goal"} <= token_set or {"health", "goals"} <= token_set:
        return "health_goals"
    if _is_routine_memory_request(tokens):
        return "routine_memory"
    if _is_household_memory_request(tokens):
        return "household_memory"
    if "mom" in token_set or "contact" in token_set:
        return "contacts.local"
    return "user_profile"


def _contact_object_from_tokens(
    tokens: tuple[str, ...],
    *,
    trusted_contact_names: tuple[str, ...] = (),
) -> str:
    trusted_contact = _matched_trusted_contact_name(tokens, trusted_contact_names)
    if trusted_contact:
        return trusted_contact
    for token in tokens:
        if token in {"mom", "dad", "caregiver"}:
            return "relationship_contact"
        if token == "someone":
            return token
    return "trusted_contact"


def _matched_trusted_contact_name(
    tokens: tuple[str, ...], trusted_contact_names: tuple[str, ...]
) -> str:
    for name in trusted_contact_names:
        normalized_name = _normalize(name)
        name_tokens = _tokenize(normalized_name)
        if name_tokens and _has_token_sequence(tokens, name_tokens):
            return normalized_name
    return ""


def _assistant_parse_score(
    intent: AssistantIntent, object_value: str, action: str
) -> float:
    score = 0.42
    if intent != "unknown":
        score += 0.33
    if object_value:
        score += 0.15
    if action:
        score += 0.1
    return round(min(score, 1.0), 3)


def _assistant_frame_complexity(uol: dict[str, Any], intent: AssistantIntent) -> float:
    slot_count = sum(
        1 for key in ("subject", "action", "object", "source", "target") if uol.get(key)
    )
    base = 0.08 + slot_count * 0.07
    if intent in {"media_playback", "social_contact"}:
        base += 0.18
    if intent in {"autobiographical_memory", "personal_memory"}:
        base += 0.14
    if intent == "assistant_status":
        base += 0.1
    if intent == "unknown":
        base += 0.28
    return round(min(base, 1.0), 3)


def _basic_nlp_debug(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    secondary_meaning_hints: tuple[str, ...],
    domain_hints: dict[str, list[str]],
    secondary_domain_hints: dict[str, list[str]],
    composition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unknown_tokens = _unknown_tokens(tokens, intent, composition)
    semantic_unknown_tokens = list(
        (composition or {}).get("semantic_unknown_tokens", [])
    )
    return {
        "schema": "melm.basic_nlp_debug.v1",
        "language": "en",
        "token_count": len(tokens),
        "question_like": "?" in text
        or tokens[:1] in {("who",), ("what",), ("why",), ("how",), ("should",)},
        "imperative_like": bool(
            tokens[:1]
            and tokens[0]
            in {
                "tell",
                "describe",
                "play",
                "call",
                "send",
                "remember",
                "forget",
                "show",
                "report",
                "summarize",
                "list",
            }
        ),
        "bounded_intent": intent,
        "primary_parse_basis": "uol_chat_frame",
        "primary_domain_evidence": _primary_domain_evidence(intent, composition),
        "secondary_hint_policy": "debug_only_never_primary_route",
        "secondary_meaning_hints": list(secondary_meaning_hints),
        "secondary_lexical_evidence": _secondary_lexical_evidence(
            intent, secondary_meaning_hints
        ),
        "secondary_domain_hints": secondary_domain_hints,
        "domain_hints": domain_hints,
        "unknown_tokens": list(unknown_tokens),
        "unknown_token_count": len(unknown_tokens),
        "semantic_unknown_tokens": semantic_unknown_tokens,
        "semantic_unknown_token_count": len(semantic_unknown_tokens),
        "token_roles": list((composition or {}).get("token_roles", [])),
        "compositional_parse": composition or {},
        "functional_parse": dict((composition or {}).get("functional_parse", {})),
        "candidate_parses": list((composition or {}).get("candidate_parses", [])),
    }


def _debug_mapping(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    uol: dict[str, Any],
    chat_frame: dict[str, Any],
    secondary_meaning_hints: tuple[str, ...],
    domain_hints: dict[str, list[str]],
    secondary_domain_hints: dict[str, list[str]],
    composition: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "stage": "basic_nlp",
            "input": "utterance",
            "output": {
                "normalized": text,
                "tokens": list(tokens),
                "secondary_meaning_hints": list(secondary_meaning_hints),
                "secondary_hint_policy": "debug_only_never_primary_route",
                "secondary_lexical_evidence": _secondary_lexical_evidence(
                    intent, secondary_meaning_hints
                ),
                "primary_domain_evidence": _primary_domain_evidence(
                    intent, composition
                ),
                "secondary_domain_hints": secondary_domain_hints,
                "domain_hints": domain_hints,
                "unknown_tokens": list(_unknown_tokens(tokens, intent, composition)),
                "semantic_unknown_tokens": list(
                    (composition or {}).get("semantic_unknown_tokens", [])
                ),
                "token_roles": list((composition or {}).get("token_roles", [])),
                "compositional_parse": composition or {},
                "functional_parse": dict(
                    (composition or {}).get("functional_parse", {})
                ),
                "candidate_parses": list(
                    (composition or {}).get("candidate_parses", [])
                ),
                "bounded_intent": intent,
            },
        },
        {
            "stage": "uol_parse",
            "input": "basic_nlp",
            "output": {
                "speech_act": uol.get("speech_act", ""),
                "subject": uol.get("subject", ""),
                "action": uol.get("action", ""),
                "object": uol.get("object", ""),
                "source": uol.get("source", ""),
                "target": uol.get("target", ""),
                "parse_score": uol.get("parse_score", 0.0),
                "slot_sources": uol.get("slot_sources", {}),
                "decomposition": uol.get("decomposition", {}),
                "complement_action": uol.get("complement_action", ""),
                "indirect_object": uol.get("indirect_object", ""),
                "relations": uol.get("relations", []),
            },
        },
        {
            "stage": "chat_frame",
            "input": "uol_parse",
            "output": {
                "intent": chat_frame.get("intent", ""),
                "domain": chat_frame.get("domain", ""),
                "route": chat_frame.get("route", ""),
                "reason": chat_frame.get("reason", ""),
                "needs_tool": bool(chat_frame.get("needs_tool", False)),
                "needs_cloud": bool(chat_frame.get("needs_cloud", False)),
                "needs_confirmation": bool(chat_frame.get("needs_confirmation", False)),
                "can_answer_locally": bool(chat_frame.get("can_answer_locally", False)),
                "capabilities": chat_frame.get("capabilities", {}),
                "frame_registry": chat_frame.get("frame_registry", ""),
                "frame_id": chat_frame.get("frame_id", ""),
                "frame_source_policy": chat_frame.get("frame_source_policy", ""),
                "primary_routing_basis": chat_frame.get("primary_routing_basis", []),
                "secondary_debug_hints": chat_frame.get("secondary_debug_hints", []),
                "secondary_hint_policy": chat_frame.get("secondary_hint_policy", ""),
                "complexity_score": chat_frame.get("complexity_score", 0.0),
            },
        },
    )


def _unknown_token_count(tokens: tuple[str, ...], intent: AssistantIntent) -> int:
    return len(_unknown_tokens(tokens, intent))


def _unknown_tokens(
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    composition: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if composition:
        resolved = {
            str(item.get("token", ""))
            for item in composition.get("token_roles", [])
            if item.get("role") != "unresolved_token"
        }
        return tuple(token for token in tokens if token not in resolved)
    known = _known_debug_tokens(intent)
    return tuple(token for token in tokens if token not in known)


def _known_debug_tokens(intent: AssistantIntent) -> set[str]:
    known = set(_structural_debug_tokens())
    if intent != "unknown":
        known.update(_tokenize(intent.replace("_", " ")))
    return known


def _structural_debug_tokens() -> set[str]:
    return {
        "a",
        "about",
        "and",
        "are",
        "at",
        "do",
        "for",
        "i",
        "is",
        "it",
        "me",
        "my",
        "our",
        "what",
        "what's",
        "who",
        "why",
        "how",
        "should",
        "that",
        "the",
        "this",
        "to",
        "today",
        "you",
        "your",
    }


def _domain_hints(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    composition: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    cleaned = dict(_secondary_domain_hints(text))
    if intent == "assistant_identity" and composition:
        cleaned["assistant_identity"] = [
            str(composition.get("pattern", "identity_token_composition")),
            *[str(item) for item in composition.get("basis", [])],
        ]
    return cleaned


def _secondary_domain_hints(text: str) -> dict[str, list[str]]:
    hints = {
        group_intent: [marker for marker in markers if _has_marker(text, marker)]
        for group_intent, markers in _secondary_meaning_hint_groups().items()
    }
    return {group_intent: markers for group_intent, markers in hints.items() if markers}


def _slot_sources(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    uol: dict[str, Any],
    composition: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    sources = {
        "subject": {
            "value": str(uol.get("subject", "")),
            "source": _subject_source(intent, composition),
        },
        "action": {
            "value": str(uol.get("action", "")),
            "source": _action_source(intent, tokens, composition),
        },
        "object": {
            "value": str(uol.get("object", "")),
            "source": _object_source(intent, text, tokens, composition),
        },
        "source": {
            "value": str(uol.get("source", "")),
            "source": _evidence_source_for_intent(intent),
        },
        "target": {
            "value": str(uol.get("target", "")),
            "source": _target_source(intent, str(uol.get("target", ""))),
        },
    }
    return sources


def _subject_source(
    intent: AssistantIntent, composition: dict[str, Any] | None = None
) -> str:
    if composition and composition.get("source") == "weighted_functional_relation":
        return f"weighted_functional_subject:{composition.get('pattern', '')}"
    if intent == "assistant_identity" and composition:
        return "second_person_deixis_resolved_to_assistant"
    if intent == "assistant_status" and composition:
        return "self_status_composition_resolves_assistant_subject"
    if intent in {"assistant_identity", "assistant_status"}:
        return "self_model_override"
    return "default_user_speaker"


def _action_source(
    intent: AssistantIntent,
    tokens: tuple[str, ...],
    composition: dict[str, Any] | None = None,
) -> str:
    if composition and composition.get("source") == "weighted_functional_relation":
        return f"weighted_functional_predicate:{composition.get('pattern', '')}"
    if intent == "assistant_identity" and composition:
        return (
            f"identity_composition:{composition.get('pattern', 'token_role_relation')}"
        )
    if intent == "assistant_identity" and "name" in tokens:
        return "name_token"
    if intent == "assistant_identity":
        return "identity_composition_unavailable"
    if intent == "assistant_status" and composition:
        return f"self_status_composition:{composition.get('pattern', 'slot_role_relation')}"
    if intent == "assistant_status":
        return "status_or_planning_uol_slots"
    if intent == "weather":
        return "weather_question_slots"
    if intent == "story":
        return "story_request_slots"
    if intent == "media_playback":
        return "media_playback_command_slots"
    if intent == "social_contact":
        return "contact_action_slots"
    if intent == "unknown":
        return "first_action_like_token"
    return "intent_action_slot_rule"


def _object_source(
    intent: AssistantIntent,
    text: str,
    tokens: tuple[str, ...],
    composition: dict[str, Any] | None = None,
) -> str:
    if composition and composition.get("source") == "weighted_functional_relation":
        return f"weighted_functional_object:{composition.get('pattern', '')}"
    if intent == "assistant_identity" and composition:
        return f"self_model_from_identity_composition:{composition.get('pattern', 'token_role_relation')}"
    if intent == "assistant_identity":
        return "self_model_from_identity_composition_unavailable"
    if intent == "assistant_status" and composition:
        return f"runtime_self_observation_from_status_composition:{composition.get('pattern', 'slot_role_relation')}"
    if intent == "assistant_status":
        return (
            "runtime_status_slots"
            if "next" not in tokens and "need" not in tokens
            else "next_steps_slots"
        )
    if intent == "personal_memory":
        memory_object = _personal_memory_object_from_text(text, tokens)
        if _is_child_memory_request(tokens):
            return "child_owned_memory_slots"
        if _is_routine_memory_request(tokens):
            return "routine_memory_slots"
        if _is_household_memory_request(tokens):
            return "household_memory_slots"
        if memory_object.startswith("facts."):
            return "owned_fact_memory_slots"
        if memory_object.startswith("profile."):
            return "profile_attribute_memory_slots"
        if memory_object == "health_goals":
            return "health_goal_memory_slots"
        if memory_object == "contacts.local":
            return "contact_memory_slots"
        return "profile_memory_slots"
    if intent == "autobiographical_memory":
        return "conversation_memory_slots"
    if intent == "story":
        return "story_request_slots"
    if intent == "weather":
        return "weather_question_slots"
    if intent == "common_sense_safety":
        return "safety_policy_slots"
    if intent == "health_advice":
        return "health_advice_slots"
    if intent == "meal_suggestion":
        return "meal_request_slots"
    if intent == "media_playback":
        return "requested_media_or_default_media_slot"
    if intent == "social_contact":
        return "requested_contact_or_trusted_contact_slot"
    return "object_slot_from_tokens"


def _evidence_source_for_intent(intent: AssistantIntent) -> str:
    return {
        "assistant_identity": "self_model",
        "assistant_status": "event_ledger",
        "story": "story_inventory",
        "weather": "weather_cache",
        "common_sense_safety": "local_policy",
        "media_playback": "media_library",
        "health_advice": "local_health_policy",
        "personal_memory": "local_memory",
        "autobiographical_memory": "event_ledger",
        "meal_suggestion": "food_inventory",
        "social_contact": "trusted_contacts",
        "social_greeting": "self_model",
        "assistant_behavior": "self_model",
        "personal_goal_advice": "functional_grammar",
        "open_domain": "functional_grammar",
        "unknown": "none",
    }.get(intent, "none")


def _target_source(intent: AssistantIntent, target: str = "") -> str:
    if intent in {"media_playback", "social_contact"}:
        return "device_action_target"
    if intent == "personal_memory" and target == "external_cloud_model":
        return "policy_boundary_target"
    if intent == "personal_memory" and target == "user":
        return "answer_to_user"
    if intent == "weather":
        return "time_scope"
    if intent in {"assistant_identity", "assistant_status"}:
        return "answer_to_user"
    if intent in {"social_greeting", "assistant_behavior"}:
        return "answer_to_user"
    if intent in {"personal_goal_advice", "open_domain"}:
        return "external_cloud_model"
    return "assistant_response"


def _intent_domain(intent: AssistantIntent) -> str:
    return {
        "assistant_identity": "self_model",
        "assistant_status": "runtime_self_observation",
        "story": "story_inventory",
        "weather": "cached_weather",
        "common_sense_safety": "local_policy",
        "media_playback": "local_device_action",
        "health_advice": "bounded_health_policy",
        "personal_memory": "personal_memory",
        "autobiographical_memory": "autobiographical_memory",
        "meal_suggestion": "food_inventory",
        "social_contact": "trusted_contact_action",
        "social_greeting": "social_protocol",
        "assistant_behavior": "self_model_behavior",
        "personal_goal_advice": "personal_goal_advice",
        "open_domain": "understood_open_domain",
        "unknown": "unknown_open_domain",
    }.get(intent, "unknown_open_domain")


def _frame_capabilities(
    intent: AssistantIntent,
    route: AssistantRoute,
    decision: AssistantDecision | None,
) -> dict[str, Any]:
    sources_by_intent = {
        "assistant_identity": ("self_model",),
        "assistant_status": ("event_ledger", "self_state"),
        "story": ("story_inventory",),
        "weather": ("weekly_weather_cache",),
        "common_sense_safety": ("local_safety_policy",),
        "media_playback": ("media_library", "local_action_executor"),
        "health_advice": ("local_health_policy", "profile_health_goals"),
        "personal_memory": ("user_facts", "preferences"),
        "autobiographical_memory": ("events", "memory_digest"),
        "meal_suggestion": ("food_inventory", "weekly_weather_cache"),
        "social_contact": ("trusted_contacts", "local_action_executor"),
        "social_greeting": ("self_model",),
        "assistant_behavior": ("self_model",),
        "personal_goal_advice": (),
        "open_domain": (),
        "unknown": (),
    }
    return {
        "local_sources": list(sources_by_intent.get(intent, ())),
        "route": route,
        "local_answer_possible": route
        in {"local_answer", "cached_tool", "device_action"},
        "tool_cache_possible": intent == "weather",
        "device_action_possible": intent in {"media_playback", "social_contact"},
        "requires_confirmation": bool(decision.device_action)
        if decision is not None
        else intent in {"media_playback", "social_contact"},
        "cloud_handoff_possible": route == "cloud_handoff"
        or intent in {"story", "personal_goal_advice", "open_domain", "unknown"},
        "external_fetch_possible": intent == "weather",
    }


def _primary_routing_basis(
    intent: AssistantIntent,
    route: AssistantRoute,
    reason: str,
    uol: dict[str, Any],
    capabilities: dict[str, Any],
    composition: dict[str, Any] | None = None,
) -> list[str]:
    basis = [
        f"bounded_intent:{intent}",
        f"uol_object:{uol.get('object', '')}",
        f"route:{route}",
        f"reason:{reason}",
    ]
    if composition:
        frame_registry = str(composition.get("frame_registry", ""))
        frame_id = str(composition.get("frame_id", ""))
        source_policy = str(composition.get("source_policy", ""))
        if frame_registry:
            basis.append(f"frame_registry:{frame_registry}")
        if frame_id:
            basis.append(f"frame_id:{frame_id}")
        if source_policy:
            basis.append(f"source_policy:{source_policy}")
        basis.append(f"composition:{composition.get('pattern', 'token_role_relation')}")
        for item in composition.get("basis", []):
            basis.append(f"token_role:{item}")
    local_sources = capabilities.get("local_sources") or []
    if local_sources:
        basis.append(f"local_sources:{','.join(str(item) for item in local_sources)}")
    if capabilities.get("requires_confirmation"):
        basis.append("confirmation_gate:required_before_side_effect")
    return basis


def _secondary_debug_hints(secondary_meaning_hints: tuple[str, ...]) -> list[str]:
    if not secondary_meaning_hints:
        return []
    return [f"secondary_debug_hint:{','.join(secondary_meaning_hints)}"]


def _primary_domain_evidence(
    intent: AssistantIntent,
    composition: dict[str, Any] | None,
) -> dict[str, Any]:
    if composition:
        return {
            "intent": intent,
            "source": str(composition.get("source", "")),
            "pattern": str(composition.get("pattern", "")),
            "frame_registry": str(composition.get("frame_registry", "")),
            "frame_id": str(composition.get("frame_id", "")),
            "source_policy": str(composition.get("source_policy", "")),
            "secondary_hint_policy": str(composition.get("secondary_hint_policy", "")),
            "basis": list(composition.get("basis", [])),
        }
    return {
        "intent": intent,
        "source": "no_local_composition"
        if intent == "unknown"
        else "uol_slot_classifier",
        "pattern": "",
        "basis": [],
    }


def _secondary_lexical_evidence(
    intent: AssistantIntent,
    secondary_meaning_hints: tuple[str, ...],
) -> list[dict[str, str]]:
    return [
        {
            "intent": intent,
            "marker": marker,
            "basis": "secondary_token_sequence",
        }
        for marker in secondary_meaning_hints
    ]


def _secondary_meaning_hints(text: str, intent: AssistantIntent) -> tuple[str, ...]:
    if intent == "assistant_identity":
        return ()
    hints = [
        marker
        for marker in _secondary_meaning_hint_groups().get(intent, ())
        if _has_marker(text, marker)
    ]
    return tuple(dict.fromkeys(hints))


def _secondary_meaning_hint_groups() -> dict[str, tuple[str, ...]]:
    return {
        "assistant_status": (
            "status",
            "ledger",
            "cloud",
            "memory",
            "missing",
            "next",
        ),
        "story": ("story", "tale", "fable", "bedtime"),
        "weather": ("weather", "forecast", "temperature", "rain"),
        "common_sense_safety": ("naked", "clothes", "wear", "school"),
        "media_playback": ("song", "music", "piano", "radio", "lofi", "sounds"),
        "health_advice": (
            "health",
            "healthy",
            "healthier",
            "doctor",
            "medicine",
            "poison",
            "breathe",
        ),
        "personal_memory": (
            "remember",
            "recall",
            "routine",
            "morning",
            "household",
            "family",
            "profile",
        ),
        "autobiographical_memory": (
            "earlier",
            "recent",
            "previous",
            "conversation",
            "sessions",
        ),
        "meal_suggestion": ("eat", "food", "meal", "breakfast", "lunch", "dinner"),
        "social_contact": ("talk", "call", "reach", "caregiver", "someone"),
    }


def _debug_notes(
    text: str,
    intent: AssistantIntent,
    route: str,
    reason: str,
) -> tuple[str, ...]:
    notes: list[str] = []
    if intent == "unknown":
        notes.append("classifier_fell_through_to_unknown")
    if route == "cloud_handoff":
        if reason == "private_memory_cloud_request":
            notes.append("private_memory_cloud_boundary_requires_policy")
        else:
            notes.append(
                "would_leave_local_runtime_without_new_local_rule_or_inventory"
            )
    if intent == "assistant_identity" and reason in {
        "pre_route_parse",
        "self_model_identity",
    }:
        notes.append("identity_should_be_local_self_model")
    if intent == "assistant_status":
        notes.append("status_should_use_local_ledger")
    if "?" not in text and intent in {"assistant_identity", "personal_memory"}:
        tokens = _tokenize(text)
        outward_request = tokens[:1] in {
            ("send",),
            ("share",),
            ("upload",),
            ("export",),
            ("give",),
            ("tell",),
        }
        if _is_question_like(text, tokens):
            notes.append("question_mapped_by_semantic_parse_not_question_mark")
        elif outward_request or _is_request_like(tokens):
            notes.append("request_mapped_by_semantic_parse_not_question_mark")
        else:
            notes.append("statement_mapped_by_semantic_parse_not_question_mark")
    return tuple(notes)


def _requested_contact(text: str, contacts: dict[str, str]) -> str:
    for name in contacts:
        if _has_marker(text, name.lower()):
            return name
    return next(iter(contacts))


def _requested_media(text: str, media_library: tuple[str, ...]) -> str:
    for title in media_library:
        if _has_marker(text, title.lower()):
            return title
    return ""


def _is_broad_personal_memory_request(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if {"who", "am", "i"} <= token_set:
        return True
    if _about_targets_self(tokens):
        return True
    if (
        "you" in token_set
        and token_set & {"remember", "know", "recall"}
        and token_set & {"me", "myself"}
    ):
        return True
    return False


def _is_routine_memory_request(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if token_set & {"routine", "schedule", "morning", "bedtime"}:
        return True
    return bool({"school", "day"} <= token_set or {"work", "day"} <= token_set)


def _is_household_memory_request(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    if token_set & {"household", "family"}:
        return True
    if {"shared", "device"} <= token_set:
        return True
    return _is_device_user_memory_question(tokens)


def _is_device_user_memory_question(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    return bool(
        "who" in token_set
        and token_set & {"use", "uses", "using"}
        and token_set & {"device", "assistant"}
    )


def _is_child_memory_request(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    child_terms = {
        "child",
        "kid",
        "son",
        "daughter",
        "child's",
        "kid's",
        "son's",
        "daughter's",
    }
    if not token_set & child_terms:
        return False
    possessive_child = bool(token_set & {"child's", "kid's", "son's", "daughter's"})
    owned_child = bool(token_set & {"my", "our"} and token_set & child_terms)
    about_child = bool("about" in token_set and token_set & child_terms)
    return possessive_child or owned_child or about_child


def _has_routine_fact(profile: LocalAssistantProfile) -> bool:
    return any("routine" in key or "schedule" in key for key in profile.facts)


def _has_household_fact(profile: LocalAssistantProfile) -> bool:
    return any("household" in key or "family" in key for key in profile.facts)


def _has_child_fact(profile: LocalAssistantProfile) -> bool:
    return any(
        any(marker in key for marker in ("child", "son", "daughter"))
        for key in profile.facts
    )


def _first_matching_fact(
    profile: LocalAssistantProfile,
    markers: tuple[str, ...],
) -> tuple[str, str]:
    for key, value in profile.facts.items():
        if any(marker in key for marker in markers):
            return key, value
    return "", ""


def _personal_summary_evidence_keys(profile: LocalAssistantProfile) -> tuple[str, ...]:
    keys: list[str] = []
    if profile.age > 0:
        keys.append("profile.age")
    if profile.location and profile.location.lower() != "unknown":
        keys.append("profile.location")
    if profile.culture and profile.culture.lower() != "unknown":
        keys.append("profile.culture")
    for fact_key in tuple(profile.facts)[:3]:
        keys.append(f"facts.{fact_key}")
    if "story_theme" in profile.preferences:
        keys.append("preferences.story_theme")
    for preference_key in tuple(profile.preferences)[:2]:
        keys.append(f"preferences.{preference_key}")
    return tuple(dict.fromkeys(keys))


def _has_personal_summary_memory(profile: LocalAssistantProfile) -> bool:
    return bool(profile.facts) or "story_theme" in profile.preferences


def _has_marker(text: str, marker: str) -> bool:
    """Token-sequence check for secondary hints and post-frame target resolution."""

    marker_tokens = _tokenize(_normalize(marker))
    if not marker_tokens:
        return False
    return _has_token_sequence(_tokenize(text), marker_tokens)


def _has_any_token_sequence(
    tokens: tuple[str, ...], sequences: tuple[tuple[str, ...], ...]
) -> bool:
    return any(_has_token_sequence(tokens, sequence) for sequence in sequences)


def _has_token_sequence(tokens: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    if not sequence:
        return False
    width = len(sequence)
    return any(
        tokens[index : index + width] == sequence
        for index in range(0, len(tokens) - width + 1)
    )


def _has_urgent_health_frame(tokens: tuple[str, ...]) -> bool:
    token_set = set(tokens)
    urgent_terms = {"bleeding", "poison", "faint", "emergency"}
    urgent_pairs = (
        ("chest", "pain"),
        ("cannot", "breathe"),
        ("can't", "breathe"),
    )
    return bool(
        token_set & urgent_terms or _has_any_token_sequence(tokens, urgent_pairs)
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())
