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

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from melm.contracts import load_food_tags, load_igbo_greetings, load_igbo_lexicon_seed, load_meal_scopes, load_router_semantic_aliases, load_yoruba_greetings, load_swahili_greetings

from .assistant_skill_meal import MealSuggestion, suggest_meal

from .functional_grammar import (
    FunctionalParse,
    _lemma,
    functional_frame_kind,
    parse_functional_relations,
    set_uol_lexicon,
)
from .assistant_lexicon_legacy import build_legacy_in_memory_lexicon
from .assistant_frame_linker import FrameCandidate, FrameLinker
from .assistant_frame_ranker import E3CandidateReranker
from .language_adapters import SyntaxGraph, build_syntax_graph, detect_language, get_adapter
from .language_adapters.igbo import seed_igbo_lexicon
from .uol_atomizer import atomize_syntax_graph
from .assistant_mood_engine import (
    MoodState,
    compute_utterance_affect,
    detect_identity_claim,
    detect_identity_probe,
    initial_mood_from_baseline,
    load_affect_lexicon,
    load_mood_regions,
    load_response_pools,
    update_session_mood,
)
from .uol_types import AffectSignal

import functools

_MORAL_ENGINE: Any | None = None

# Maps surface entity types to verb_states.v1.json semantic class labels
_PATIENT_TYPE_CLASS_MAP: dict[str, str] = {
    "person": "person", "people": "person", "someone": "person", "anyone": "person",
    "man": "person", "woman": "person", "child": "person", "kid": "person",
    "animal": "living_thing.animal", "dog": "living_thing.animal", "cat": "living_thing.animal",
    "pet": "living_thing.animal",
    "clothes": "clothing_item", "shirt": "clothing_item", "pants": "clothing_item",
    "shorts": "clothing_item", "jacket": "clothing_item", "coat": "clothing_item",
    "clothing": "clothing_item", "dress": "clothing_item", "shoe": "clothing_item",
    "wall": "physical_object", "table": "physical_object",
    "group": "group", "team": "group", "crowd": "group",
}

def _resolve_patient_type(patient_raw: str) -> str:
    """Map surface patient text to verb_states semantic class label."""
    return _PATIENT_TYPE_CLASS_MAP.get(patient_raw, "person")

def _lazy_moral_engine():
    global _MORAL_ENGINE
    if _MORAL_ENGINE is None:
        from melm.appliance.reasoning.implications import derive_moral_context
        from melm.contracts import load_contract_json
        verb_data = load_contract_json("verb_states.v1.json")
        valence_data = load_contract_json("state_valences.v1.json").get("valences", {})
        _MORAL_ENGINE = functools.lru_cache(maxsize=128)(
            lambda v, p: derive_moral_context(v, p, verb_data, valence_data)
        )
    return _MORAL_ENGINE


def _extract_verb(functional_parse=None, uol_act=None):
    """Extract verb lemma from whichever parse is available."""
    if functional_parse is not None:
        action = getattr(functional_parse, "action", None)
        if action and isinstance(action, str):
            return action.strip().lower()
    if uol_act is not None:
        content = uol_act.get("content") if isinstance(uol_act, dict) else None
        if content and isinstance(content, list) and len(content) > 0:
            first = content[0] if isinstance(content[0], dict) else {}
            pred = first.get("predicate", {}) if isinstance(first, dict) else {}
            lemma = pred.get("lemma") if isinstance(pred, dict) else None
            if lemma and isinstance(lemma, str):
                return lemma.strip().lower()
    return ""


def _extract_patient_type(functional_parse=None, uol_act=None):
    """Extract patient/object entity type from parse."""
    if functional_parse is not None:
        obj = getattr(functional_parse, "object", None)
        if obj and isinstance(obj, str):
            return obj.strip().lower()
    if uol_act is not None:
        content = uol_act.get("content") if isinstance(uol_act, dict) else None
        if content and isinstance(content, list) and len(content) > 0:
            first = content[0] if isinstance(content[0], dict) else {}
            for role in first.get("roles", []) if isinstance(first, dict) else []:
                if isinstance(role, dict) and role.get("role") in ("theme", "patient"):
                    return role.get("value", "").lower()
    return ""

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
    language_preference: str = "english"
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
    user_id: str = "default"


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
    semantic_classes_activated: frozenset[str] = frozenset()
    slot_states: dict[str, str] = field(default_factory=dict)  # slot_name → state constant
    functional_parse: dict[str, Any] | None = None
    uol_act: dict[str, Any] | None = None
    utterance_affect: Any = None
    session_mood: Any = None
    intent_occurrence: int = 0
    rapid_occurrence: int = 0
    active_user_id: str = "default"
    familiarity: int = 0
    # Boundary fields for cross-turn behavior context (slice 1).
    prev_mood: Any = None
    prev_intent: str = ""
    ambient_valence: float = 0.0
    ambient_valence_delta: float = 0.0
    # Reasoning-layer outputs marshalled to synthesis (slice 4).
    reasoning_result: dict | None = None
    reasoning_provenance: dict | None = None
    refusal_signal: str | None = None


@dataclass(frozen=True)
class _ParseBundle:
    language: str
    text: str
    tokens: tuple[str, ...]
    syntax_graph: SyntaxGraph
    functional_parse: FunctionalParse | None
    uol_act: dict[str, Any] | None
    lemmas: tuple[str, ...] = ()
    informal_affect: tuple[dict[str, Any], ...] = ()
    last_intent: str = ""


def _build_parse_bundle(utterance: str, last_intent: str = "") -> _ParseBundle:
    detected_lang, _conf = detect_language(utterance)
    adapter = get_adapter(detected_lang)
    # Layer 0 surface repair (slang/abbreviation/typo expansion) before
    # normalize/tokenize, so "gimme a story" parses like "give me a story".
    if adapter is not None and hasattr(adapter, "correct"):
        utterance = adapter.correct(utterance)
    text = adapter.normalize(utterance) if adapter else _normalize(utterance)
    tokens = adapter.tokenize(text) if adapter else _tokenize(text)

    # Phase 2: Strip noise tokens from parse, collect as informal affect signals
    from melm.contracts.validation import load_noise_tokens, ContractValidationError
    try:
        noise = load_noise_tokens()
    except (ContractValidationError, OSError):
        noise = {}
    content_tokens = []
    informal_affect = []
    for t in tokens:
        entry = noise.get(t.lower())
        if entry and entry.get("strip_from_parse", True):
            v = entry.get("valence", 0.0)
            a = entry.get("arousal", 0.0)
            if v != 0.0 or a != 0.0:
                informal_affect.append({
                    "valence": v,
                    "arousal": a,
                    "tags": entry.get("tags", []),
                    "confidence": 0.4,
                    "source": "informal",
                })
        else:
            content_tokens.append(t)
    stripped = tuple(content_tokens) if content_tokens else tokens
    lemmas = tuple(_lemma(t) for t in stripped)

    syntax_graph = (
        adapter.tag(stripped)
        if adapter is not None
        else build_syntax_graph(detected_lang, stripped, stripped)
    )
    parse = parse_functional_relations(
        stripped,
        question_mark="?" in text,
        language=detected_lang,
    )
    act = atomize_syntax_graph(syntax_graph)
    return _ParseBundle(
        language=detected_lang,
        text=text,
        tokens=stripped,
        syntax_graph=syntax_graph,
        functional_parse=parse,
        uol_act=act.to_dict() if act is not None else None,
        lemmas=lemmas,
        informal_affect=tuple(informal_affect),
        last_intent=last_intent,
    )


def _aggregate_informal_affect(entries: tuple[dict[str, Any], ...]) -> AffectSignal | None:
    """Aggregate stripped informal tokens (lol/haha/yay/ugh) into one signal.

    Used as a low-priority fallback when no lexicon/UOL/perception affect is
    present, so a bare ``"haha"`` or ``"ugh"`` still registers speaker mood
    instead of being silently dropped.
    """
    valid = [e for e in entries if e]
    if not valid:
        return None
    n = len(valid)
    valence = sum(float(e.get("valence", 0.0)) for e in valid) / n
    arousal = sum(float(e.get("arousal", 0.0)) for e in valid) / n
    tags: set[str] = set()
    for e in valid:
        tags.update(e.get("tags", []))
    confidence = max(float(e.get("confidence", 0.4)) for e in valid)
    return AffectSignal(
        valence=valence,
        arousal=arousal,
        confidence=confidence,
        source="informal",
        dominant_tags=tuple(sorted(tags)),
    )


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
        *,
        question_like: bool | None = None,
        functional_parse: FunctionalParse | None = None,
    ) -> AssistantFrameMatch | None:
        composition = _compose_primary_frame(
            text,
            tokens,
            intent,
            question_like=question_like,
            functional_parse=functional_parse,
        )
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
        store: Any = None,
    ) -> None:
        self.profile = profile or LocalAssistantProfile()
        self.store = store
        self._mood_state: dict[str, MoodState] = {}
        self._rapid_state: dict[str, dict[str, Any]] = {}
        self._mood_regions_cache: list[dict[str, Any]] = []
        self._affect_lexicon_cache: dict[str, Any] = {}
        self._pools_cache: dict[str, Any] = {}
        self._mood_regions_loaded: bool = False

    def _build_parse_bundle(self, utterance: str, last_intent: str = "") -> _ParseBundle:
        return _build_parse_bundle(utterance, last_intent=last_intent)

    def _mood_regions(self) -> list[dict[str, Any]]:
        if not self._mood_regions_loaded:
            self._mood_regions_cache = load_mood_regions()
            self._mood_regions_loaded = True
        return self._mood_regions_cache

    def _affect_lexicon(self) -> dict[str, Any]:
        if not self._affect_lexicon_cache:
            self._affect_lexicon_cache = load_affect_lexicon()
        return self._affect_lexicon_cache

    def _pools(self) -> dict[str, Any]:
        if not self._pools_cache:
            self._pools_cache = load_response_pools()
        return self._pools_cache

    def _mood_regions_list(self) -> list[dict[str, Any]]:
        raw = self._mood_regions()
        moods = raw.get("moods", {})
        return list(moods.values()) if isinstance(moods, dict) else moods

    def _load_or_init_mood(
        self, user_id: str,
    ) -> MoodState:
        from datetime import datetime, timezone
        regions = self._mood_regions_list()
        mood = initial_mood_from_baseline(user_id, self.store, regions)
        if not mood.last_updated:
            mood.last_updated = datetime.now(timezone.utc).isoformat()
        return mood

    def _infer_intent_hint(
        self, uol_act: dict[str, Any] | None,
    ) -> str | None:
        if uol_act is None:
            return None
        content = uol_act.get("content", [])
        if not content:
            return None
        main = content[0]
        pred = main.get("predicate", {})
        pred_id = str(pred.get("id", "")).strip().lower()
        affect = self._affect_lexicon()
        entry = affect.get(pred_id) or affect.get(pred_id.rstrip("s"))
        if entry is not None:
            return entry.get("intent_hint")
        return None

    def _build_turn_context(
        self, utterance: str, parse_bundle: _ParseBundle,
    ) -> dict[str, Any]:
        user_id = self.profile.user_id
        session_id = self.store.current_session_id() if self.store is not None else ""
        if user_id not in self._mood_state:
            self._mood_state[user_id] = self._load_or_init_mood(user_id)
            self._rapid_state[user_id] = {"count": 0, "last_utterance": ""}
        current_mood = self._mood_state[user_id]
        rapid = self._rapid_state[user_id]

        # Previous-turn state from the persistent store. The router is rebuilt
        # per turn (kernel constructs a fresh one), so cross-turn context must
        # come from the store, not router instance state.
        prev_mood = None
        prev_intent = ""
        prev_ambient = 0.0
        if self.store is not None:
            try:
                prev_mood = self.store.current_mood_state(session_id, user_id)
            except Exception:
                prev_mood = None
            try:
                prev_intent = self.store.previous_intent(session_id)
            except Exception:
                prev_intent = ""
            try:
                ambient = self.store.load_ambient_mood()
                if ambient is not None:
                    prev_ambient = float(ambient.get("valence", 0.0) or 0.0)
            except Exception:
                prev_ambient = 0.0

        familiarity = 0
        if self.store is not None:
            try:
                summaries = self.store.query_session_summaries(user_id, limit=100)
                familiarity = len(summaries) if summaries else 0
            except Exception:
                familiarity = 0

        affect = AffectSignal()
        uol_act = parse_bundle.uol_act
        affect_lexicon = self._affect_lexicon()
        inferred = compute_utterance_affect(parse_bundle.lemmas, uol_act, affect_lexicon)
        if inferred is not None and inferred.confidence > 0:
            affect = inferred
        elif parse_bundle.informal_affect:
            # Fallback: bare informal tokens (lol/haha/ugh) carry the only mood.
            informal = _aggregate_informal_affect(parse_bundle.informal_affect)
            if informal is not None:
                affect = informal

        # Rapid repetition: track consecutive same-utterance text
        prev_text = rapid.get("last_utterance", "")
        normalized = utterance.strip().lower()
        if normalized and normalized == prev_text:
            rapid["count"] += 1
        else:
            rapid["count"] = 0
            rapid["last_utterance"] = normalized

        short_circuit = self._apply_short_circuits(
            utterance, parse_bundle, affect, current_mood, rapid,
        )
        regions = self._mood_regions_list()
        updated_mood = update_session_mood(
            current_mood, affect, user_id, session_id,
            current_mood.turn_count + 1, regions,
        )
        self._mood_state[user_id] = updated_mood

        ambient_valence = prev_ambient
        ambient_valence_delta = float(updated_mood.valence) - prev_ambient

        return {
            "user_id": user_id,
            "session_id": session_id,
            "utterance_affect": affect,
            "session_mood": updated_mood,
            "prev_mood": prev_mood,
            "prev_intent": prev_intent,
            "ambient_valence": ambient_valence,
            "ambient_valence_delta": ambient_valence_delta,
            "short_circuit": short_circuit,
            "rapid_count": rapid["count"],
            "familiarity": familiarity,
        }

    def _try_reasoning(
        self, utterance: str, parse_bundle: _ParseBundle,
    ) -> AssistantDecision | None:
        """Detect + solve a deterministic reasoning task, or None to fall through.

        Gated by the ``reasoning`` capability family. On a missing-fact refusal,
        emits a typed refusal decision; on success, an evidence-bound result.
        """
        if not _is_family_installed("reasoning"):
            return None
        # Knowledge typing — store factual claims as structured propositions
        knowledge_dec = self._knowledge_decision(utterance, parse_bundle)
        if knowledge_dec is not None:
            return knowledge_dec
        itinerary = self._itinerary_decision(utterance, parse_bundle)
        if itinerary is not None:
            return itinerary
        try:
            from .reasoning import detect_reasoning_task, solve
            task = detect_reasoning_task(
                parse_bundle.text, parse_bundle.tokens, parse_bundle.uol_act,
            )
            if task is None:
                return None
            if task.get("task") == "self_query":
                return self._self_query_decision(utterance, task)
            result, answer, refusal = solve(task)
        except Exception:
            return None
        intent = f"reasoning:{task.get('task', '')}"
        if refusal is not None:
            return AssistantDecision(
                utterance=utterance, intent=intent, route="local_answer",
                answer=answer or "I need a bit more information to answer that.",
                refusal_signal=refusal, evidence_keys=("reasoning.refusal",),
                confidence=0.9, reason=f"reasoning_refusal:{refusal}",
            )
        return AssistantDecision(
            utterance=utterance, intent=intent, route="local_answer",
            answer=answer, reasoning_result=result,
            evidence_keys=("reasoning.result",), confidence=0.95,
            reason="reasoning_solved",
        )

    def _knowledge_decision(
        self, utterance: str, parse_bundle: _ParseBundle,
    ) -> AssistantDecision | None:
        """Try to classify and store a factual claim from UOL atoms.

        Returns None when the utterance isn't a factual claim, or when the
        knowledge subsystem isn't available. Returns a decision with the
        stored-fact acknowledgment or a contradiction prompt.
        """
        if self.store is None:
            return None
        from .assistant_knowledge import classify_knowledge, extract_proposition
        uol_act = parse_bundle.uol_act
        kt = classify_knowledge(uol_act, utterance)
        if kt not in ("static_fact", "negated_fact", "opinion"):
            return None
        prop = extract_proposition(uol_act)
        if prop is None:
            return None
        import uuid
        eid = f"wf_{uuid.uuid4().hex[:12]}"
        polarity = "asserted" if kt != "negated_fact" else "negated"
        self.store.set_world_fact(
            eid, prop["subject"], prop["relation"], prop["object"],
            polarity=polarity, provenance="user",
            confidence=prop.get("confidence", 0.6),
            source_utterance=utterance,
        )
        from melm.contracts.validation import load_knowledge_types
        kt_data = load_knowledge_types()
        contradictions = self.store.find_contradicting_facts(
            prop["subject"], prop["relation"], prop["object"], polarity,
        )
        if contradictions:
            existing = contradictions[0]
            prop_str = f"{prop['subject']} {prop['relation']} {prop['object']}"
            if existing.get("confidence", 0) >= prop.get("confidence", 0.6):
                answer = kt_data.get("truth_arbitration", {}).get(
                    "contradiction_prompt", "That differs from what I have."
                ).replace("{proposition}", prop_str)
                return AssistantDecision(
                    utterance=utterance, intent="personal_memory", route="local_answer",
                    answer=answer, reasoning_result={"task": "knowledge_contradiction",
                        "proposition": prop_str, "stored_polarity": existing.get("polarity", "")},
                    evidence_keys=(f"world_fact.{eid}",), confidence=0.9,
                    reason="knowledge_contradiction",
                )
        # Positive ack
        prop_str = f"{prop['subject']} {prop['relation']} {prop['object']}"
        if kt == "negated_fact":
            negated = f"{prop['subject']} is not {prop['object']}"
            answer = kt_data.get("truth_arbitration", {}).get(
                "negate_ack", "I will remember that {negation}."
            ).replace("{negation}", negated)
        elif kt == "opinion":
            answer = f"I hear your opinion about {prop['subject']}."
        else:
            answer = kt_data.get("truth_arbitration", {}).get(
                "assert_ack", "I will remember that {proposition}."
            ).replace("{proposition}", prop_str)
        return AssistantDecision(
            utterance=utterance, intent="personal_memory", route="local_answer",
            answer=answer,
            reasoning_result={"task": "knowledge_write", "proposition": prop_str, "type": kt},
            evidence_keys=(f"world_fact.{eid}",), confidence=0.9,
            reason="knowledge_write",
        )

    def _itinerary_decision(
        self, utterance: str, parse_bundle: _ParseBundle,
    ) -> AssistantDecision | None:
        """Build/answer a multi-turn itinerary scenario from the geo atlas.

        Turn 1 parses + stores the journey; later turns query the bound scenario
        (duration / total distance / displacement / location-at-time). Returns
        None when there is neither a new itinerary nor a scenario query, so other
        routing proceeds.
        """
        if self.store is None:
            return None
        try:
            from melm.contracts import load_geo_atlas
            from .reasoning import clock
            from .reasoning.itinerary import (
                detect_itinerary_query, parse_itinerary, solve_itinerary,
            )
            atlas = load_geo_atlas()
            place_names = list(atlas.get("places", {}).keys())
            text = parse_bundle.text
            session_id = self.store.current_session_id()
            query = detect_itinerary_query(text)
            parsed = parse_itinerary(text, place_names)
            if parsed is not None:
                self.store.set_current_scenario(session_id, parsed)
            scenario = self.store.get_current_scenario(session_id)
            if scenario is None:
                return None
            if query is None:
                if parsed is None:
                    return None
                seq = " -> ".join(p.title() for p in scenario["places"])
                return AssistantDecision(
                    utterance=utterance, intent="reasoning:itinerary", route="local_answer",
                    answer=f"Got it - I'll track this journey: {seq}.",
                    reasoning_result={"task": "itinerary", "query": "summary",
                                      "places": scenario["places"]},
                    evidence_keys=("reasoning.result",), confidence=0.9,
                    reason="reasoning_solved",
                )
            result, answer, refusal = solve_itinerary(scenario, atlas, query, clock.now())
        except Exception:
            return None
        if refusal is not None:
            return AssistantDecision(
                utterance=utterance, intent="reasoning:itinerary", route="local_answer",
                answer=answer or "I need a bit more detail about that journey.",
                refusal_signal=refusal, evidence_keys=("reasoning.refusal",),
                confidence=0.9, reason=f"reasoning_refusal:{refusal}",
            )
        return AssistantDecision(
            utterance=utterance, intent="reasoning:itinerary", route="local_answer",
            answer=answer, reasoning_result=result, evidence_keys=("reasoning.result",),
            confidence=0.95, reason="reasoning_solved",
        )

    def _self_query_decision(
        self, utterance: str, task: dict[str, Any],
    ) -> AssistantDecision | None:
        """Answer a self-referential probe locally from the identity contract,
        grounding 'feeling' in the current operating mood."""
        from melm.contracts import load_assistant_identity
        category = str(task.get("category", ""))
        templates = load_assistant_identity()
        text = templates.get(f"reflection_{category}", "")
        if not text:
            return None
        if category == "feeling":
            mood = self._mood_state.get(self.profile.user_id)
            mood_id = getattr(mood, "mood_id", "neutral") if mood else "neutral"
            text = text.replace("{mood}", mood_id or "neutral")
        return AssistantDecision(
            utterance=utterance, intent="reasoning:self_query", route="local_answer",
            answer=text, reasoning_result={"task": "self_query", "category": category},
            evidence_keys=("reasoning.result",), confidence=0.95, reason="reasoning_solved",
        )

    def _occurrence_for(self, intent: str) -> int:
        """Per-(session,intent) occurrence for the current turn (1-based).

        Reads the store's O(1) in-memory tally of PRIOR turns and adds one for
        the current turn. Never scans history (anti-regression invariant 5).
        """
        if self.store is None:
            return 0
        try:
            sid = self.store.current_session_id()
            return self.store.get_intent_tally(sid, intent) + 1
        except Exception:
            return 0

    def _attach_turn_context(
        self, decision: AssistantDecision, turn_context: dict[str, Any],
    ) -> AssistantDecision:
        """Marshal turn context onto the decision (single source of truth)."""
        return replace(
            decision,
            utterance_affect=turn_context.get("utterance_affect"),
            session_mood=turn_context.get("session_mood"),
            intent_occurrence=self._occurrence_for(decision.intent),
            rapid_occurrence=turn_context.get("rapid_count", 0),
            active_user_id=turn_context.get("user_id", "default"),
            prev_mood=turn_context.get("prev_mood"),
            prev_intent=turn_context.get("prev_intent", ""),
            ambient_valence=turn_context.get("ambient_valence", 0.0),
            ambient_valence_delta=turn_context.get("ambient_valence_delta", 0.0),
            familiarity=turn_context.get("familiarity", 0),
        )

    def _apply_short_circuits(
        self,
        utterance: str,
        parse_bundle: _ParseBundle,
        affect: AffectSignal | None,
        mood: MoodState,
        rapid: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "is_short_circuit": False,
            "reason": "",
            "override_intent": None,
        }
        uol_act = parse_bundle.uol_act
        tokens = parse_bundle.tokens
        # P0-1: Identity claim detection
        if uol_act is not None and detect_identity_claim(uol_act, tokens) is not None:
            result["is_short_circuit"] = True
            result["reason"] = "identity_switch"
            result["override_intent"] = "assistant_identity"
            return result
        # P0-2: Identity probe detection (assistant-directed + UOL identity predicate)
        tokens_lower = [t.lower() for t in tokens]
        assistant_directed = (
            "you" in tokens_lower or "your" in tokens_lower
        )
        is_genuine_identity = True
        if uol_act is not None:
            identity_predicates = {"who", "what"}
            content = uol_act.get("content", [])
            if content:
                main = content[0] if isinstance(content, list) else content[0]
                if isinstance(main, dict):
                    pred_id = str(main.get("predicate", {}).get("id", "")).lower().strip()
                    if pred_id not in identity_predicates:
                        is_genuine_identity = False
                    else:
                        themes = [
                            str(r.get("value", ""))
                            for r in main.get("roles", [])
                            if r.get("role") == "theme"
                        ]
                        # If there's a verb-like theme (not a PROPN), it's not identity
                        for tv in themes:
                            if tv and not tv[0].isupper():
                                is_genuine_identity = False
                                break
        if uol_act is not None and assistant_directed and is_genuine_identity and detect_identity_probe(uol_act, tokens):
            result["is_short_circuit"] = True
            result["reason"] = "identity_probe_detected"
            result["override_intent"] = "assistant_identity"
            return result
        # P0-3: Complaint detected — acknowledge as assistant_behavior, NOT a greeting.
        if affect is not None and affect.is_complaint:
            result["is_short_circuit"] = True
            result["reason"] = "complaint_acknowledged"
            result["override_intent"] = "assistant_behavior"
            return result
        # P0-4: Rapid repetition
        if rapid.get("count", 0) >= 2:
            result["is_short_circuit"] = True
            result["reason"] = "rapid_repetition"
            result["override_intent"] = "assistant_status"
            return result
        # P0-5: Perception urgency
        if affect is not None and affect.source == "perception" and affect.confidence >= 0.9:
            result["is_short_circuit"] = True
            result["reason"] = "perception_urgency_high"
            result["override_intent"] = "common_sense_safety"
            return result
        return result

    def handle(self, utterance: str, last_intent: str = "") -> AssistantDecision:
        collector: set[str] = set()
        set_semantic_class_collector(collector)
        parse_bundle = self._build_parse_bundle(utterance, last_intent=last_intent)
        turn_context = self._build_turn_context(utterance, parse_bundle)
        sc = turn_context.get("short_circuit", {})
        if sc.get("is_short_circuit"):
            set_semantic_class_collector(None)
            override_intent = sc.get("override_intent", "")
            sc_reason = sc.get("reason", "short_circuit")
            if override_intent == "assistant_identity":
                decision = self._assistant_identity(utterance)
            elif override_intent == "social_greeting":
                decision = self._greeting(utterance)
            elif override_intent == "assistant_status":
                decision = self._assistant_status(utterance)
            elif override_intent == "assistant_behavior":
                decision = self._assistant_behavior(utterance)
                if sc_reason == "complaint_acknowledged":
                    decision = replace(
                        decision,
                        reason=sc_reason,
                        answer=(
                            "I hear you're not satisfied with how I responded. "
                            "Tell me what went wrong and I'll try to do better."
                        ),
                    )
            elif override_intent == "common_sense_safety":
                decision = AssistantDecision(
                    utterance=utterance,
                    intent="common_sense_safety",
                    route="local_answer",
                    answer=(
                        "That sounds urgent. If there may be danger, "
                        "move to safety and alert someone now."
                    ),
                    evidence_keys=("local_safety_policy.perception_urgency",),
                    confidence=0.90,
                    reason=sc_reason,
                )
            else:
                decision = AssistantDecision(
                    utterance=utterance,
                    intent=override_intent,
                    route="local_answer",
                    answer="",
                    evidence_keys=("self_model.purpose",),
                    confidence=0.90,
                    reason=sc_reason,
                )
            return self._attach_turn_context(decision, turn_context)
        try:
            decision = self._route_impl(utterance, parse_bundle=parse_bundle, turn_context=turn_context)
        finally:
            set_semantic_class_collector(None)
        if collector:
            decision = replace(decision, semantic_classes_activated=frozenset(collector))
        if parse_bundle.functional_parse is not None:
            decision = replace(decision, functional_parse=parse_bundle.functional_parse.to_dict())
            if parse_bundle.uol_act is not None:
                decision = replace(decision, uol_act=parse_bundle.uol_act)
        decision = self._attach_turn_context(decision, turn_context)
        return decision

    def _route_impl(self, utterance: str, parse_bundle: _ParseBundle | None = None, turn_context: dict[str, Any] | None = None) -> AssistantDecision:
        parse_bundle = parse_bundle or self._build_parse_bundle(utterance)
        # Novelty detection (best-effort side-effect)
        if self.store is not None and parse_bundle is not None:
            try:
                unknown = getattr(parse_bundle, "semantic_unknown_tokens", None)
                if unknown:
                    from .assistant_skill_novelty import detect_novelty, record_novelty_candidates
                    lex_ref = getattr(self, "_uol_lexicon_ref", None)
                    candidates = detect_novelty(parse_bundle, lex_ref, self.store)
                    if candidates:
                        record_novelty_candidates(self.store, candidates)
            except Exception:
                pass
        detected_lang = parse_bundle.language
        text = parse_bundle.text
        tokens = parse_bundle.tokens
        intent = _classify_intent_from_uol_slots(
            text,
            tokens,
            trusted_contact_names=tuple(self.profile.contacts),
            language=detected_lang,
            parse_bundle=parse_bundle,
            uol_act=parse_bundle.uol_act,
        )
        # Commitment extraction (best-effort side-effect)
        if self.store is not None:
            try:
                from .assistant_skill_commitments import extract_commitment, record_commitment
                commitment = extract_commitment(utterance, parse_bundle)
                if commitment is not None:
                    commitment.session_id = getattr(parse_bundle, "session_id", "") if parse_bundle else ""
                    record_commitment(self.store, commitment)
            except Exception:
                pass
        # Reasoning task signatures outrank closed-intent routing (e.g. quantity
        # arithmetic outranks meal_suggestion). Falls through when no task matches.
        reasoning_decision = self._try_reasoning(utterance, parse_bundle)
        if reasoning_decision is not None:
            return reasoning_decision
        if intent != "open_domain" and not _is_family_installed(intent):
            return AssistantDecision(
                utterance=utterance,
                intent=intent,
                route="open_domain",
                answer="That capability is not available on this device.",
                cloud_needed=False,
                confidence=0.60,
                reason=f"family_not_installed:{intent}",
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
            return self._assistant_identity(utterance, parse_bundle)
        if intent == "assistant_status":
            return self._assistant_status(utterance)
        if intent == "weather":
            return self._weather(utterance)
        if intent == "common_sense_safety":
            return self._safety(utterance, parse_bundle)
        if intent == "health_advice":
            return self._health(utterance)
        music_gen = self._music_generation(text, parse_bundle, turn_context)
        if music_gen is not None:
            return music_gen
        music_disc = self._music_discovery(text, parse_bundle, turn_context)
        if music_disc is not None:
            return music_disc
        if intent == "media_playback":
            return self._media(utterance)
        if intent == "personal_memory":
            return self._personal_memory(utterance)
        if intent == "meal_suggestion":
            return self._meal(utterance)
        if intent == "social_contact":
            return self._contact(utterance)
        if intent == "autobiographical_memory":
            return AssistantDecision(
                utterance=utterance,
                intent="autobiographical_memory",
                route="clarify",
                answer="I do not have earlier conversation memory to replay yet.",
                local_memory_used=True,
                confidence=0.74,
                reason="autobiographical_memory_empty",
            )
        if intent == "social_greeting":
            return self._greeting(utterance)
        if intent == "assistant_behavior":
            return self._assistant_behavior(utterance)
        if intent == "personal_goal_advice":
            return _cloud(utterance, intent, reason="understood_personal_goal_advice")
        if intent == "open_domain":
            return AssistantDecision(
                utterance=utterance,
                intent=intent,
                route="local_answer",
                answer="",
                evidence_keys=("self_model.purpose",),
                confidence=0.74,
                reason="understood_open_domain",
            )
        # Phase 7: Pre-fallthrough absurdity check — if we're about to return
        # unknown, check if the raw text has a clear social/intent pattern.
        if intent == "unknown":
            if _detect_social_status(text):
                return self._assistant_status(utterance)
            from melm.appliance.reasoning.typability import classify_utterance_tokens
            tokens = text.split()
            if tokens and classify_utterance_tokens(tokens) == "gibberish":
                return AssistantDecision(
                    utterance=utterance,
                    intent="unknown",
                    route="local_answer",
                    answer="I did not understand that — could you rephrase?",
                    confidence=0.3,
                    reason="gibberish_detected",
                )
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
        answer = "Hi. What would you like help with?"
        culture = self.profile.culture.lower()
        lang = self.profile.language_preference.lower()
        try:
            if culture == "yoruba" or lang == "yoruba":
                greeting_data = load_yoruba_greetings()
                greeting = greeting_data["greetings"].get("general", "Báwo ni?")
                answer = f"{greeting}. What would you like help with?"
            elif culture == "swahili" or lang == "swahili":
                greeting_data = load_swahili_greetings()
                greeting = greeting_data["greetings"].get("general", "Habari")
                answer = f"{greeting}. What would you like help with?"
            elif culture == "igbo" or lang == "igbo":
                greeting_data = load_igbo_greetings()
                greeting = greeting_data["greetings"].get("general", "Ndeewo")
                answer = f"{greeting}. What would you like help with?"
        except Exception:
            pass
        pools = self._pools()
        pool = pools.get("social_greeting", [])
        if pool and not answer:
            answer = pool[0].get("text", answer)
        return AssistantDecision(
            utterance=utterance,
            intent="social_greeting",
            route="local_answer",
            answer=answer,
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

    def _assistant_identity(self, utterance: str, parse_bundle: _ParseBundle | None = None) -> AssistantDecision:
        answer = (
            "I am MELM Local Assistant OS. I run local-first on this device, "
            "using local memory, cached tools, and confirmed actions before "
            "asking a larger model."
        )
        pools = self._pools()
        pool = pools.get("assistant_identity", [])
        if pool:
            answer = pool[0].get("text", answer)
        # Derive identity action from composition for evidence-key tagging
        tokens = _tokenize(utterance)
        composition = _identity_composition(utterance, tokens)
        action = composition.get("action", "identify") if composition else "identify"
        # "why" follow-up after identity explanation — override action
        stripped = utterance.strip().lower().rstrip("?").strip()
        if action == "identify" and parse_bundle is not None and \
           parse_bundle.last_intent == "assistant_identity" and \
           stripped in ("why", "why not"):
            action = "explain_identity"
        evidence = (
            "self_model.name",
            "self_model.purpose",
            "self_model.local_capabilities",
            "self_model.limits",
        )
        if action in ("suggest_name", "name_awareness", "name_origin", "explain_identity"):
            evidence = evidence + (f"identity_action:{action}",)
        return AssistantDecision(
            utterance=utterance,
            intent="assistant_identity",
            route="local_answer",
            answer=answer,
            evidence_keys=evidence,
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

    def _safety(self, utterance: str, parse_bundle: _ParseBundle | None = None) -> AssistantDecision:
        text = _normalize(utterance)
        tokens = _tokenize(text)
        token_set = set(tokens)
        # Use implication engine for safety detection
        fp = parse_bundle.functional_parse if parse_bundle is not None else None
        ua = parse_bundle.uol_act if parse_bundle is not None else None
        verb = _extract_verb(fp, ua)
        patient = _extract_patient_type(fp, ua)
        patient_cls = _resolve_patient_type(patient) if patient else "person"
        if verb:
            mc = _lazy_moral_engine()(verb, patient_cls)
            if not mc.has_implication:
                from melm.appliance.reasoning.implications import record_verb_candidate
                record_verb_candidate(verb, patient_cls, utterance)
            if mc.policy_triggers:
                return AssistantDecision(
                    utterance=utterance,
                    intent="common_sense_safety",
                    route="local_answer",
                    answer="",
                    evidence_keys=("local_safety_policy.clothing_public_school",),
                    reason="safety_policy_triggered",
                )
        # School-clothing-weather policy: weather-contextual clothing advice
        _nudity_terms = {"naked", "undressed"}
        _school_clothing_terms = {"wear", "clothes", "coat", "raincoat"}
        if (
            _school_clothing_terms & token_set
            and "school" in token_set
            and not (_nudity_terms & token_set)
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
                answer="",
                evidence_keys=("weekly_weather.today", "facts.school"),
                local_memory_used=True,
                confidence=0.91,
                reason="school_clothing_weather_policy",
            )
        # Contract-driven nudity/modesty safety check
        try:
            from melm.contracts import load_safety_policies
            policies = load_safety_policies()
        except Exception:
            policies = {}
        clothing = policies.get("public_clothing", {})
        triggers = set(clothing.get("triggers", []))
        if triggers and triggers & token_set:
            return AssistantDecision(
                utterance=utterance,
                intent="common_sense_safety",
                route="local_answer",
                answer="",
                evidence_keys=("local_safety_policy.clothing_public_school",),
                reason="local_common_sense_policy",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="common_sense_safety",
            route="local_answer",
            answer="",
            evidence_keys=("local_safety_policy.clothing_public_school",),
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
            answer="",
            evidence_keys=("health_goals", "local_health_safety_policy"),
            local_memory_used=True,
            reason="bounded_general_health_guidance",
        )

    def _music_generation(
        self,
        text: str,
        parse_bundle: _ParseBundle,
        turn_context: dict[str, Any],
    ) -> AssistantDecision | None:
        """Route to music generation (MIDI) when user asks to play an instrument.

        Only fires when the utterance is specifically about composing/playing
        instrumental music — NOT when a track title (e.g. "focus piano") happens
        to contain an instrument keyword.
        """
        instrument_keywords = {"piano", "guitar", "violin", "flute", "drums", "keyboard"}
        text_lower = text.lower()
        # Strip common punctuation so "piano." matches "piano"
        clean = "".join(ch for ch in text_lower if ch.isalnum() or ch.isspace())

        # Bail if any media library title is a substring of the utterance —
        # the user is requesting a known track, not MIDI composition.
        media_lib = self.profile.media_library
        if media_lib and any(title.lower() in clean for title in media_lib):
            return None

        has_play = any(word in clean.split() for word in ("play", "sing", "perform"))
        has_instrument = any(kw in clean.split() for kw in instrument_keywords)
        if has_play and has_instrument:
            return AssistantDecision(
                utterance=text,
                intent="music_generation",
                route="local_answer",
                answer=f"I'll compose some {text_lower.replace('play', '').strip()} for you.",
                evidence_keys=(),
                cloud_needed=False,
                external_fetch_needed=False,
                privacy_exposure=False,
                local_memory_used=False,
                device_action=False,
            )
        return None

    def _music_discovery(
        self,
        text: str,
        parse_bundle: _ParseBundle,
        turn_context: dict[str, Any],
    ) -> AssistantDecision | None:
        """Route to music discovery when user asks to find/download music."""
        text_lower = text.lower()
        clean = "".join(ch for ch in text_lower if ch.isalnum() or ch.isspace())
        discovery_verbs = {"find", "search", "download", "look", "get", "discover"}
        has_discovery_verb = any(word in clean.split() for word in discovery_verbs)
        has_music = any(word in clean.split() for word in ("music", "song", "tune", "audio", "songs"))
        instrument_keywords = {"piano", "guitar", "violin", "flute", "drums", "keyboard"}
        if has_discovery_verb and has_music and not any(kw in clean.split() for kw in instrument_keywords):
            from .assistant_music_discovery import MusicDiscoverer
            discoverer = MusicDiscoverer()
            results = discoverer.search_inventory(text_lower, self.store)
            if results:
                title = results[0].get("title", text_lower)
                return AssistantDecision(
                    utterance=text,
                    intent="media_playback",
                    route="device_action",
                    answer=f"Playing {title}.",
                    evidence_keys=(),
                    cloud_needed=False,
                    external_fetch_needed=False,
                    privacy_exposure=False,
                    local_memory_used=True,
                    device_action=True,
                    reason="local_media_action",
                )
            return AssistantDecision(
                utterance=text,
                intent="music_discovery",
                route="local_answer",
                answer=discoverer.offer_search(text_lower),
                evidence_keys=(),
                cloud_needed=False,
                external_fetch_needed=False,
                privacy_exposure=False,
                local_memory_used=False,
                device_action=False,
            )
        return None

    def _recall_profile_attribute(
        self, utterance: str, attribute: str,
    ) -> AssistantDecision | None:
        """Recall a structured profile field (name/location/age) locally.

        Never hands to cloud/model: a missing field clarifies instead of guessing.
        """
        if attribute == "name":
            value, key, phrase = self.profile.user_name, "profile.user_name", "Your name is {v}."
        elif attribute == "location":
            value, key, phrase = self.profile.location, "profile.location", "You live in {v}."
        elif attribute == "age":
            value = str(self.profile.age) if getattr(self.profile, "age", 0) else ""
            key, phrase = "profile.age", "You are {v}."
        else:
            return None
        value = (value or "").strip()
        if not value:
            return AssistantDecision(
                utterance=utterance,
                intent="personal_memory",
                route="clarify",
                answer=f"I don't have your {attribute} stored yet. Tell me and I'll remember it locally.",
                local_memory_used=True,
                confidence=0.74,
                reason="personal_memory_empty",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="personal_memory",
            route="local_answer",
            answer=phrase.format(v=value),
            evidence_keys=(key,),
            local_memory_used=True,
            confidence=0.95,
            reason="personal_memory_recall",
        )

    def _personal_memory(self, utterance: str) -> AssistantDecision:
        text = _normalize(utterance)
        tokens = _tokenize(text)
        attribute = _profile_attribute_requested(tokens)
        if attribute is not None:
            recall = self._recall_profile_attribute(utterance, attribute)
            if recall is not None:
                return recall
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
    preferences: dict[str, str] | None = None,
    weather: str = "",
    utterance: str = "",
) -> LocalMealChoice:
    result = suggest_meal(foods, preferences=preferences, weather=weather, utterance=utterance)
    return LocalMealChoice(
        items=result.items,
        backups=result.backups,
        reason_tags=result.reason_tags,
        meal_scope=result.meal_scope,
        warm_note=result.warm_note,
    )


def _clean_food_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


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


def _food_tags(food: str) -> set[str]:
    tokens = _tokenize(food.lower())
    tags: set[str] = set()
    mapping = load_food_tags()
    for marker, marker_tags in mapping.items():
        if _has_token_sequence(tokens, _tokenize(marker)):
            tags.update(marker_tags)
    return tags or {"food"}


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


def _has_private_context_frame(
    tokens: tuple[str, ...],
) -> bool:
    token_set = set(tokens)
    private_terms = _semantic_family_terms(
        tokens,
        semantic_classes={
            "social_relation",
            "personal_attribute",
            "child_relation",
            "household_concept",
            "routine_concept",
            "public_place",
            "autobiographical_event",
        },


    )
    owned_memory_subject = bool(private_terms)
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

    parse_bundle = _build_parse_bundle(utterance)
    normalized = parse_bundle.text
    tokens = parse_bundle.tokens
    question_like = _question_like_from_parse_bundle(parse_bundle)
    intent = (
        decision.intent
        if decision is not None
        else _classify_intent_from_uol_slots(
            normalized,
            tokens,
            language=parse_bundle.language,
            parse_bundle=parse_bundle,
            uol_act=parse_bundle.uol_act,
        )
    )
    frame_match = AssistantFrameRegistry.match(
        normalized,
        tokens,
        intent,
        question_like=question_like,
        functional_parse=parse_bundle.functional_parse,
    )
    composition = frame_match.to_composition() if frame_match is not None else None
    route = decision.route if decision is not None else _route_hint(intent, composition)
    reason = (
        decision.reason
        if decision is not None
        else _route_reason_hint(intent, composition)
    )
    secondary_meaning_hints = _secondary_meaning_hints(normalized, intent)
    secondary_domain_hints = _secondary_domain_hints(normalized)
    domain_hints = _domain_hints(normalized, tokens, intent, composition)
    uol = _assistant_uol(
        normalized,
        tokens,
        intent,
        composition,
        uol_act=parse_bundle.uol_act,
    )
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

    parse_bundle = _build_parse_bundle(utterance)
    return _self_status_composition(
        parse_bundle.text,
        parse_bundle.tokens,
        question_like=_question_like_from_parse_bundle(parse_bundle),
    )


def compose_autobiographical_memory_frame(utterance: str) -> dict[str, Any] | None:
    """Return the primary autobiographical-memory UOL composition used by router and kernel."""

    parse_bundle = _build_parse_bundle(utterance)
    scope = _autobiographical_memory_scope(parse_bundle)
    if not scope:
        return None
    return _compose_primary_frame(
        parse_bundle.text,
        parse_bundle.tokens,
        "autobiographical_memory",
        question_like=_question_like_from_parse_bundle(parse_bundle),
        functional_parse=parse_bundle.functional_parse,
    )


def classify_autobiographical_memory_scope(utterance: str) -> str:
    """Classify the structural scope of an autobiographical-memory request."""

    return _autobiographical_memory_scope(_build_parse_bundle(utterance))


def _classify_intent(text: str) -> AssistantIntent:
    parse_bundle = _build_parse_bundle(text)
    return _classify_intent_from_uol_slots(
        parse_bundle.text,
        parse_bundle.tokens,
        language=parse_bundle.language,
        parse_bundle=parse_bundle,
        uol_act=parse_bundle.uol_act,
    )


def _semantic_classes_from_parse(parse: Any) -> frozenset[str]:
    """Collect all semantic classes from parse candidates."""
    classes: set[str] = set()
    for cand in parse.candidates:
        sc = cand.get("semantic_class", "")
        if sc:
            classes.add(sc)
    return frozenset(classes)


def _classify_from_functional_parse(
    functional_parse: Any,
    tokens: tuple[str, ...],
    text: str,
) -> str | None:
    """UOL-first intent classification from FunctionalParse atoms.

    Returns an intent string when the parse clearly indicates a known
    frame; returns ``None`` only when the parse is genuinely ambiguous.
    """
    # Urgent health detection via implication engine (contract-driven)
    verb = _extract_verb(functional_parse=functional_parse)
    if verb:
        patient = _extract_patient_type(functional_parse=functional_parse)
        mc = _lazy_moral_engine()(verb, patient or "person")
        if not mc.has_implication:
            from melm.appliance.reasoning.implications import record_verb_candidate
            record_verb_candidate(verb, patient or "person", text)
        if mc.harm_severity == "high":
            return "health_advice"
    # Fallback: contract-driven health frame (reads health_disclaimers.v1.json)
    if _has_urgent_health_frame(tokens):
        return "health_advice"

    # Structured composition fallbacks work from tokens — no parse required.
    # They are UOL compositions, not raw keyword guards.
    if _identity_composition(text, tokens) is not None:
        return "assistant_identity"
    if _self_status_composition(text, tokens) is not None:
        return "assistant_status"

    if functional_parse is None:
        # Fragment-based classification for short utterances without parseable predicates
        token_set = set(tokens)
        if "your" in token_set and "name" in token_set and "?" in text:
            return "assistant_identity"
        return None

    speech_act = functional_parse.speech_act
    subject = functional_parse.subject
    action = functional_parse.action
    obj = functional_parse.object
    obj_words = set(obj.split()) if obj else set()
    sem_classes = _semantic_classes_from_parse(functional_parse)

    # Assistant identity: "Who are you?" / "What is your name?"
    # Check token_roles for assistant deixis (possessor=assistant, grammatical_subject=assistant)
    assistant_refs_in_roles = any(
        tr.get("meaning") == "assistant" or tr.get("lemma") in ("your", "you", "yourself")
        for tr in (functional_parse.token_roles if functional_parse else ())
    )
    has_assistant_ref = subject == "assistant" or any(
        w in obj for w in ("your", "you", "yourself")
    ) or assistant_refs_in_roles or "your" in tokens
    if speech_act in {"wh_question", "yes_no_question", "question"}:
        if has_assistant_ref and action == "be" and ("name" in obj or "identity" in obj or "purpose" in obj):
            return "assistant_identity"
        if action == "be" and subject == "assistant" and ("who" in tokens or "why" in tokens) and obj not in {"calling", "doing"}:
            return "assistant_identity"
        if action == "be" and subject == "assistant" and "what" in tokens and ("kind" in tokens or "assistant" in tokens):
            return "assistant_identity"
        # Capability questions: "What can you help me with?", "What can you do?"
        # Exclude past tense "did" to avoid catching "What did you do?" (status)
        if subject == "assistant" and action in {"help", "do", "can"} and ("what" in tokens or "who" in tokens) and not obj and "did" not in tokens:
            return "assistant_identity"
        # Identity challenge: "you don't know who you are?"
        if action == "know" and "who" in tokens and ("you" in tokens or "your" in tokens):
            return "assistant_identity"

    # Imperative self-description: "Describe yourself"
    if speech_act in {"request", "command"} and action in {"describe", "tell", "say"} and ("assistant" in obj or "yourself" in obj or "you" in obj):
        return "assistant_identity"

    # Assistant status: "What is your status?" / "How are you doing?" / "What did you do?"
    if speech_act in {"wh_question", "yes_no_question", "question"}:
        assistant_ref_question = subject == "assistant" or any(
            w in obj for w in ("your", "you", "yourself")
        )
        if assistant_ref_question and action == "be" and ("status" in obj or "health" in obj or "condition" in obj):
            return "assistant_status"
        if action in {"do", "feel"} and subject == "assistant" and ("status" in obj or "health" in obj or "condition" in obj):
            return "assistant_status"
        # Bare assistant-action questions ("What did you do?") — no concrete object
        if subject == "assistant" and action in {"do", "did"} and not obj:
            return "assistant_status"
        # "how are you" / "how are you feeling" — no object after "be"/"feel"
        if subject == "assistant" and action in {"be", "feel"} and not obj:
            return "assistant_status"
        # Assistant progress/completion questions ("What have you done so far?")
        if subject == "assistant" and speech_act in {"wh_question", "yes_no_question"} and "done" in tokens:
            return "assistant_status"

    # Assistant status request: "Show your memory ledger", "Tell me your status"
    if speech_act in {"request", "command"} and action in {"show", "tell", "display"} and ("assistant" in obj or has_assistant_ref) and any(w in obj for w in ("status", "ledger", "memory", "health")):
        return "assistant_status"

    # Meal suggestion: "What should I eat?", "gini m ga eri?"
    if (
        speech_act in {"wh_question", "yes_no_question", "request"}
        and action in {"eat", "cook", "prepare"}
        and subject in {"user", "i", "we"}
    ):
        return "meal_suggestion"

    # Story request: "Tell me a story", "Read me a tale"
    if (
        speech_act in {"request", "command", "wh_question"}
        and action in {"tell", "read", "make", "give"}
        and bool(obj_words & {"story", "tale", "fable", "narrative"})
    ):
        return "story"

    # Weather: "Will it rain today?", "Is it snowing?"
    if (
        speech_act in {"wh_question", "yes_no_question", "request"}
        and action in {"rain", "snow", "forecast"}
    ):
        return "weather"

    # Media playback: "Play a song", "Start some music"
    if (
        speech_act in {"request", "command"}
        and action in {"play", "start"}
        and bool(obj_words & {"music", "song", "audio", "media"})
    ):
        return "media_playback"

    # Health advice — parse-level health-domain semantic classes
    has_health_sem = bool(
        sem_classes
        & {"health_domain", "health_condition", "advice_action", "medical_procedure"}
    )
    if has_health_sem:
        return "health_advice"

    # Common sense safety — clothing + public place OR undress state
    has_clothing = "clothing_item" in sem_classes
    has_public = "public_place" in sem_classes
    has_undress = "undress_state" in sem_classes
    if has_undress or (has_clothing and has_public):
        return "common_sense_safety"

    # Social contact — social relation + contact action
    has_social_relation = bool(
        sem_classes & {"social_relation", "child_relation"}
    )
    is_contact_action = action in {"call", "phone", "ring", "reach", "contact"}
    if has_social_relation and is_contact_action:
        return "social_contact"

    # Personal memory / autobiographical — memory recall semantic class
    has_memory_sem = bool(
        sem_classes
        & {"memory_recall", "personal_attribute", "autobiographical_event"}
    )
    if has_memory_sem:
        return "personal_memory"

    if _detect_social_status(text):
        return "assistant_status"

    return None


def _semantic_classes_from_atom(atom: dict[str, Any]) -> set[str]:
    """Collect semantic classes from an atom's predicate and roles."""
    classes: set[str] = set()
    predicate = atom.get("predicate", {})
    sc = str(predicate.get("semantic_class", "")).strip().lower()
    if sc and sc != "unknown":
        classes.add(sc)
    # Also check roles for semantic class annotations
    for role in atom.get("roles", []):
        role_sc = str(role.get("semantic_class", "")).strip().lower()
        if role_sc:
            classes.add(role_sc)
    return classes


def _classify_from_atoms(
    uol_act: dict[str, Any] | None,
    tokens: tuple[str, ...],
    text: str,
) -> str | None:
    """Atom-based intent classification from UolAct dict.

    Returns an intent string when the atom clearly indicates a known frame;
    returns None when genuinely ambiguous.
    """
    if uol_act is None:
        return None

    # Token-level composition checks (still use tokens for these)
    if _identity_composition(text, tokens) is not None:
        return "assistant_identity"
    if _self_status_composition(text, tokens) is not None:
        return "assistant_status"
    # Social-status fallback: detect "how are you" patterns in raw text
    # when UOL atoms may overfit and miss a clear status question.
    if _detect_social_status(text):
        return "assistant_status"

    # Extract atom structure
    act_type = str(uol_act.get("act", ""))  # "question", "request", "command", "claim", etc.
    content = uol_act.get("content", [])

    if not content:
        verb = _extract_verb(uol_act=uol_act)
        if verb:
            mc = _lazy_moral_engine()(verb, "person")
            if not mc.has_implication:
                from melm.appliance.reasoning.implications import record_verb_candidate
                record_verb_candidate(verb, "person", text)
            if mc.harm_severity == "high":
                return "health_advice"
        if _has_urgent_health_frame(tokens):
            return "health_advice"
        if _detect_social_status(text):
            return "assistant_status"
        token_set = set(tokens)
        if "your" in token_set and "name" in token_set and "?" in text:
            return "assistant_identity"
        return None

    main_atom = content[0] if content else {}
    predicate = main_atom.get("predicate", {})
    pred_id = str(predicate.get("id", "")).strip().lower()
    sem_class = str(predicate.get("semantic_class", "")).strip().lower()
    roles = main_atom.get("roles", [])
    context = main_atom.get("context", {})
    polarity = str(context.get("polarity", "positive")).strip().lower()
    modality = str(context.get("modality", "assertive")).strip().lower()
    negation_scope = bool(context.get("negation_scope", False))
    negated_predicate = polarity == "negative" or negation_scope
    blocked_action_match = negated_predicate or modality == "counterfactual"

    if not negated_predicate:
        verb = _extract_verb(uol_act=uol_act)
        if verb:
            patient = _extract_patient_type(uol_act=uol_act)
            mc = _lazy_moral_engine()(verb, patient or "person")
            if not mc.has_implication:
                from melm.appliance.reasoning.implications import record_verb_candidate
                record_verb_candidate(verb, patient or "person", text)
            if mc.harm_severity == "high":
                return "health_advice"
        if _has_urgent_health_frame(tokens):
            return "health_advice"

    # Collect theme values from roles
    theme_values = {
        str(r.get("value", "")).strip().lower()
        for r in roles
        if r.get("role") == "theme"
    }
    # Collect agent values
    agent_values = {
        str(r.get("value", "")).strip().lower()
        for r in roles
        if r.get("role") == "agent"
    }

    # Map UolAct.act → speech_act equivalents
    is_question_act = act_type in {"question"}
    is_request_act = act_type in {"request", "command"}

    # Semantic class checks
    health_sem_classes = {"health_domain", "health_condition", "advice_action", "medical_procedure"}
    has_health_sem = any(
        sc in sem_class or sc in sem_class.replace(".", "_")
        for sc in health_sem_classes
    ) or bool(health_sem_classes & _semantic_classes_from_atom(main_atom))

    # Meal suggestion: eat/cook/prepare + question/request + user agent
    if (
        not blocked_action_match
        and
        (is_question_act or is_request_act)
        and pred_id in {"eat", "cook", "prepare", "eri", "nri"}
        and (agent_values & {"user", "i", "we"} or not agent_values)
    ):
        return "meal_suggestion"

    # Story: tell/read/make/give + story/tale in theme
    if (
        not blocked_action_match
        and
        (is_question_act or is_request_act)
        and pred_id in {"tell", "read", "make", "give"}
        and bool(theme_values & {"story", "tale", "fable", "narrative"})
    ):
        return "story"

    # Weather: rain/snow/forecast predicate
    if (
        not blocked_action_match
        and
        (is_question_act or is_request_act)
        and pred_id in {"rain", "snow", "forecast", "weather"}
    ):
        return "weather"

    # Media playback: play/start + music/song theme
    if (
        not blocked_action_match
        and
        is_request_act
        and pred_id in {"play", "start"}
        and bool(theme_values & {"music", "song", "audio", "media", "radio"})
    ):
        return "media_playback"

    # Health advice from semantic class
    if has_health_sem and not blocked_action_match:
        return "health_advice"

    # Common sense safety: clothing_item or undress_state class
    atom_sem_classes = _semantic_classes_from_atom(main_atom)
    if not blocked_action_match and (
        "undress_state" in atom_sem_classes or (
            "clothing_item" in atom_sem_classes and "public_place" in atom_sem_classes
        )
    ):
        return "common_sense_safety"

    # Social contact: call/phone + social_relation
    has_social_relation = "social_relation" in atom_sem_classes or "child_relation" in atom_sem_classes
    if not blocked_action_match and has_social_relation and pred_id in {"call", "phone", "ring", "reach", "contact"}:
        return "social_contact"

    # Personal memory: memory_recall semantic class
    if bool(atom_sem_classes & {"memory_recall", "personal_attribute", "autobiographical_event"}):
        return "personal_memory"

    return None


def is_question_like_act(uol_act: dict[str, Any] | None) -> bool:
    """True when the UolAct speech act is a question type."""
    if uol_act is None:
        return False
    return str(uol_act.get("act", "")) == "question"


def is_request_like_act(uol_act: dict[str, Any] | None) -> bool:
    """True when the UolAct speech act is a request or command."""
    if uol_act is None:
        return False
    return str(uol_act.get("act", "")) in {"request", "command"}


def _frame_linker_candidates_from_atoms(
    uol_act: dict[str, Any],
    tokens: tuple[str, ...],
) -> list[FrameCandidate]:
    linker = _get_frame_linker()
    return linker.score_atoms(uol_act, _IN_MEMORY_LEXICON, tokens=tokens)


def _frame_linker_candidate_matches(
    candidates: list[FrameCandidate],
    frame_id: str,
    *,
    use_margin: bool = False,
) -> bool:
    if not candidates:
        return False
    top = candidates[0]
    needed = _FRAME_LINKER_CONFIRMATION_MARGIN if use_margin else 0.0
    return bool(top.frame_id == frame_id and top.score >= top.threshold + needed)


def _classify_intent_from_uol_slots(
    text: str,
    tokens: tuple[str, ...],
    trusted_contact_names: tuple[str, ...] = (),
    language: str = "en",
    parse_bundle: _ParseBundle | None = None,
    uol_act: dict[str, Any] | None = None,
) -> AssistantIntent:
    # "why" follow-up: if the only token is "why" and the previous turn was assistant_identity
    if tokens == ("why",) and parse_bundle is not None and parse_bundle.last_intent == "assistant_identity":
        return "assistant_identity"

    functional_parse = (
        parse_bundle.functional_parse
        if parse_bundle is not None
        else parse_functional_relations(tokens, question_mark="?" in text, language=language)
    )
    atom_routing = uol_act is not None
    atom_intent = (
        _classify_from_atoms(uol_act, tokens, text)
        if atom_routing
        else _classify_from_functional_parse(functional_parse, tokens, text)
    )
    if atom_intent is not None:
        return atom_intent

    # Phase 7: Auto-correction — if the parse was derailed by an unknown predicate
    # and there are multiple atoms, try re-classification ignoring the first unknown atom.
    if atom_intent is None and uol_act is not None:
        try:
            content = uol_act.get("content", [])
            if len(content) > 1:
                first_pred = content[0].get("predicate", {}).get("semantic_class", "")
                if first_pred in ("unknown", "semantic_unknown") or not first_pred:
                    filtered = {"act": uol_act.get("act", ""), "content": list(content[1:])}
                    corrected = _classify_from_atoms(filtered, tokens, text)
                    if corrected is not None:
                        return corrected
        except Exception:
            pass

    question_like = (
        is_question_like_act(uol_act)
        if atom_routing
        else _speech_act_is_question(functional_parse.speech_act) if functional_parse is not None else _surface_question_like(text, tokens)
    )
    request_like = (
        is_request_like_act(uol_act)
        if atom_routing
        else _speech_act_is_request(functional_parse.speech_act) if functional_parse is not None else _surface_request_like(tokens)
    )
    atom_candidates = (
        _frame_linker_candidates_from_atoms(uol_act, tokens)
        if atom_routing and uol_act is not None
        else None
    )

    def _route_frame_match(
        frame_id: str,
        *,
        collector_classes: frozenset[str] = frozenset(),
        use_margin: bool = False,
    ) -> bool:
        if collector_classes:
            _semantic_family_terms(tokens, semantic_classes=collector_classes)
        if atom_candidates is not None:
            return _frame_linker_candidate_matches(
                atom_candidates,
                frame_id,
                use_margin=use_margin,
            )
        return _classify_from_frame_linker(
            text,
            tokens,
            frame_id,
            collector_classes=collector_classes,
            use_margin=use_margin,
        )

    # Secondary: frame linker scoring for intents not yet fully atomized
    # (operates on semantic classes, not raw English keywords)
    # Story requires an explicit request verb AND question/request speech act
    # to avoid "What is a story?" (definition) or "The same people tell stories" (statement)
    story_action_present = bool(set(tokens) & {"tell", "read", "make", "give"})
    story_speech_act_ok = request_like or question_like
    if story_action_present and story_speech_act_ok and _route_frame_match(
        "story",
        collector_classes=frozenset({"narrative_content"}),
        use_margin=False,
    ):
        return "story"
    if _route_frame_match(
        "weather",
        collector_classes=frozenset({"weather_phenomenon"}),
        use_margin=False,
    ):
        return "weather"
    if _route_frame_match(
        "common_sense_safety",
        collector_classes=frozenset({"clothing_item", "public_place", "undress_state"}),
        use_margin=False,
    ):
        return "common_sense_safety"
    if set(tokens) & {"play", "start"} and _route_frame_match(
        "media_playback",
        collector_classes=frozenset({
            "media_content", "media_descriptor",
            "physical_object.instrument",
            "physical_object.media_source",
        }),
        use_margin=False,
    ):
        return "media_playback"
    if _route_frame_match(
        "health_advice",
        collector_classes=frozenset({"health_domain", "health_condition", "advice_action"}),
        use_margin=False,
    ):
        return "health_advice"
    if _is_private_cloud_export_request(text, tokens):
        return "personal_memory"
    # Social contact — frame linker + trusted name pre-filter
    # Requires explicit contact-action signal AND request/question structure
    # to avoid "Send my child's age to cloud" → social_contact
    has_social_relation = any(
        "social_relation" in _IN_MEMORY_LEXICON.get(t, frozenset())
        or "child_relation" in _IN_MEMORY_LEXICON.get(t, frozenset())
        for t in tokens
    )
    has_trusted_name = bool(
        trusted_contact_names and _matched_trusted_contact_name(tokens, trusted_contact_names)
    )
    if not (has_social_relation or has_trusted_name):
        pass  # skip social_contact entirely
    else:
        has_contact_action_token = bool(set(tokens) & {"call", "phone", "ring", "reach"})
        has_talk_context = bool(set(tokens) & {"need", "help", "please"})
        contact_semantic = _semantic_family_terms(tokens, semantic_classes={"contact_action"})
        comm_semantic = _semantic_family_terms(tokens, semantic_classes={"communication_action"})
        is_request_or_question = request_like or question_like
        # Path A: explicit contact-action tokens OR contact semantic + structure
        if has_contact_action_token or (contact_semantic and is_request_or_question):
            if _route_frame_match(
                "social_contact",
                collector_classes=frozenset({"contact_action", "communication_action", "social_relation"}),
                use_margin=False,
            ):
                return "social_contact"
        # Path B: communication_action semantic class with talk/question context only
        if comm_semantic and (has_talk_context or question_like):
            if _route_frame_match(
                "social_contact",
                collector_classes=frozenset({"contact_action", "communication_action", "social_relation"}),
                use_margin=False,
            ):
                return "social_contact"
    if _route_frame_match(
        "personal_memory",
        collector_classes=frozenset({"memory_recall", "personal_attribute", "child_relation", "social_relation"}),
        use_margin=False,
    ):
        return "personal_memory"
    token_set = set(tokens)
    # Structural fallbacks for patterns not yet expressible as atoms.
    if _is_child_memory_request(tokens):
        return "personal_memory"
    memory_cognition = _semantic_family_terms(tokens, semantic_classes={"memory_recall"})
    memory_frame = question_like or request_like or bool(memory_cognition)
    if _is_routine_memory_request(tokens):
        owned_or_recalled = bool(
            token_set & {"my", "our", "me", "i"} or memory_cognition or _about_targets_self(tokens)
        )
        if memory_frame and owned_or_recalled:
            return "personal_memory"
    if _is_household_memory_request(tokens):
        owned_or_recalled = bool(
            token_set & {"my", "our", "we", "us", "this"} or memory_cognition or _is_device_user_memory_question(tokens)
        )
        if memory_frame and owned_or_recalled:
            return "personal_memory"
    if {"who", "am", "i"} <= token_set:
        return "personal_memory"
    if _profile_attribute_requested(tokens) is not None:
        return "personal_memory"
    first_person_targets = {"me", "my", "myself", "i"}
    if memory_cognition and token_set & first_person_targets:
        if _route_frame_match(
            "personal_memory",
            collector_classes=frozenset({"memory_recall", "personal_attribute", "child_relation", "social_relation"}),
        ):
            return "personal_memory"
    if _about_targets_self(tokens):
        return "personal_memory"
    # Autobiographical fallback: token + semantic structure when atom coverage misses
    if _autobiographical_question_or_command(text, tokens, question_like=question_like):
        if _autobiographical_long_horizon_frame(text, tokens, question_like=question_like):
            return "autobiographical_memory"
        if _autobiographical_session_summary_frame(text, tokens, question_like=question_like):
            return "autobiographical_memory"
        if _autobiographical_latest_event_frame(text, tokens, question_like=question_like):
            return "autobiographical_memory"
        if _route_frame_match(
            "autobiographical_memory",
            collector_classes=frozenset({"autobiographical_event", "autobiographical_action", "temporal_descriptor"}),
            use_margin=False,
        ):
            return "autobiographical_memory"
        if token_set & {"we", "our"} and (token_set & {"talk", "talked", "conversation", "conversations"}):
            if (
                _semantic_family_terms(tokens, semantic_classes={"autobiographical_action"})
                or _semantic_family_terms(tokens, semantic_classes={"temporal_descriptor"})
                or _semantic_family_terms(tokens, semantic_classes={"communication_action"})
            ):
                return "autobiographical_memory"
        if token_set & {"we", "our"} and (token_set & {"discuss", "discussed", "discussion", "chat", "chatted"}):
            return "autobiographical_memory"
    # Meal suggestion fallback: frame linker when atoms miss non-English or edge cases
    if _route_frame_match(
        "meal_suggestion",
        collector_classes=frozenset({"food_item"}),
        use_margin=False,
    ):
        return "meal_suggestion"

    functional_kind = functional_frame_kind(functional_parse)
    if functional_kind in {
        "social_greeting",
        "assistant_behavior",
        "personal_goal_advice",
        "open_domain",
    }:
        return functional_kind  # type: ignore[return-value]

    # Frame linker catches high-confidence cases that even functional grammar
    # misses. Uses margin for non-migrated intents (prevents false positives).
    linker = _get_frame_linker()
    candidates = (
        atom_candidates
        if atom_candidates is not None
        else linker.score(
            tokens,
            _IN_MEMORY_LEXICON,
            is_question_like=question_like,
            is_request_like=request_like,
        )
    )
    if candidates:
        # Apply UOL-aware reranker when functional parse is available.
        if functional_parse is not None and atom_candidates is None:
            reranker = _get_frame_reranker()
            reranked = reranker.rerank(
                candidates, tokens, _IN_MEMORY_LEXICON,
                is_question_like=question_like,
                is_request_like=request_like,
                token_roles=functional_parse.token_roles,
            )
            top = reranked[0]
        else:
            top = candidates[0]
        if top.frame_id in _FRAME_LINKER_MIGRATED_INTENTS:
            # Migrated intents require a non-required contribution
            # (structure, action, or optional) to prevent bare
            # required-class matches like "hi-fi audio" -> media_playback.
            return top.intent if top.score > top.threshold else "unknown"
        if top.score >= top.threshold + _FRAME_LINKER_CONFIRMATION_MARGIN:
            return top.intent
    return "unknown"


def _is_assistant_identity_request(
    text: str, tokens: tuple[str, ...] | None = None
) -> bool:
    return _identity_composition(text, tokens or _tokenize(text)) is not None


def _is_assistant_status_request(
    text: str, tokens: tuple[str, ...] | None = None
) -> bool:
    return _self_status_composition(text, tokens or _tokenize(text)) is not None




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



_IN_MEMORY_LEXICON: dict[str, frozenset[str]] = build_legacy_in_memory_lexicon()
# Wire the growing lexicon into the UOL grammar so acquired verbs lemmatize
# and receive a semantic class even when not in the hardcoded _VERBS dict.
set_uol_lexicon(_IN_MEMORY_LEXICON)


def refresh_in_memory_lexicon(store: Any) -> int:
    """Merge active store-backed lexical senses into the routing lexicon.

    Queries ``lexemes`` JOIN ``lexical_senses`` for status='active' rows and
    adds any term not already present in ``_IN_MEMORY_LEXICON``.  Existing
    legacy entries are preserved (reserved/policy terms cannot be overwritten
    at runtime).

    Returns the number of new terms added.  Safe to call after every
    successful ``lexicon_ingest()``; the operation is idempotent.
    """
    global _IN_MEMORY_LEXICON
    try:
        rows = store.connection.execute(
            """
            SELECT lx.normalized_lemma, ls.semantic_class_id
            FROM lexemes lx
            JOIN lexical_senses ls ON ls.lexeme_id = lx.lexeme_id
            WHERE ls.status = 'active'
            """
        ).fetchall()
    except Exception:
        return 0
    added = 0
    for row in rows:
        lemma = str(row["normalized_lemma"]).strip().lower()
        class_id = str(row["semantic_class_id"]).strip().lower()
        if not lemma or not class_id:
            continue
        existing = _IN_MEMORY_LEXICON.get(lemma)
        if existing is None:
            _IN_MEMORY_LEXICON[lemma] = frozenset({class_id})
            added += 1
        elif class_id not in existing:
            _IN_MEMORY_LEXICON[lemma] = existing | frozenset({class_id})
    set_uol_lexicon(_IN_MEMORY_LEXICON)
    return added
# Seed Igbo lemmas from contract so the UOL parser recognises them.
try:
    _igbo_lexicon_data = load_igbo_lexicon_seed()
    seed_igbo_lexicon(_IN_MEMORY_LEXICON, _igbo_lexicon_data.get("entries", []))
except Exception:
    pass
# Frame linker requires score >= template threshold + this margin to fire.
# This prevents false positives on borderline matches while still catching
# high-confidence cases that keyword classifiers miss.
_FRAME_LINKER_CONFIRMATION_MARGIN = 0.20
# Intents handled by the UOL atom classifier first; frame linker is secondary fallback.
# These require stricter scoring to avoid false positives when atoms return None.
_FRAME_LINKER_MIGRATED_INTENTS: frozenset[str] = frozenset({
    "weather", "story", "media_playback",
    "autobiographical_memory", "meal_suggestion",
    "common_sense_safety", "social_contact",
    "health_advice",
})
_FRAME_LINKER: FrameLinker | None = None
_FRAME_RERANKER: E3CandidateReranker | None = None
# Capability-manifest state: installed set + all managed families.
# Initialised lazily; overridable via replace_installed_families().
_INSTALLED_FAMILIES: frozenset[str] | None = None
_ALL_MANAGED_FAMILIES: frozenset[str] | None = None


def _is_family_installed(family: str) -> bool:
    """Check whether *family* is installed per the capability manifest.

    Families not listed in the manifest (unmanaged) are allowed through.
    Only explicitly-marked-not-installed families are blocked.
    """
    installed, managed = _get_capability_manifest()
    if family not in managed:
        return True
    return family in installed


def _get_capability_manifest() -> tuple[frozenset[str], frozenset[str]]:
    global _INSTALLED_FAMILIES, _ALL_MANAGED_FAMILIES
    if _INSTALLED_FAMILIES is None:
        path = Path(__file__).resolve().parent.parent / "contracts" / "default_capability_manifest.v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("families", {})
        _ALL_MANAGED_FAMILIES = frozenset(raw.keys())
        _INSTALLED_FAMILIES = frozenset(
            k for k, v in raw.items() if v.get("installed")
        )
    return _INSTALLED_FAMILIES, _ALL_MANAGED_FAMILIES


def replace_installed_families(
    installed: frozenset[str] | None,
    managed: frozenset[str] | None = None,
) -> None:
    """Override capability-manifest state (used for testing)."""
    global _INSTALLED_FAMILIES, _ALL_MANAGED_FAMILIES, _CAPABILITY_PAYLOAD
    _INSTALLED_FAMILIES = installed
    _ALL_MANAGED_FAMILIES = managed
    _CAPABILITY_PAYLOAD = None


_CAPABILITY_PAYLOAD: dict[str, Any] | None = None


def _capability_payload() -> dict[str, Any]:
    global _CAPABILITY_PAYLOAD
    if _CAPABILITY_PAYLOAD is None:
        path = Path(__file__).resolve().parent.parent / "contracts" / "default_capability_manifest.v1.json"
        _CAPABILITY_PAYLOAD = json.loads(path.read_text(encoding="utf-8"))
    return _CAPABILITY_PAYLOAD


def _capability_flag(family: str, key: str, default: bool = False) -> bool:
    """Read a nested capability sub-flag, e.g. ``mood_affect.creative_behaviors``.

    Unlike ``_is_family_installed`` (family install state only), this reads a
    sub-key of an installed family. Returns *default* when the family is absent
    or not installed.
    """
    fam = _capability_payload().get("families", {}).get(family)
    if not isinstance(fam, dict) or not fam.get("installed"):
        return default
    return bool(fam.get(key, default))


def _classify_from_frame_linker(
    text: str,
    tokens: tuple[str, ...],
    frame_id: str,
    *,
    collector_classes: frozenset[str] = frozenset(),
    use_margin: bool = False,
    parse_bundle: _ParseBundle | None = None,
) -> bool:
    """Check if *frame_id* is the top-scoring candidate above its effective threshold.

    Migrated intents (validated templates) use the bare template threshold.
    Non-migrated fallback uses *use_margin* to prevent false positives.
    """
    if collector_classes:
        _semantic_family_terms(tokens, semantic_classes=collector_classes)
    parse_bundle = parse_bundle or _build_parse_bundle(text)
    candidates = _frame_linker_candidates_for_parse_bundle(parse_bundle)
    needed = _FRAME_LINKER_CONFIRMATION_MARGIN if use_margin else 0.0
    if not candidates:
        return False
    top = candidates[0]
    return bool(
        top.frame_id == frame_id
        and top.score >= top.threshold + needed
    )


def _get_frame_linker() -> FrameLinker:
    global _FRAME_LINKER
    if _FRAME_LINKER is None:
        _FRAME_LINKER = FrameLinker()
    return _FRAME_LINKER


def _question_like_from_parse_bundle(parse_bundle: _ParseBundle) -> bool:
    if parse_bundle.uol_act is not None:
        return is_question_like_act(parse_bundle.uol_act)
    if parse_bundle.functional_parse is not None:
        return _speech_act_is_question(parse_bundle.functional_parse.speech_act)
    return _surface_question_like(parse_bundle.text, parse_bundle.tokens)


def _request_like_from_parse_bundle(parse_bundle: _ParseBundle) -> bool:
    if parse_bundle.uol_act is not None:
        return is_request_like_act(parse_bundle.uol_act)
    if parse_bundle.functional_parse is not None:
        return _speech_act_is_request(parse_bundle.functional_parse.speech_act)
    return _surface_request_like(parse_bundle.tokens)


def _speech_act_is_question(speech_act: str) -> bool:
    return speech_act in {"question", "wh_question", "yes_no_question"}


def _speech_act_is_request(speech_act: str) -> bool:
    return speech_act in {"request", "command"}


def _frame_linker_candidates_for_parse_bundle(
    parse_bundle: _ParseBundle,
) -> list[FrameCandidate]:
    if parse_bundle.uol_act is not None:
        return _frame_linker_candidates_from_atoms(
            parse_bundle.uol_act,
            parse_bundle.tokens,
        )
    linker = _get_frame_linker()
    return linker.score(
        parse_bundle.tokens,
        _IN_MEMORY_LEXICON,
        is_question_like=_question_like_from_parse_bundle(parse_bundle),
        is_request_like=_request_like_from_parse_bundle(parse_bundle),
    )


def _get_frame_reranker() -> E3CandidateReranker:
    global _FRAME_RERANKER
    if _FRAME_RERANKER is None:
        _FRAME_RERANKER = E3CandidateReranker()
    return _FRAME_RERANKER


def replace_in_memory_lexicon(
    lexicon: dict[str, frozenset[str]],
) -> None:
    _IN_MEMORY_LEXICON.clear()
    _IN_MEMORY_LEXICON.update(lexicon)


def inject_lexicon_entry(lemma: str, class_id: str) -> None:
    """Inject or update a single lemma→class mapping in the runtime lexicon.

    Adds *class_id* to the existing frozenset for *lemma* if the lemma already
    exists, otherwise creates a new entry.  Does not affect other entries.
    """
    existing = _IN_MEMORY_LEXICON.get(lemma, frozenset())
    _IN_MEMORY_LEXICON[lemma] = frozenset(existing | {class_id})


# Collector for activated semantic classes during a single ``handle()`` call.
# Reset via ``set_semantic_class_collector`` before routing, read after.
_SEMANTIC_CLASS_COLLECTOR: set[str] | None = None


def set_semantic_class_collector(collector: set[str] | None) -> None:
    global _SEMANTIC_CLASS_COLLECTOR
    _SEMANTIC_CLASS_COLLECTOR = collector


def _semantic_family_terms(
    tokens: tuple[str, ...],
    semantic_classes: set[str],
) -> set[str]:
    result: set[str] = set()
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(tokens):
            compound = f"{token}_{tokens[i + 1]}"
            if _IN_MEMORY_LEXICON.get(compound, frozenset()) & semantic_classes:
                result.add(compound)
                skip_next = True
                continue
        if _IN_MEMORY_LEXICON.get(token, frozenset()) & semantic_classes:
            result.add(token)
    if result and _SEMANTIC_CLASS_COLLECTOR is not None:
        matched_classes: set[str] = set()
        for term in result:
            matched_classes.update(_IN_MEMORY_LEXICON.get(term, frozenset()) & semantic_classes)
        _SEMANTIC_CLASS_COLLECTOR.update(matched_classes)
    return result


def rebuild_entity_lexicon_index(
    store: Any,
) -> None:
    """Inject entity labels into _IN_MEMORY_LEXICON for entity-aware routing.

    Reads entities WHERE kind IN ('person', 'place', 'object', 'event_type') and
    injects their label (and canonical_lemma, if different) into _IN_MEMORY_LEXICON
    with the entity's semantic_class_id. Multi-word labels are injected as
    underscore-joined compound keys for bigram matching in _semantic_family_terms.
    """
    rows = store.connection.execute(
        """
        SELECT e.label, e.canonical_lemma, e.semantic_class_id
        FROM entities e
        WHERE e.kind IN ('person', 'place', 'object', 'event_type', 'event_instance')
        """
    ).fetchall()
    for row in rows:
        label = str(row["label"]).strip().lower()
        lemma = str(row["canonical_lemma"]).strip().lower() if row["canonical_lemma"] else ""
        class_id = str(row["semantic_class_id"])
        if not class_id:
            continue
        terms: set[str] = {label}
        if lemma and lemma != label:
            terms.add(lemma)
        for term in terms:
            for t in (term, term.replace(" ", "_")):
                existing = _IN_MEMORY_LEXICON.get(t, frozenset())
                _IN_MEMORY_LEXICON[t] = frozenset(existing | {class_id})




def _has_contact_target(
    tokens: tuple[str, ...],
    trusted_contact_names: tuple[str, ...] = (),



) -> bool:
    relation_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"social_relation", "child_relation"},


    )
    return bool(
        relation_terms
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




def _autobiographical_question_or_command(
    text: str,
    tokens: tuple[str, ...],
    *,
    question_like: bool | None = None,
) -> bool:
    is_question = (
        _surface_question_like(text, tokens) if question_like is None else question_like
    )
    return is_question or tokens[:1] in {
        ("summarize",),
        ("recap",),
        ("show",),
        ("list",),
        ("tell",),
    }


def _autobiographical_memory_scope(parse_bundle: _ParseBundle) -> str:
    text = parse_bundle.text
    tokens = parse_bundle.tokens
    token_set = set(tokens)
    question_like = _question_like_from_parse_bundle(parse_bundle)
    if not _autobiographical_question_or_command(
        text,
        tokens,
        question_like=question_like,
    ):
        return ""
    if _autobiographical_long_horizon_frame(
        text,
        tokens,
        question_like=question_like,
    ):
        return "long_horizon"
    if _autobiographical_session_summary_frame(
        text,
        tokens,
        question_like=question_like,
    ):
        return "session_summary"
    if _autobiographical_latest_event_frame(
        text,
        tokens,
        question_like=question_like,
    ):
        return "latest_event"
    if _classify_from_frame_linker(
        text,
        tokens,
        "autobiographical_memory",
        collector_classes=frozenset(
            {
                "autobiographical_event",
                "autobiographical_action",
                "temporal_descriptor",
            }
        ),
        use_margin=False,
        parse_bundle=parse_bundle,
    ):
        return "event_query"
    if token_set & {"we", "our"} and (
        token_set & {"talk", "talked", "conversation", "conversations"}
    ):
        if (
            _semantic_family_terms(tokens, semantic_classes={"autobiographical_action"})
            or _semantic_family_terms(tokens, semantic_classes={"temporal_descriptor"})
            or _semantic_family_terms(tokens, semantic_classes={"communication_action"})
        ):
            return "event_query"
    if token_set & {"we", "our"} and (
        token_set & {"discuss", "discussed", "discussion", "chat", "chatted"}
    ):
        return "event_query"
    return ""


def _autobiographical_long_horizon_frame(
    text: str,
    tokens: tuple[str, ...],
    *,
    question_like: bool | None = None,
) -> bool:
    token_set = set(tokens)
    if not _autobiographical_question_or_command(text, tokens, question_like=question_like):
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


def _autobiographical_session_summary_frame(
    text: str,
    tokens: tuple[str, ...],
    *,
    question_like: bool | None = None,
) -> bool:
    token_set = set(tokens)
    if not _autobiographical_question_or_command(text, tokens, question_like=question_like):
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


def _autobiographical_latest_event_frame(
    text: str,
    tokens: tuple[str, ...],
    *,
    question_like: bool | None = None,
) -> bool:
    token_set = set(tokens)
    if not _autobiographical_question_or_command(text, tokens, question_like=question_like):
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



def _surface_question_like(text: str, tokens: tuple[str, ...]) -> bool:
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
        ("where",),
    }


def _surface_request_like(tokens: tuple[str, ...]) -> bool:
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
    return tuple(re.findall(r"[a-z0-9']+", text.lower()))


def _compose_primary_frame(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    *,
    question_like: bool | None = None,
    functional_parse: FunctionalParse | None = None,
) -> dict[str, Any] | None:
    if intent == "assistant_identity":
        return _identity_composition(text, tokens)
    if intent == "assistant_status":
        return _self_status_composition(text, tokens, question_like=question_like)
    if intent in {
        "social_greeting",
        "assistant_behavior",
        "personal_goal_advice",
        "open_domain",
    }:
        return _compose_functional_frame(
            text,
            tokens,
            intent,
            functional_parse=functional_parse,
        )
    return _compose_semantic_frame(
        text,
        tokens,
        intent,
    )


def _assistant_frame_id(composition: dict[str, Any]) -> str:
    intent = str(composition.get("intent", "unknown"))
    pattern = str(composition.get("pattern", "unmatched"))
    return f"{intent}.{pattern}"


def _compose_semantic_frame(
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


def _compose_functional_frame(
    text: str,
    tokens: tuple[str, ...],
    intent: AssistantIntent,
    *,
    functional_parse: FunctionalParse | None = None,
) -> dict[str, Any] | None:
    parse = functional_parse or parse_functional_relations(tokens, question_mark="?" in text)
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
    aliases = load_router_semantic_aliases().get("object_role_tokens", {})
    tokens.update(set(aliases.get(intent, [])))
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


def _detect_social_status(text: str) -> bool:
    """Check if raw text contains a "how are you" -type social question.

    Used as a fallback when UOL atoms overfit on noise tokens (e.g.
    "lol, how are you?" where "lol" becomes the action).
    """
    lower = text.lower().strip().rstrip("?.,!;: ")
    # Avoid false positives: "you doing" matches "what are you doing" (action Q, not status Q).
    # Only match when the first word is "how" (how are you doing, how you doing).
    if "you doing" in lower and not lower.startswith("how"):
        return False
    patterns = ("how are you", "how's it going", "how do you feel", "how are you doing", "how you doing")
    return any(p in lower for p in patterns)


def _self_status_composition(
    text: str,
    tokens: tuple[str, ...],
    *,
    question_like: bool | None = None,
) -> dict[str, Any] | None:
    token_set = set(tokens)
    if not tokens:
        return None
    if "think" in token_set and (
        token_set & {"i", "me", "my"} or token_set & {"health", "eat", "food"}
    ):
        return None
    autobiographical_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"autobiographical_event"},
    )
    if autobiographical_terms:
        return None
    command_like = tokens[:1] in {("show",), ("report",), ("summarize",), ("list",)}
    question_or_command = (
        _surface_question_like(text, tokens) if question_like is None else question_like
    ) or command_like
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
    # Name suggestion/awareness patterns — checked only when no higher-priority pattern matched
    if not pattern:
        name_tokens = {"name", "named", "call", "calling"}
        if name_tokens & set(tokens):
            if "what" in tokens and "should" in tokens and "i" in tokens and "you" in tokens:
                pattern = "what_modal_i_call_you"
                action = "suggest_name"
                focus = "name"
                basis = ("what:attribute_question", "should:deontic_modal", "i:self_deixis", "call:label_action", "you:assistant_deixis")
            elif "name" in tokens and \
                 ("your" in tokens or ("you" in tokens and "my" not in tokens) or "yourself" in tokens) and \
                 any(q in tokens for q in ("do", "what", "did", "have", "who", "is", "are")):
                if "yourself" in tokens:
                    action = "name_origin"
                else:
                    action = "name_awareness"
                focus = "name"
                pattern = "name_awareness_question"
                basis = ("name:identity_attribute", "your:assistant_possessive")
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


def _profile_attribute_requested(tokens: tuple[str, ...]) -> str | None:
    """First-person attribute recall question → 'name' | 'location' | 'age' | None.

    Catches 'what is my name', 'where do I live', 'how old am I'. Requires a
    question frame plus a first-person marker so it never fires on third-person
    or non-questions. Returns None for anything else (falls through to routing).
    """
    token_set = set(tokens)
    # Third-party / relational questions ("how old is my child", "my son's name")
    # belong to child/household memory, not the user's own attributes.
    if token_set & {
        "child", "children", "son", "daughter", "kid", "kids", "baby",
        "family", "household", "wife", "husband", "mom", "dad", "mother", "father",
    }:
        return None
    if not (token_set & {"my", "i", "me", "mine", "am"}):
        return None
    if not (token_set & {"what", "where", "who", "how", "whats"}):
        return None
    if "name" in token_set and (token_set & {"my", "mine"}):
        return "name"
    if "location" in token_set or ("where" in token_set and token_set & {"live", "from", "located", "stay"}):
        return "location"
    if "age" in token_set or ("how" in token_set and "old" in token_set):
        return "age"
    return None


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
    *,
    uol_act: dict[str, Any] | None = None,
) -> dict[str, Any]:
    speech_act = "request"
    if uol_act is not None:
        act_type = str(uol_act.get("act", ""))
        if act_type == "question":
            speech_act = "question"
        elif act_type == "claim":
            speech_act = "statement"
        elif act_type:
            speech_act = act_type
    elif "?" in text or tokens[:1] in {("who",), ("what",), ("why",), ("how",)}:
        speech_act = "question"
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
    if intent == "personal_goal_advice":
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
    groups = load_router_semantic_aliases().get("secondary_hint_groups", {})
    return {intent: tuple(tokens) for intent, tokens in groups.items()}


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
        if _surface_question_like(text, tokens):
            notes.append("question_mapped_by_semantic_parse_not_question_mark")
        elif outward_request or _surface_request_like(tokens):
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
    memory_terms = _semantic_family_terms(
        tokens, semantic_classes=frozenset({"memory_recall"}),
    )
    if "you" in token_set and memory_terms and token_set & {"me", "myself"}:
        return True
    return False


def _is_routine_memory_request(
    tokens: tuple[str, ...],
) -> bool:
    token_set = set(tokens)
    routine_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"routine_concept"},
    )
    if routine_terms:
        return True
    day_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"temporal_descriptor"},
    )
    return bool(
        ({"school"} & day_terms or {"work"} & day_terms)
        and {"day"} & day_terms
    )


def _is_household_memory_request(
    tokens: tuple[str, ...],



) -> bool:
    token_set = set(tokens)
    household_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"household_concept"},


    )
    if household_terms:
        return True
    owner_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"owner_concept", "hardware_entity"},


    )
    if len(owner_terms) >= 2:
        return True
    return _is_device_user_memory_question(
        tokens,


    )


def _is_device_user_memory_question(
    tokens: tuple[str, ...],



) -> bool:
    token_set = set(tokens)
    action_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"action_verb"},


    )
    hardware_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"hardware_entity"},


    )
    return bool(
        "who" in token_set
        and action_terms
        and hardware_terms
    )


def _is_child_memory_request(
    tokens: tuple[str, ...],



) -> bool:
    token_set = set(tokens)
    child_terms = _semantic_family_terms(
        tokens,
        semantic_classes={"child_relation"},


    )
    if not child_terms:
        return False
    possessive_child = bool(child_terms & {"child's", "kid's", "son's", "daughter's"})
    owned_child = bool(token_set & {"my", "our"} and child_terms)
    about_child = bool("about" in token_set and child_terms)
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


_URGENT_HEALTH_CACHE: dict[str, Any] | None = None

def _has_urgent_health_frame(tokens: tuple[str, ...]) -> bool:
    global _URGENT_HEALTH_CACHE
    if _URGENT_HEALTH_CACHE is None:
        from melm.contracts import load_contract_json
        payload = load_contract_json("health_disclaimers.v1.json")
        _URGENT_HEALTH_CACHE = {
            "urgent_terms": set(payload.get("urgent_terms", [])),
            "urgent_pairs": tuple(tuple(pair) for pair in payload.get("urgent_pairs", [])),
        }
    token_set = set(tokens)
    return bool(
        token_set & _URGENT_HEALTH_CACHE["urgent_terms"]
        or _has_any_token_sequence(tokens, _URGENT_HEALTH_CACHE["urgent_pairs"])
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())
