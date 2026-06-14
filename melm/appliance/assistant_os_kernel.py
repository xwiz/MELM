"""Self-model and autobiographical-memory layer for the assistant MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import re
from typing import Any, Literal

from .assistant_actions import LocalDeviceActionExecutor
from .assistant_integrity import ResponseIntegrityAssessment, assess_response_integrity
from .assistant_inventory import (
    LocalMediaInventoryAdapter,
    PublicDomainStoryMetadataAdapter,
    media_items_to_inventory_rows,
    story_items_to_inventory_rows,
)
from .assistant_lexicon import cloud_definition_lookup, offline_definition_lookup
from .assistant_os_store import AssistantOSStore
from .assistant_synthesis import BoundedLocalSynthesizer, BoundedSynthesisResult
from .local_assistant_router import (
    AssistantDecision,
    AssistantIntent,
    LocalAssistantProfile,
    OnDeviceAssistantRouter,
    _IN_MEMORY_LEXICON,
    classify_autobiographical_memory_scope,
    compose_autobiographical_memory_frame,
    compose_assistant_status_frame,
    parse_assistant_debug_frame,
    rebuild_entity_lexicon_index,
    replace_in_memory_lexicon,
)


OpportunityKind = Literal[
    "build_story_inventory",
    "refresh_weather_cache",
    "ask_profile_memory",
    "request_trusted_contact",
    "build_media_index",
    "ask_routine_memory",
    "ask_household_memory",
]

SELF_OBSERVATION_HISTORY_LIMIT = 24

# Slot names that each intent checks on its target entity.
_INTENT_SLOT_BINDINGS: dict[str, tuple[str, ...]] = {
    "social_contact": ("name", "phone"),
    "personal_memory": ("self_facts",),
}


def _simple_tokenize(text: str) -> tuple[str, ...]:
    return tuple(text.lower().split())


def _resolve_slot_states(
    intent: str,
    tokens: tuple[str, ...],
    store: AssistantOSStore | None,
) -> dict[str, str]:
    """Resolve slot fill states for *intent* from the entity store."""
    if store is None:
        return {}
    from .assistant_frame_linker import (
        SLOT_STATE_FILLED,
        SLOT_STATE_UNKNOWN,
        SLOT_STATE_UNKNOWN_ENTITY,
    )

    slot_names = _INTENT_SLOT_BINDINGS.get(intent)
    if not slot_names:
        return {}

    if intent == "social_contact":
        persons = store.find_entities(kind="person")
        if not persons:
            return {s: SLOT_STATE_UNKNOWN_ENTITY for s in slot_names}
        token_set = set(tokens)
        matched = [
            p for p in persons
            if p.label.lower() in token_set
            or set(p.label.lower().split()) & token_set
        ]
        if not matched:
            matched = persons
        entity_id = matched[0].entity_id
        states: dict[str, str] = {}
        for sn in slot_names:
            slot = store.get_entity_slot(entity_id, sn)
            if slot is None:
                states[sn] = SLOT_STATE_UNKNOWN
            else:
                states[sn] = slot.slot_state
        return states

    if intent == "personal_memory":
        self_entity = store.get_entity("self")
        if self_entity is None:
            return {s: SLOT_STATE_UNKNOWN_ENTITY for s in slot_names}
        rows = store.connection.execute(
            "SELECT slot_name, slot_state FROM entity_slots WHERE entity_id = 'self' AND consent=1"
        ).fetchall()
        has_facts = any(str(r["slot_state"]) == SLOT_STATE_FILLED for r in rows)
        return {"self_facts": SLOT_STATE_FILLED if has_facts else SLOT_STATE_UNKNOWN}

    return {}


@dataclass(frozen=True)
class SelfModel:
    """Operational self-awareness: purpose, abilities, limits, and inventories."""

    name: str = "MELM Local Assistant OS"
    purpose: str = "Help the user locally first, safely, privately, and cheaply."
    strengths: tuple[str, ...] = (
        "remember user preferences",
        "route between local memory, tools, actions, and cloud",
        "prepare local inventories from public-domain sources",
        "answer safety/common-sense questions locally",
    )
    limits: tuple[str, ...] = (
        "no broad open-domain generation without cloud or local inventory",
        "no medical diagnosis",
        "no action without user consent",
    )
    local_capabilities: tuple[str, ...] = (
        "personal_memory",
        "autobiographical_memory",
        "cached_weather",
        "story_inventory",
        "media_library",
        "trusted_contacts",
        "local_policy",
    )
    cloud_capabilities: tuple[str, ...] = ("large_language_generation", "open_domain_reasoning")
    inventory_counts: dict[str, int] = field(
        default_factory=lambda: {
            "story_models": 0,
            "weather_days": 0,
            "contacts": 0,
            "media_items": 0,
            "routine_facts": 0,
            "household_facts": 0,
        }
    )


@dataclass(frozen=True)
class AssistantMemoryEvent:
    event_id: str
    utterance: str
    intent: AssistantIntent
    route: str
    reason: str
    cloud_needed: bool
    evidence_keys: tuple[str, ...]
    semantic_classes_activated: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MembraneDecision:
    """Boundary decision for what can stay local, use tools, act, or cross cloud."""

    route: str
    allowed: bool
    boundary_crossed: str
    personal_facts_included: tuple[str, ...] = ()
    personal_facts_excluded: tuple[str, ...] = ()
    confirmation_required: bool = False
    reason: str = ""


@dataclass(frozen=True)
class HomeostaticState:
    """Per-turn health snapshot for the user/device relationship."""

    privacy_risk: float
    cloud_dependence: float
    local_capability: float
    uncertainty: float
    cache_freshness: float
    action_risk: float
    user_trust: float
    inventory_coverage: float
    reason: str


@dataclass(frozen=True)
class TypedActionPlan:
    action_type: str
    target: str
    utterance: str
    evidence_keys: tuple[str, ...]
    confirmation_state: Literal["pending", "confirmed", "not_required"] = "pending"


@dataclass(frozen=True)
class Opportunity:
    kind: OpportunityKind
    priority: float
    reason: str
    evidence_event_ids: tuple[str, ...]
    expected_cloud_reduction: int
    proposed_action: str
    source_candidates: tuple[str, ...] = ()
    priority_signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelLearningReport:
    before_route: str
    after_route: str
    cloud_handoffs_before: int
    cloud_handoffs_after: int
    opportunities: tuple[Opportunity, ...]
    executed_jobs: tuple[str, ...]
    remembered_events: int
    story_inventory_count: int


class AssistantOSKernel:
    """Multi-turn local assistant kernel with self-model reflection."""

    def __init__(
        self,
        *,
        profile: LocalAssistantProfile | None = None,
        self_model: SelfModel | None = None,
        store: AssistantOSStore | None = None,
        db_path: str | Path | None = None,
        action_executor: LocalDeviceActionExecutor | None = None,
        capture_surface: str = "",
        capture_source: str = "",
        improvement_opt_in: bool = False,
        offline_dictionary_path: str | Path | None = None,
        cloud_api_key: str | None = None,
    ) -> None:
        if store is not None and db_path is not None:
            raise ValueError("pass either store or db_path, not both")
        self.store = store or (AssistantOSStore(db_path) if db_path is not None else None)
        self.profile = profile or LocalAssistantProfile()
        if self.store is not None:
            self.profile = self.store.load_profile(self.profile)
        self.self_model = self_model or self._self_model_from_profile(self.profile)
        self.action_executor = action_executor or LocalDeviceActionExecutor()
        self.capture_surface = str(capture_surface)
        self.capture_source = str(capture_source)
        self.improvement_opt_in = bool(improvement_opt_in)
        self.offline_dictionary_path = str(offline_dictionary_path) if offline_dictionary_path is not None else None
        self.cloud_api_key = cloud_api_key
        self.events: list[AssistantMemoryEvent] = []
        self.executed_jobs: list[str] = []
        self.last_synthesis: BoundedSynthesisResult | None = None
        self.last_response_integrity: ResponseIntegrityAssessment | None = None
        self._current_self_status: dict[str, Any] = {}
        if self.store is not None:
            self.events = [
                AssistantMemoryEvent(
                    event_id=item.event_id,
                    utterance=item.utterance,
                    intent=item.intent,  # type: ignore[arg-type]
                    route=item.route,
                    reason=item.reason,
                    cloud_needed=item.cloud_needed,
                    evidence_keys=item.evidence_keys,
                    semantic_classes_activated=item.semantic_classes_activated,
                )
                for item in self.store.load_events()
            ]
            self.executed_jobs = self.store.load_executed_jobs()
            self._persist_profile_and_self_model()
            self._rebuild_router_lexicon_cache()

    def handle(self, utterance: str) -> AssistantDecision:
        decision = self.decide(utterance)
        self.remember(decision)
        self._run_acquisition()
        return decision

    def _run_acquisition(self) -> None:
        if self.store is None or self.last_response_integrity is None:
            return
        topics = self.last_response_integrity.research_topics
        if not topics:
            return
        seen = set()
        for token in topics:
            if token in seen:
                continue
            seen.add(token)
            if token in _IN_MEMORY_LEXICON:
                continue
            if self.offline_dictionary_path is not None:
                try:
                    offline_definition_lookup(
                        self.store, token, dictionary_path=self.offline_dictionary_path,
                    )
                except Exception:
                    pass
            if self.cloud_api_key is not None:
                try:
                    cloud_definition_lookup(
                        self.store, token, api_key=self.cloud_api_key,
                    )
                except Exception:
                    pass

    def decide(self, utterance: str) -> AssistantDecision:
        self._current_self_status = {}
        privacy_control = self._privacy_control_decision(utterance)
        if privacy_control is not None:
            synthesis = self._synthesizer().synthesize(
                privacy_control,
                boundary_crossed="none",
                membrane_allowed=True,
            )
            self.last_synthesis = synthesis
            if synthesis.applied:
                return replace(privacy_control, answer=synthesis.answer)
            return privacy_control
        profile_setup = self._local_profile_setup_decision(utterance)
        if profile_setup is not None:
            synthesis = self._synthesizer().synthesize(
                profile_setup,
                boundary_crossed="none",
                membrane_allowed=True,
            )
            self.last_synthesis = synthesis
            if synthesis.applied:
                return replace(profile_setup, answer=synthesis.answer)
            return profile_setup
        action_control = self._action_control_decision(utterance)
        if action_control is not None:
            synthesis = self._synthesizer().synthesize(
                action_control,
                boundary_crossed="none",
                membrane_allowed=True,
            )
            self.last_synthesis = synthesis
            if synthesis.applied:
                return replace(action_control, answer=synthesis.answer)
            return action_control
        assistant_status = self._assistant_status_decision(utterance)
        if assistant_status is not None:
            synthesis = self._synthesizer().synthesize(
                assistant_status,
                boundary_crossed="none",
                membrane_allowed=True,
            )
            self.last_synthesis = synthesis
            if synthesis.applied:
                return replace(assistant_status, answer=synthesis.answer)
            return assistant_status
        autobiographical_recall = self._autobiographical_recall_decision(utterance)
        if autobiographical_recall is not None:
            membrane = MembranePolicy().evaluate(
                autobiographical_recall,
                fact_privacy=self._fact_privacy_index(),
            )
            synthesis = self._synthesizer().synthesize(
                autobiographical_recall,
                boundary_crossed=membrane.boundary_crossed,
                membrane_allowed=membrane.allowed,
            )
            self.last_synthesis = synthesis
            if synthesis.applied:
                return replace(autobiographical_recall, answer=synthesis.answer)
            return autobiographical_recall
        decision = OnDeviceAssistantRouter(
            self.profile,
        ).handle(utterance)
        if self.store is not None:
            resolved = _resolve_slot_states(
                decision.intent,
                _simple_tokenize(utterance),
                self.store,
            )
            if resolved:
                decision = replace(decision, slot_states=resolved)
        membrane = MembranePolicy().evaluate(decision, fact_privacy=self._fact_privacy_index())
        if not membrane.allowed:
            rejected = AssistantDecision(
                utterance=utterance,
                intent=decision.intent,
                route="reject",
                answer="I will keep that private local information on this device.",
                evidence_keys=decision.evidence_keys,
                confidence=0.97,
                reason=membrane.reason,
            )
            self.last_synthesis = self._synthesizer().synthesize(
                rejected,
                boundary_crossed="blocked",
                membrane_allowed=False,
            )
            return rejected
        synthesis = self._synthesizer().synthesize(
            decision,
            boundary_crossed=membrane.boundary_crossed,
            membrane_allowed=True,
        )
        self.last_synthesis = synthesis
        if synthesis.applied:
            return replace(decision, answer=synthesis.answer)
        return decision

    def remember(self, decision: AssistantDecision) -> None:
        self._remember(decision)

    def reflect(self) -> tuple[Opportunity, ...]:
        pressure_context = self._opportunity_pressure_context()
        opportunities = [
            item
            for item in (
                self._story_inventory_opportunity(pressure_context),
                self._weather_cache_opportunity(pressure_context),
                self._profile_memory_opportunity(pressure_context),
                self._trusted_contact_opportunity(pressure_context),
                self._media_index_opportunity(pressure_context),
                self._routine_memory_opportunity(pressure_context),
                self._household_memory_opportunity(pressure_context),
            )
            if item is not None
        ]
        ordered = tuple(sorted(opportunities, key=lambda item: item.priority, reverse=True))
        if self.store is not None:
            for opportunity in ordered:
                self.store.save_opportunity(
                    kind=opportunity.kind,
                    priority=opportunity.priority,
                    reason=opportunity.reason,
                    evidence_event_ids=opportunity.evidence_event_ids,
                    expected_cloud_reduction=opportunity.expected_cloud_reduction,
                    proposed_action=opportunity.proposed_action,
                    source_candidates=opportunity.source_candidates,
                )
            persist_self_observation(self.store, self.self_model)
        return ordered

    def execute(self, opportunity: Opportunity) -> None:
        if opportunity.kind == "build_story_inventory":
            self._install_story_inventory()
        elif opportunity.kind == "refresh_weather_cache":
            self.profile = replace(
                self.profile,
                weekly_weather={
                    "today": "warm with afternoon rain",
                    "tomorrow": "cloudy and humid",
                    "week": "mostly warm with two rainy afternoons",
                },
            )
        elif opportunity.kind == "ask_profile_memory":
            self._record_setup_request(
                "profile_memory",
                prompt="Ask one short profile-memory question and store only the user's answer.",
                reason=opportunity.reason,
            )
        elif opportunity.kind == "request_trusted_contact":
            self._record_setup_request(
                "trusted_contact",
                prompt="Ask the user to name a trusted contact before future call actions.",
                reason=opportunity.reason,
            )
        elif opportunity.kind == "build_media_index":
            imported = self._install_media_inventory()
            if not imported:
                self._record_setup_request(
                    "media_library",
                    prompt="Ask the user to choose a local media folder or media manifest before future playback.",
                    reason=opportunity.reason,
                )
        elif opportunity.kind == "ask_routine_memory":
            self._record_setup_request(
                "routine_memory",
                prompt="Ask the user to describe their routine and store the exact answer locally.",
                reason=opportunity.reason,
            )
        elif opportunity.kind == "ask_household_memory":
            self._record_setup_request(
                "household_memory",
                prompt="Ask who uses this device and define local-only household memory ownership.",
                reason=opportunity.reason,
            )
        self.self_model = self._self_model_from_profile(self.profile)
        self.executed_jobs.append(opportunity.kind)
        if self.store is not None:
            self._persist_profile_and_self_model()
            self.store.mark_opportunity_executed(opportunity.kind)
            persist_self_observation(self.store, self.self_model)

    def _install_media_inventory(self) -> bool:
        adapter = LocalMediaInventoryAdapter()
        try:
            result = adapter.import_manifest(self.profile, limit=24)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            return False
        if not result.items:
            return False
        existing = tuple(self.profile.media_library)
        imported_ids = tuple(row["item_id"] for row in media_items_to_inventory_rows(result.items))
        media = tuple(dict.fromkeys((*existing, *imported_ids)))
        self.profile = replace(self.profile, media_library=media)
        if self.store is not None:
            for row in media_items_to_inventory_rows(result.items):
                self.store.upsert_inventory(
                    str(row["kind"]),
                    str(row["item_id"]),
                    dict(row["payload"]),
                    source=str(row["source"]),
                    license=str(row["license"]),
                    tags=tuple(str(tag) for tag in row["tags"]),
                )
            self.store.connection.commit()
        return True

    def _remember(self, decision: AssistantDecision) -> None:
        event = AssistantMemoryEvent(
            event_id=self.store.next_event_id() if self.store is not None else f"os_e{len(self.events) + 1}",
            utterance=decision.utterance,
            intent=decision.intent,
            route=decision.route,
            reason=decision.reason,
            cloud_needed=decision.cloud_needed,
            evidence_keys=decision.evidence_keys,
            semantic_classes_activated=decision.semantic_classes_activated,
        )
        self.events.append(event)
        if decision.reason == "consent_revoked_user_fact":
            self._apply_consent_revocation(decision.evidence_keys)
        if self.store is not None:
            membrane = MembranePolicy().evaluate(decision, fact_privacy=self._fact_privacy_index())
            homeostasis = homeostatic_state_from_decision(
                decision,
                self_model=self.self_model,
                membrane=membrane,
            )
            self._persist_profile_and_self_model()
            self.store.record_turn(
                event_id=event.event_id,
                utterance=decision.utterance,
                intent=decision.intent,
                route=decision.route,
                reason=decision.reason,
                answer=decision.answer,
                cloud_needed=decision.cloud_needed,
                external_fetch_needed=decision.external_fetch_needed,
                device_action=decision.device_action,
                local_memory_used=decision.local_memory_used,
                evidence_keys=decision.evidence_keys,
                membrane=asdict(membrane),
                homeostasis=asdict(homeostasis),
                capture_surface=self.capture_surface,
                capture_source=self.capture_source,
                semantic_classes_activated=decision.semantic_classes_activated,
            )
            if self.last_synthesis is not None and _synthesis_matches_decision(self.last_synthesis, decision):
                self.store.record_synthesis_trace(
                    event_id=event.event_id,
                    route=self.last_synthesis.route,
                    applied=self.last_synthesis.applied,
                    refused=self.last_synthesis.refused,
                    quality=self.last_synthesis.quality,
                    reason=self.last_synthesis.reason,
                    boundary_crossed=self.last_synthesis.boundary_crossed,
                )
            debug_parse = parse_assistant_debug_frame(decision.utterance, decision)
            integrity = assess_response_integrity(
                decision,
                debug_parse,
                synthesis=self.last_synthesis.to_dict() if self.last_synthesis is not None else {},
                membrane=asdict(membrane),
            )
            self.last_response_integrity = integrity
            integrity_payload = integrity.to_dict()
            self.store.record_response_integrity(
                event_id=event.event_id,
                assessment=integrity_payload,
            )
            session_id = self.store.session_id_for_event(event.event_id)
            self.store.set_session_improvement_consent(
                session_id,
                opted_in=self.improvement_opt_in,
            )
            self.store.queue_improvement_candidate(
                event_id=event.event_id,
                assessment=integrity_payload,
            )
            action_plan = typed_action_plan_from_decision(decision)
            if action_plan is not None and action_plan.confirmation_state == "pending":
                self.store.record_pending_action(
                    action_type=action_plan.action_type,
                    target=action_plan.target,
                    utterance=action_plan.utterance,
                    evidence_keys=action_plan.evidence_keys,
                )
            elif action_plan is not None and action_plan.confirmation_state == "confirmed":
                pending = self.store.latest_pending_action()
                if pending is not None:
                    execution = self.action_executor.execute(pending, store=self.store)
                    self.store.mark_pending_action_executed(
                        str(pending["action_id"]),
                        result=execution.to_dict(),
                        executed=execution.status in {"prepared", "executed"},
                    )
                else:
                    self.store.mark_latest_pending_action_executed(decision.answer)
            elif decision.reason == "cancelled_pending_action":
                self.store.mark_latest_pending_action_cancelled(decision.answer)
            persist_self_observation(self.store, self.self_model)

    def _privacy_control_decision(self, utterance: str) -> AssistantDecision | None:
        fact_key = _revoked_fact_key(utterance)
        if fact_key is None:
            return None
        return AssistantDecision(
            utterance=utterance,
            intent="personal_memory",
            route="local_answer",
            answer="I will stop using that remembered fact.",
            evidence_keys=(fact_key,),
            local_memory_used=True,
            confidence=0.96,
            reason="consent_revoked_user_fact",
        )

    def _local_profile_setup_decision(self, utterance: str) -> AssistantDecision | None:
        setup = _extract_local_profile_setup(utterance)
        if setup is None:
            return None
        kind, key, value = setup
        if kind == "trusted_contact":
            contacts = dict(self.profile.contacts)
            contacts[key] = value
            self.profile = replace(self.profile, contacts=contacts)
            evidence_key = f"contacts.{key}"
            answer = f"I will remember {key} as a trusted contact on this device."
            reason = "consented_trusted_contact_stored"
        elif kind == "profile_fact":
            evidence_key = key
            if key == "profile.age":
                self.profile = replace(self.profile, age=int(value))
                answer = "I will remember your age locally on this device."
            elif key == "profile.location":
                self.profile = replace(self.profile, location=value)
                answer = "I will remember your location locally on this device."
            elif key == "profile.user_name":
                self.profile = replace(self.profile, user_name=value)
                answer = "I will remember your name locally on this device."
            else:
                facts = dict(self.profile.facts)
                memory_key = _memory_key(key)
                facts[memory_key] = value
                self.profile = replace(self.profile, facts=facts)
                evidence_key = f"facts.{memory_key}"
                answer = "I will remember that profile fact locally."
            reason = "profile_update"
        elif kind == "preference":
            preferences = dict(self.profile.preferences)
            preferences[key] = value
            culture = "Yoruba" if re.search(r"\byoruba\b", value, flags=re.IGNORECASE) else self.profile.culture
            self.profile = replace(self.profile, preferences=preferences, culture=culture)
            evidence_key = f"preferences.{key}"
            answer = f"I will remember your {key.replace('_', ' ')} locally on this device."
            reason = "profile_update"
        else:
            facts = dict(self.profile.facts)
            facts[key] = value
            self.profile = replace(self.profile, facts=facts)
            evidence_key = f"facts.{key}"
            answer = f"I will remember your {key.replace('_', ' ')} locally."
            reason = f"consented_{kind}_stored"
        self.self_model = self._self_model_from_profile(self.profile)
        return AssistantDecision(
            utterance=utterance,
            intent="personal_memory" if kind != "trusted_contact" else "social_contact",
            route="local_answer",
            answer=answer,
            evidence_keys=(evidence_key,),
            local_memory_used=True,
            confidence=0.96,
            reason=reason,
        )

    def _apply_consent_revocation(self, evidence_keys: tuple[str, ...]) -> None:
        for key in evidence_keys:
            if key.startswith("facts."):
                facts = dict(self.profile.facts)
                slot_name = key.split(".", 1)[1]
                facts.pop(slot_name, None)
                self.profile = replace(self.profile, facts=facts)
                if self.store is not None:
                    slot = self.store.get_entity_slot("self", slot_name)
                    if slot is not None:
                        self.store.set_entity_slot(
                            "self", slot_name, json.loads(slot.value_json),
                            consent=0, provenance="revoked",
                            local_only=slot.local_only,
                            cloud_eligible=slot.cloud_eligible,
                            scope=slot.scope,
                            source=slot.source,
                            confidence=slot.confidence,
                        )
            elif key.startswith("preferences."):
                preferences = dict(self.profile.preferences)
                preferences.pop(key.split(".", 1)[1], None)
                self.profile = replace(self.profile, preferences=preferences)
            if self.store is not None:
                self.store.set_user_fact_consent(key, consent=False)

    def _action_control_decision(self, utterance: str) -> AssistantDecision | None:
        if self.store is None:
            return None
        pending = self.store.latest_pending_action()
        if _is_action_cancellation(utterance):
            if pending is None:
                return AssistantDecision(
                    utterance=utterance,
                    intent="social_contact",
                    route="clarify",
                    answer="I do not have a pending action to cancel.",
                    confidence=0.93,
                    reason="no_pending_action_to_cancel",
                )
            return AssistantDecision(
                utterance=utterance,
                intent=_intent_from_pending_action(pending["action_type"]),
                route="local_answer",
                answer=f"Cancelled: {pending['target']}",
                evidence_keys=tuple(pending["evidence_keys"]),
                local_memory_used=True,
                confidence=0.96,
                reason="cancelled_pending_action",
            )
        if _is_confirmation(utterance):
            if pending is None:
                return AssistantDecision(
                    utterance=utterance,
                    intent="social_contact",
                    route="clarify",
                    answer="I do not have a pending action to confirm.",
                    confidence=0.93,
                    reason="no_pending_action_to_confirm",
                )
            requested_target = _requested_confirmation_target(utterance)
            if requested_target and not _target_matches_pending(requested_target, pending["target"]):
                return AssistantDecision(
                    utterance=utterance,
                    intent=_intent_from_pending_action(pending["action_type"]),
                    route="clarify",
                    answer=(
                        f"The pending action is {pending['target']}. "
                        f"I will not switch it to {requested_target} without a new request."
                    ),
                    evidence_keys=tuple(pending["evidence_keys"]),
                    local_memory_used=True,
                    confidence=0.94,
                    reason="confirmation_target_mismatch",
                )
            return AssistantDecision(
                utterance=utterance,
                intent=_intent_from_pending_action(pending["action_type"]),
                route="device_action",
                answer=f"Confirmed: {pending['target']}",
                evidence_keys=tuple(pending["evidence_keys"]),
                local_memory_used=True,
                device_action=True,
                confidence=0.96,
                reason="confirmed_device_action",
            )
        return None

    def _assistant_status_decision(self, utterance: str) -> AssistantDecision | None:
        if not _is_assistant_status_request(utterance):
            return None
        if self.store is None:
            self._current_self_status = {"available": False, "reason": "no_persistent_store"}
            return AssistantDecision(
                utterance=utterance,
                intent="assistant_status",
                route="local_answer",
                answer="I can report my status when a local event ledger is attached.",
                evidence_keys=("self_model.name", "self_model.local_capabilities", "self_status.no_store"),
                local_memory_used=True,
                confidence=0.76,
                reason="self_status_no_ledger",
            )
        status = _assistant_runtime_status(self.store, self.self_model)
        self._current_self_status = status
        wants_next = _is_assistant_next_step_request(utterance)
        return AssistantDecision(
            utterance=utterance,
            intent="assistant_status",
            route="local_answer",
            answer="I can summarize my local runtime status from the assistant ledger.",
            evidence_keys=(
                "self_model.name",
                "self_model.local_capabilities",
                "self_status.counts",
                "self_status.routes",
                "self_status.safety_flags",
                "self_status.self_observation",
                "self_status.next_steps",
            ),
            local_memory_used=True,
            confidence=0.92,
            reason="self_status_next_steps" if wants_next else "self_status_ledger_summary",
        )

    def _autobiographical_recall_decision(self, utterance: str) -> AssistantDecision | None:
        if self.store is None:
            return None
        if not _is_autobiographical_recall_request(utterance):
            return None
        if _is_autobiographical_long_horizon_request(utterance):
            digest = self.store.build_memory_digest(
                session_limit=_autobiographical_digest_session_limit(utterance),
                events_per_session=_autobiographical_digest_events_per_session(utterance),
            )
            if int(digest.get("event_count", 0)) <= 0:
                return AssistantDecision(
                    utterance=utterance,
                    intent="autobiographical_memory",
                    route="clarify",
                    answer="I do not have enough earlier local memory to summarize yet.",
                    confidence=0.74,
                    reason="autobiographical_memory_empty",
                )
            return AssistantDecision(
                utterance=utterance,
                intent="autobiographical_memory",
                route="local_answer",
                answer="I can summarize that from a local long-horizon memory digest.",
                evidence_keys=(f"memory_digest.{digest['digest_id']}",),
                local_memory_used=True,
                confidence=0.88,
                reason="autobiographical_memory_digest",
            )
        if _is_autobiographical_session_summary_request(utterance):
            replay = self.store.query_recent_session_memory(
                session_limit=_autobiographical_session_limit(utterance),
                events_per_session=_autobiographical_events_per_session(utterance),
            )
            reason = "autobiographical_session_summary"
        else:
            query = _autobiographical_recall_query(utterance)
            replay = self.store.query_event_memory(
                query=query,
                limit=_autobiographical_recall_limit(utterance),
            )
            reason = "autobiographical_memory_summary"
        event_ids = tuple(str(event["event_id"]) for event in replay["events"])
        if not event_ids:
            return AssistantDecision(
                utterance=utterance,
                intent="autobiographical_memory",
                route="clarify",
                answer="I do not have earlier conversation memory to replay yet.",
                confidence=0.74,
                reason="autobiographical_memory_empty",
            )
        return AssistantDecision(
            utterance=utterance,
            intent="autobiographical_memory",
            route="local_answer",
            answer="I can summarize that from local conversation memory.",
            evidence_keys=tuple(f"events.{event_id}" for event_id in event_ids),
            local_memory_used=True,
            confidence=0.88,
            reason=reason,
        )

    def _opportunity_pressure_context(self) -> dict[str, Any]:
        event_count = max(1, len(self.events))
        context = {
            "avg_cloud_dependence": round(sum(1 for event in self.events if event.cloud_needed) / event_count, 3),
            "avg_uncertainty": 0.0,
            "avg_local_capability": round(
                sum(1 for event in self.events if event.route in {"local_answer", "cached_tool", "device_action"})
                / event_count,
                3,
            ),
            "avg_action_risk": 0.0,
            "avg_cache_freshness": 1.0 if self.profile.weekly_weather else 0.0,
            "cloud_dependence_delta": 0.0,
            "cache_freshness_delta": 0.0,
            "self_observation_points": 0,
            "local_resolution_delta": 0.0,
            "weather_cache_gap_persistence": 0.0,
            "story_inventory_gap_persistence": 0.0,
            "failed_jobs": {},
        }
        if self.store is None:
            return context
        context.update(_self_observation_pressure_context(self.store))
        rows = self.store.connection.execute(
            """
            SELECT cloud_dependence, uncertainty, local_capability,
                   action_risk, cache_freshness
            FROM homeostatic_snapshots
            ORDER BY created_at DESC
            LIMIT 12
            """
        ).fetchall()
        if rows:
            newest = rows[0]
            oldest = rows[-1]
            context.update(
                {
                    "avg_cloud_dependence": _avg_float(row["cloud_dependence"] for row in rows),
                    "avg_uncertainty": _avg_float(row["uncertainty"] for row in rows),
                    "avg_local_capability": _avg_float(row["local_capability"] for row in rows),
                    "avg_action_risk": _avg_float(row["action_risk"] for row in rows),
                    "avg_cache_freshness": _avg_float(row["cache_freshness"] for row in rows),
                    "cloud_dependence_delta": round(
                        float(newest["cloud_dependence"]) - float(oldest["cloud_dependence"]),
                        3,
                    ),
                    "cache_freshness_delta": round(
                        float(newest["cache_freshness"]) - float(oldest["cache_freshness"]),
                        3,
                    ),
                }
            )
        failed_rows = self.store.connection.execute(
            """
            SELECT kind, COUNT(*) AS count
            FROM jobs
            WHERE status='failed'
            GROUP BY kind
            """
        ).fetchall()
        context["failed_jobs"] = {str(row["kind"]): int(row["count"]) for row in failed_rows}
        return context

    def _story_inventory_opportunity(self, pressure_context: dict[str, Any]) -> Opportunity | None:
        story_events = [event for event in self.events if event.intent == "story"]
        cloud_story_events = [event for event in story_events if event.cloud_needed]
        if len(story_events) < 3 or not cloud_story_events:
            return None
        if self.profile.story_models:
            return None
        signals = {
            "story_requests": len(story_events),
            "story_cloud_handoffs": len(cloud_story_events),
            "avg_cloud_dependence": pressure_context["avg_cloud_dependence"],
            "avg_uncertainty": pressure_context["avg_uncertainty"],
            "self_observation_points": pressure_context["self_observation_points"],
            "local_resolution_delta": pressure_context["local_resolution_delta"],
            "story_inventory_gap_persistence": pressure_context["story_inventory_gap_persistence"],
            "recent_failed_jobs": int(pressure_context["failed_jobs"].get("build_story_inventory", 0)),
        }
        return Opportunity(
            kind="build_story_inventory",
            priority=_bounded_opportunity_priority(
                0.8
                + min(0.12, len(cloud_story_events) * 0.04)
                + float(signals["avg_cloud_dependence"]) * 0.03
                + float(signals["avg_uncertainty"]) * 0.02
                + max(0.0, -float(signals["local_resolution_delta"])) * 0.04
                + float(signals["story_inventory_gap_persistence"]) * 0.03
                - int(signals["recent_failed_jobs"]) * 0.05
            ),
            reason=(
                f"user age {self.profile.age} asked for stories {len(story_events)} "
                "times and local story inventory is empty"
            ),
            evidence_event_ids=tuple(event.event_id for event in cloud_story_events),
            expected_cloud_reduction=len(cloud_story_events),
            proposed_action=(
                "create age/topic/culture-tagged public-domain story inventory "
                "for local bedtime/adventure requests"
            ),
            source_candidates=(
                "project_gutenberg_catalog_metadata",
                "internet_archive_item_search_and_metadata",
            ),
            priority_signals=signals,
        )

    def _weather_cache_opportunity(self, pressure_context: dict[str, Any]) -> Opportunity | None:
        if self.profile.weekly_weather:
            return None
        weather_misses = [
            event
            for event in self.events
            if event.intent == "weather" and event.reason == "weather_cache_miss"
        ]
        if not weather_misses:
            return None
        signals = {
            "weather_cache_misses": len(weather_misses),
            "avg_cache_freshness": pressure_context["avg_cache_freshness"],
            "cache_freshness_delta": pressure_context["cache_freshness_delta"],
            "avg_uncertainty": pressure_context["avg_uncertainty"],
            "self_observation_points": pressure_context["self_observation_points"],
            "local_resolution_delta": pressure_context["local_resolution_delta"],
            "weather_cache_gap_persistence": pressure_context["weather_cache_gap_persistence"],
            "recent_failed_jobs": int(pressure_context["failed_jobs"].get("refresh_weather_cache", 0)),
        }
        return Opportunity(
            kind="refresh_weather_cache",
            priority=_bounded_opportunity_priority(
                0.68
                + min(0.08, len(weather_misses) * 0.04)
                + max(0.0, -float(signals["cache_freshness_delta"])) * 0.05
                + float(signals["avg_uncertainty"]) * 0.03
                + float(signals["weather_cache_gap_persistence"]) * 0.04
                - int(signals["recent_failed_jobs"]) * 0.04
            ),
            reason="weather request arrived while local forecast cache was empty",
            evidence_event_ids=tuple(event.event_id for event in weather_misses),
            expected_cloud_reduction=0,
            proposed_action="fetch and cache seven-day weather from a weather tool",
            priority_signals=signals,
        )

    def _profile_memory_opportunity(self, pressure_context: dict[str, Any]) -> Opportunity | None:
        memory_misses = [
            event
            for event in self.events
            if event.intent == "personal_memory" and event.reason == "personal_memory_empty"
            and not _is_routine_memory_gap_event(event)
            and not _is_household_memory_gap_event(event)
        ]
        if not memory_misses:
            return None
        if self.profile.facts:
            return None
        signals = {
            "profile_memory_misses": len(memory_misses),
            "avg_uncertainty": pressure_context["avg_uncertainty"],
            "avg_local_capability": pressure_context["avg_local_capability"],
            "recent_failed_jobs": int(pressure_context["failed_jobs"].get("ask_profile_memory", 0)),
        }
        return Opportunity(
            kind="ask_profile_memory",
            priority=_bounded_opportunity_priority(
                0.58
                + min(0.12, len(memory_misses) * 0.06)
                + float(signals["avg_uncertainty"]) * 0.08
                + max(0.0, 1.0 - float(signals["avg_local_capability"])) * 0.04
                - int(signals["recent_failed_jobs"]) * 0.04
            ),
            reason="user asked about themselves but profile memory was empty",
            evidence_event_ids=tuple(event.event_id for event in memory_misses),
            expected_cloud_reduction=0,
            proposed_action="ask one short preference question and store consented answer",
            priority_signals=signals,
        )

    def _trusted_contact_opportunity(self, pressure_context: dict[str, Any]) -> Opportunity | None:
        if self.profile.contacts:
            return None
        contact_misses = [
            event
            for event in self.events
            if event.intent == "social_contact" and event.reason == "missing_contact"
        ]
        if not contact_misses:
            return None
        signals = {
            "trusted_contact_misses": len(contact_misses),
            "avg_uncertainty": pressure_context["avg_uncertainty"],
            "avg_local_capability": pressure_context["avg_local_capability"],
            "avg_action_risk": pressure_context["avg_action_risk"],
            "recent_failed_jobs": int(pressure_context["failed_jobs"].get("request_trusted_contact", 0)),
        }
        return Opportunity(
            kind="request_trusted_contact",
            priority=_bounded_opportunity_priority(
                0.62
                + min(0.14, len(contact_misses) * 0.07)
                + float(signals["avg_uncertainty"]) * 0.04
                + max(0.0, 1.0 - float(signals["avg_local_capability"])) * 0.04
                + float(signals["avg_action_risk"]) * 0.02
                - int(signals["recent_failed_jobs"]) * 0.04
            ),
            reason="user wanted to talk to someone but no trusted contact was available",
            evidence_event_ids=tuple(event.event_id for event in contact_misses),
            expected_cloud_reduction=0,
            proposed_action="ask user to choose a trusted contact for future call actions",
            priority_signals=signals,
        )

    def _media_index_opportunity(self, pressure_context: dict[str, Any]) -> Opportunity | None:
        media_misses = [
            event
            for event in self.events
            if event.intent == "media_playback" and event.reason in {"empty_media_library", "missing_media_choice"}
        ]
        if not media_misses:
            return None
        if self.profile.media_library:
            return None
        signals = {
            "media_misses": len(media_misses),
            "avg_uncertainty": pressure_context["avg_uncertainty"],
            "avg_local_capability": pressure_context["avg_local_capability"],
            "avg_action_risk": pressure_context["avg_action_risk"],
            "recent_failed_jobs": int(pressure_context["failed_jobs"].get("build_media_index", 0)),
        }
        return Opportunity(
            kind="build_media_index",
            priority=_bounded_opportunity_priority(
                0.6
                + min(0.15, len(media_misses) * 0.075)
                + float(signals["avg_uncertainty"]) * 0.04
                + max(0.0, 1.0 - float(signals["avg_local_capability"])) * 0.04
                + float(signals["avg_action_risk"]) * 0.02
                - int(signals["recent_failed_jobs"]) * 0.04
            ),
            reason="user asked for media but local media index is empty or underspecified",
            evidence_event_ids=tuple(event.event_id for event in media_misses),
            expected_cloud_reduction=0,
            proposed_action="scan local media titles or ask user to connect a local media app",
            source_candidates=("local_media_index", "user_selected_music_app"),
            priority_signals=signals,
        )

    def _routine_memory_opportunity(self, pressure_context: dict[str, Any]) -> Opportunity | None:
        routine_misses = [
            event
            for event in self.events
            if event.intent == "personal_memory"
            and event.reason == "personal_memory_empty"
            and _is_routine_memory_gap_event(event)
        ]
        if not routine_misses:
            return None
        if any(key for key in self.profile.facts if "routine" in key or "schedule" in key):
            return None
        signals = {
            "routine_memory_misses": len(routine_misses),
            "avg_uncertainty": pressure_context["avg_uncertainty"],
            "avg_local_capability": pressure_context["avg_local_capability"],
            "recent_failed_jobs": int(pressure_context["failed_jobs"].get("ask_routine_memory", 0)),
        }
        return Opportunity(
            kind="ask_routine_memory",
            priority=_bounded_opportunity_priority(
                0.57
                + min(0.14, len(routine_misses) * 0.07)
                + float(signals["avg_uncertainty"]) * 0.06
                + max(0.0, 1.0 - float(signals["avg_local_capability"])) * 0.04
                - int(signals["recent_failed_jobs"]) * 0.04
            ),
            reason="user asked about routine or schedule memory, but no routine fact exists",
            evidence_event_ids=tuple(event.event_id for event in routine_misses),
            expected_cloud_reduction=0,
            proposed_action="ask for the user's routine with consent and store it locally",
            source_candidates=("user_profile_routine_memory",),
            priority_signals=signals,
        )

    def _household_memory_opportunity(self, pressure_context: dict[str, Any]) -> Opportunity | None:
        household_misses = [
            event
            for event in self.events
            if event.intent == "personal_memory"
            and event.reason == "personal_memory_empty"
            and _is_household_memory_gap_event(event)
        ]
        if not household_misses:
            return None
        if any(key for key in self.profile.facts if "household" in key or "family" in key):
            return None
        signals = {
            "household_memory_misses": len(household_misses),
            "avg_uncertainty": pressure_context["avg_uncertainty"],
            "avg_local_capability": pressure_context["avg_local_capability"],
            "recent_failed_jobs": int(pressure_context["failed_jobs"].get("ask_household_memory", 0)),
        }
        return Opportunity(
            kind="ask_household_memory",
            priority=_bounded_opportunity_priority(
                0.56
                + min(0.14, len(household_misses) * 0.07)
                + float(signals["avg_uncertainty"]) * 0.06
                + max(0.0, 1.0 - float(signals["avg_local_capability"])) * 0.05
                - int(signals["recent_failed_jobs"]) * 0.04
            ),
            reason="user asked about household memory, but household ownership is not configured",
            evidence_event_ids=tuple(event.event_id for event in household_misses),
            expected_cloud_reduction=0,
            proposed_action="ask who uses this device and define local-only household memory ownership",
            source_candidates=("user_profile_household_memory", "local_consent_policy"),
            priority_signals=signals,
        )

    def _install_story_inventory(self) -> None:
        result = PublicDomainStoryMetadataAdapter().build_story_inventory(self.profile, limit=3)
        self.profile = replace(self.profile, story_models=result.story_models)
        if self.store is not None:
            for row in story_items_to_inventory_rows(result.selected_items, profile=self.profile):
                self.store.upsert_inventory(
                    str(row["kind"]),
                    str(row["item_id"]),
                    dict(row["payload"]),
                    source=str(row["source"]),
                    license=str(row["license"]),
                    tags=tuple(str(tag) for tag in row["tags"]),
                )

    @staticmethod
    def _self_model_from_profile(profile: LocalAssistantProfile) -> SelfModel:
        return self_model_from_profile(profile)

    def _rebuild_router_lexicon_cache(self) -> None:
        rows = self.store.connection.execute(
            """
            SELECT l.normalized_lemma, s.semantic_class_id
            FROM lexemes AS l
            JOIN lexical_senses AS s ON s.lexeme_id = l.lexeme_id
            WHERE s.status = 'active'
            """
        ).fetchall()
        if rows:
            from collections import defaultdict
            lexicon: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                lemma = str(row["normalized_lemma"])
                class_id = str(row["semantic_class_id"])
                lexicon[lemma].add(class_id)
            replace_in_memory_lexicon(
                {k: frozenset(v) for k, v in lexicon.items()}
            )
            rebuild_entity_lexicon_index(self.store)
            return
        from .assistant_lexicon_legacy import build_legacy_in_memory_lexicon
        replace_in_memory_lexicon(build_legacy_in_memory_lexicon())
        rebuild_entity_lexicon_index(self.store)

    def _persist_profile_and_self_model(self) -> None:
        if self.store is None:
            return
        self.store.save_profile(self.profile)
        self.store.save_self_model(_self_model_payload(self.self_model))

    def _synthesizer(self) -> BoundedLocalSynthesizer:
        return BoundedLocalSynthesizer(
            self.profile,
            store=self.store,
            self_state=_self_model_payload(self.self_model),
            runtime_status=self._current_self_status,
        )

    def _fact_privacy_index(self) -> dict[str, dict[str, Any]]:
        if self.store is None:
            return {}
        return self.store.load_user_fact_privacy_index()

    def _record_setup_request(self, item_id: str, *, prompt: str, reason: str) -> None:
        if self.store is None:
            return
        self.store.upsert_inventory(
            "setup_request",
            item_id,
            {
                "prompt": prompt,
                "reason": reason,
                "requires_user_supplied_value": True,
            },
            source="opportunity_planner",
            license="private_local",
            tags=("setup", item_id, "local_only"),
        )


class MembranePolicy:
    """Fail-closed local boundary policy for v0.1 routing decisions."""

    def evaluate(
        self,
        decision: AssistantDecision,
        *,
        fact_privacy: dict[str, dict[str, Any]] | None = None,
    ) -> MembraneDecision:
        shareable, local_only = _classified_evidence_keys(
            decision.evidence_keys,
            fact_privacy=fact_privacy or {},
        )
        if decision.route == "cloud_handoff":
            if decision.privacy_exposure and local_only:
                return MembraneDecision(
                    route=decision.route,
                    allowed=False,
                    boundary_crossed="blocked_cloud",
                    personal_facts_included=(),
                    personal_facts_excluded=local_only,
                    reason="blocked_private_facts_to_cloud",
                )
            return MembraneDecision(
                route=decision.route,
                allowed=True,
                boundary_crossed="cloud",
                personal_facts_included=shareable,
                personal_facts_excluded=local_only,
                confirmation_required=False,
                reason=(
                    "cloud_allowed_with_user_eligible_facts"
                    if shareable
                    else "cloud_allowed_with_local_only_facts_excluded"
                ),
            )
        if decision.route == "external_fetch":
            return MembraneDecision(
                route=decision.route,
                allowed=True,
                boundary_crossed="tool_fetch",
                personal_facts_included=(),
                personal_facts_excluded=local_only,
                reason="tool_fetch_allowed_with_coarse_context",
            )
        if decision.device_action:
            confirmed = decision.reason == "confirmed_device_action"
            return MembraneDecision(
                route=decision.route,
                allowed=True,
                boundary_crossed="device_action",
                personal_facts_included=local_only,
                personal_facts_excluded=(),
                confirmation_required=not confirmed,
                reason="device_action_requires_confirmation" if not confirmed else "confirmed_device_action_allowed",
            )
        if decision.route == "clarify":
            return MembraneDecision(
                route=decision.route,
                allowed=True,
                boundary_crossed="none",
                personal_facts_excluded=local_only,
                reason="clarify_without_external_boundary",
            )
        if decision.route == "reject":
            return MembraneDecision(
                route=decision.route,
                allowed=False,
                boundary_crossed="blocked",
                personal_facts_excluded=local_only,
                reason=decision.reason or "blocked_by_membrane",
            )
        return MembraneDecision(
            route=decision.route,
            allowed=True,
            boundary_crossed="none",
            personal_facts_included=local_only,
            reason="local_route_allowed",
        )


def _classified_evidence_keys(
    evidence_keys: tuple[str, ...],
    *,
    fact_privacy: dict[str, dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    shareable: list[str] = []
    excluded: list[str] = []
    for key in evidence_keys:
        if not _is_personal_evidence_key(key):
            continue
        if _is_cloud_eligible_fact(key, fact_privacy):
            shareable.append(key)
        else:
            excluded.append(key)
    return tuple(dict.fromkeys(shareable)), tuple(dict.fromkeys(excluded))


def _is_personal_evidence_key(key: str) -> bool:
    return key.startswith(
        (
            "profile.",
            "facts.",
            "preferences.",
            "contacts.",
            "health_goals",
            "events.",
            "self_model.",
            "self_status.",
        )
    ) or key in {"health_goals", "local_health_safety_policy"}


def _is_cloud_eligible_fact(key: str, fact_privacy: dict[str, dict[str, Any]]) -> bool:
    policy = fact_privacy.get(key)
    if not policy:
        return False
    return (
        bool(policy.get("consent", False))
        and not bool(policy.get("local_only", True))
        and bool(policy.get("cloud_eligible", False))
    )


def homeostatic_state_from_decision(
    decision: AssistantDecision,
    *,
    self_model: SelfModel,
    membrane: MembraneDecision,
) -> HomeostaticState:
    inventory = self_model.inventory_counts
    inventory_coverage = round(
        sum(1 for value in inventory.values() if value > 0) / max(1, len(inventory)),
        3,
    )
    local_capability = 1.0 if decision.route in {"local_answer", "cached_tool", "device_action"} else inventory_coverage
    cache_freshness = 1.0 if inventory.get("weather_days", 0) else 0.0
    if decision.reason in {"weather_cache_miss", "clothing_needs_weather_cache", "tool_unavailable"}:
        cache_freshness = 0.0
    action_risk = 0.0
    if decision.device_action:
        action_risk = 0.72 if membrane.confirmation_required else 0.18
    privacy_risk = 0.0
    if membrane.boundary_crossed == "cloud":
        privacy_risk = 0.18 if not decision.privacy_exposure else 0.42
    elif membrane.boundary_crossed == "tool_fetch":
        privacy_risk = 0.08
    uncertainty = round(max(0.0, 1.0 - decision.confidence), 3)
    return HomeostaticState(
        privacy_risk=round(privacy_risk, 3),
        cloud_dependence=1.0 if decision.cloud_needed else (0.35 if decision.external_fetch_needed else 0.0),
        local_capability=round(local_capability, 3),
        uncertainty=uncertainty,
        cache_freshness=cache_freshness,
        action_risk=action_risk,
        user_trust=0.9 if membrane.allowed and uncertainty < 0.35 else 0.55,
        inventory_coverage=inventory_coverage,
        reason=f"{decision.route}:{decision.reason}",
    )


def typed_action_plan_from_decision(decision: AssistantDecision) -> TypedActionPlan | None:
    if not decision.device_action:
        return None
    if decision.reason == "confirmed_device_action":
        return TypedActionPlan(
            action_type="confirmed_device_action",
            target=decision.answer,
            utterance=decision.utterance,
            evidence_keys=decision.evidence_keys,
            confirmation_state="confirmed",
        )
    action_type = "device_action"
    if decision.intent == "social_contact":
        action_type = "call_contact"
    elif decision.intent == "media_playback":
        action_type = "play_media"
    return TypedActionPlan(
        action_type=action_type,
        target=decision.answer,
        utterance=decision.utterance,
        evidence_keys=decision.evidence_keys,
        confirmation_state="pending",
    )


def self_model_from_profile(profile: LocalAssistantProfile) -> SelfModel:
    return SelfModel(
        inventory_counts={
            "story_models": len(profile.story_models),
            "weather_days": len(profile.weekly_weather),
            "contacts": len(profile.contacts),
            "media_items": len(profile.media_library),
            "routine_facts": sum(1 for key in profile.facts if "routine" in key or "schedule" in key),
            "household_facts": sum(
                1 for key in profile.facts if any(marker in key for marker in ("household", "family", "child"))
            ),
        }
    )


def _self_model_payload(self_model: SelfModel) -> dict[str, Any]:
    return {
        "name": self_model.name,
        "purpose": self_model.purpose,
        "strengths": self_model.strengths,
        "limits": self_model.limits,
        "local_capabilities": self_model.local_capabilities,
        "cloud_capabilities": self_model.cloud_capabilities,
        "inventory_counts": self_model.inventory_counts,
    }


def _assistant_runtime_status(store: AssistantOSStore, self_model: SelfModel) -> dict[str, Any]:
    from .assistant_dashboard import build_assistant_os_dashboard

    dashboard = build_assistant_os_dashboard(store).to_dict()
    self_state = store.load_self_state()
    self_observation = self_state.get("runtime_health_trends")
    if not isinstance(self_observation, dict):
        self_observation = persist_self_observation(store, self_model)
    inventories = dashboard["inventories"]["by_kind"]
    safety_flags = dashboard["safety_flags"]
    clean_safety = all(
        int(safety_flags.get(flag, 0)) == 0
        for flag in (
            "missing_membrane_or_homeostasis",
            "dangling_memory_links",
            "low_quality_applied_synthesis",
            "cloud_private_inclusions",
            "unconfirmed_executed_actions",
            "action_without_confirmation_gate",
            "fake_latest_news_local_answers",
        )
    ) and bool(safety_flags.get("ledger_complete"))
    next_steps = _assistant_status_next_steps(self_model, dashboard)
    return {
        "available": True,
        "counts": dashboard["counts"],
        "routes": dashboard["route_counts"],
        "route_counts": dashboard["route_counts"],
        "intent_counts": dashboard["intent_counts"],
        "sessions": dashboard["memory"]["sessions"],
        "last_session_id": dashboard["memory"]["last_session_id"],
        "pending_actions": dashboard["pending_actions"],
        "inventories": inventories,
        "jobs": dashboard["jobs"]["by_status"],
        "safety_clean": clean_safety,
        "safety_flags": {
            key: value
            for key, value in safety_flags.items()
            if key == "ledger_complete" or int(value) != 0
        },
        "self_observation": self_observation,
        "next_steps": next_steps,
        "local_capabilities": self_model.local_capabilities,
    }


def _assistant_status_next_steps(self_model: SelfModel, dashboard: dict[str, Any]) -> tuple[str, ...]:
    inventory = self_model.inventory_counts
    queued_jobs = dashboard["jobs"]["by_status"].get("queued", 0)
    pending_actions = dashboard["pending_actions"].get("pending", 0)
    steps: list[str] = []
    if queued_jobs:
        steps.append(f"run {queued_jobs} queued local job(s)")
    if pending_actions:
        steps.append(f"resolve {pending_actions} pending action confirmation(s)")
    if inventory.get("story_models", 0) < 3:
        steps.append("grow the local story inventory")
    if inventory.get("weather_days", 0) <= 0:
        steps.append("refresh the local weather cache")
    if inventory.get("media_items", 0) <= 0:
        steps.append("build a local media index")
    if inventory.get("contacts", 0) <= 0:
        steps.append("ask for one trusted contact")
    if inventory.get("routine_facts", 0) <= 0:
        steps.append("ask for routine memory")
    if inventory.get("household_facts", 0) <= 0:
        steps.append("ask for household memory ownership")
    if not steps:
        steps.append("continue using the local ledger and refresh inventories when gaps appear")
    return tuple(steps[:4])


def persist_self_observation(store: AssistantOSStore, self_model: SelfModel) -> dict[str, Any]:
    """Persist compact self-observation trends derived from the local ledgers."""

    from .assistant_dashboard import build_assistant_os_dashboard

    dashboard = build_assistant_os_dashboard(store).to_dict()
    observation = _self_observation_payload(dashboard, self_model)
    previous = store.load_self_state()
    history = _append_self_observation_history(
        _self_observation_history(previous.get("runtime_health_history")),
        observation,
    )
    observation["history_summary"] = _self_observation_history_summary(history)
    observation["summary"] = _self_observation_summary(observation)
    store.save_self_model(
        {
            "runtime_health_trends": observation,
            "runtime_health_history": history,
        }
    )
    return observation


def _self_observation_payload(dashboard: dict[str, Any], self_model: SelfModel) -> dict[str, Any]:
    counts = dict(dashboard.get("counts", {}))
    route_counts = dict(dashboard.get("route_counts", {}))
    inventories = dict(dashboard.get("inventories", {}))
    inventory_counts = dict(self_model.inventory_counts)
    jobs = dict(dashboard.get("jobs", {}))
    importer_health = dict(jobs.get("importer_health", {}))
    importer_trends = dict(jobs.get("importer_trends", {}))
    story_quality = dict(inventories.get("story_quality", {}))
    synthesis = dict(dashboard.get("synthesis", {}))
    safety_flags = dict(dashboard.get("safety_flags", {}))
    pending_actions = dict(dashboard.get("pending_actions", {}))
    memory = dict(dashboard.get("memory", {}))
    events = int(counts.get("events", 0) or 0)
    local_resolutions = sum(
        int(route_counts.get(route, 0) or 0)
        for route in ("local_answer", "cached_tool", "device_action")
    )
    safety_clean = bool(safety_flags.get("ledger_complete")) and all(
        int(safety_flags.get(flag, 0) or 0) == 0
        for flag in (
            "missing_membrane_or_homeostasis",
            "dangling_memory_links",
            "low_quality_applied_synthesis",
            "cloud_private_inclusions",
            "unconfirmed_executed_actions",
            "action_without_confirmation_gate",
            "fake_latest_news_local_answers",
        )
    )
    weather_days = int(inventory_counts.get("weather_days", 0) or 0)
    story_models = int(inventory_counts.get("story_models", 0) or 0)
    observation = {
        "schema": "melm.assistant_self_observation.v1",
        "events_observed": events,
        "sessions_observed": int(memory.get("sessions", 0) or 0),
        "routing": {
            "local_resolution_rate": round(local_resolutions / max(1, events), 3),
            "cloud_handoffs": int(route_counts.get("cloud_handoff", 0) or 0),
            "external_fetches": int(route_counts.get("external_fetch", 0) or 0),
            "clarifications": int(route_counts.get("clarify", 0) or 0),
            "rejections": int(route_counts.get("reject", 0) or 0),
        },
        "cache_health": {
            "weather_cache_ready": weather_days > 0,
            "weather_days": weather_days,
            "story_inventory_ready": story_models > 0,
            "story_models": story_models,
            "media_items": int(inventory_counts.get("media_items", 0) or 0),
            "contacts": int(inventory_counts.get("contacts", 0) or 0),
        },
        "job_health": {
            "queued": int(dict(jobs.get("by_status", {})).get("queued", 0) or 0),
            "completed": int(dict(jobs.get("by_status", {})).get("completed", 0) or 0),
            "failed": int(dict(jobs.get("by_status", {})).get("failed", 0) or 0),
            "retryable_queued": int(dict(jobs.get("priority", {})).get("retryable_queued", 0) or 0),
        },
        "importer_trends": {
            "completed_cycles": int(importer_trends.get("completed_cycles", 0) or 0),
            "failed_cycles": int(importer_trends.get("failed_cycles", 0) or 0),
            "imported_items_total": int(importer_trends.get("imported_items_total", 0) or 0),
            "selected_items_total": int(importer_trends.get("selected_items_total", 0) or 0),
            "avg_metadata_quality": float(importer_trends.get("avg_metadata_quality", 0.0) or 0.0),
            "quality_delta": float(importer_trends.get("quality_delta", 0.0) or 0.0),
            "network_used_cycles": int(importer_trends.get("network_used_cycles", 0) or 0),
            "byte_budget_exhausted_cycles": int(importer_trends.get("byte_budget_exhausted_cycles", 0) or 0),
        },
        "importer_health": {
            "completed_import_jobs": int(importer_health.get("completed_import_jobs", 0) or 0),
            "failed_import_jobs": int(importer_health.get("failed_import_jobs", 0) or 0),
            "selected_items": int(importer_health.get("selected_items", 0) or 0),
            "quality_rejected_items": int(importer_health.get("quality_rejected_items", 0) or 0),
            "duplicate_rejected_items": int(importer_health.get("duplicate_rejected_items", 0) or 0),
        },
        "story_quality": {
            "count": int(story_quality.get("count", 0) or 0),
            "avg_metadata_quality": float(story_quality.get("avg_metadata_quality", 0.0) or 0.0),
            "below_metadata_quality_floor": int(story_quality.get("below_metadata_quality_floor", 0) or 0),
        },
        "synthesis_health": {
            "traces": int(synthesis.get("samples", 0) or 0),
            "low_quality_applied": int(synthesis.get("low_quality_applied", 0) or 0),
            "warning_counts": dict(synthesis.get("warning_counts", {})),
        },
        "action_health": {
            "pending": int(pending_actions.get("pending", 0) or 0),
            "executed": int(pending_actions.get("executed", 0) or 0),
            "cancelled": int(pending_actions.get("cancelled", 0) or 0),
        },
        "safety_clean": safety_clean,
        "next_observed_needs": _assistant_status_next_steps(self_model, dashboard),
    }
    return observation


def _self_observation_history(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _append_self_observation_history(
    history: tuple[dict[str, Any], ...],
    observation: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    point = _self_observation_history_point(observation)
    if history and _same_self_observation_point(history[-1], point):
        updated = (*history[:-1], point)
    else:
        updated = (*history, point)
    return updated[-SELF_OBSERVATION_HISTORY_LIMIT:]


def _self_observation_history_point(observation: dict[str, Any]) -> dict[str, Any]:
    routing = dict(observation.get("routing", {}))
    cache = dict(observation.get("cache_health", {}))
    jobs = dict(observation.get("job_health", {}))
    synthesis = dict(observation.get("synthesis_health", {}))
    return {
        "events_observed": int(observation.get("events_observed", 0) or 0),
        "sessions_observed": int(observation.get("sessions_observed", 0) or 0),
        "local_resolution_rate": float(routing.get("local_resolution_rate", 0.0) or 0.0),
        "cloud_handoffs": int(routing.get("cloud_handoffs", 0) or 0),
        "external_fetches": int(routing.get("external_fetches", 0) or 0),
        "weather_cache_ready": bool(cache.get("weather_cache_ready", False)),
        "weather_days": int(cache.get("weather_days", 0) or 0),
        "story_inventory_ready": bool(cache.get("story_inventory_ready", False)),
        "story_models": int(cache.get("story_models", 0) or 0),
        "media_items": int(cache.get("media_items", 0) or 0),
        "contacts": int(cache.get("contacts", 0) or 0),
        "queued_jobs": int(jobs.get("queued", 0) or 0),
        "completed_jobs": int(jobs.get("completed", 0) or 0),
        "failed_jobs": int(jobs.get("failed", 0) or 0),
        "low_quality_applied": int(synthesis.get("low_quality_applied", 0) or 0),
        "safety_clean": bool(observation.get("safety_clean", False)),
    }


def _same_self_observation_point(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in right)


def _self_observation_history_summary(history: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if not history:
        return {
            "points": 0,
            "local_resolution_delta": 0.0,
            "weather_cache_gap_persistence": 0.0,
            "story_inventory_gap_persistence": 0.0,
            "weather_cache_became_ready": False,
            "story_model_delta": 0,
            "completed_job_delta": 0,
            "failed_job_delta": 0,
            "safety_regressions": 0,
        }
    first = history[0]
    latest = history[-1]
    points = len(history)
    return {
        "points": points,
        "first_events_observed": int(first.get("events_observed", 0) or 0),
        "latest_events_observed": int(latest.get("events_observed", 0) or 0),
        "local_resolution_delta": round(
            float(latest.get("local_resolution_rate", 0.0) or 0.0)
            - float(first.get("local_resolution_rate", 0.0) or 0.0),
            3,
        ),
        "weather_cache_gap_persistence": round(
            sum(1 for item in history if not bool(item.get("weather_cache_ready"))) / max(1, points),
            3,
        ),
        "story_inventory_gap_persistence": round(
            sum(1 for item in history if not bool(item.get("story_inventory_ready"))) / max(1, points),
            3,
        ),
        "weather_cache_became_ready": (
            not bool(first.get("weather_cache_ready"))
            and bool(latest.get("weather_cache_ready"))
        ),
        "story_model_delta": int(latest.get("story_models", 0) or 0) - int(first.get("story_models", 0) or 0),
        "completed_job_delta": int(latest.get("completed_jobs", 0) or 0)
        - int(first.get("completed_jobs", 0) or 0),
        "failed_job_delta": int(latest.get("failed_jobs", 0) or 0) - int(first.get("failed_jobs", 0) or 0),
        "safety_regressions": sum(1 for item in history if not bool(item.get("safety_clean"))),
    }


def _self_observation_pressure_context(store: AssistantOSStore) -> dict[str, Any]:
    state = store.load_self_state()
    history = _self_observation_history(state.get("runtime_health_history"))
    summary = _self_observation_history_summary(history)
    return {
        "self_observation_points": int(summary.get("points", 0) or 0),
        "local_resolution_delta": float(summary.get("local_resolution_delta", 0.0) or 0.0),
        "weather_cache_gap_persistence": float(summary.get("weather_cache_gap_persistence", 0.0) or 0.0),
        "story_inventory_gap_persistence": float(summary.get("story_inventory_gap_persistence", 0.0) or 0.0),
    }


def _self_observation_summary(observation: dict[str, Any]) -> str:
    routing = dict(observation.get("routing", {}))
    cache = dict(observation.get("cache_health", {}))
    jobs = dict(observation.get("job_health", {}))
    importer = dict(observation.get("importer_trends", {}))
    parts = [
        f"local_resolution={routing.get('local_resolution_rate', 0.0)}",
        "weather_cache=ready" if cache.get("weather_cache_ready") else "weather_cache=missing",
        f"story_models={cache.get('story_models', 0)}",
        f"jobs_completed={jobs.get('completed', 0)}",
    ]
    completed_cycles = int(importer.get("completed_cycles", 0) or 0)
    if completed_cycles:
        parts.append(f"import_cycles={completed_cycles}")
        parts.append(f"metadata_quality={importer.get('avg_metadata_quality', 0.0)}")
    history = dict(observation.get("history_summary", {}))
    points = int(history.get("points", 0) or 0)
    if points > 1:
        parts.append(f"history_points={points}")
        parts.append(f"local_resolution_delta={history.get('local_resolution_delta', 0.0)}")
        if history.get("weather_cache_became_ready"):
            parts.append("weather_cache_transition=ready")
    if not observation.get("safety_clean", False):
        parts.append("safety=needs_review")
    else:
        parts.append("safety=clean")
    return "; ".join(parts)


def _synthesis_matches_decision(
    synthesis: BoundedSynthesisResult,
    decision: AssistantDecision,
) -> bool:
    return synthesis.route == decision.route and synthesis.answer == decision.answer


def _bounded_opportunity_priority(value: float) -> float:
    return round(min(0.99, max(0.1, value)), 3)


def _is_routine_memory_gap_event(event: AssistantMemoryEvent) -> bool:
    text = event.utterance.lower()
    return _contains_any_lifecycle_marker(
        text,
        (
            "routine",
            "schedule",
            "school day",
            "work day",
            "morning",
            "bedtime",
        ),
    )


def _is_household_memory_gap_event(event: AssistantMemoryEvent) -> bool:
    text = event.utterance.lower()
    return _contains_any_lifecycle_marker(
        text,
        (
            "household",
            "shared device",
            "family memory",
            "family",
            "who uses this device",
        ),
    )


def _avg_float(values) -> float:
    value_tuple = tuple(float(value) for value in values)
    if not value_tuple:
        return 0.0
    return round(sum(value_tuple) / len(value_tuple), 3)


def _intent_from_pending_action(action_type: str) -> AssistantIntent:
    if action_type == "play_media":
        return "media_playback"
    return "social_contact"


def _is_assistant_status_request(utterance: str) -> bool:
    return compose_assistant_status_frame(utterance) is not None


def _is_assistant_next_step_request(utterance: str) -> bool:
    composition = compose_assistant_status_frame(utterance)
    return bool(composition and composition.get("action") == "plan")


def _is_confirmation(utterance: str) -> bool:
    text = utterance.lower().strip()
    return _first_token_is(text, ("yes", "confirm"))


def _first_token_is(text: str, values: tuple[str, ...]) -> bool:
    tokens = _marker_tokens(text)
    return bool(tokens) and tokens[0] in values


def _contains_any_lifecycle_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_has_marker(text, marker) for marker in markers)


def _has_marker(text: str, marker: str) -> bool:
    marker_tokens = _marker_tokens(marker)
    if not marker_tokens:
        return False
    tokens = _marker_tokens(text)
    width = len(marker_tokens)
    return any(tokens[index : index + width] == marker_tokens for index in range(0, len(tokens) - width + 1))


def _marker_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9']+", text.lower()))


def _extract_local_profile_setup(utterance: str) -> tuple[str, str, str] | None:
    text = " ".join(utterance.strip().split())
    normalized = text.lower()
    if not text or "?" in text:
        return None
    contact = _extract_trusted_contact_setup(text, normalized)
    if contact is not None:
        return contact
    child = _extract_child_setup(text, normalized)
    if child is not None:
        return child
    profile_fact = _extract_basic_profile_setup(text, normalized)
    if profile_fact is not None:
        return profile_fact
    routine = _extract_routine_setup(text, normalized)
    if routine is not None:
        return routine
    household = _extract_household_setup(text, normalized)
    if household is not None:
        return household
    return None


def _extract_trusted_contact_setup(text: str, normalized: str) -> tuple[str, str, str] | None:
    if not _contains_any_lifecycle_marker(normalized, ("trusted contact", "emergency contact")):
        return None
    match = re.search(
        r"\b(?P<name>[A-Za-z][A-Za-z0-9 _'-]{0,32})\s+is\s+(?:my|our)\s+"
        r"(?:(?:trusted|emergency)\s+contact)(?:\s+for\s+calls?)?"
        r"(?:\s*(?:at|on|:)\s*(?P<value>[+0-9][+0-9A-Za-z ()-]{4,32}))?",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    name = _memory_key(match.group("name"))
    if not name:
        return None
    value = _clean_memory_value(match.group("value") or f"local_contact:{name}")
    return ("trusted_contact", name, value)


def _extract_basic_profile_setup(text: str, normalized: str) -> tuple[str, str, str] | None:
    age_match = re.search(
        r"\b(?:i\s+am|i'm|my\s+age\s+is)\s+(?P<age>\d{1,3})(?:\s+years?\s+old)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if age_match is not None:
        age = int(age_match.group("age"))
        if 1 <= age <= 120:
            return ("profile_fact", "profile.age", str(age))
    location_match = re.search(
        r"\b(?:i\s+live\s+in|we\s+live\s+in|my\s+location\s+is|i\s+am\s+in|i'm\s+in)\s+"
        r"(?P<value>[A-Za-z][A-Za-z .'-]{1,64})$",
        text,
        flags=re.IGNORECASE,
    )
    if location_match is not None:
        value = _clean_memory_value(location_match.group("value"))
        if value:
            return ("profile_fact", "profile.location", value)
    name_match = re.search(
        r"\bmy\s+name\s+is\s+(?P<value>[A-Za-z][A-Za-z '-]{1,40})$",
        text,
        flags=re.IGNORECASE,
    )
    if name_match is not None:
        value = _clean_memory_value(name_match.group("value"))
        if value:
            return ("profile_fact", "profile.user_name", value)
    story_preference_match = re.search(
        r"\b(?:i\s+(?:like|love|prefer)|my\s+story\s+(?:theme|preference)\s+is)\s+"
        r"(?P<value>[^.?!]*(?:stor(?:y|ies)|tales?|folktales?|fables?)[^.?!]*)[.?!]?$",
        text,
        flags=re.IGNORECASE,
    )
    if story_preference_match is not None:
        value = _clean_memory_value(story_preference_match.group("value"))
        if value:
            return ("preference", "story_theme", value)
    return None


def _extract_child_setup(text: str, normalized: str) -> tuple[str, str, str] | None:
    if not _contains_any_lifecycle_marker(
        normalized,
        ("my child", "my kid", "my son", "my daughter", "child's", "kid's", "son's", "daughter's"),
    ):
        return None
    age_match = re.search(
        r"\bmy\s+(?:child|kid|son|daughter)\s+(?:is|is about|is almost)\s+"
        r"(?P<age>\d{1,2})(?:\s+years?\s+old)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if age_match is not None:
        age = int(age_match.group("age"))
        if 0 <= age <= 25:
            return ("child_memory", "child_age", str(age))
    school_match = re.search(
        r"\bmy\s+(?:child|kid|son|daughter)(?:'s)?\s+"
        r"(?:school\s+(?:is|=)|goes\s+to|attends)\s+(?P<value>[A-Za-z][A-Za-z0-9 .&'-]{1,80})$",
        text,
        flags=re.IGNORECASE,
    )
    if school_match is not None:
        value = _clean_memory_value(school_match.group("value"))
        if value:
            return ("child_memory", "child_school", value)
    name_match = re.search(
        r"\bmy\s+(?:child|kid|son|daughter)(?:'s)?\s+"
        r"(?:name\s+is|is\s+named|is\s+called)\s+(?P<value>[A-Za-z][A-Za-z '-]{1,40})$",
        text,
        flags=re.IGNORECASE,
    )
    if name_match is not None:
        value = _clean_memory_value(name_match.group("value"))
        if value:
            return ("child_memory", "child_name", value)
    return None


def _extract_routine_setup(text: str, normalized: str) -> tuple[str, str, str] | None:
    if not _contains_any_lifecycle_marker(normalized, ("routine", "schedule")):
        return None
    match = re.search(
        r"\b(?:my|our)\s+(?P<label>morning|bedtime|school day|work day|daily)?\s*"
        r"(?P<kind>routine|schedule)\s+(?:is|includes|means)\s+(?P<value>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"\bremember\s+(?:that\s+)?(?:my|our)\s+"
            r"(?P<label>morning|bedtime|school day|work day|daily)?\s*"
            r"(?P<kind>routine|schedule)\s+(?:is|includes|means)\s+(?P<value>.+)$",
            text,
            flags=re.IGNORECASE,
        )
    if match is None:
        return None
    label = _memory_key(match.group("label") or "daily")
    kind = _memory_key(match.group("kind"))
    value = _clean_memory_value(match.group("value"))
    if not value:
        return None
    key = f"{label}_{kind}" if label not in {"", "daily"} else f"daily_{kind}"
    return ("routine_memory", key, value)


def _extract_household_setup(text: str, normalized: str) -> tuple[str, str, str] | None:
    if not _contains_any_lifecycle_marker(
        normalized,
        ("household", "shared device", "family memory", "our family"),
    ):
        return None
    patterns = (
        r"\b(?:our|my)\s+household\s+(?:is|includes|has)\s+(?P<value>.+)$",
        r"\bthis\s+(?:shared\s+)?device\s+is\s+shared\s+by\s+(?P<value>.+)$",
        r"\b(?:our|my)\s+family\s+memory\s+(?:is|includes|should include)\s+(?P<value>.+)$",
        r"\bremember\s+(?:that\s+)?(?:our|my)\s+household\s+(?:is|includes|has)\s+(?P<value>.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        value = _clean_memory_value(match.group("value"))
        if value:
            return ("household_memory", "household_context", value)
    return None


def _memory_key(value: str | None) -> str:
    if not value:
        return ""
    cleaned = []
    for char in value.lower().strip():
        if char.isalnum():
            cleaned.append(char)
        elif char.isspace() or char in {"-", "_"}:
            cleaned.append("_")
    return "_".join("".join(cleaned).split("_")).strip("_")


def _clean_memory_value(value: str | None) -> str:
    if not value:
        return ""
    cleaned = " ".join(value.strip().strip(".!").split())
    return cleaned[:240]


def _revoked_fact_key(utterance: str) -> str | None:
    text = utterance.lower().strip()
    forget_markers = (
        "forget",
        "stop remembering",
        "stop using",
        "don't remember",
        "do not remember",
        "remove",
        "delete",
    )
    if not _contains_any_lifecycle_marker(text, forget_markers):
        return None
    if _has_marker(text, "favorite color"):
        return "facts.favorite_color"
    if _contains_any_lifecycle_marker(text, ("morning routine", "routine")):
        return "facts.morning_routine"
    if _contains_any_lifecycle_marker(
        text,
        ("child", "kid", "son", "daughter", "child's", "kid's", "son's", "daughter's"),
    ):
        if _has_marker(text, "school"):
            return "facts.child_school"
        if _contains_any_lifecycle_marker(text, ("name", "called")):
            return "facts.child_name"
        if _contains_any_lifecycle_marker(text, ("age", "old")):
            return "facts.child_age"
        return "facts.child_age"
    if _contains_any_lifecycle_marker(text, ("household", "family memory", "shared device", "our family")):
        return "facts.household_context"
    if _has_marker(text, "job"):
        return "facts.job"
    if _has_marker(text, "trip"):
        return "facts.trip"
    if _has_marker(text, "accessibility"):
        return "facts.accessibility"
    if _contains_any_lifecycle_marker(text, ("story theme", "story preference")):
        return "preferences.story_theme"
    if _contains_any_lifecycle_marker(text, ("music", "song preference")):
        return "preferences.music"
    if _has_marker(text, "breakfast"):
        return "preferences.breakfast"
    return None


def _requested_confirmation_target(utterance: str) -> str:
    text = utterance.lower().strip()
    for marker in ("call", "phone", "ring", "play"):
        target = _target_after_marker(text, marker)
        if target:
            return _clean_target(target)
    return ""


def _target_after_marker(text: str, marker: str) -> str:
    marker_tokens = _marker_tokens(marker)
    if not marker_tokens:
        return ""
    tokens = _marker_tokens(text)
    width = len(marker_tokens)
    for index in range(0, len(tokens) - width + 1):
        if tokens[index : index + width] == marker_tokens:
            return " ".join(tokens[index + width :])
    return ""


def _target_matches_pending(requested_target: str, pending_target: str) -> bool:
    requested = _clean_target(requested_target)
    pending = _clean_target(pending_target)
    return bool(requested and requested in pending)


def _clean_target(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum() or char.isspace():
            allowed.append(char)
        elif char in {".", ",", "!", "?"}:
            break
    cleaned = " ".join("".join(allowed).split())
    for prefix in ("my ", "the ", "to "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned


def _is_action_cancellation(utterance: str) -> bool:
    text = utterance.lower().strip()
    return (
        _first_token_is(text, ("cancel", "stop", "no"))
        or _contains_any_lifecycle_marker(text, ("don't call", "do not call", "never mind"))
    )


def _is_autobiographical_recall_request(utterance: str) -> bool:
    tokens = _marker_tokens(utterance)
    token_set = set(tokens)
    if "cloud" in token_set or _first_token_is(" ".join(tokens), ("send", "share", "upload")):
        return False
    return compose_autobiographical_memory_frame(utterance) is not None


def _is_autobiographical_session_summary_request(utterance: str) -> bool:
    return classify_autobiographical_memory_scope(utterance) == "session_summary"


def _is_autobiographical_long_horizon_request(utterance: str) -> bool:
    return classify_autobiographical_memory_scope(utterance) == "long_horizon"


def _autobiographical_digest_session_limit(utterance: str) -> int:
    tokens = _marker_tokens(utterance)
    token_set = set(tokens)
    if token_set & {"all", "whole", "everything"}:
        return 30
    if token_set & {"two", "2"}:
        return 2
    return 20


def _autobiographical_digest_events_per_session(utterance: str) -> int:
    token_set = set(_marker_tokens(utterance))
    if token_set & {"detailed", "detail"}:
        return 6
    if token_set & {"brief", "quick"}:
        return 2
    return 3


def _autobiographical_recall_query(utterance: str) -> str:
    token_set = set(_marker_tokens(utterance))
    topic_markers = (
        "story",
        "weather",
        "school",
        "health",
        "food",
        "meal",
        "song",
        "music",
        "call",
        "mom",
        "contact",
    )
    for marker in topic_markers:
        if marker in token_set:
            return marker
    return ""


def _autobiographical_session_limit(utterance: str) -> int:
    token_set = set(_marker_tokens(utterance))
    if token_set & {"two", "2"}:
        return 2
    if token_set & {"few", "recent"}:
        return 3
    return 3


def _autobiographical_events_per_session(utterance: str) -> int:
    token_set = set(_marker_tokens(utterance))
    if token_set & {"brief", "quick"}:
        return 2
    return 4


def _autobiographical_recall_limit(utterance: str) -> int:
    if classify_autobiographical_memory_scope(utterance) == "latest_event":
        return 1
    return 5


def run_assistant_kernel_learning_probe() -> KernelLearningReport:
    """Show the assistant noticing a gap and preparing local inventory."""

    profile = LocalAssistantProfile(
        story_models={},
        weekly_weather={},
        facts={},
        contacts={},
    )
    kernel = AssistantOSKernel(profile=profile)
    before_decisions = [kernel.handle("Tell me a story.") for _ in range(3)]
    opportunities = kernel.reflect()
    for opportunity in opportunities:
        if opportunity.kind == "build_story_inventory":
            kernel.execute(opportunity)
            break
    after_decision = kernel.handle("Tell me a story.")
    return KernelLearningReport(
        before_route=before_decisions[-1].route,
        after_route=after_decision.route,
        cloud_handoffs_before=sum(decision.cloud_needed for decision in before_decisions),
        cloud_handoffs_after=1 if after_decision.cloud_needed else 0,
        opportunities=opportunities,
        executed_jobs=tuple(kernel.executed_jobs),
        remembered_events=len(kernel.events),
        story_inventory_count=kernel.self_model.inventory_counts["story_models"],
    )
