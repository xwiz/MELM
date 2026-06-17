"""Bounded local synthesis for the Local Assistant OS MVP.

The synthesizer is intentionally deterministic and small. It does not try to be
an open-ended language model; it turns an already routed, membrane-approved
decision plus its evidence keys into a local answer trace with citations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from melm.contracts import load_answer_templates, load_assistant_identity, load_health_disclaimers, load_safety_policies

from .assistant_authority import (
    AnswerPlan,
    AuthorityEvidenceItem,
    AuthorityEvidencePacket,
    AuthorityInfo,
    DecoderResult,
    VerificationResult,
    build_answer_plan,
    build_evidence_packet,
    verify_answer,
)
from .assistant_decoder import ConstrainedDecoder, build_decoding_grammar
from .assistant_os_store import AssistantOSStore
from .assistant_skill_meal import format_meal_answer
from .assistant_skill_memory import autobiographical_digest_summary, autobiographical_memory_summary, autobiographical_session_summary, personal_memory_summary
from .assistant_skill_story import format_story_answer, format_story_frame
from .local_assistant_router import AssistantDecision, LocalAssistantProfile


SYNTHESIZABLE_ROUTES = {"local_answer", "cached_tool"}
SYNTHESIS_QUALITY_FLOOR = 0.65


def _load_answer_template(intent: str, reason: str) -> str | None:
    templates = load_answer_templates()
    entry = templates.get(intent)
    if entry is None:
        return None
    if "template" in entry:
        return entry["template"]
    if "templates" in entry and reason in entry["templates"]:
        return entry["templates"][reason]
    return None


def _render_contract_answer(
    decision: AssistantDecision,
    evidence: tuple[SynthesisEvidence, ...],
    profile: LocalAssistantProfile,
) -> str | None:
    templates = load_answer_templates()
    entry = templates.get(decision.intent)
    if entry is None:
        return None
    if "reason_gate" in entry:
        gate = entry["reason_gate"]
        if gate.startswith("!"):
            if decision.reason == gate[1:]:
                return None
        elif decision.reason != gate:
            return None
    if "template" in entry:
        template = entry["template"]
    elif "templates" in entry and decision.reason in entry["templates"]:
        template = entry["templates"][decision.reason]
    else:
        return None
    required = entry.get("requires_evidence", [])
    for kind in required:
        if _first_kind(evidence, kind) is None:
            return None
    try:
        return _format_answer_template(
            template, decision, evidence, profile,
        )
    except (KeyError, ValueError):
        return None


def _format_answer_template(
    template: str,
    decision: AssistantDecision,
    evidence: tuple[SynthesisEvidence, ...],
    profile: LocalAssistantProfile,
) -> str:
    evidence_map = {item.kind: item.value for item in evidence}
    weather = _first_kind(evidence, "weather")
    location = _first_kind(evidence, "profile")
    place = location.value if location is not None else profile.location
    weather_val = weather.value if weather is not None else ""
    summary = _autobiographical_memory_summary(evidence)
    goals = [item.value for item in evidence if item.kind == "health_goal"]
    goal_text = "; ".join(goals) if goals else "basic care"
    preference = _first_kind(evidence, "preference")
    contact = _first_kind(evidence, "contact")
    title = preference.value if preference is not None else _cancelled_title(decision.answer)
    label = _contact_label(contact, decision.answer) if contact is not None else "the contact"
    return template.format(
        place=place,
        weather=weather_val,
        summary=summary or "",
        goal_text=goal_text,
        title=title,
        label=label,
    )


def _handle_story(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    story = _first_kind(evidence, "story_model")
    if story is not None:
        return self._story_answer(story)
    return None


def _handle_identity(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    summary = _assistant_identity_summary(evidence)
    return summary if summary else None


def _handle_status(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    summary = _assistant_status_summary(decision, self.runtime_status)
    return summary if summary else None


def _handle_health_advice(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    if decision.reason == "urgent_health_safety_escalation":
        return _urgent_health_answer(decision.utterance)
    return None


def _handle_meal_suggestion(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    foods = [item.value for item in evidence if item.kind == "food_inventory"]
    weather = _first_kind(evidence, "weather")
    return format_meal_answer(
        foods,
        weather=weather.value if weather is not None else "",
        utterance=decision.utterance,
        preferences=self.profile.preferences,
        answer=decision.answer,
    )


def _handle_common_sense_safety(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    weather = _first_kind(evidence, "weather")
    if weather is not None and "school_clothing_weather_policy" in decision.reason:
        template = _load_answer_template(decision.intent, "school_clothing_weather_policy")
        return template if template else "Wear school clothes and carry rain protection because the forecast mentions rain."
    if decision.reason == "local_common_sense_policy":
        return _public_clothing_safety_answer(decision.utterance)
    return None


def _handle_autobiographical_memory(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    if decision.reason == "autobiographical_memory_digest":
        summary = _autobiographical_digest_summary(evidence)
    elif decision.reason == "autobiographical_session_summary":
        summary = _autobiographical_session_summary(evidence)
    else:
        summary = _autobiographical_memory_summary(evidence)
    if summary:
        template = _load_answer_template(decision.intent, decision.reason)
        if template:
            return template.format(summary=summary)
        return f"From local conversation memory: {summary}."
    return None


def _handle_personal_memory(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    summary = _personal_memory_summary(evidence)
    if summary:
        template = _load_answer_template(decision.intent, "personal_memory_summary")
        if template:
            return template.format(summary=summary)
        return f"I know this from local memory: {summary}."
    return None


def _handle_media_playback(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    if decision.reason == "cancelled_pending_action":
        preference = _first_kind(evidence, "preference")
        title = preference.value if preference is not None else _cancelled_title(decision.answer)
        template = _load_answer_template(decision.intent, "cancelled_pending_action")
        if template:
            return template.format(title=title)
        return (
            f"Cancelled. I will not play {title} now. "
            "There is no pending media action left to execute unless you ask again."
        )
    return None


def _handle_social_contact(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    if decision.reason == "cancelled_pending_action":
        contact = _first_kind(evidence, "contact")
        label = _contact_label(contact, decision.answer)
        template = _load_answer_template(decision.intent, "cancelled_pending_action")
        if template:
            return template.format(label=label)
        return (
            f"Cancelled. I will not call {label} now. "
            "The pending trusted-contact action is cleared, so no call will run unless you ask again."
        )
    if decision.reason == "consented_trusted_contact_stored":
        contact = _first_kind(evidence, "contact")
        if contact is not None:
            label = contact.key.split(".", 1)[1].replace("_", " ")
            template = _load_answer_template(decision.intent, "consented_trusted_contact_stored")
            if template:
                return template.format(label=label)
            return f"I will remember {label} as a trusted contact on this device."
    return None


def _handle_open_domain(
    synthesizer: "BoundedLocalSynthesizer",
    decision: AssistantDecision,
    evidence: tuple,
) -> str | None:
    from .assistant_skill_research import extract_topic, extract_action, format_open_domain_answer
    topic = extract_topic(decision.functional_parse)
    action = extract_action(decision.functional_parse)
    learned = [e for e in evidence if e.kind == "learned_fact"]
    if learned:
        return decision.answer
    return format_open_domain_answer(topic=topic or "that", action=action)


def _handle_unknown(
    synthesizer: "BoundedLocalSynthesizer",
    decision: AssistantDecision,
    evidence: tuple,
) -> str | None:
    from .assistant_skill_research import extract_topic, extract_action, format_open_domain_answer
    topic = extract_topic(decision.functional_parse)
    action = extract_action(decision.functional_parse)
    learned = [e for e in evidence if e.kind == "learned_fact"]
    if learned:
        return decision.answer
    return format_open_domain_answer(topic=topic or "that", action=action)


_ANSWER_HANDLERS: dict[str, Any] = {
    "story": _handle_story,
    "assistant_identity": _handle_identity,
    "assistant_status": _handle_status,
    "health_advice": _handle_health_advice,
    "meal_suggestion": _handle_meal_suggestion,
    "common_sense_safety": _handle_common_sense_safety,
    "autobiographical_memory": _handle_autobiographical_memory,
    "personal_memory": _handle_personal_memory,
    "media_playback": _handle_media_playback,
    "social_contact": _handle_social_contact,
    "open_domain": _handle_open_domain,
    "unknown": _handle_unknown,
}


@dataclass(frozen=True)
class SynthesisEvidence:
    key: str
    kind: str
    value: str
    source: str
    license: str
    local_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "value": self.value,
            "source": self.source,
            "license": self.license,
            "local_only": self.local_only,
        }


@dataclass(frozen=True)
class BoundedSynthesisResult:
    route: str
    applied: bool
    refused: bool
    answer: str
    citations: tuple[str, ...]
    evidence: tuple[SynthesisEvidence, ...]
    admitted_evidence_count: int
    reason: str
    boundary_crossed: str
    quality: dict[str, Any] = field(default_factory=dict)
    authority: AuthorityInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "applied": self.applied,
            "refused": self.refused,
            "answer": self.answer,
            "citations": list(self.citations),
            "evidence": [item.to_dict() for item in self.evidence],
            "admitted_evidence_count": self.admitted_evidence_count,
            "reason": self.reason,
            "boundary_crossed": self.boundary_crossed,
            "quality": self.quality,
        }


class BoundedLocalSynthesizer:
    """Cited verbalizer for local/cache decisions after membrane approval."""

    def __init__(
        self,
        profile: LocalAssistantProfile,
        *,
        store: AssistantOSStore | None = None,
        self_state: dict[str, Any] | None = None,
        runtime_status: dict[str, Any] | None = None,
        decoder: ConstrainedDecoder | None = None,
    ) -> None:
        self.profile = profile
        self.store = store
        self.self_state = self_state or {}
        self.runtime_status = runtime_status or {}
        self.decoder = decoder

    def synthesize(
        self,
        decision: AssistantDecision,
        *,
        boundary_crossed: str,
        membrane_allowed: bool,
    ) -> BoundedSynthesisResult:
        if not membrane_allowed:
            return self._refused(decision, boundary_crossed, "membrane_blocked")
        if decision.route not in SYNTHESIZABLE_ROUTES:
            return self._refused(decision, boundary_crossed, "route_not_synthesizable")
        if decision.reason == "consent_revoked_user_fact":
            evidence = (_policy("local_privacy_policy.consent_revocation", "revoked facts are removed from active memory"),)
            answer = (
                "Done. I removed that remembered fact from active local memory. "
                "I will not use it in future answers unless you tell me again."
            )
            return BoundedSynthesisResult(
                route=decision.route,
                applied=True,
                refused=False,
                answer=answer,
                citations=tuple(item.key for item in evidence),
                evidence=evidence,
                admitted_evidence_count=len(evidence),
                reason=f"bounded_synthesis:{decision.reason}",
                boundary_crossed=boundary_crossed,
                quality=_synthesis_quality(
                    decision,
                    applied=True,
                    refused=False,
                    answer=answer,
                    citations=tuple(item.key for item in evidence),
                    evidence=evidence,
                    reason=f"bounded_synthesis:{decision.reason}",
                    boundary_crossed=boundary_crossed,
                ),
            )

        evidence = tuple(
            item
            for key in decision.evidence_keys
            for item in self._resolve_evidence(key)
        )
        if not evidence:
            return self._refused(decision, boundary_crossed, "no_bound_evidence")

        template_answer = self._answer(decision, evidence)
        evidence_items = tuple(
            AuthorityEvidenceItem(
                key=item.key, kind=item.kind, value=str(item.value),
                source=item.source, license=item.license,
                local_only=item.local_only,
            )
            for item in evidence
        )
        packet = build_evidence_packet(decision.evidence_keys, evidence_items, boundary_crossed)
        plan = build_answer_plan(decision, packet)
        answer = self._decode_verified(plan, evidence, decision, template_answer, packet)
        verification = verify_answer(plan, answer, packet)
        authority = AuthorityInfo(evidence_packet=packet, answer_plan=plan, verification=verification)
        return BoundedSynthesisResult(
            route=decision.route,
            applied=True,
            refused=False,
            answer=answer,
            citations=tuple(item.key for item in evidence),
            evidence=evidence,
            admitted_evidence_count=len(evidence),
            reason=f"bounded_synthesis:{decision.reason}",
            boundary_crossed=boundary_crossed,
            quality=_synthesis_quality(
                decision,
                applied=True,
                refused=False,
                answer=answer,
                citations=tuple(item.key for item in evidence),
                evidence=evidence,
                reason=f"bounded_synthesis:{decision.reason}",
                boundary_crossed=boundary_crossed,
            ),
            authority=authority,
        )

    def _decode(
        self,
        plan: AnswerPlan,
        evidence: tuple[SynthesisEvidence, ...],
        decision: AssistantDecision,
    ) -> DecoderResult:
        answer = self._answer(decision, evidence)
        tokens_generated = len(answer.split()) if answer else 0
        return DecoderResult(answer=answer, decoder="template", tokens_generated=tokens_generated)

    def _decode_verified(
        self,
        plan: AnswerPlan,
        evidence: tuple[SynthesisEvidence, ...],
        decision: AssistantDecision,
        template_answer: str,
        packet: AuthorityEvidencePacket,
    ) -> str:
        if self.decoder is None:
            return template_answer
        evidence_entities = tuple(
            item.key for item in evidence
            if item.kind in ("story_model", "weather", "user_fact", "profile", "contact", "health_goal", "food_inventory", "preference")
        )
        grammar = build_decoding_grammar(plan, template_answer, evidence_entities)
        result = self.decoder.dispatch(plan, grammar)
        if result is not None and result.decoder != "template":
            v = verify_answer(plan, result.answer, packet)
            if v.passed:
                return result.answer
        return template_answer

    def _answer(
        self,
        decision: AssistantDecision,
        evidence: tuple[SynthesisEvidence, ...],
    ) -> str:
        handler = _ANSWER_HANDLERS.get(decision.intent)
        if handler is not None:
            result = handler(self, decision, evidence)
            if result is not None:
                return result
        return _render_contract_answer(decision, evidence, self.profile) or decision.answer

    def _story_answer(self, evidence: SynthesisEvidence) -> str:
        payload = self._story_payload(evidence)
        if not payload:
            frame = self._story_frame(evidence)
            payload = {
                "title": _title_from_evidence(evidence),
                "summary": format_story_frame(
                    frame,
                    name=self.profile.user_name,
                    location=self.profile.location,
                    culture=self.profile.culture,
                ),
            }
        title = str(payload.get("title") or _title_from_evidence(evidence))
        summary = str(payload.get("summary") or evidence.value)
        topics = _string_tuple(payload.get("topics"))
        cultures = _string_tuple(payload.get("cultures"))
        return format_story_answer(
            title, summary, topics, cultures,
            name=self.profile.user_name,
            location=self.profile.location,
            culture=self.profile.culture,
        )

    def _story_payload(self, evidence: SynthesisEvidence) -> dict[str, Any]:
        if self.store is None or not evidence.key.startswith("story_models."):
            return {}
        item_id = evidence.key.split(".", 1)[1]
        return self.store.load_inventory("story_model").get(item_id, {})

    def _story_frame(self, evidence: SynthesisEvidence) -> str:
        if self.store is not None and evidence.key.startswith("story_models."):
            item_id = evidence.key.split(".", 1)[1]
            payload = self.store.load_inventory("story_model").get(item_id, {})
            frame = _story_frame_from_payload(payload)
            if frame:
                return frame
        item_id = evidence.key.split(".", 1)[1] if "." in evidence.key else evidence.key
        return self.profile.story_models.get(item_id, evidence.value)

    def _resolve_evidence(self, key: str) -> tuple[SynthesisEvidence, ...]:
        if key.startswith("self_model."):
            field = key.split(".", 1)[1]
            value = self.self_state.get(field)
            if value is None and self.store is not None:
                value = self.store.load_self_state().get(field)
            if value is not None:
                text = _self_model_value_text(value)
                if text:
                    return (
                        SynthesisEvidence(
                            key=key,
                            kind="self_model",
                            value=text,
                            source="assistant_self_state",
                            license="local_runtime",
                            local_only=True,
                        ),
                    )
        if key.startswith("self_status."):
            field = key.split(".", 1)[1]
            value = self.runtime_status.get(field)
            if value is None and field == "no_store":
                value = "no persistent ledger is attached"
            if value is not None:
                text = _self_model_value_text(value)
                if text:
                    return (
                        SynthesisEvidence(
                            key=key,
                            kind="self_status",
                            value=text,
                            source="assistant_runtime_ledger",
                            license="private_local",
                            local_only=True,
                        ),
                    )
        if key.startswith("story_models."):
            item_id = key.split(".", 1)[1]
            payload = self._inventory_payload("story_model", item_id)
            title = str(payload.get("title") or item_id)
            summary = str(payload.get("summary") or _story_frame_from_payload(payload) or self.profile.story_models.get(item_id, ""))
            source, license_name = self._inventory_source_license("story_model", item_id)
            return (
                SynthesisEvidence(
                    key=key,
                    kind="story_model",
                    value=f"{title}: {summary}",
                    source=source,
                    license=license_name,
                    local_only=False,
                ),
            )
        if key.startswith("profile."):
            field = key.split(".", 1)[1]
            value = str(getattr(self.profile, field, ""))
            if value:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="profile",
                        value=value,
                        source="user_profile",
                        license="private_local",
                        local_only=True,
                    ),
                )
        if key.startswith("facts."):
            fact_key = key.split(".", 1)[1]
            value = ""
            source = "user_profile"
            local_only = True
            if self.store is not None:
                slot = self.store.get_entity_slot("self", fact_key)
                if slot is not None and slot.consent:
                    raw = json.loads(slot.value_json) if slot.value_json else ""
                    value = str(raw) if raw else ""
                    source = slot.source
                    local_only = slot.scope == "private_local" or "local" in slot.scope
            if not value:
                value = self.profile.facts.get(fact_key, "")
            if value:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="user_fact",
                        value=value,
                        source=source,
                        license="private_local",
                        local_only=local_only,
                    ),
                )
        if key.startswith("preferences."):
            pref_key = key.split(".", 1)[1]
            value = self.profile.preferences.get(pref_key, "")
            if value:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="preference",
                        value=value,
                        source="user_profile",
                        license="private_local",
                        local_only=True,
                    ),
                )
        if key == "health_goals":
            return tuple(
                SynthesisEvidence(
                    key=f"health_goals.{index}",
                    kind="health_goal",
                    value=goal,
                    source="user_profile",
                    license="private_local",
                    local_only=True,
                )
                for index, goal in enumerate(self.profile.health_goals)
            )
        if key == "local_health_safety_policy":
            return (_policy(key, "general health guidance is bounded and escalates danger"),)
        if key == "local_safety_policy.clothing_public_school":
            return (_policy(key, "public school clothing requires proper clothes"),)
        if key.startswith("weekly_weather."):
            item_id = key.split(".", 1)[1]
            forecast = self.profile.weekly_weather.get(item_id, "")
            if not forecast and self.store is not None:
                forecast = str(self.store.load_inventory("weather").get(item_id, {}).get("forecast", ""))
            if forecast:
                source, license_name = self._inventory_source_license("weather", item_id)
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="weather",
                        value=forecast,
                        source=source,
                        license=license_name,
                        local_only=False,
                    ),
                )
        if key == "food_inventory":
            return tuple(
                SynthesisEvidence(
                    key=f"food_inventory.{item}",
                    kind="food_inventory",
                    value=item,
                    source="local_food_inventory",
                    license="local_seed",
                    local_only=False,
                )
                for item in self.profile.food_inventory
            )
        if key == "media_library":
            return tuple(
                SynthesisEvidence(
                    key=f"media_library.{item}",
                    kind="media",
                    value=item,
                    source="local_media_index",
                    license="local_device",
                    local_only=False,
                )
                for item in self.profile.media_library
            )
        if key.startswith("contacts."):
            contact_key = key.split(".", 1)[1]
            value = ""
            source = "user_profile"
            local_only = True
            if self.store is not None:
                entity_id = f"contact:{contact_key}"
                entity = self.store.get_entity(entity_id)
                if entity is not None:
                    phone_slot = self.store.get_entity_slot(entity_id, "phone")
                    if phone_slot is not None and phone_slot.consent:
                        raw = json.loads(phone_slot.value_json) if phone_slot.value_json else ""
                        value = str(raw) if raw else ""
                        source = phone_slot.source
                        local_only = phone_slot.scope == "private_local" or "local" in phone_slot.scope
            if not value:
                value = self.profile.contacts.get(contact_key, "")
            if value:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="contact",
                        value=value,
                        source=source,
                        license="private_local",
                        local_only=local_only,
                    ),
                )
        if key.startswith("events.") and self.store is not None:
            event_id = key.split(".", 1)[1]
            row = self.store.connection.execute(
                """
                SELECT session_id, utterance, intent, route, reason
                FROM events
                WHERE event_id=?
                """,
                (event_id,),
            ).fetchone()
            if row is not None:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="event_memory",
                        value=(
                            f"{row['intent']} via {row['route']}: "
                            f"{row['utterance']} ({row['reason']})"
                        ),
                        source=f"assistant_event_ledger:{row['session_id']}",
                        license="private_local",
                        local_only=True,
                    ),
                )
        if key.startswith("memory_digest.") and self.store is not None:
            digest_id = key.split(".", 1)[1]
            payload = self.store.load_memory_digest(digest_id)
            summary = str(payload.get("summary", ""))
            if summary:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="memory_digest",
                        value=summary,
                        source="assistant_event_ledger_compactor",
                        license="private_local",
                        local_only=True,
                    ),
                )
        if key.startswith("learned_fact.") and self.store is not None:
            entity_id = key.split(".", 1)[1]
            slots = self.store.get_entity_slots(entity_id)
            slot_map = {}
            for s in slots:
                try:
                    slot_map[s.slot_name] = json.loads(s.value_json) if s.value_json else ""
                except json.JSONDecodeError:
                    slot_map[s.slot_name] = s.value_json
            summary = str(slot_map.get("summary", ""))
            if summary:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="learned_fact",
                        value=summary,
                        source=str(slot_map.get("source", "assistant_research")),
                        license="local_runtime",
                        local_only=True,
                    ),
                )
        return ()

    def _inventory_payload(self, kind: str, item_id: str) -> dict[str, Any]:
        if self.store is None:
            return {}
        return self.store.load_inventory(kind).get(item_id, {})

    def _inventory_source_license(self, kind: str, item_id: str) -> tuple[str, str]:
        if self.store is None:
            return ("profile_or_local_runtime", "local_runtime")
        row = self.store.connection.execute(
            """
            SELECT source, license
            FROM inventories
            WHERE kind=? AND item_id=?
            """,
            (kind, item_id),
        ).fetchone()
        if row is None:
            return ("profile_or_local_runtime", "local_runtime")
        return str(row["source"]), str(row["license"])

    def _refused(
        self,
        decision: AssistantDecision,
        boundary_crossed: str,
        reason: str,
    ) -> BoundedSynthesisResult:
        return BoundedSynthesisResult(
            route=decision.route,
            applied=False,
            refused=True,
            answer=decision.answer,
            citations=(),
            evidence=(),
            admitted_evidence_count=0,
            reason=reason,
            boundary_crossed=boundary_crossed,
            quality=_synthesis_quality(
                decision,
                applied=False,
                refused=True,
                answer=decision.answer,
                citations=(),
                evidence=(),
                reason=reason,
                boundary_crossed=boundary_crossed,
            ),
        )


def _policy(key: str, value: str) -> SynthesisEvidence:
    return SynthesisEvidence(
        key=key,
        kind="policy",
        value=value,
        source="local_policy",
        license="local_policy",
        local_only=False,
    )


def _self_model_value_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={value[key]}" for key in sorted(value))
    return str(value)


def _first_kind(
    evidence: tuple[SynthesisEvidence, ...],
    kind: str,
) -> SynthesisEvidence | None:
    return next((item for item in evidence if item.kind == kind), None)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _title_from_evidence(evidence: SynthesisEvidence) -> str:
    return evidence.value.split(":", 1)[0] if ":" in evidence.value else "the selected story"


def _story_frame_from_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("narrative_frame") or payload.get("template") or "")


def _personal_memory_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    return personal_memory_summary(evidence)


def _assistant_identity_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    values = {item.key: item.value for item in evidence if item.kind == "self_model"}
    name = values.get("self_model.name", "MELM Local Assistant OS")
    purpose = values.get("self_model.purpose", "help locally first")
    capabilities = _split_self_model_list(values.get("self_model.local_capabilities", ""))
    limits = _split_self_model_list(values.get("self_model.limits", ""))
    capability_text = _join_short_list(
        tuple(item.replace("_", " ") for item in capabilities[:4]),
        fallback="local memory, cached tools, confirmed actions, and local policy",
    )
    limit_text = _join_short_list(
        tuple(item for item in limits[:2]),
        fallback="I ask for help when local evidence is not enough.",
    )
    purpose_text = purpose[0].lower() + purpose[1:] if purpose else "help locally first"
    templates = load_assistant_identity()
    return templates["introduction"].format(
        name=name, purpose=purpose_text,
        capability_text=capability_text, limit_text=limit_text,
    )


def _assistant_status_summary(decision: AssistantDecision, status: dict[str, Any]) -> str:
    templates = load_assistant_identity()
    if not status.get("available", False):
        return templates["status_unavailable"]
    counts = _status_dict(status.get("counts"))
    routes = _status_dict(status.get("route_counts"))
    inventories = _status_dict(status.get("inventories"))
    pending = _status_dict(status.get("pending_actions"))
    events = int(counts.get("events", 0))
    sessions = int(status.get("sessions", 0) or 0)
    synthesis_traces = int(counts.get("synthesis_traces", 0))
    pending_count = int(pending.get("pending", 0))
    route_text = _status_counts_text(routes, fallback="no routed turns yet")
    inventory_text = _status_counts_text(inventories, fallback="no inventory rows yet")
    safety = "clean" if bool(status.get("safety_clean", False)) else "needs review"
    next_steps = tuple(str(item) for item in status.get("next_steps", ()) if str(item))
    next_text = _join_short_list(next_steps, fallback="continue using the local ledger and refresh inventories when gaps appear")
    observation_text = _self_observation_status_text(status.get("self_observation"))
    key = "status_next_steps" if decision.reason == "self_status_next_steps" else "status_default"
    return templates[key].format(
        events=events, sessions=sessions, synthesis_traces=synthesis_traces,
        pending_count=pending_count, safety=safety, observation_text=observation_text,
        next_text=next_text, route_text=route_text, inventory_text=inventory_text,
    )


def _status_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _status_counts_text(counts: dict[str, Any], *, fallback: str) -> str:
    items = [
        f"{key}={counts[key]}"
        for key in sorted(counts)
        if key and int(counts.get(key, 0) or 0) != 0
    ]
    return ", ".join(items[:6]) if items else fallback


def _self_observation_status_text(value: Any) -> str:
    observation = value if isinstance(value, dict) else {}
    if not observation:
        return "no local trend snapshot yet"
    summary = str(observation.get("summary", ""))
    if summary:
        return summary
    routing = _status_dict(observation.get("routing"))
    cache = _status_dict(observation.get("cache_health"))
    jobs = _status_dict(observation.get("job_health"))
    parts = [
        f"local_resolution={routing.get('local_resolution_rate', 0.0)}",
        "weather_cache=ready" if cache.get("weather_cache_ready") else "weather_cache=missing",
        f"story_models={cache.get('story_models', 0)}",
        f"jobs_completed={jobs.get('completed', 0)}",
    ]
    return "; ".join(parts)


def _split_self_model_list(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _join_short_list(items: tuple[str, ...], *, fallback: str) -> str:
    if not items:
        return fallback
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _autobiographical_memory_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    return autobiographical_memory_summary(evidence)


def _autobiographical_session_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    return autobiographical_session_summary(evidence)


def _autobiographical_digest_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    return autobiographical_digest_summary(evidence)


def _urgent_health_answer(utterance: str) -> str:
    text = utterance.lower()
    responses = load_health_disclaimers()
    for key, entry in responses.items():
        if key == "fallback":
            continue
        for trigger in entry.get("triggers", []):
            if trigger in text:
                return entry["text"]
    return responses.get("fallback", {}).get("text", "")


def _public_clothing_safety_answer(utterance: str) -> str:
    text = utterance.lower()
    policies = load_safety_policies()
    clothing = policies.get("public_clothing", {})
    destinations = clothing.get("destinations", {})
    template = clothing.get("template", "")
    dest_phrase = destinations.get("default", {}).get("phrase", "going outside")
    for key, dentry in destinations.items():
        if key == "default":
            continue
        for trigger in dentry.get("triggers", []):
            if trigger in text:
                dest_phrase = dentry.get("phrase", dest_phrase)
                break
    return template.replace("{destination}", dest_phrase)


def _cancelled_title(answer: str) -> str:
    cleaned = answer.replace("Cancelled:", "").replace("Playing", "").strip()
    return cleaned.rstrip(".") or "that audio"


def _synthesis_quality(
    decision: AssistantDecision,
    *,
    applied: bool,
    refused: bool,
    answer: str,
    citations: tuple[str, ...],
    evidence: tuple[SynthesisEvidence, ...],
    reason: str,
    boundary_crossed: str,
) -> dict[str, Any]:
    expected_refusal = (
        decision.route not in SYNTHESIZABLE_ROUTES
        or boundary_crossed.startswith("blocked")
        or reason in {"membrane_blocked", "route_not_synthesizable", "no_bound_evidence"}
    )
    route_discipline = 1.0 if (refused == expected_refusal or applied != expected_refusal) else 0.0
    citation_coverage = _citation_coverage(citations, evidence)
    evidence_strength = min(1.0, len(evidence) / _target_evidence_count(decision))
    source_diversity = min(1.0, len({item.source for item in evidence}) / max(1, min(3, len(evidence))))
    answer_specificity = _answer_specificity(decision, answer, evidence)
    local_privacy_discipline = 1.0
    if boundary_crossed == "cloud" and any(item.local_only for item in evidence):
        local_privacy_discipline = 0.0
    if expected_refusal:
        score = 0.75 * route_discipline + 0.25 * local_privacy_discipline
    else:
        score = (
            0.3 * route_discipline
            + 0.25 * citation_coverage
            + 0.2 * evidence_strength
            + 0.15 * answer_specificity
            + 0.05 * source_diversity
            + 0.05 * local_privacy_discipline
        )
    warnings = []
    if not expected_refusal and citation_coverage < 1.0:
        warnings.append("citation_gap")
    if not expected_refusal and answer_specificity < 0.5:
        warnings.append("generic_answer")
    if route_discipline < 1.0:
        warnings.append("route_discipline_failure")
    if local_privacy_discipline < 1.0:
        warnings.append("private_evidence_boundary_failure")
    return {
        "score": round(score, 3),
        "route_discipline": round(route_discipline, 3),
        "citation_coverage": round(citation_coverage, 3),
        "evidence_strength": round(evidence_strength, 3),
        "answer_specificity": round(answer_specificity, 3),
        "source_diversity": round(source_diversity, 3),
        "local_privacy_discipline": round(local_privacy_discipline, 3),
        "citation_count": len(citations),
        "evidence_count": len(evidence),
        "expected_refusal": expected_refusal,
        "warnings": warnings,
    }


def _citation_coverage(
    citations: tuple[str, ...],
    evidence: tuple[SynthesisEvidence, ...],
) -> float:
    if not evidence:
        return 1.0 if not citations else 0.0
    evidence_keys = {item.key for item in evidence}
    cited_keys = set(citations)
    return len(evidence_keys & cited_keys) / max(1, len(evidence_keys))


def _contact_label(contact: SynthesisEvidence | None, answer: str) -> str:
    if contact is not None and "." in contact.key:
        return contact.key.split(".", 1)[1].replace("_", " ")
    cleaned = answer.replace("Cancelled:", "").replace("I can call", "").strip()
    return cleaned.rstrip(".") or "that contact"


def _target_evidence_count(decision: AssistantDecision) -> int:
    from melm.contracts import load_contract_json
    payload = load_contract_json("answer_templates.v1.json")
    targets = payload.get("evidence_count_targets", {})
    return targets.get(decision.intent, 1)


def _answer_specificity(
    decision: AssistantDecision,
    answer: str,
    evidence: tuple[SynthesisEvidence, ...],
) -> float:
    text = answer.lower()
    score = 0.0
    if len(answer.split()) >= 8:
        score += 0.25
    if len(answer.split(".")) >= 3:
        score += 0.15
    evidence_terms = _evidence_terms(evidence)
    if evidence_terms:
        matches = sum(1 for term in evidence_terms if term and term in text)
        score += min(0.45, matches * 0.15)
    from melm.contracts import load_contract_json
    payload = load_contract_json("answer_templates.v1.json")
    phrases = payload.get("answer_specificity_phrases", {})
    intent_phrases = phrases.get(decision.intent, [])
    if isinstance(intent_phrases, str):
        intent_phrases = [intent_phrases]
    if intent_phrases and any(p in text for p in intent_phrases):
        score += 0.15
    bonuses = payload.get("answer_specificity_bonuses", [])
    for entry in bonuses:
        if entry.get("intent") == decision.intent and entry.get("reason", "") == decision.reason:
            for group in entry.get("triggers", []):
                if all(p in text for p in group):
                    score += entry.get("bonus", 0.0)
                    break
    return min(1.0, score)


def _evidence_terms(evidence: tuple[SynthesisEvidence, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    for item in evidence:
        if item.kind == "profile":
            terms.append(item.value.lower())
        elif item.kind == "story_model":
            terms.append(_title_from_evidence(item).lower())
        elif item.kind in {"health_goal", "food_inventory", "weather", "user_fact", "preference"}:
            terms.extend(word for word in item.value.lower().split() if len(word) >= 4)
        elif item.kind == "event_memory":
            terms.extend(word for word in item.value.lower().split() if len(word) >= 4)
        elif item.kind == "self_model":
            terms.extend(word for word in item.value.lower().replace("_", " ").split() if len(word) >= 4)
        elif item.kind == "self_status":
            terms.extend(word for word in item.value.lower().replace("_", " ").split() if len(word) >= 4)
        elif item.kind in {"policy", "media", "contact"}:
            terms.extend(word for word in item.value.lower().split() if len(word) >= 5)
    return tuple(dict.fromkeys(terms))
