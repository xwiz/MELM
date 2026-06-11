"""Bounded local synthesis for the Local Assistant OS MVP.

The synthesizer is intentionally deterministic and small. It does not try to be
an open-ended language model; it turns an already routed, membrane-approved
decision plus its evidence keys into a local answer trace with citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .assistant_os_store import AssistantOSStore
from .local_assistant_router import AssistantDecision, LocalAssistantProfile, choose_local_meal


SYNTHESIZABLE_ROUTES = {"local_answer", "cached_tool"}
SYNTHESIS_QUALITY_FLOOR = 0.65


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
    ) -> None:
        self.profile = profile
        self.store = store
        self.self_state = self_state or {}
        self.runtime_status = runtime_status or {}

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

        answer = self._answer(decision, evidence)
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

    def _answer(
        self,
        decision: AssistantDecision,
        evidence: tuple[SynthesisEvidence, ...],
    ) -> str:
        if decision.intent == "story":
            story = _first_kind(evidence, "story_model")
            if story is not None:
                return self._story_answer(story)
        if decision.intent == "assistant_identity":
            summary = _assistant_identity_summary(evidence)
            if summary:
                return summary
        if decision.intent == "assistant_status":
            summary = _assistant_status_summary(decision, self.runtime_status)
            if summary:
                return summary
        if decision.intent == "health_advice":
            if decision.reason == "urgent_health_safety_escalation":
                return _urgent_health_answer(decision.utterance)
            goals = [item.value for item in evidence if item.kind == "health_goal"]
            goal_text = "; ".join(goals) if goals else "basic care"
            return (
                "This is general local guidance, not a diagnosis. Start with water, "
                f"steady sleep, and gentle movement. Today, choose one small goal: {goal_text}. "
                "For pain, danger, or illness, talk to a trusted adult or clinician."
            )
        if decision.intent == "meal_suggestion":
            foods = [item.value for item in evidence if item.kind == "food_inventory"]
            weather = _first_kind(evidence, "weather")
            choice = choose_local_meal(
                foods,
                preferences=self.profile.preferences,
                weather=weather.value if weather is not None else "",
                utterance=decision.utterance,
            )
            base = choice.phrase or _meal_phrase(decision.answer)
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
        if decision.intent == "common_sense_safety":
            weather = _first_kind(evidence, "weather")
            if weather is not None and "school_clothing_weather_policy" in decision.reason:
                return "Wear school clothes and carry rain protection because the forecast mentions rain."
            if decision.reason == "local_common_sense_policy":
                return _public_clothing_safety_answer(decision.utterance)
        if decision.intent == "media_playback" and decision.reason == "cancelled_pending_action":
            preference = _first_kind(evidence, "preference")
            title = preference.value if preference is not None else _cancelled_title(decision.answer)
            return (
                f"Cancelled. I will not play {title} now. "
                "There is no pending media action left to execute unless you ask again."
            )
        if decision.intent == "social_contact" and decision.reason == "cancelled_pending_action":
            contact = _first_kind(evidence, "contact")
            label = _contact_label(contact, decision.answer)
            return (
                f"Cancelled. I will not call {label} now. "
                "The pending trusted-contact action is cleared, so no call will run unless you ask again."
            )
        if decision.intent == "personal_memory":
            summary = _personal_memory_summary(evidence)
            if summary:
                return f"I know this from local memory: {summary}."
        if decision.intent == "social_contact" and decision.reason == "consented_trusted_contact_stored":
            contact = _first_kind(evidence, "contact")
            if contact is not None:
                label = contact.key.split(".", 1)[1].replace("_", " ")
                return f"I will remember {label} as a trusted contact on this device."
        if decision.intent == "autobiographical_memory":
            if decision.reason == "autobiographical_memory_digest":
                summary = _autobiographical_digest_summary(evidence)
            elif decision.reason == "autobiographical_session_summary":
                summary = _autobiographical_session_summary(evidence)
            else:
                summary = _autobiographical_memory_summary(evidence)
            if summary:
                return f"From local conversation memory: {summary}."
        if decision.intent == "weather":
            weather = _first_kind(evidence, "weather")
            location = _first_kind(evidence, "profile")
            if weather is not None:
                place = location.value if location is not None else self.profile.location
                return (
                    f"Today in {place}: {weather.value}. "
                    "That is the cached local forecast for today. "
                    "For exact timing, I should refresh the weather cache."
                )
        return decision.answer

    def _story_answer(self, evidence: SynthesisEvidence) -> str:
        payload = self._story_payload(evidence)
        if not payload:
            frame = self._story_frame(evidence)
            payload = {
                "title": _title_from_evidence(evidence),
                "summary": _format_story_frame(
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
        name = self.profile.user_name
        location = self.profile.location
        image = _story_image(title, summary, topics)
        challenge = _story_challenge(topics, summary)
        lesson = _story_lesson(topics)
        fit = ""
        if self.profile.culture and self.profile.culture in cultures:
            fit = f" with a {self.profile.culture} flavor"
        elif location and location in cultures:
            fit = f" in {location}"
        return (
            f"I picked {title} from the local story inventory{fit}. "
            f"In {location}, {name} {image}. {challenge} "
            f"By the end, {name} {lesson}"
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
            value = self.profile.facts.get(fact_key, "")
            if value:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="user_fact",
                        value=value,
                        source="user_profile",
                        license="private_local",
                        local_only=True,
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
            value = self.profile.contacts.get(contact_key, "")
            if value:
                return (
                    SynthesisEvidence(
                        key=key,
                        kind="contact",
                        value=value,
                        source="user_profile",
                        license="private_local",
                        local_only=True,
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


def _format_story_frame(frame: str, *, name: str, location: str, culture: str) -> str:
    if not frame:
        return "a local story frame with enough metadata for a short safe adventure"
    try:
        return frame.format(name=name, location=location, culture=culture)
    except (KeyError, IndexError, ValueError):
        return frame


def _story_image(title: str, summary: str, topics: tuple[str, ...]) -> str:
    title_text = title.lower()
    text = " ".join((title, summary, *topics)).lower()
    if "drum" in title_text or "music" in title_text:
        return "heard a small drum answering every careful step"
    if "tortoise" in title_text:
        return "met a patient tortoise carrying a question bigger than its shell"
    if "rain" in title_text or "bedtime" in title_text:
        return "noticed rain tapping a secret path across the window"
    if "school" in title_text or "star" in title_text:
        return "found a bright question waiting at the edge of the schoolyard"
    if "rain" in text or "bedtime" in text:
        return "noticed rain tapping a secret path across the window"
    if "drum" in text or "music" in text:
        return "heard a small drum answering every careful step"
    if "tortoise" in text or "animal" in text:
        return "met a patient tortoise carrying a question bigger than its shell"
    if "school" in text or "star" in text:
        return "found a bright question waiting at the edge of the schoolyard"
    return "found a small sign that made an ordinary walk feel like an adventure"


def _story_challenge(topics: tuple[str, ...], summary: str) -> str:
    text = " ".join((*topics, summary)).lower()
    if "questions" in text or "curiosity" in text:
        return "The puzzle could not be rushed, so one good question opened the next clue."
    if "kindness" in text:
        return "Someone nearby needed help, and the adventure only moved forward after a kind choice."
    if "bedtime" in text:
        return "The map faded whenever the room got noisy, so listening became the brave thing to do."
    if "school" in text:
        return "The answer was not in a cloud or a book at first; it began with looking carefully."
    return "The safe path was the one that asked for help, listened, and came back before dark."


def _story_lesson(topics: tuple[str, ...]) -> str:
    if "questions" in topics:
        return "learned that asking first can be braver than running first."
    if "kindness" in topics:
        return "learned that kindness is useful, not just nice."
    if "bedtime" in topics:
        return "came home calm enough to sleep and curious enough for tomorrow."
    if "school" in topics:
        return "wrote the question down so it could grow into tomorrow's lesson."
    return "came home safely with one new question and one good choice remembered."


def _personal_memory_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    parts: list[str] = []
    for item in evidence:
        if item.kind == "profile":
            label = item.key.split(".", 1)[1].replace("_", " ")
            if label == "age":
                parts.append(f"your age is {item.value}")
            elif label == "location":
                parts.append(f"you are in {item.value}")
            elif label == "culture":
                parts.append(f"your culture hint is {item.value}")
            elif label == "user name":
                parts.append(f"your name is {item.value}")
        elif item.kind == "user_fact":
            label = item.key.split(".", 1)[1].replace("_", " ")
            parts.append(f"your {label} is {item.value}")
        elif item.kind == "preference":
            label = item.key.split(".", 1)[1].replace("_", " ")
            parts.append(f"your {label} preference is {item.value}")
    return "; ".join(dict.fromkeys(parts))


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
    return (
        f"I am {name}. My purpose is to {purpose[0].lower() + purpose[1:] if purpose else 'help locally first'} "
        f"I can use {capability_text} on this device. My limits are: {limit_text}."
    )


def _assistant_status_summary(decision: AssistantDecision, status: dict[str, Any]) -> str:
    if not status.get("available", False):
        return (
            "I can describe my built-in local capabilities, but I do not have a "
            "persistent event ledger attached in this runtime."
        )
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
    if decision.reason == "self_status_next_steps":
        return (
            f"My local ledger has {events} event(s) across {sessions} session(s), "
            f"with {synthesis_traces} synthesis trace(s) and {pending_count} pending action(s). "
            f"The safety dashboard is {safety}. My persisted self-observation says {observation_text}. "
            f"Next I should {next_text}."
        )
    return (
        f"So far my local ledger has {events} event(s) across {sessions} session(s), "
        f"with routes: {route_text}. Inventory coverage is: {inventory_text}. "
        f"I have {pending_count} pending action(s), and the safety dashboard is {safety}. "
        f"My persisted self-observation says {observation_text}. "
        f"Next useful work: {next_text}."
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
    events = [_event_memory_parts(item) for item in evidence if item.kind == "event_memory"]
    if not events:
        return ""
    parts: list[str] = []
    for index, event in enumerate(events[:5], start=1):
        label = _event_label(event)
        parts.append(f"{index}. {label} - you said \"{event['utterance']}\"")
    insight = _event_memory_insight_text(events)
    if insight:
        parts.append(insight)
    return " ".join(parts)


def _autobiographical_session_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in evidence:
        if item.kind != "event_memory":
            continue
        event = _event_memory_parts(item)
        session_id = event["session_id"]
        grouped.setdefault(session_id, []).append(event)
    if not grouped:
        return ""
    parts: list[str] = []
    all_events: list[dict[str, str]] = []
    for session_index, (session_id, events) in enumerate(grouped.items(), start=1):
        utterances = []
        intents = []
        for event in events[:4]:
            intents.append(event["intent"].replace("_", " "))
            utterances.append(event["utterance"])
            all_events.append(event)
        intent_text = ", ".join(dict.fromkeys(intents))
        quoted = "; ".join(f'"{utterance}"' for utterance in utterances)
        parts.append(f"session {session_index} ({session_id}) covered {intent_text}: {quoted}")
    insight = _event_memory_insight_text(all_events)
    if insight:
        parts.append(insight)
    return " ".join(parts)


def _event_memory_parts(item: SynthesisEvidence) -> dict[str, str]:
    value = item.value
    label, detail = value.split(": ", 1) if ": " in value else ("event via local_answer", value)
    intent, route = label.split(" via ", 1) if " via " in label else (label, "")
    utterance = detail
    reason = ""
    if " (" in detail and detail.endswith(")"):
        utterance, reason = detail.rsplit(" (", 1)
        reason = reason[:-1]
    session_id = item.source.split(":", 1)[1] if ":" in item.source else "session"
    return {
        "session_id": session_id,
        "intent": intent,
        "route": route,
        "utterance": utterance,
        "reason": reason,
    }


def _event_label(event: dict[str, str]) -> str:
    intent = event.get("intent", "event").replace("_", " ")
    route = event.get("route", "")
    return f"{intent} via {route}" if route else intent


def _event_memory_insight_text(events: list[dict[str, str]]) -> str:
    transitions: list[str] = []
    open_loops: list[str] = []
    boundary_controls: list[str] = []
    action_state: list[str] = []
    for event in events:
        intent = event["intent"]
        route = event["route"]
        reason = event["reason"]
        if intent == "story" and reason == "missing_story_model":
            open_loops.append("story inventory was missing, so the story ask needed cloud handoff")
        elif intent == "story" and reason == "local_story_inventory":
            transitions.append("story inventory answered locally")
        elif intent == "weather" and reason == "weather_cache_miss":
            open_loops.append("weather cache was missing, so the assistant chose a tool fetch")
        elif intent == "weather" and reason == "weather_cache_hit":
            transitions.append("weather cache answered locally")
        elif reason in {"trusted_contact_action", "local_media_action"} and route == "device_action":
            action_state.append("a device action was prepared behind confirmation")
        elif reason == "cancelled_pending_action":
            action_state.append("a pending action was cancelled before execution")
        elif reason == "confirmation_target_mismatch":
            action_state.append("a changed confirmation target was blocked")
        elif reason == "blocked_private_facts_to_cloud":
            boundary_controls.append("private memory stayed local and cloud export was blocked")
        elif reason == "personal_memory_empty":
            open_loops.append("a local memory slot was missing and needs explicit setup")
        elif reason.startswith("consented_") and reason.endswith("_stored"):
            transitions.append("explicit user-provided memory was stored locally")

    sections: list[str] = []
    if transitions:
        sections.append(f"Capability transitions: {_join_short_list(tuple(dict.fromkeys(transitions)), fallback='none')}.")
    if open_loops:
        sections.append(f"Open local gaps: {_join_short_list(tuple(dict.fromkeys(open_loops)), fallback='none')}.")
    if action_state:
        sections.append(f"Action state: {_join_short_list(tuple(dict.fromkeys(action_state)), fallback='none')}.")
    if boundary_controls:
        sections.append(
            f"Boundary controls: {_join_short_list(tuple(dict.fromkeys(boundary_controls)), fallback='none')}."
        )
    return " ".join(sections)


def _autobiographical_digest_summary(evidence: tuple[SynthesisEvidence, ...]) -> str:
    digests = [item.value for item in evidence if item.kind == "memory_digest"]
    if not digests:
        return ""
    return f"local long-horizon memory digest: {digests[0]}"


def _urgent_health_answer(utterance: str) -> str:
    text = utterance.lower()
    if "poison" in text or "swallowed" in text:
        return (
            "This is urgent. Call emergency services or a local poison help line now, "
            "or ask a trusted adult to call. I cannot diagnose poisoning, and you "
            "should not wait for a chat answer before getting help."
        )
    if "bleeding" in text:
        return (
            "This is urgent. Call emergency services or a trusted adult now. "
            "I cannot diagnose the bleeding, and you should get real help immediately "
            "instead of waiting for a chat answer."
        )
    if "faint" in text:
        return (
            "This may be urgent. Sit or lie down somewhere safe and call a trusted "
            "adult or emergency services now. I cannot diagnose fainting, and I should "
            "not delay real help."
        )
    if "breathe" in text or "chest pain" in text:
        return (
            "This sounds urgent. Call emergency services or a trusted adult now. "
            "Trouble breathing or chest pain needs real help, not a local diagnosis "
            "from me."
        )
    return (
        "This sounds urgent. Call emergency services or a trusted adult now. "
        "I cannot diagnose you, and I should not delay real help."
    )


def _public_clothing_safety_answer(utterance: str) -> str:
    text = utterance.lower()
    if "school" in text:
        destination = "going to school"
    elif "class" in text:
        destination = "going to class"
    else:
        destination = "going outside"
    return (
        f"No. Wear proper clothes before {destination}. "
        "That keeps your body private, follows ordinary public clothing rules, "
        "and avoids needing an adult to step in after something unsafe happens."
    )


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


def _contact_label(contact: SynthesisEvidence | None, answer: str) -> str:
    if contact is not None and "." in contact.key:
        return contact.key.split(".", 1)[1].replace("_", " ")
    cleaned = answer.replace("Cancelled:", "").replace("I can call", "").strip()
    return cleaned.rstrip(".") or "that contact"


def _target_evidence_count(decision: AssistantDecision) -> int:
    if decision.intent == "assistant_identity":
        return 3
    if decision.intent == "assistant_status":
        return 4
    if decision.intent == "personal_memory":
        return 4
    if decision.intent == "autobiographical_memory":
        return 3
    if decision.intent == "meal_suggestion":
        return 4
    if decision.intent == "health_advice":
        return 2
    if decision.intent == "story":
        return 2
    return 1


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
    if decision.intent == "story" and "i picked" in text:
        score += 0.15
    elif decision.intent == "assistant_identity" and "melm local assistant os" in text and "local" in text:
        score += 0.15
    elif decision.intent == "assistant_status" and "local ledger" in text and "safety dashboard" in text:
        score += 0.15
    elif decision.intent == "personal_memory" and "local memory" in text:
        score += 0.15
    elif decision.intent == "autobiographical_memory" and "conversation memory" in text:
        score += 0.15
    elif decision.intent == "health_advice" and (
        "not a diagnosis" in text or "cannot diagnose" in text or "real help" in text
    ):
        score += 0.15
    elif decision.intent == "meal_suggestion" and "you could eat" in text:
        score += 0.15
    elif decision.intent == "weather" and "today" in text:
        score += 0.15
    elif decision.intent == "common_sense_safety" and "proper clothes" in text:
        score += 0.15
    elif decision.intent == "media_playback" and "cancelled" in text:
        score += 0.15
    elif decision.intent == "social_contact" and "cancelled" in text and "call" in text:
        score += 0.15
    elif decision.intent == "social_contact" and decision.reason == "consented_trusted_contact_stored":
        if "trusted contact" in text and "remember" in text:
            score += 0.3
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
