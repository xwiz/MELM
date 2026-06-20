"""Bounded local synthesis for the Local Assistant OS MVP.

The synthesizer is intentionally deterministic and small. It does not try to be
an open-ended language model; it turns an already routed, membrane-approved
decision plus its evidence keys into a local answer trace with citations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from melm.contracts import load_answer_templates, load_assistant_identity, load_health_disclaimers, load_prompt_seeds, load_safety_policies
from .assistant_skill_research import ResearchProvider

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
from .assistant_skill_story_planning import plan_story
from .assistant_story_prompt_pipeline import StoryPromptPipeline
from .local_assistant_router import AssistantDecision, LocalAssistantProfile
from melm.appliance.reasoning.implications import MoralContext


SYNTHESIZABLE_ROUTES = {"local_answer", "cached_tool"}
SYNTHESIS_QUALITY_FLOOR = 0.65

# Cached contract data for moral cognition (loaded once, reused across turns)
_VERB_STATES_CACHE: dict | None = None
_VALENCE_DATA_CACHE: dict | None = None


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
    evidence_set = set(decision.evidence_keys) if decision.evidence_keys else set()

    if "identity_action:suggest_name" in evidence_set:
        return _handle_identity_name_suggestion(self, decision, evidence)

    if "identity_action:name_awareness" in evidence_set:
        return _handle_identity_name_awareness(self, decision, evidence)

    if "identity_action:name_origin" in evidence_set:
        return _handle_identity_name_origin(self, decision, evidence)

    if "identity_action:explain_identity" in evidence_set:
        return _handle_identity_explain(self, decision, evidence)

    return _handle_identity_derived(self, decision, evidence)


def _handle_identity_derived(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    try:
        if self.store is not None:
            from .assistant_skill_self_identity import analyze_user_identity, derive_identity_narrative
            user_id = getattr(self.profile, "user_id", "default")
            identity = analyze_user_identity(self.store, user_id)
            if identity is not None:
                mood_id = "neutral"
                if decision.session_mood is not None:
                    mood_id = getattr(decision.session_mood, "mood_id", "neutral")
                narrative = derive_identity_narrative(identity, mood_id)
                if narrative is not None:
                    return narrative
    except Exception:
        pass
    summary = _assistant_identity_summary(evidence)
    return summary if summary else None


def _handle_identity_name_suggestion(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    try:
        if self.store is not None:
            from .assistant_skill_self_identity import analyze_user_identity, get_name_awareness_template
            user_id = getattr(self.profile, "user_id", "default")
            identity = analyze_user_identity(self.store, user_id)
            if identity is not None:
                result = get_name_awareness_template(identity, "name_suggestion")
                if result is not None:
                    return result
    except Exception:
        pass
    summary = _assistant_identity_summary(evidence)
    return summary if summary else None


def _handle_identity_name_awareness(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    try:
        if self.store is not None:
            from .assistant_skill_self_identity import analyze_user_identity, get_name_awareness_template
            user_id = getattr(self.profile, "user_id", "default")
            identity = analyze_user_identity(self.store, user_id)
            if identity is not None:
                key = "has_name" if identity.has_name else "no_name"
                result = get_name_awareness_template(identity, key)
                if result is not None:
                    return result
    except Exception:
        pass
    summary = _assistant_identity_summary(evidence)
    return summary if summary else None


def _handle_identity_name_origin(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    return "My name was given to me. It's part of who I am now."


def _handle_identity_explain(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    try:
        if self.store is not None:
            from .assistant_skill_self_identity import analyze_user_identity, derive_identity_explanation
            user_id = getattr(self.profile, "user_id", "default")
            identity = analyze_user_identity(self.store, user_id)
            if identity is not None:
                result = derive_identity_explanation(identity)
                if result is not None:
                    return result
    except Exception:
        pass
    return "I see myself based on the things you ask me to do most often, especially the conversations that feel most meaningful."


def _handle_identity_name_given(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    try:
        if self.store is not None:
            state = self.store.load_self_state() if hasattr(self.store, "load_self_state") else {}
            given_name = state.get("given_name", None)
            if given_name:
                return f"You can call me {given_name}. That's the name you gave me."
    except Exception:
        pass
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


def _handle_social_greeting(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    """Handle social_greeting intents with optional context injection."""
    pool_result = _pool_select(decision, self.profile)
    if pool_result is not None:
        return pool_result
    return None


def _handle_common_sense_safety(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    weather = _first_kind(evidence, "weather")
    if weather is not None and "school_clothing_weather_policy" in decision.reason:
        template = _load_answer_template(decision.intent, "school_clothing_weather_policy")
        return template if template else "Wear school clothes and carry rain protection because the forecast mentions rain."
    if decision.reason == "local_common_sense_policy":
        return _public_clothing_safety_answer(decision.utterance)
    if decision.reason == "safety_policy_triggered":
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
    # Guard: knowledge write/contradiction decisions already have their pre-rendered answer
    if decision.reason in ("knowledge_write", "knowledge_contradiction"):
        return None
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


def _handle_music_generation(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    """Generate MIDI music from the request description."""
    try:
        from .assistant_music_midi import MusicDescription, MidiRenderer
        desc = MusicDescription()
        text_lower = decision.utterance.lower()
        if "calm" in text_lower or "relaxing" in text_lower:
            desc = MusicDescription(genre="classical", mood="calm", tempo_bpm=80)
        elif "happy" in text_lower or "upbeat" in text_lower:
            desc = MusicDescription(genre="waltz", mood="happy", tempo_bpm=130)
        elif "jazz" in text_lower or "blues" in text_lower:
            desc = MusicDescription(genre="jazz", mood="calm", tempo_bpm=120)
        elif "sad" in text_lower or "melancholy" in text_lower:
            desc = MusicDescription(genre="classical", mood="sad", tempo_bpm=60, mode="minor")
        renderer = MidiRenderer()
        midi_bytes = renderer.render(desc)
        import tempfile, os
        out_dir = os.path.join(os.path.dirname(__file__), "..", "generated_music")
        os.makedirs(out_dir, exist_ok=True)
        counter = 1
        while os.path.exists(os.path.join(out_dir, f"{desc.genre}_{desc.mood}_{counter:03d}.mid")):
            counter += 1
        fname = f"{desc.genre}_{desc.mood}_{counter:03d}.mid"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(midi_bytes)
        return f"I composed a {desc.mood} {desc.genre} piece for you. Check {fname}"
    except Exception as exc:
        return f"I tried to compose some music but ran into an issue: {exc}"


def _handle_music_discovery(self: BoundedLocalSynthesizer, decision: AssistantDecision, evidence: tuple[SynthesisEvidence, ...]) -> str | None:
    """Handle music discovery response."""
    return decision.answer


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
    # Synthesis renders admitted evidence only — auto-research is owned solely
    # by the kernel (assistant_os_kernel.py decision-shaping). No fetching here.
    from .assistant_skill_research import (
        extract_topic, extract_action, format_open_domain_answer,
    )
    learned = [e for e in evidence if e.kind == "learned_fact"]
    if learned:
        # Kernel already populated the learned answer on the decision.
        return decision.answer
    topic = extract_topic(decision.functional_parse)
    action = extract_action(decision.functional_parse)
    return format_open_domain_answer(topic=topic or "that", action=action)


def _handle_unknown(
    synthesizer: "BoundedLocalSynthesizer",
    decision: AssistantDecision,
    evidence: tuple,
) -> str | None:
    # Synthesis renders admitted evidence only — auto-research is owned solely
    # by the kernel (assistant_os_kernel.py decision-shaping). No fetching here.
    from .assistant_skill_research import (
        extract_topic, extract_action, format_open_domain_answer,
    )
    # Social-pattern fallback: if utterance looks like a "how are you"
    # question despite UOL parse derailment (e.g. "lol, how are you?"),
    # answer with mood state instead of a generic handoff.
    from .local_assistant_router import _detect_social_status
    if _detect_social_status(decision.utterance):
        mood = decision.session_mood
        if mood and hasattr(mood, "mood_id") and mood.mood_id:
            mood_label = str(mood.mood_id).replace("_", " ").title()
            return f"I'm feeling {mood_label.lower()}. Thanks for asking."
        return "I'm doing well, thanks for asking."
    learned = [e for e in evidence if e.kind == "learned_fact"]
    if learned:
        # Kernel already populated the learned answer on the decision.
        return decision.answer
    topic = extract_topic(decision.functional_parse)
    action = extract_action(decision.functional_parse)
    return format_open_domain_answer(topic=topic or "that", action=action)


def _pool_select(
    decision: AssistantDecision,
    profile: LocalAssistantProfile,
) -> str | None:
    from datetime import datetime
    from melm.contracts import load_contract_json
    from .assistant_mood_engine import _build_pool_key, _resolve_template
    try:
        pools = load_contract_json("response_pools.v1.json")
    except Exception:
        return None
    mood = decision.session_mood
    mood_id = getattr(mood, "mood_id", "neutral") if mood is not None else "neutral"
    occ = decision.intent_occurrence
    familiarity = getattr(decision, "familiarity", 0)
    hour = datetime.now().hour
    sensory_tag = "generic"
    affect = getattr(decision, "utterance_affect", None)
    if affect is not None:
        tags = getattr(affect, "dominant_tags", None) or getattr(affect, "tags", None)
        if tags:
            sensory = [t for t in tags if t.startswith("sensory.")]
            if sensory:
                sensory_tag = sensory[0]
    key = _build_pool_key(decision.intent, mood, occ, str(familiarity), str(hour), sensory_tag)
    pool = _resolve_template(pools, key)
    if not pool:
        return None
    seed = _template_seed(decision, key)
    index = seed % len(pool)
    entry = pool[index]
    text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
    return text.format(name=profile.user_name, location=profile.location) if text else None


def _template_seed(
    decision: AssistantDecision,
    key: str,
) -> int:
    raw = f"{decision.intent}:{decision.reason}:{decision.utterance}"
    h = 0
    for ch in raw.encode("utf-8"):
        h = ((h << 5) - h) + ch
        h &= 0xFFFFFFFF
    return abs(h)


_ALWAYS_RESPOND_INTENTS: frozenset[str] = frozenset({
    "common_sense_safety", "health_advice", "social_greeting",
    "assistant_identity", "identity_switch", "identity_probe_detected",
    "music_generation", "music_discovery",
})

_SHORT_CIRCUIT_REASONS: frozenset[str] = frozenset({
    "complaint_acknowledged", "rapid_repetition", "perception_urgency_high",
    "identity_switch", "identity_probe_detected",
})

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
    "music_generation": _handle_music_generation,
    "music_discovery": _handle_music_discovery,
    "social_greeting": _handle_social_greeting,
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
    decoder_used: str = ""

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
            "decoder_used": self.decoder_used,
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
        research_provider: ResearchProvider | None = None,
    ) -> None:
        self.profile = profile
        self.store = store
        self.self_state = self_state or {}
        self.runtime_status = runtime_status or {}
        self.decoder = decoder
        self.research_provider = research_provider

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
                decoder_used="",
            )

        # Reasoning-layer outcomes are evidence-bound and resolved BEFORE the
        # generic no_bound_evidence gate (slice 4). A typed refusal renders a
        # clarification; a solver result renders its copy-slot answer.
        if decision.refusal_signal:
            return self._render_refusal(decision, boundary_crossed)
        if decision.reasoning_result is not None:
            return self._render_reasoning(decision, boundary_crossed)

        evidence = tuple(
            item
            for key in decision.evidence_keys
            for item in self._resolve_evidence(key)
        )
        if not evidence:
            return self._refused(decision, boundary_crossed, "no_bound_evidence")

        template_answer = self._answer(decision, evidence)
        template_answer = self._apply_creative_behaviors(decision, template_answer)
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
        answer, decoder_used = self._decode_verified(plan, evidence, decision, template_answer, packet)
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
            decoder_used=decoder_used,
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
    ) -> tuple[str, str]:
        """Return (answer, decoder_name).  decoder_name is 'template' when no backend is used."""
        if self.decoder is None:
            return template_answer, "template"
        evidence_entities = tuple(
            item.key for item in evidence
            if item.kind in ("story_model", "weather", "user_fact", "profile", "contact", "health_goal", "food_inventory", "preference")
        )

        # Look up contract prompt seed for model-preferred intents.
        template_hint = template_answer
        model_preferred: set[str] = set()
        try:
            seeds = load_prompt_seeds()
            model_preferred = set(seeds.get("model_preferred_intents", []))
            if decision.intent in model_preferred:
                seed = seeds.get("seeds", {}).get(decision.intent, {})
                if seed:
                    system = seed.get("system", "")
                    user_msg = decision.utterance
                    if decision.intent == "story":
                        personal_facts: tuple[str, ...] = ()
                        recent_context: tuple[str, ...] = ()
                        if self.store is not None:
                            try:
                                facts = self.store.find_entities(kind="user_fact")
                                personal_facts = tuple(
                                    getattr(f, "canonical_lemma", "") or getattr(f, "label", "")
                                    for f in facts[-5:] if getattr(f, "canonical_lemma", "") or getattr(f, "label", "")
                                )
                                memories = self.store.find_entities(kind="personal_experience")
                                recent_context = tuple(
                                    getattr(m, "label", "") for m in memories[-2:] if getattr(m, "label", "")
                                )
                            except Exception:
                                pass
                        valence = getattr(decision.session_mood, "valence", 0.0) if decision.session_mood else 0.0
                        arousal = getattr(decision.session_mood, "arousal", 0.0) if decision.session_mood else 0.0
                        story_plan = plan_story(
                            utterance=decision.utterance,
                            functional_parse=decision.functional_parse,
                            user_name=self.profile.user_name,
                            location=self.profile.location,
                            culture=self.profile.culture,
                            age=getattr(self.profile, "age", 0),
                            personal_facts=personal_facts,
                            recent_context=recent_context,
                            valence=valence,
                            arousal=arousal,
                        )
                        pipeline = StoryPromptPipeline()
                        template_hint = pipeline.build_string(story_plan)
                    else:
                        template_hint = f"{system}\n\n{user_msg}"
        except Exception:
            pass

        grammar = build_decoding_grammar(plan, template_hint, evidence_entities)

        # Hybrid dispatch: model-preferred intents get llamacpp first,
        # all others stay on template for speed / determinism.
        if decision.intent in model_preferred and "llamacpp" in self.decoder.available:
            previous = self.decoder.preferred()
            self.decoder.preferred("llamacpp")
            try:
                result = self.decoder.dispatch(plan, grammar)
                if result is not None and result.decoder != "template":
                    v = verify_answer(plan, result.answer, packet)
                    if v.passed:
                        answer = result.answer
                        if decision.intent == "story":
                            name = self.profile.user_name or "the character"
                            answer += f"\n\nWhat do you think {name} learned from this story?"
                        return answer, result.decoder
            finally:
                self.decoder.preferred(previous)

        result = self.decoder.dispatch(plan, grammar)
        if result is not None and result.decoder != "template":
            v = verify_answer(plan, result.answer, packet)
            if v.passed:
                answer = result.answer
                if decision.intent == "story":
                    name = self.profile.user_name or "the character"
                    answer += f"\n\nWhat do you think {name} learned from this story?"
                return answer, result.decoder
        return template_answer, "template"

    def _answer(
        self,
        decision: AssistantDecision,
        evidence: tuple[SynthesisEvidence, ...],
    ) -> str:
        mood = decision.session_mood
        affect = decision.utterance_affect
        if mood is not None:
            is_listening = getattr(mood, "is_listening", False)
            if is_listening and decision.intent not in _ALWAYS_RESPOND_INTENTS:
                return "..."
        pool_intents = {"social_greeting", "assistant_identity", "assistant_status"}
        if decision.reason in _SHORT_CIRCUIT_REASONS or decision.intent in pool_intents:
            pool_result = _pool_select(decision, self.profile)
            if pool_result is not None:
                return pool_result
        # Moral cognition check — short-circuit for urgent harm
        from melm.appliance.reasoning.implications import derive_moral_context
        global _VERB_STATES_CACHE, _VALENCE_DATA_CACHE
        if _VERB_STATES_CACHE is None:
            from melm.contracts import load_contract_json
            _VERB_STATES_CACHE = load_contract_json("verb_states.v1.json")
            _VALENCE_DATA_CACHE = load_contract_json("state_valences.v1.json").get("valences", {})
        from .local_assistant_router import _extract_verb
        verb = _extract_verb(uol_act=decision.uol_act)
        if verb:
            mc = derive_moral_context(verb, "person", _VERB_STATES_CACHE, _VALENCE_DATA_CACHE)
            if mc.harm_severity == "high":
                return _urgent_harm_answer(decision, mc)
        handler = _ANSWER_HANDLERS.get(decision.intent)
        if handler is not None:
            result = handler(self, decision, evidence)
            if result is not None:
                result = _enforce_response_mode(result, decision.session_mood)
                result = self._maybe_emoji(decision, result)
                return result
        fallback = _render_contract_answer(decision, evidence, self.profile)
        if fallback is None and decision.uol_act is not None:
            from .assistant_decoder_atom import AtomTemplateBackend
            _atom_gen = getattr(self, '_atom_backend', None)
            if _atom_gen is None:
                _atom_gen = AtomTemplateBackend()
                self._atom_backend = _atom_gen
            fallback = _atom_gen.generate(decision.intent, decision.uol_act, evidence)
        if fallback is None:
            fallback = decision.answer
        fallback = _enforce_response_mode(fallback, decision.session_mood)
        return self._maybe_emoji(decision, fallback)

    def _maybe_emoji(self, decision: AssistantDecision, answer: str) -> str:
        """Prepend mood emoji if capability enabled (default off for portability)."""
        from .local_assistant_router import _capability_flag
        if _capability_flag("mood_affect", "mood_emoji", False):
            emoji_map = {
                "neutral": "\U0001F610",
                "calm": "\U0001F60C",
                "happy": "\U0001F60A",
                "excited": "\U0001F929",
                "curious": "\U0001F914",
                "annoyed": "\U0001F612",
                "frustrated": "\U0001F624",
                "hurt": "\U0001F622",
                "sad": "\U0001F61E",
                "listening": "\U0001F442",
            }
            mood_id = getattr(decision.session_mood, "mood_id", "neutral") if decision.session_mood else "neutral"
            emoji = emoji_map.get(mood_id, "\U0001F610")
            return f"{emoji} {answer}"
        return answer

    def _get_behavior_engine(self):
        """Lazily create the BehaviorEngine, homed on the store so cooldown
        state persists across turns (the synthesizer is rebuilt per turn)."""
        if self.store is None:
            return None
        engine = getattr(self.store, "_behavior_engine", None)
        if engine is None:
            from .assistant_behavior_engine import BehaviorEngine
            engine = BehaviorEngine()
            self.store._behavior_engine = engine
        return engine

    def _apply_creative_behaviors(self, decision: AssistantDecision, answer: str) -> str:
        """Mood-gated post-processor (capability-gated, off by default).

        Zero overhead when disabled: the gate is checked before any context is
        built or engine created. Reasoning results / refusals are protected from
        replacement or truncation.
        """
        from .local_assistant_router import _capability_flag
        if not _capability_flag("mood_affect", "creative_behaviors", False):
            return answer
        engine = self._get_behavior_engine()
        if engine is None:
            return answer
        from .assistant_behavior_engine import apply_behaviors, build_behavior_context
        results = engine.evaluate(build_behavior_context(decision))
        if not results:
            return answer
        protect = bool(
            getattr(decision, "reasoning_result", None)
            or getattr(decision, "refusal_signal", None)
        )
        return apply_behaviors(answer, results, protect=protect)

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

        if key.startswith("deferred_task."):
            entity_id = key.split(".", 1)[1]
            value = ""
            if self.store is not None:
                slot = self.store.get_entity_slot(entity_id, "topic")
                if slot is not None and slot.value_json:
                    value = str(json.loads(slot.value_json))
            if value:
                return (SynthesisEvidence(
                    key=key, kind="deferred_task", value=value,
                    source="entity_store", license="private_local", local_only=True,
                ),)

        if key.startswith("novelty_candidate."):
            entity_id = key.split(".", 1)[1]
            value = ""
            if self.store is not None:
                slot = self.store.get_entity_slot(entity_id, "surface_form")
                if slot is not None and slot.value_json:
                    value = str(json.loads(slot.value_json))
            if value:
                return (SynthesisEvidence(
                    key=key, kind="novelty_candidate", value=value,
                    source="entity_store", license="private_local", local_only=True,
                ),)

        if key.startswith("user_commitment."):
            entity_id = key.split(".", 1)[1]
            value = ""
            if self.store is not None:
                slot = self.store.get_entity_slot(entity_id, "topic")
                if slot is not None and slot.value_json:
                    value = str(json.loads(slot.value_json))
            if value:
                return (SynthesisEvidence(
                    key=key, kind="user_commitment", value=value,
                    source="entity_store", license="private_local", local_only=True,
                ),)

        if key.startswith("epistemic_state."):
            entity_id = key.split(".", 1)[1]
            value = ""
            if self.store is not None:
                slot = self.store.get_entity_slot(entity_id, "state_type")
                if slot is not None and slot.value_json:
                    value = str(json.loads(slot.value_json))
            if value:
                return (SynthesisEvidence(
                    key=key, kind="epistemic_state", value=value,
                    source="entity_store", license="private_local", local_only=True,
                ),)

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

    def _bounded_reasoning_result(
        self,
        decision: AssistantDecision,
        boundary_crossed: str,
        answer: str,
        evidence: tuple[SynthesisEvidence, ...],
        reason: str,
    ) -> BoundedSynthesisResult:
        """Build an applied result for a reasoning answer/refusal, reusing the
        bounded-synthesis quality + citation machinery."""
        citations = tuple(item.key for item in evidence)
        return BoundedSynthesisResult(
            route=decision.route,
            applied=True,
            refused=False,
            answer=answer,
            citations=citations,
            evidence=evidence,
            admitted_evidence_count=len(evidence),
            reason=reason,
            boundary_crossed=boundary_crossed,
            quality=_synthesis_quality(
                decision,
                applied=True,
                refused=False,
                answer=answer,
                citations=citations,
                evidence=evidence,
                reason=reason,
                boundary_crossed=boundary_crossed,
            ),
            decoder_used="",
        )

    def _render_reasoning(
        self, decision: AssistantDecision, boundary_crossed: str,
    ) -> BoundedSynthesisResult:
        result = decision.reasoning_result or {}
        # The solver/router pre-renders the faithful copy-slot answer onto the
        # decision; the structured result is bound as evidence for citation.
        answer = decision.answer or str(result.get("answer", ""))
        evidence = (
            SynthesisEvidence(
                key="reasoning.result",
                kind="reasoning",
                value=json.dumps(result, sort_keys=True, default=str),
                source="local_reasoner",
                license="local",
                local_only=True,
            ),
        )
        answer = self._apply_creative_behaviors(decision, answer)  # protected
        task = str(result.get("task", "")) if isinstance(result, dict) else ""
        return self._bounded_reasoning_result(
            decision, boundary_crossed, answer, evidence, reason=f"reasoning:{task}",
        )

    def _render_refusal(
        self, decision: AssistantDecision, boundary_crossed: str,
    ) -> BoundedSynthesisResult:
        answer = decision.answer or "I need a bit more information to answer that."
        evidence = (
            SynthesisEvidence(
                key="reasoning.refusal",
                kind="clarification",
                value=str(decision.refusal_signal),
                source="local_reasoner",
                license="local",
                local_only=True,
            ),
        )
        answer = self._apply_creative_behaviors(decision, answer)  # protected
        return self._bounded_reasoning_result(
            decision, boundary_crossed, answer, evidence,
            reason=f"reasoning_refusal:{decision.refusal_signal}",
        )

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
            decoder_used="",
        )


def _enforce_response_mode(answer: str, mood: Any) -> str:
    """Truncate answer to response_mode.max_words from mood_states contract."""
    if mood is None:
        return answer
    mode = getattr(mood, "response_mode", "normal")
    try:
        from melm.contracts import load_contract_json
        states = load_contract_json("mood_states.v1.json")
        modes_cfg = states.get("response_modes", {})
        mode_cfg = modes_cfg.get(mode, {})
        max_words = mode_cfg.get("max_words", 0)
        if max_words and len(answer.split()) > max_words:
            words = answer.split()
            return " ".join(words[:max_words]) + "..."
    except Exception:
        pass
    return answer


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


def _urgent_harm_answer(decision: AssistantDecision, ctx: MoralContext) -> str:
    """Short-circuit answer for urgent harm detected by implication engine."""
    if ctx.consent_status == "not_consented":
        return ("I understand you're describing a situation that could cause "
                "harm. Please don't act on this — talk to someone you trust instead.")
    if ctx.harm_severity == "high":
        return ("That sounds concerning. Please get help from a trusted adult "
                "or professional right away.")
    return ("Please be careful — that could cause harm. "
            "It's important to treat others with respect.")


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
