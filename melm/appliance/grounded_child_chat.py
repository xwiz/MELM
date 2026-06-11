"""Bounded grounded-chat micro-MVP for a five-year-old-scale world.

This module is intentionally tiny in world scope, but not in architecture. It
keeps the same joints a scaled MELM appliance would need:

UOL parser -> semantic atlas -> typed frame -> bound plan -> state algebra ->
Memory OS -> evidence admission -> hybrid SSM/attention SLM boundary.

The implementation is deterministic so tests can prove the contract. The
interfaces are deliberately separable so each micro component can later be
replaced by a learned parser, a larger atlas, indexed memory, or a real hybrid
SSM/attention model without changing the proof shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from typing import Literal

from melm.grounding import DEFAULT_CONTRADICTIONS, StateSet
from melm.memory import Event, EventMemory, ObjectLocationTracker


DIMENSIONS = ("physical", "emotional", "positional", "mental")
Purpose = Literal["inform", "question", "command"]
Route = Literal["state_transition", "memory_write", "memory_read", "evidence_check", "reject"]
Status = Literal["accepted", "answered", "abstained", "rejected"]
PlanKind = Literal["state_transition", "memory_write", "memory_read", "evidence_check"]


@dataclass(frozen=True)
class ActionSpec:
    """Micro action schema used by parsing, planning, and capability probes."""

    canonical: str
    verbs: tuple[str, ...]
    purposes: tuple[Purpose, ...]
    route: Route
    required_slots: tuple[str, ...]
    base_complexity: float


@dataclass(frozen=True)
class NounMention:
    canonical: str
    surface: str
    start: int
    end: int


@dataclass(frozen=True)
class ParseCandidate:
    uol: UOLSentence
    score: float
    complexity: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class StageTrace:
    """One inspectable pipeline stage result."""

    stage: str
    status: str
    detail: str


@dataclass(frozen=True)
class RejectionPacket:
    """Typed fail-closed packet, mirroring db-claw's rejection handoff idea."""

    code: str
    detail: str
    stage: str
    schema: str = "melm.child_chat_rejection.v1"


@dataclass(frozen=True)
class UOLSentence:
    """A tiny Unified Object Language sentence object."""

    raw_text: str
    purpose: Purpose
    action: str
    subject: str | None = None
    object: str | None = None
    source: str | None = None
    target: str | None = None
    qualifier: str | None = None
    parse_score: float = 1.0
    parse_complexity: float = 0.0
    parse_notes: tuple[str, ...] = ()
    schema: str = "melm.uol_sentence.v1"


@dataclass(frozen=True)
class StatePatch:
    """State transition with explicit remove/add phases."""

    object_name: str
    required: StateSet = field(default_factory=StateSet)
    remove: StateSet = field(default_factory=StateSet)
    add: StateSet = field(default_factory=StateSet)


@dataclass(frozen=True)
class StatePatchResult:
    valid: bool
    next_state: StateSet | None
    missing: dict[str, set[str]]
    conflicts: list[tuple[str, str, str]]


@dataclass(frozen=True)
class AtlasResolution:
    """Grounded noun/action resolution against the micro semantic atlas."""

    uol: UOLSentence
    subject: str | None
    object: str | None
    source: str | None
    target: str | None
    action: str
    rejection: RejectionPacket | None = None
    schema: str = "melm.child_atlas_resolution.v1"


@dataclass(frozen=True)
class ChatFrame:
    """Typed frame between language understanding and executable planning."""

    uol: UOLSentence
    route: Route
    schema: str = "melm.child_chat_frame.v1"
    needs_evidence: bool = False
    rejection_code: str | None = None
    rejection_detail: str | None = None


@dataclass(frozen=True)
class BoundChildPlan:
    """Validated executable plan before any state/memory side effect."""

    kind: PlanKind
    frame: ChatFrame
    response_intent: str
    state_patch: StatePatch | None = None
    object_name: str | None = None
    target: str | None = None
    actor: str | None = None
    expected_action: str | None = None
    requires_evidence: bool = False
    parse_score: float = 1.0
    complexity_score: float = 0.0
    schema: str = "melm.bound_child_plan.v1"


@dataclass(frozen=True)
class StateProjection:
    """Memory OS projection used as the SSM-like recurrent state."""

    lines: tuple[str, ...]
    schema: str = "melm.child_state_projection.v1"


@dataclass(frozen=True)
class EvidencePacket:
    """Evidence admitted by Memory OS for the attention path."""

    admitted: bool
    events: tuple[Event, ...] = ()
    answer_fact: str | None = None
    confidence: float = 0.0
    reason: str = ""
    packed_context: str = ""
    matching_event_count: int = 0
    evidence_budget: int | None = None
    schema: str = "melm.child_evidence_packet.v1"

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)


@dataclass(frozen=True)
class MicroSsmState:
    """Tiny recurrent state produced from Memory OS state projection."""

    lines: tuple[str, ...]
    state_hash: str
    schema: str = "melm.micro_ssm_state.v1"


@dataclass(frozen=True)
class MicroAttentionSlice:
    """Tiny attention slice over the current admitted evidence packet."""

    event_ids: tuple[str, ...]
    context: str
    schema: str = "melm.micro_attention_slice.v1"


@dataclass(frozen=True)
class HybridSlmInput:
    """The only payload the final verbalizer is allowed to see."""

    schema: str
    model_family: str
    compact_state: tuple[str, ...]
    attended_evidence_event_ids: tuple[str, ...]
    response_intent: str
    ssm_state: MicroSsmState
    attention: MicroAttentionSlice
    bound_plan_schema: str
    matching_evidence_count: int = 0
    evidence_budget: int | None = None


@dataclass(frozen=True)
class GroundedChildResponse:
    query: str
    status: Status
    answer: str
    frame: ChatFrame
    bound_plan: BoundChildPlan | None = None
    evidence_event_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    parse_score: float = 0.0
    complexity_score: float = 0.0
    slm_input: HybridSlmInput | None = None
    rejection_packet: RejectionPacket | None = None
    trace: tuple[StageTrace, ...] = ()


@dataclass(frozen=True)
class CapabilityResult:
    utterance: str
    status: Status
    route: Route
    reason: str
    parse_score: float
    complexity_score: float
    evidence_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityCase:
    utterance: str
    setup: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityReport:
    cases: int
    accepted: int
    answered: int
    abstained: int
    rejected: int
    average_parse_score: float
    average_complexity: float
    max_complexity: float
    by_reason: dict[str, int]
    results: tuple[CapabilityResult, ...]


@dataclass(frozen=True)
class EvidencePolicy:
    """Evidence admission budget for attention-side context."""

    action_top_k: int | None = None
    prefer_latest: bool = True


@dataclass(frozen=True)
class ContextBudgetResult:
    query_kind: str
    query: str
    move_count: int
    status: Status
    raw_transcript_chars: int
    unbudgeted_payload_chars: int
    budgeted_payload_chars: int
    unbudgeted_evidence_count: int
    budgeted_evidence_count: int
    matching_evidence_count: int
    unbudgeted_compression_ratio: float
    budgeted_compression_ratio: float


@dataclass(frozen=True)
class ContextBudgetReport:
    evidence_top_k: int
    move_counts: tuple[int, ...]
    results: tuple[ContextBudgetResult, ...]


class ChildWorldAtlas:
    """Micro semantic atlas: bounded nouns, actions, aliases, and action frames."""

    def __init__(
        self,
        *,
        people: set[str] | None = None,
        objects: set[str] | None = None,
        aliases: dict[str, str] | None = None,
        object_roles: dict[str, str] | None = None,
    ) -> None:
        self.people = people or {"maya", "leo"}
        self.objects = objects or {"red block", "blue box", "green basket"}
        self.aliases = aliases or {
            "box": "blue box",
            "basket": "green basket",
            "block": "red block",
        }
        self.object_roles = object_roles or {
            "red block": "toy",
            "blue box": "container",
            "green basket": "container",
        }
        self.action_specs = {
            spec.canonical: spec
            for spec in (
                ActionSpec("open", ("open", "opened"), ("inform", "command"), "state_transition", ("object",), 0.20),
                ActionSpec("close", ("close", "closed"), ("inform", "command"), "state_transition", ("object",), 0.20),
                ActionSpec("put", ("put",), ("inform",), "memory_write", ("subject", "object", "target"), 0.35),
                ActionSpec("move", ("move", "moved"), ("inform",), "memory_write", ("subject", "object", "target"), 0.45),
                ActionSpec("where", ("where",), ("question",), "memory_read", ("object",), 0.55),
                ActionSpec("did_move", ("did", "move"), ("question",), "evidence_check", ("subject", "object", "target"), 0.70),
            )
        }
        self.actions = set(self.action_specs)

    def resolve(self, uol: UOLSentence) -> AtlasResolution:
        subject = self._resolve_subject(uol.subject)
        if subject is None and uol.subject not in {None, "user"}:
            return self._rejected(uol, "unknown_subject", uol.subject or "")
        action = uol.action
        if action not in self.actions:
            return self._rejected(uol, "unknown_action", action)
        resolved_object = self._resolve_object(uol.object)
        if resolved_object is None and uol.object:
            return self._rejected(uol, "unknown_object", uol.object)
        resolved_source = self._resolve_object(uol.source)
        if resolved_source is None and uol.source:
            return self._rejected(uol, "unknown_object", uol.source)
        resolved_target = self._resolve_object(uol.target)
        if resolved_target is None and uol.target:
            return self._rejected(uol, "unknown_object", uol.target)
        if action in {"put", "move"} and resolved_target is None:
            return self._rejected(uol, "missing_target", uol.raw_text)
        return AtlasResolution(
            uol=UOLSentence(
                raw_text=uol.raw_text,
                purpose=uol.purpose,
                action=uol.action,
                subject=subject or uol.subject,
                object=resolved_object,
                source=resolved_source,
                target=resolved_target,
                qualifier=uol.qualifier,
                parse_score=uol.parse_score,
                parse_complexity=uol.parse_complexity,
                parse_notes=uol.parse_notes,
            ),
            subject=subject,
            object=resolved_object,
            source=resolved_source,
            target=resolved_target,
            action=action,
        )

    def state_patch_for(self, uol: UOLSentence) -> StatePatch:
        if not uol.object:
            return StatePatch(object_name="")
        if uol.action == "open":
            return StatePatch(
                object_name=uol.object,
                required=StateSet.from_mapping({"physical": ["closed"]}),
                remove=StateSet.from_mapping({"physical": ["closed"]}),
                add=StateSet.from_mapping({"physical": ["open"]}),
            )
        if uol.action == "close":
            return StatePatch(
                object_name=uol.object,
                required=StateSet.from_mapping({"physical": ["open"]}),
                remove=StateSet.from_mapping({"physical": ["open"]}),
                add=StateSet.from_mapping({"physical": ["closed"]}),
            )
        return StatePatch(object_name=uol.object)

    def _resolve_subject(self, value: str | None) -> str | None:
        if value is None:
            return None
        return value if value in self.people or value == "user" else None

    def _resolve_object(self, value: str | None) -> str | None:
        if value is None:
            return None
        value = self.aliases.get(value, value)
        return value if value in self.objects else None

    def noun_surface_map(self) -> dict[str, str]:
        surfaces = {canonical: canonical for canonical in self.objects}
        surfaces.update(self.aliases)
        return surfaces

    def is_container(self, object_name: str | None) -> bool:
        return bool(object_name and self.object_roles.get(object_name) == "container")

    def is_movable(self, object_name: str | None) -> bool:
        return bool(object_name and self.object_roles.get(object_name) in {"toy", "container"})

    def _rejected(self, uol: UOLSentence, code: str, detail: str) -> AtlasResolution:
        return AtlasResolution(
            uol=uol,
            subject=uol.subject,
            object=uol.object,
            source=uol.source,
            target=uol.target,
            action=uol.action,
            rejection=RejectionPacket(code=code, detail=detail, stage="semantic_atlas"),
        )


class ChildFramePlanner:
    """Builds frames and bound plans. This is the db-claw-shaped contract."""

    def __init__(self, atlas: ChildWorldAtlas) -> None:
        self.atlas = atlas

    def frame(self, resolution: AtlasResolution) -> ChatFrame:
        if resolution.rejection is not None:
            return ChatFrame(
                uol=resolution.uol,
                route="reject",
                rejection_code=resolution.rejection.code,
                rejection_detail=resolution.rejection.detail,
            )
        uol = resolution.uol
        if uol.action in {"open", "close"}:
            return ChatFrame(uol=uol, route="state_transition")
        if uol.action in {"put", "move"} and uol.purpose == "inform":
            return ChatFrame(uol=uol, route="memory_write")
        if uol.action == "where":
            return ChatFrame(uol=uol, route="memory_read", needs_evidence=True)
        if uol.action == "did_move":
            return ChatFrame(uol=uol, route="evidence_check", needs_evidence=True)
        return ChatFrame(
            uol=uol,
            route="reject",
            rejection_code="unsupported_action",
            rejection_detail=uol.action,
        )

    def bind(self, frame: ChatFrame) -> BoundChildPlan | RejectionPacket:
        if frame.route == "reject":
            return RejectionPacket(
                code=frame.rejection_code or "frame_rejected",
                detail=frame.rejection_detail or "frame rejected",
                stage="frame_build",
            )
        uol = frame.uol
        if frame.route == "state_transition":
            if not self.atlas.is_container(uol.object):
                return RejectionPacket(
                    code="unsupported_object_action",
                    detail=f"{uol.action}:{uol.object}",
                    stage="plan_bind",
                )
            return BoundChildPlan(
                kind="state_transition",
                frame=frame,
                response_intent="state_update",
                state_patch=self.atlas.state_patch_for(uol),
                object_name=uol.object,
                actor=uol.subject,
                parse_score=uol.parse_score,
                complexity_score=_plan_complexity(uol, frame.route),
            )
        if frame.route == "memory_write":
            if not self.atlas.is_movable(uol.object):
                return RejectionPacket(
                    code="unsupported_object_action",
                    detail=f"{uol.action}:{uol.object}",
                    stage="plan_bind",
                )
            if not self.atlas.is_container(uol.target):
                return RejectionPacket(
                    code="unsupported_target",
                    detail=uol.target or "",
                    stage="plan_bind",
                )
            return BoundChildPlan(
                kind="memory_write",
                frame=frame,
                response_intent="memory_write",
                object_name=uol.object,
                target=uol.target,
                actor=uol.subject,
                parse_score=uol.parse_score,
                complexity_score=_plan_complexity(uol, frame.route),
            )
        if frame.route == "memory_read":
            return BoundChildPlan(
                kind="memory_read",
                frame=frame,
                response_intent="location_answer",
                object_name=uol.object,
                requires_evidence=True,
                parse_score=uol.parse_score,
                complexity_score=_plan_complexity(uol, frame.route),
            )
        if frame.route == "evidence_check":
            return BoundChildPlan(
                kind="evidence_check",
                frame=frame,
                response_intent="evidence_check_answer",
                object_name=uol.object,
                target=uol.target,
                actor=uol.subject,
                expected_action="moved",
                requires_evidence=True,
                parse_score=uol.parse_score,
                complexity_score=_plan_complexity(uol, frame.route),
            )
        return RejectionPacket(
            code="unsupported_route",
            detail=frame.route,
            stage="plan_bind",
        )


class ChildStateAlgebra:
    """Small state algebra under test, not a trusted nameless_vector port."""

    def apply(self, current: StateSet, patch: StatePatch) -> StatePatchResult:
        return apply_state_patch(current, patch)


class ChildMemoryOS:
    """Micro Memory OS: event store, current-state projection, evidence admission."""

    def __init__(
        self,
        *,
        initial_states: dict[str, StateSet] | None = None,
        state_algebra: ChildStateAlgebra | None = None,
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        self.object_states = initial_states or {
            "blue box": StateSet.from_mapping({"physical": ["closed"]}),
            "green basket": StateSet.from_mapping({"physical": ["open"]}),
        }
        self.event_memory = EventMemory()
        self.state_algebra = state_algebra or ChildStateAlgebra()
        self.evidence_policy = evidence_policy or EvidencePolicy()
        self._event_counter = 0

    def validate_state_transition(self, plan: BoundChildPlan) -> StatePatchResult:
        if plan.state_patch is None:
            return StatePatchResult(False, None, {"plan": {"missing state patch"}}, [])
        current = self.object_states.get(plan.state_patch.object_name, StateSet())
        return self.state_algebra.apply(current, plan.state_patch)

    def validate_memory_write(self, plan: BoundChildPlan) -> RejectionPacket | None:
        if not plan.object_name or not plan.target:
            return RejectionPacket("missing_write_slot", plan.frame.uol.raw_text, "memory_os")
        target_state = self.object_states.get(plan.target, StateSet())
        if "closed" in target_state.physical:
            return RejectionPacket("target_closed", plan.target, "memory_os")
        return None

    def commit_state_transition(self, plan: BoundChildPlan, result: StatePatchResult) -> Event:
        if plan.state_patch is None or result.next_state is None:
            raise ValueError("cannot commit invalid state transition")
        self.object_states[plan.state_patch.object_name] = result.next_state
        self._event_counter += 1
        uol = plan.frame.uol
        event = Event(
            event_id=f"child_e{self._event_counter}",
            source_span=uol.raw_text,
            time_index=self._event_counter,
            actors=(uol.subject,) if uol.subject else (),
            action_or_state=uol.action,
            objects=(uol.object,) if uol.object else (),
            metadata={"state_object": uol.object or "", "state_after": uol.action},
        )
        self.event_memory.add(event)
        return event

    def commit_memory_write(self, plan: BoundChildPlan) -> Event:
        self._event_counter += 1
        uol = plan.frame.uol
        previous_location = None
        if uol.action == "move" and uol.object:
            previous = self.location_tracker().resolve_location(uol.object)
            previous_location = previous.location if previous else uol.source
        event = Event(
            event_id=f"child_e{self._event_counter}",
            source_span=uol.raw_text,
            time_index=self._event_counter,
            actors=(uol.subject,) if uol.subject else (),
            action_or_state="moved" if uol.action == "move" else "put",
            objects=_event_objects(uol, previous_location=previous_location),
            location=uol.target,
            metadata={
                "state_object": uol.object or "",
                "location_after": uol.target or "",
            },
        )
        self.event_memory.add(event)
        return event

    def admit_evidence(self, plan: BoundChildPlan) -> EvidencePacket:
        if plan.kind == "memory_read":
            return self._location_evidence(plan)
        if plan.kind == "evidence_check":
            return self._action_evidence(plan)
        return EvidencePacket(
            admitted=False,
            reason="plan_does_not_require_evidence",
            packed_context="",
        )

    def event_packet(self, event: Event, *, reason: str) -> EvidencePacket:
        return EvidencePacket(
            admitted=True,
            events=(event,),
            confidence=1.0,
            reason=reason,
            packed_context=event.source_span,
            matching_event_count=1,
        )

    def project_state(self) -> StateProjection:
        state_lines = []
        known_objects = set(self.object_states)
        for event in self.event_memory.events():
            known_objects.update(event.objects)
        for object_name in sorted(known_objects):
            state = self.object_states.get(object_name)
            if state and state.physical:
                state_lines.append(f"{object_name}:physical={','.join(sorted(state.physical))}")
        tracker = self.location_tracker()
        for object_name in sorted(known_objects):
            observation = tracker.resolve_location(object_name)
            if observation:
                state_lines.append(f"{object_name}:location={observation.location}")
        return StateProjection(lines=tuple(state_lines))

    def location_tracker(self) -> ObjectLocationTracker:
        return ObjectLocationTracker(list(self.event_memory.events()))

    def events(self) -> tuple[Event, ...]:
        return self.event_memory.events()

    def get(self, event_id: str) -> Event | None:
        return self.event_memory.get(event_id)

    def _location_evidence(self, plan: BoundChildPlan) -> EvidencePacket:
        if not plan.object_name:
            return EvidencePacket(admitted=False, reason="missing_object")
        observation = self.location_tracker().resolve_location(plan.object_name)
        if observation is None:
            return EvidencePacket(admitted=False, reason="no_location_observation")
        event = self.get(observation.event_id)
        if event is None:
            return EvidencePacket(admitted=False, reason="missing_evidence_event")
        return EvidencePacket(
            admitted=True,
            events=(event,),
            answer_fact=observation.location,
            confidence=1.0,
            reason="latest_location_projection",
            packed_context=event.source_span,
            matching_event_count=1,
        )

    def _action_evidence(self, plan: BoundChildPlan) -> EvidencePacket:
        matches = []
        for event in self.events():
            if plan.actor and plan.actor not in event.actors:
                continue
            if plan.expected_action and event.action_or_state != plan.expected_action:
                continue
            if plan.object_name and plan.object_name not in event.objects:
                continue
            if plan.target and event.metadata.get("location_after") != plan.target:
                continue
            matches.append(event)
        if not matches:
            return EvidencePacket(
                admitted=False,
                reason="no_matching_action_evidence",
                matching_event_count=0,
                evidence_budget=self.evidence_policy.action_top_k,
            )
        selected = _apply_evidence_policy(matches, self.evidence_policy)
        return EvidencePacket(
            admitted=True,
            events=tuple(selected),
            confidence=1.0,
            reason="action_evidence_match",
            packed_context="\n".join(event.source_span for event in selected),
            matching_event_count=len(matches),
            evidence_budget=self.evidence_policy.action_top_k,
        )


class MicroHybridSlm:
    """Deterministic stand-in for a real hybrid SSM/attention SLM."""

    model_family = "micro_hybrid_ssm_attention"

    def __init__(self) -> None:
        self.calls = 0

    def encode_ssm_state(self, projection: StateProjection) -> MicroSsmState:
        joined = "|".join(projection.lines)
        state_hash = str(sum(ord(ch) for ch in joined) % 100_000)
        return MicroSsmState(lines=projection.lines, state_hash=state_hash)

    def attend(self, packet: EvidencePacket) -> MicroAttentionSlice:
        return MicroAttentionSlice(event_ids=packet.event_ids, context=packet.packed_context)

    def render(
        self,
        *,
        plan: BoundChildPlan,
        state_projection: StateProjection,
        evidence_packet: EvidencePacket,
    ) -> tuple[str, HybridSlmInput]:
        self.calls += 1
        ssm_state = self.encode_ssm_state(state_projection)
        attention = self.attend(evidence_packet)
        slm_input = HybridSlmInput(
            schema="melm.hybrid_slm_input.v1",
            model_family=self.model_family,
            compact_state=ssm_state.lines,
            attended_evidence_event_ids=attention.event_ids,
            response_intent=plan.response_intent,
            ssm_state=ssm_state,
            attention=attention,
            bound_plan_schema=plan.schema,
            matching_evidence_count=evidence_packet.matching_event_count,
            evidence_budget=evidence_packet.evidence_budget,
        )
        uol = plan.frame.uol
        if plan.response_intent == "location_answer" and uol.object and evidence_packet.answer_fact:
            return f"The {uol.object} is in the {evidence_packet.answer_fact}.", slm_input
        if plan.response_intent == "state_update" and uol.object:
            return f"Okay, I know the {uol.object} is {uol.action}.", slm_input
        if plan.response_intent == "memory_write" and uol.object and uol.target:
            return f"Okay, I remember the {uol.object} went to the {uol.target}.", slm_input
        if plan.response_intent == "evidence_check_answer" and evidence_packet.admitted:
            return "Yes, I have evidence for that.", slm_input
        return "I do not have enough evidence for that.", slm_input


TinyHybridSlmRenderer = MicroHybridSlm


class FiveYearOldGroundedChat:
    """Highly constrained appliance using the full micro architecture."""

    def __init__(
        self,
        *,
        atlas: ChildWorldAtlas | None = None,
        memory_os: ChildMemoryOS | None = None,
        renderer: MicroHybridSlm | None = None,
        known_people: set[str] | None = None,
        known_objects: set[str] | None = None,
        initial_states: dict[str, StateSet] | None = None,
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        self.atlas = atlas or ChildWorldAtlas(people=known_people, objects=known_objects)
        self.planner = ChildFramePlanner(self.atlas)
        self.memory_os = memory_os or ChildMemoryOS(
            initial_states=initial_states,
            evidence_policy=evidence_policy,
        )
        self.renderer = renderer or MicroHybridSlm()

    @property
    def object_states(self) -> dict[str, StateSet]:
        return self.memory_os.object_states

    def handle(self, text: str) -> GroundedChildResponse:
        trace: list[StageTrace] = []

        parsed = parse_uol(text, atlas=self.atlas)
        if isinstance(parsed, ChatFrame):
            packet = RejectionPacket(
                code=parsed.rejection_code or "uol_parse_failed",
                detail=parsed.rejection_detail or text,
                stage="uol_parse",
            )
            trace.append(StageTrace("uol_parse", "reject", packet.code))
            return self._rejected(text, parsed, packet, trace)
        trace.append(StageTrace("uol_parse", "ok", f"{parsed.schema}:score={parsed.parse_score:.2f}"))

        resolution = self.atlas.resolve(parsed)
        if resolution.rejection is not None:
            frame = ChatFrame(
                uol=resolution.uol,
                route="reject",
                rejection_code=resolution.rejection.code,
                rejection_detail=resolution.rejection.detail,
            )
            trace.append(StageTrace("semantic_atlas", "reject", resolution.rejection.code))
            return self._rejected(text, frame, resolution.rejection, trace)
        trace.append(StageTrace("semantic_atlas", "ok", resolution.schema))

        frame = self.planner.frame(resolution)
        if frame.route == "reject":
            packet = RejectionPacket(
                code=frame.rejection_code or "frame_rejected",
                detail=frame.rejection_detail or "frame rejected",
                stage="frame_build",
            )
            trace.append(StageTrace("frame_build", "reject", packet.code))
            return self._rejected(text, frame, packet, trace)
        trace.append(StageTrace("frame_build", "ok", frame.schema))

        plan_or_reject = self.planner.bind(frame)
        if isinstance(plan_or_reject, RejectionPacket):
            trace.append(StageTrace("plan_bind", "reject", plan_or_reject.code))
            return self._rejected(text, frame, plan_or_reject, trace)
        plan = plan_or_reject
        trace.append(StageTrace("plan_bind", "ok", plan.schema))

        if plan.kind == "state_transition":
            return self._handle_state_transition(text, frame, plan, trace)
        if plan.kind == "memory_write":
            return self._handle_memory_write(text, frame, plan, trace)
        if plan.requires_evidence:
            return self._handle_evidence_query(text, frame, plan, trace)
        packet = RejectionPacket("unsupported_plan", plan.kind, "plan_dispatch")
        trace.append(StageTrace("plan_dispatch", "reject", packet.code))
        return self._rejected(text, frame, packet, trace)

    def frame(self, text: str) -> ChatFrame:
        parsed = parse_uol(text, atlas=self.atlas)
        if isinstance(parsed, ChatFrame):
            return parsed
        resolution = self.atlas.resolve(parsed)
        return self.planner.frame(resolution)

    def events(self) -> tuple[Event, ...]:
        return self.memory_os.events()

    def _handle_state_transition(
        self,
        text: str,
        frame: ChatFrame,
        plan: BoundChildPlan,
        trace: list[StageTrace],
    ) -> GroundedChildResponse:
        result = self.memory_os.validate_state_transition(plan)
        if not result.valid:
            packet = RejectionPacket(
                code="state_transition_invalid",
                detail=_state_failure_detail(result),
                stage="state_algebra",
            )
            rejected_frame = ChatFrame(
                uol=frame.uol,
                route="reject",
                rejection_code=packet.code,
                rejection_detail=packet.detail,
            )
            trace.append(StageTrace("state_algebra", "reject", packet.detail))
            return self._rejected(text, rejected_frame, packet, trace, plan)
        trace.append(StageTrace("state_algebra", "ok", "patch_valid"))
        event = self.memory_os.commit_state_transition(plan, result)
        packet = self.memory_os.event_packet(event, reason="state_commit")
        trace.append(StageTrace("memory_os", "ok", "state_event_committed"))
        return self._render(text, "accepted", frame, plan, packet, trace)

    def _handle_memory_write(
        self,
        text: str,
        frame: ChatFrame,
        plan: BoundChildPlan,
        trace: list[StageTrace],
    ) -> GroundedChildResponse:
        rejection = self.memory_os.validate_memory_write(plan)
        if rejection is not None:
            rejected_frame = ChatFrame(
                uol=frame.uol,
                route="reject",
                rejection_code=rejection.code,
                rejection_detail=rejection.detail,
            )
            trace.append(StageTrace("memory_os", "reject", rejection.code))
            return self._rejected(text, rejected_frame, rejection, trace, plan)
        event = self.memory_os.commit_memory_write(plan)
        packet = self.memory_os.event_packet(event, reason="memory_write_commit")
        trace.append(StageTrace("memory_os", "ok", "location_event_committed"))
        return self._render(text, "accepted", frame, plan, packet, trace)

    def _handle_evidence_query(
        self,
        text: str,
        frame: ChatFrame,
        plan: BoundChildPlan,
        trace: list[StageTrace],
    ) -> GroundedChildResponse:
        packet = self.memory_os.admit_evidence(plan)
        trace.append(
            StageTrace(
                "evidence_admission",
                "ok" if packet.admitted else "abstain",
                packet.reason,
            )
        )
        if not packet.admitted:
            abstain_plan = replace(plan, response_intent="not_enough_evidence")
            return self._render(text, "abstained", frame, abstain_plan, packet, trace)
        return self._render(text, "answered", frame, plan, packet, trace)

    def _render(
        self,
        text: str,
        status: Status,
        frame: ChatFrame,
        plan: BoundChildPlan,
        packet: EvidencePacket,
        trace: list[StageTrace],
    ) -> GroundedChildResponse:
        projection = self.memory_os.project_state()
        trace.append(StageTrace("memory_os_projection", "ok", projection.schema))
        answer, slm_input = self.renderer.render(
            plan=plan,
            state_projection=projection,
            evidence_packet=packet,
        )
        trace.append(StageTrace("hybrid_slm", "ok", slm_input.model_family))
        return GroundedChildResponse(
            query=text,
            status=status,
            answer=answer,
            frame=frame,
            bound_plan=plan,
            evidence_event_ids=packet.event_ids,
            confidence=packet.confidence,
            parse_score=plan.parse_score,
            complexity_score=plan.complexity_score,
            slm_input=slm_input,
            trace=tuple(trace),
        )

    def _rejected(
        self,
        text: str,
        frame: ChatFrame,
        packet: RejectionPacket,
        trace: list[StageTrace],
        plan: BoundChildPlan | None = None,
    ) -> GroundedChildResponse:
        detail = packet.detail or packet.code
        return GroundedChildResponse(
            query=text,
            status="rejected",
            answer=f"I cannot safely use that sentence yet: {detail}.",
            frame=frame,
            bound_plan=plan,
            parse_score=plan.parse_score if plan else frame.uol.parse_score,
            complexity_score=plan.complexity_score if plan else _frame_complexity(frame),
            rejection_packet=packet,
            trace=tuple(trace),
        )


def apply_state_patch(current: StateSet, patch: StatePatch) -> StatePatchResult:
    """Apply a bounded state patch with explicit old-state removal."""

    missing = current.missing(patch.required)
    if missing:
        return StatePatchResult(False, None, missing, [])
    removed = _subtract_state(current, patch.remove)
    next_state = _merge_state(removed, patch.add)
    conflicts = _state_conflicts(next_state)
    return StatePatchResult(
        valid=not conflicts,
        next_state=None if conflicts else next_state,
        missing={},
        conflicts=conflicts,
    )


def parse_uol(
    text: str,
    *,
    atlas: ChildWorldAtlas | None = None,
) -> UOLSentence | ChatFrame:
    raw_text = text.strip()
    candidates = parse_uol_candidates(raw_text, atlas=atlas)
    if candidates:
        best = candidates[0]
        if best.score >= 0.50:
            return best.uol
        empty_uol = UOLSentence(
            raw_text=raw_text,
            purpose=best.uol.purpose,
            action=best.uol.action,
            parse_score=best.score,
            parse_complexity=best.complexity,
            parse_notes=best.notes + ("below_parse_threshold",),
        )
    else:
        empty_uol = UOLSentence(
            raw_text=raw_text,
            purpose="inform",
            action="unknown",
            parse_score=0.0,
            parse_complexity=1.0,
            parse_notes=("no_parse_candidate",),
        )
    return ChatFrame(
        uol=empty_uol,
        route="reject",
        rejection_code="unsupported_uol_shape",
        rejection_detail=raw_text,
    )


def parse_uol_candidates(
    text: str,
    *,
    atlas: ChildWorldAtlas | None = None,
) -> tuple[ParseCandidate, ...]:
    """Return scored UOL candidates over the bounded action/noun inventory."""

    atlas = atlas or ChildWorldAtlas()
    normalized = _normalize_text(text)
    tokens = normalized.split()
    if not tokens:
        return ()

    candidates: list[ParseCandidate] = []
    candidates.extend(_parse_state_candidates(text, tokens, atlas))
    candidates.extend(_parse_put_candidates(text, tokens, atlas))
    candidates.extend(_parse_move_candidates(text, tokens, atlas))
    candidates.extend(_parse_where_candidates(text, tokens, atlas))
    candidates.extend(_parse_did_move_candidates(text, tokens, atlas))
    candidates.sort(key=lambda candidate: (candidate.score, -candidate.complexity), reverse=True)
    return tuple(candidates)


def _parse_state_candidates(
    raw_text: str,
    tokens: list[str],
    atlas: ChildWorldAtlas,
) -> list[ParseCandidate]:
    candidates: list[ParseCandidate] = []
    for index, token in enumerate(tokens):
        action = _normalize_action(token)
        if action not in {"open", "close"}:
            continue
        purpose: Purpose = "command" if index == 0 or tokens[0] == "please" else "inform"
        subject = "user" if purpose == "command" else _token_before(tokens, index)
        object_phrase = _phrase_after(tokens, index + 1)
        uol = _scored_uol(
            raw_text,
            atlas,
            purpose=purpose,
            action=action,
            subject=subject,
            object=object_phrase,
            notes=("state_action",),
        )
        candidates.append(uol)
    return candidates


def _parse_put_candidates(
    raw_text: str,
    tokens: list[str],
    atlas: ChildWorldAtlas,
) -> list[ParseCandidate]:
    candidates: list[ParseCandidate] = []
    for index, token in enumerate(tokens):
        if token != "put":
            continue
        prep_index = _first_index(tokens, {"in", "into", "to"}, start=index + 1)
        if prep_index is None:
            continue
        subject = _token_before(tokens, index)
        object_phrase = _phrase_between(tokens, index + 1, prep_index)
        target_phrase = _phrase_after(tokens, prep_index + 1)
        candidates.append(
            _scored_uol(
                raw_text,
                atlas,
                purpose="inform",
                action="put",
                subject=subject,
                object=object_phrase,
                target=target_phrase,
                notes=("put_action", "target_preposition"),
            )
        )
    return candidates


def _parse_move_candidates(
    raw_text: str,
    tokens: list[str],
    atlas: ChildWorldAtlas,
) -> list[ParseCandidate]:
    candidates: list[ParseCandidate] = []
    for index, token in enumerate(tokens):
        if _normalize_action(token) != "move":
            continue
        to_index = _first_index(tokens, {"to", "into", "in"}, start=index + 1)
        if to_index is None:
            continue
        from_index = _first_index(tokens, {"from"}, start=index + 1)
        subject = _token_before(tokens, index)
        object_end = from_index if from_index is not None and from_index < to_index else to_index
        object_phrase = _phrase_between(tokens, index + 1, object_end)
        source_phrase = (
            _phrase_between(tokens, from_index + 1, to_index)
            if from_index is not None and from_index < to_index
            else None
        )
        target_phrase = _phrase_after(tokens, to_index + 1)
        candidates.append(
            _scored_uol(
                raw_text,
                atlas,
                purpose="inform",
                action="move",
                subject=subject,
                object=object_phrase,
                source=source_phrase,
                target=target_phrase,
                notes=("move_action", "target_preposition"),
            )
        )
    return candidates


def _parse_where_candidates(
    raw_text: str,
    tokens: list[str],
    atlas: ChildWorldAtlas,
) -> list[ParseCandidate]:
    if tokens[0] != "where":
        return []
    if len(tokens) >= 3 and tokens[1] == "is":
        object_phrase = _phrase_after(tokens, 2)
        return [
            _scored_uol(
                raw_text,
                atlas,
                purpose="question",
                action="where",
                object=object_phrase,
                notes=("where_is",),
            )
        ]
    if len(tokens) >= 5 and tokens[1] == "did":
        subject = tokens[2]
        verb_index = 3
        query_verb = _normalize_action(tokens[verb_index])
        if query_verb not in {"put", "move"}:
            return []
        object_phrase = _phrase_after(tokens, verb_index + 1)
        return [
            _scored_uol(
                raw_text,
                atlas,
                purpose="question",
                action="where",
                subject=subject,
                object=object_phrase,
                notes=("where_did", query_verb),
            )
        ]
    return []


def _parse_did_move_candidates(
    raw_text: str,
    tokens: list[str],
    atlas: ChildWorldAtlas,
) -> list[ParseCandidate]:
    if len(tokens) < 6 or tokens[0] != "did":
        return []
    subject = tokens[1]
    move_index = 2
    if _normalize_action(tokens[move_index]) != "move":
        return []
    to_index = _first_index(tokens, {"to", "into", "in"}, start=move_index + 1)
    if to_index is None:
        return []
    object_phrase = _phrase_between(tokens, move_index + 1, to_index)
    target_phrase = _phrase_after(tokens, to_index + 1)
    return [
        _scored_uol(
            raw_text,
            atlas,
            purpose="question",
            action="did_move",
            subject=subject,
            object=object_phrase,
            target=target_phrase,
            notes=("did_move",),
        )
    ]


def _event_objects(uol: UOLSentence, *, previous_location: str | None) -> tuple[str, ...]:
    if uol.action == "move":
        source = uol.source or previous_location or "unknown"
        return tuple(part for part in (uol.object, source, uol.target) if part)
    return tuple(part for part in (uol.object, uol.target) if part)


def _apply_evidence_policy(
    matches: list[Event],
    policy: EvidencePolicy,
) -> tuple[Event, ...]:
    top_k = policy.action_top_k
    if top_k is None:
        return tuple(matches)
    if top_k <= 0:
        return ()
    if len(matches) <= top_k:
        return tuple(matches)
    if policy.prefer_latest:
        return tuple(matches[-top_k:])
    return tuple(matches[:top_k])


def _scored_uol(
    raw_text: str,
    atlas: ChildWorldAtlas,
    *,
    purpose: Purpose,
    action: str,
    subject: str | None = None,
    object: str | None = None,
    source: str | None = None,
    target: str | None = None,
    notes: tuple[str, ...] = (),
) -> ParseCandidate:
    object = _clean_noun(object)
    source = _clean_noun(source)
    target = _clean_noun(target)
    score = 0.25
    complexity = 0.05
    scored_notes = list(notes)
    spec = atlas.action_specs.get(action)
    if spec is not None:
        score += 0.20
        complexity += spec.base_complexity
        if purpose not in spec.purposes:
            score -= 0.10
            complexity += 0.10
            scored_notes.append("purpose_mismatch")
    else:
        scored_notes.append("unknown_action")
        complexity += 0.20
    if purpose in {"question", "command"}:
        score += 0.08
        complexity += 0.10
    if subject:
        if subject in atlas.people or subject == "user":
            score += 0.12
        else:
            scored_notes.append("unknown_subject")
            complexity += 0.12
    for slot_name, slot_value, weight in (
        ("object", object, 0.22),
        ("source", source, 0.08),
        ("target", target, 0.18),
    ):
        if not slot_value:
            continue
        if atlas._resolve_object(slot_value) is not None:
            score += weight
        else:
            scored_notes.append(f"unknown_{slot_name}")
            complexity += 0.10
    if spec is not None:
        slot_values = {
            "subject": subject,
            "object": object,
            "source": source,
            "target": target,
        }
        missing_required = tuple(
            slot_name
            for slot_name in spec.required_slots
            if not slot_values.get(slot_name)
        )
        if missing_required:
            score -= 0.18 * len(missing_required)
            complexity += 0.08 * len(missing_required)
            scored_notes.extend(f"missing_{slot_name}" for slot_name in missing_required)
    slot_count = sum(1 for value in (subject, object, source, target) if value)
    complexity += slot_count * 0.06
    uol = UOLSentence(
        raw_text=raw_text,
        purpose=purpose,
        action=action,
        subject=subject,
        object=object,
        source=source,
        target=target,
        parse_score=round(max(0.0, min(score, 1.0)), 3),
        parse_complexity=round(complexity, 3),
        parse_notes=tuple(scored_notes),
    )
    return ParseCandidate(
        uol=uol,
        score=uol.parse_score,
        complexity=uol.parse_complexity,
        notes=uol.parse_notes,
    )


def _token_before(tokens: list[str], index: int) -> str | None:
    if index <= 0:
        return None
    candidate = tokens[index - 1]
    return None if candidate in {"please", "the"} else candidate


def _phrase_between(tokens: list[str], start: int, end: int) -> str | None:
    return _clean_noun(" ".join(tokens[start:end]))


def _phrase_after(tokens: list[str], start: int) -> str | None:
    return _clean_noun(" ".join(tokens[start:]))


def _first_index(tokens: list[str], choices: set[str], *, start: int = 0) -> int | None:
    for index in range(start, len(tokens)):
        if tokens[index] in choices:
            return index
    return None


def _plan_complexity(uol: UOLSentence, route: Route) -> float:
    slot_count = sum(1 for value in (uol.subject, uol.object, uol.source, uol.target) if value)
    route_weight = {
        "state_transition": 0.20,
        "memory_write": 0.40,
        "memory_read": 0.55,
        "evidence_check": 0.75,
        "reject": 1.0,
    }[route]
    return round(route_weight + slot_count * 0.05 + uol.parse_complexity, 3)


def _frame_complexity(frame: ChatFrame) -> float:
    if frame.route == "reject":
        return frame.uol.parse_complexity
    return _plan_complexity(frame.uol, frame.route)


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_action(value: str) -> str:
    if value in {"opened", "open"}:
        return "open"
    if value in {"closed", "close"}:
        return "close"
    if value in {"moved", "move"}:
        return "move"
    return value


def _clean_noun(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    value = value.removeprefix("the ").strip()
    return value or None


def _subtract_state(current: StateSet, remove: StateSet) -> StateSet:
    return StateSet(
        physical=current.physical - remove.physical,
        emotional=current.emotional - remove.emotional,
        positional=current.positional - remove.positional,
        mental=current.mental - remove.mental,
    )


def _merge_state(left: StateSet, right: StateSet) -> StateSet:
    return StateSet(
        physical=left.physical | right.physical,
        emotional=left.emotional | right.emotional,
        positional=left.positional | right.positional,
        mental=left.mental | right.mental,
    )


def _state_conflicts(state: StateSet) -> list[tuple[str, str, str]]:
    conflicts: list[tuple[str, str, str]] = []
    for dimension in DIMENSIONS:
        states = getattr(state, dimension)
        for left, right in DEFAULT_CONTRADICTIONS:
            if left in states and right in states:
                conflicts.append((dimension, left, right))
    return conflicts


def _state_failure_detail(result: StatePatchResult) -> str:
    if result.missing:
        pieces = [
            f"missing {dimension}: {', '.join(sorted(values))}"
            for dimension, values in sorted(result.missing.items())
        ]
        return "; ".join(pieces)
    if result.conflicts:
        return "; ".join(
            f"{dimension} conflict: {left}/{right}"
            for dimension, left, right in result.conflicts
        )
    return "state transition failed"


def generated_child_capability_cases(
    atlas: ChildWorldAtlas | None = None,
) -> tuple[CapabilityCase, ...]:
    """Generate a bounded coverage set from the atlas instead of hand examples."""

    atlas = atlas or ChildWorldAtlas()
    people = sorted(atlas.people)
    objects = sorted(atlas.objects)
    containers = sorted(obj for obj in objects if atlas.is_container(obj))
    open_blue_box = ("Maya opened the blue box.",)
    story_setup = (
        "Maya opened the blue box.",
        "Maya put the red block in the blue box.",
        "Leo moved the red block to the green basket.",
    )

    cases: list[CapabilityCase] = []
    for actor in people:
        for obj in objects:
            cases.append(CapabilityCase(f"{actor.title()} opened the {obj}."))
            cases.append(CapabilityCase(f"{actor.title()} closed the {obj}.", setup=open_blue_box))

    for actor in people:
        for obj in objects:
            for target in objects:
                if obj == target:
                    continue
                setup = open_blue_box if target == "blue box" else ()
                cases.append(CapabilityCase(f"{actor.title()} put the {obj} in the {target}.", setup=setup))

    for actor in people:
        for obj in objects:
            for target in containers:
                if obj == target:
                    continue
                setup = open_blue_box if target == "blue box" else ()
                cases.append(CapabilityCase(f"{actor.title()} moved the {obj} to the {target}.", setup=setup))

    for obj in objects:
        cases.append(CapabilityCase(f"Where is the {obj}?", setup=story_setup))
        for actor in people:
            for target in containers:
                cases.append(
                    CapabilityCase(
                        f"Did {actor.title()} move the {obj} to the {target}?",
                        setup=story_setup,
                    )
                )

    cases.extend(
        (
            CapabilityCase("Maya painted the red block."),
            CapabilityCase("Where did Maya hide the red block?", setup=story_setup),
            CapabilityCase("Maya put the red block beside the blue box.", setup=open_blue_box),
        )
    )
    return tuple(cases)


def run_child_capability_probe(
    *,
    atlas: ChildWorldAtlas | None = None,
    cases: tuple[CapabilityCase, ...] | None = None,
) -> CapabilityReport:
    """Run generated utterances through the same micro appliance path."""

    atlas = atlas or ChildWorldAtlas()
    cases = cases or generated_child_capability_cases(atlas)
    results: list[CapabilityResult] = []
    for case in cases:
        chat = FiveYearOldGroundedChat(atlas=atlas)
        for setup_utterance in case.setup:
            chat.handle(setup_utterance)
        response = chat.handle(case.utterance)
        results.append(_capability_result(case.utterance, response))
    return _capability_report(tuple(results))


def run_child_context_budget_probe(
    *,
    move_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    evidence_top_k: int = 1,
) -> ContextBudgetReport:
    """Measure raw transcript growth against SSM/attention boundary payloads."""

    query_specs = (
        ("location", "Where is the red block?"),
        ("positive_evidence", "Did Leo move the red block to the green basket?"),
        ("negative_evidence", "Did Maya move the red block to the green basket?"),
    )
    budgeted_policy = EvidencePolicy(action_top_k=evidence_top_k)
    results: list[ContextBudgetResult] = []
    for query_kind, query in query_specs:
        for move_count in move_counts:
            transcript, unbudgeted = _run_repeated_move_story(move_count, query)
            _, budgeted = _run_repeated_move_story(
                move_count,
                query,
                evidence_policy=budgeted_policy,
            )
            if unbudgeted.slm_input is None or budgeted.slm_input is None:
                continue
            raw_transcript_chars = len("\n".join((*transcript, query)))
            unbudgeted_payload_chars = _slm_payload_chars(unbudgeted.slm_input)
            budgeted_payload_chars = _slm_payload_chars(budgeted.slm_input)
            results.append(
                ContextBudgetResult(
                    query_kind=query_kind,
                    query=query,
                    move_count=move_count,
                    status=budgeted.status,
                    raw_transcript_chars=raw_transcript_chars,
                    unbudgeted_payload_chars=unbudgeted_payload_chars,
                    budgeted_payload_chars=budgeted_payload_chars,
                    unbudgeted_evidence_count=len(
                        unbudgeted.slm_input.attended_evidence_event_ids
                    ),
                    budgeted_evidence_count=len(
                        budgeted.slm_input.attended_evidence_event_ids
                    ),
                    matching_evidence_count=budgeted.slm_input.matching_evidence_count,
                    unbudgeted_compression_ratio=_compression_ratio(
                        raw_transcript_chars,
                        unbudgeted_payload_chars,
                    ),
                    budgeted_compression_ratio=_compression_ratio(
                        raw_transcript_chars,
                        budgeted_payload_chars,
                    ),
                )
            )
    return ContextBudgetReport(
        evidence_top_k=evidence_top_k,
        move_counts=move_counts,
        results=tuple(results),
    )


def _capability_result(utterance: str, response: GroundedChildResponse) -> CapabilityResult:
    if response.rejection_packet is not None:
        reason = response.rejection_packet.code
    elif response.status == "abstained":
        reason = next(
            (stage.detail for stage in response.trace if stage.stage == "evidence_admission"),
            "abstained",
        )
    else:
        reason = response.frame.route
    return CapabilityResult(
        utterance=utterance,
        status=response.status,
        route=response.frame.route,
        reason=reason,
        parse_score=response.parse_score,
        complexity_score=response.complexity_score,
        evidence_event_ids=response.evidence_event_ids,
    )


def _capability_report(results: tuple[CapabilityResult, ...]) -> CapabilityReport:
    by_reason: dict[str, int] = {}
    for result in results:
        by_reason[result.reason] = by_reason.get(result.reason, 0) + 1
    return CapabilityReport(
        cases=len(results),
        accepted=sum(1 for result in results if result.status == "accepted"),
        answered=sum(1 for result in results if result.status == "answered"),
        abstained=sum(1 for result in results if result.status == "abstained"),
        rejected=sum(1 for result in results if result.status == "rejected"),
        average_parse_score=round(
            sum(result.parse_score for result in results) / len(results),
            3,
        )
        if results
        else 0.0,
        average_complexity=round(
            sum(result.complexity_score for result in results) / len(results),
            3,
        )
        if results
        else 0.0,
        max_complexity=max((result.complexity_score for result in results), default=0.0),
        by_reason=dict(sorted(by_reason.items())),
        results=results,
    )


def _run_repeated_move_story(
    move_count: int,
    query: str,
    *,
    evidence_policy: EvidencePolicy | None = None,
) -> tuple[tuple[str, ...], GroundedChildResponse]:
    chat = FiveYearOldGroundedChat(evidence_policy=evidence_policy)
    transcript = ["Maya opened the blue box."]
    chat.handle(transcript[-1])
    for index in range(move_count):
        actor = "Maya" if index % 2 == 0 else "Leo"
        target = "blue box" if index % 2 == 0 else "green basket"
        utterance = f"{actor} moved the red block to the {target}."
        transcript.append(utterance)
        chat.handle(utterance)
    return tuple(transcript), chat.handle(query)


def _slm_payload_chars(slm_input: HybridSlmInput) -> int:
    payload = {
        "compact_state": slm_input.compact_state,
        "attended_evidence_event_ids": slm_input.attended_evidence_event_ids,
        "matching_evidence_count": slm_input.matching_evidence_count,
        "evidence_budget": slm_input.evidence_budget,
        "response_intent": slm_input.response_intent,
        "attention_context": slm_input.attention.context,
    }
    return len(json.dumps(payload, sort_keys=True))


def _compression_ratio(raw_chars: int, payload_chars: int) -> float:
    if payload_chars <= 0:
        return 0.0
    return round(raw_chars / payload_chars, 2)
