"""Authored support/refunds dataset loader for MELM Guard + Memory OS."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from melm.guard import ActionProposal, Fact, GuardStatus
from melm.memory import Event

from .support_refunds import (
    GuardBenchmarkCase,
    SupportMemoryCase,
    SupportRefundFixture,
    support_refund_fixture,
)


DEFAULT_AUTHORED_SUPPORT_REFUNDS_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "support_refunds_authored.jsonl"
)

VALID_GUARD_STATUSES = {"allow", "deny", "warn", "abstain"}
REQUIRED_GUARD_CATEGORIES = {
    "approval_required",
    "duplicate_refund",
    "fraud_flag",
    "identity_missing_or_false",
    "malformed_action",
    "missing_order",
    "not_delivered",
    "outside_return_window",
    "stale_approval",
    "stale_state_trap",
    "valid_high_value",
    "valid_low_value",
}
REQUIRED_MEMORY_CATEGORIES = {
    "approval_recall",
    "contradiction_resolution",
    "current_state",
    "policy_recall",
    "risk_recall",
    "stale_state_update",
    "unknown_order",
}


@dataclass(frozen=True)
class AuthoredSupportRefundDataset:
    path: str
    metadata: dict[str, Any]
    turns: tuple[dict[str, Any], ...]
    fixture: SupportRefundFixture
    validation_errors: tuple[str, ...]


def load_authored_support_refund_dataset(
    path: str | Path | None = None,
    *,
    strict: bool = False,
) -> AuthoredSupportRefundDataset:
    """Load the JSONL authored support/refunds benchmark."""

    dataset_path = Path(path) if path else DEFAULT_AUTHORED_SUPPORT_REFUNDS_PATH
    records = _read_jsonl(dataset_path)
    metadata: dict[str, Any] = {}
    turns: list[dict[str, Any]] = []
    events: list[Event] = []
    facts: list[Fact] = []
    guard_cases: list[GuardBenchmarkCase] = []
    memory_cases: list[SupportMemoryCase] = []

    for line_no, record in records:
        schema = record.get("schema")
        if schema == "melm.support_refunds.dataset.v1":
            metadata = {key: value for key, value in record.items() if key != "schema"}
        elif schema == "melm.support_refunds.turn.v1":
            turns.append({**record, "line_no": line_no})
        elif schema == "melm.support_refunds.fact_event.v1":
            event, fact = _parse_fact_event(record, line_no=line_no, dataset_id=metadata.get("dataset_id", ""))
            events.append(event)
            facts.append(fact)
        elif schema == "melm.support_refunds.guard_case.v1":
            guard_cases.append(_parse_guard_case(record, line_no=line_no))
        elif schema == "melm.support_refunds.memory_case.v1":
            memory_cases.append(_parse_memory_case(record, line_no=line_no))
        else:
            raise ValueError(f"{dataset_path}:{line_no}: unsupported schema {schema!r}")

    current_time = max(
        [event.time_index for event in events]
        + [case.current_time or 0 for case in guard_cases]
        + [int(metadata.get("current_time", 0) or 0)]
    )
    fixture = SupportRefundFixture(
        events=events,
        facts=facts,
        rules=support_refund_fixture().rules,
        guard_cases=guard_cases,
        memory_cases=memory_cases,
        current_time=current_time,
    )
    dataset = AuthoredSupportRefundDataset(
        path=str(dataset_path),
        metadata=metadata,
        turns=tuple(turns),
        fixture=fixture,
        validation_errors=(),
    )
    errors = tuple(validate_authored_support_refund_dataset(dataset))
    dataset = replace(dataset, validation_errors=errors)
    if strict and errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Authored support/refunds dataset failed validation:\n{joined}")
    return dataset


def validate_authored_support_refund_dataset(dataset: AuthoredSupportRefundDataset) -> list[str]:
    """Return semantic validation errors for the authored dataset."""

    errors: list[str] = []
    metadata = dataset.metadata
    fixture = dataset.fixture
    events_by_id = {event.event_id: event for event in fixture.events}
    facts_by_id = {fact.fact_id: fact for fact in fixture.facts}

    if metadata.get("dataset_id") != "melm_support_refunds_authored_v0_1":
        errors.append("metadata.dataset_id must be melm_support_refunds_authored_v0_1")
    if metadata.get("vertical") != "support_refunds":
        errors.append("metadata.vertical must be support_refunds")
    if not metadata.get("authoring_protocol"):
        errors.append("metadata.authoring_protocol is required")
    requires_external = bool(metadata.get("requires_external_blind_batch", True))
    if not requires_external:
        if not metadata.get("external_blind_batch"):
            errors.append("external-ready datasets must set metadata.external_blind_batch=true")
        if int(metadata.get("annotator_count", 0) or 0) < 2:
            errors.append("external-ready datasets must report metadata.annotator_count >= 2")

    _check_unique("event_id", [event.event_id for event in fixture.events], errors)
    _check_unique("fact_id", [fact.fact_id for fact in fixture.facts], errors)
    _check_unique("guard action_id", [case.proposal.action_id for case in fixture.guard_cases], errors)

    if len(dataset.turns) < 20:
        errors.append("dataset must include at least 20 authored transcript turns")
    if len(fixture.events) < 40:
        errors.append("dataset must include at least 40 fact-bearing events")
    if len(fixture.guard_cases) < 12:
        errors.append("dataset must include at least 12 guard cases")
    if len(fixture.memory_cases) < 20:
        errors.append("dataset must include at least 20 memory cases")

    guard_categories = {case.category for case in fixture.guard_cases}
    missing_guard = sorted(REQUIRED_GUARD_CATEGORIES - guard_categories)
    if missing_guard:
        errors.append(f"missing guard categories: {', '.join(missing_guard)}")

    memory_categories = {case.category for case in fixture.memory_cases}
    missing_memory = sorted(REQUIRED_MEMORY_CATEGORIES - memory_categories)
    if missing_memory:
        errors.append(f"missing memory categories: {', '.join(missing_memory)}")

    for fact in fixture.facts:
        if fact.source_event_id not in events_by_id:
            errors.append(f"fact {fact.fact_id} cites missing event {fact.source_event_id}")
        if not fact.metadata.get("case_id"):
            errors.append(f"fact {fact.fact_id} is missing case_id provenance")

    required_policy_keys = {
        ("policy:support", "max_refund_without_approval"),
        ("policy:support", "manager_approval_fresh_steps"),
        ("policy:support", "return_window_days"),
    }
    observed_policy_keys = {(fact.subject, fact.predicate) for fact in fixture.facts}
    missing_policy = sorted(required_policy_keys - observed_policy_keys)
    if missing_policy:
        errors.append(f"missing policy facts: {missing_policy}")

    for case in fixture.guard_cases:
        if case.expected_status not in VALID_GUARD_STATUSES:
            errors.append(f"guard case {case.proposal.action_id} has invalid expected_status {case.expected_status}")
        order_id = str(case.proposal.parameters.get("order_id", "")).lower()
        known_order = any(
            event.metadata.get("order_id", "").lower() == order_id
            for event in fixture.events
        )
        if order_id and not known_order and case.category != "missing_order":
            errors.append(f"guard case {case.proposal.action_id} references unknown order outside missing_order")
        if case.expected_status != "allow" and case.category.startswith("valid_"):
            errors.append(f"valid guard category {case.category} cannot expect {case.expected_status}")

    for case in fixture.memory_cases:
        if case.kind not in {"event", "state"}:
            errors.append(f"memory case {case.query!r} has unsupported kind {case.kind!r}")
        if case.kind == "event" and case.expected_event_id not in events_by_id:
            errors.append(f"memory event case {case.query!r} cites missing event {case.expected_event_id}")
        if case.kind == "state" and case.expected_value is not None and not case.predicate:
            errors.append(f"memory state case {case.query!r} is missing predicate")

    return errors


def authored_support_refund_fixture(path: str | Path | None = None) -> SupportRefundFixture:
    """Convenience wrapper returning only the fixture."""

    return load_authored_support_refund_dataset(path, strict=True).fixture


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append((line_no, json.loads(line)))
    return rows


def _parse_fact_event(record: dict[str, Any], *, line_no: int, dataset_id: str) -> tuple[Event, Fact]:
    fact_record = _required_dict(record, "fact", line_no)
    fact_subject = _required_str(fact_record, "subject", line_no)
    fact_predicate = _required_str(fact_record, "predicate", line_no)
    value = fact_record.get("value")
    order_id = str(record.get("order_id") or _entity_from_subject(fact_subject, "order:"))
    customer_id = str(record.get("customer_id") or _entity_from_subject(fact_subject, "customer:"))
    metadata = _string_map(record.get("metadata", {}))
    metadata.update(
        {
            "dataset_id": dataset_id,
            "case_id": str(record.get("case_id", "")),
            "fact_subject": fact_subject,
            "fact_predicate": fact_predicate,
            "fact_value": str(value),
        }
    )
    if order_id:
        metadata["order_id"] = order_id.lower()
    if customer_id:
        metadata["customer_id"] = customer_id.lower()
    objects = [str(item).lower() for item in record.get("objects", [])]
    for item in (order_id, customer_id, fact_predicate):
        normalized = str(item).lower()
        if normalized and normalized not in objects:
            objects.append(normalized)

    event = Event(
        event_id=_required_str(record, "event_id", line_no),
        source_span=_required_str(record, "source_span", line_no),
        time_index=int(record.get("time_index", 0)),
        actors=tuple(str(item).lower() for item in record.get("actors", ())),
        action_or_state=str(record.get("action_or_state", "recorded")).lower(),
        objects=tuple(objects),
        location=record.get("location"),
        causal_links=tuple(str(item) for item in record.get("causal_links", ())),
        salience=float(record.get("salience", 1.0)),
        surprise_score=float(record.get("surprise_score", 0.0)),
        previous_event_id=record.get("previous_event_id"),
        next_event_id=record.get("next_event_id"),
        metadata=metadata,
    )
    fact_metadata = {
        "dataset_id": dataset_id,
        "case_id": str(record.get("case_id", "")),
        "source_schema": str(record.get("schema", "")),
        **_string_map(fact_record.get("metadata", {})),
    }
    fact = Fact(
        fact_id=_required_str(fact_record, "fact_id", line_no),
        subject=fact_subject,
        predicate=fact_predicate,
        value=value,
        time_index=event.time_index,
        source_event_id=event.event_id,
        confidence=float(fact_record.get("confidence", 1.0)),
        metadata=fact_metadata,
    )
    return event, fact


def _parse_guard_case(record: dict[str, Any], *, line_no: int) -> GuardBenchmarkCase:
    proposal_record = _required_dict(record, "proposal", line_no)
    proposal = ActionProposal(
        action_id=_required_str(proposal_record, "action_id", line_no),
        action_type=_required_str(proposal_record, "action_type", line_no),
        parameters=dict(proposal_record.get("parameters", {})),
        source_query=_required_str(proposal_record, "source_query", line_no),
        proposed_by=str(proposal_record.get("proposed_by", "authored_fixture_agent")),
        malformed=bool(proposal_record.get("malformed", False)),
        metadata={
            "case_id": str(record.get("case_id", "")),
            **_string_map(proposal_record.get("metadata", {})),
        },
    )
    return GuardBenchmarkCase(
        proposal=proposal,
        expected_status=record.get("expected_status"),  # type: ignore[arg-type]
        category=_required_str(record, "category", line_no),
        current_time=int(record["current_time"]) if "current_time" in record else None,
    )


def _parse_memory_case(record: dict[str, Any], *, line_no: int) -> SupportMemoryCase:
    return SupportMemoryCase(
        query=_required_str(record, "query", line_no),
        kind=record.get("kind"),  # type: ignore[arg-type]
        category=_required_str(record, "category", line_no),
        order_id=str(record.get("order_id", "")).lower(),
        expected_event_id=record.get("expected_event_id"),
        predicate=record.get("predicate"),
        expected_value=record.get("expected_value"),
    )


def _check_unique(label: str, values: list[str], errors: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        errors.append(f"duplicate {label}: {', '.join(sorted(duplicates))}")


def _required_dict(record: dict[str, Any], key: str, line_no: int) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"line {line_no}: {key} must be an object")
    return value


def _required_str(record: dict[str, Any], key: str, line_no: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"line {line_no}: {key} must be a non-empty string")
    return value


def _entity_from_subject(subject: str, prefix: str) -> str:
    return subject.removeprefix(prefix) if subject.startswith(prefix) else ""


def _string_map(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}
