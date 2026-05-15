"""JSONL import/export helpers for MELM memory benchmarks."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from melm.memory import Event

from .evidence import EvidenceCase
from .synthetic_episodic import EpisodicCase


EVENT_TUPLE_FIELDS = ("actors", "objects", "causal_links")


def save_episodic_benchmark(
    events: list[Event],
    cases: list[EpisodicCase],
    *,
    events_path: str | Path,
    cases_path: str | Path,
) -> None:
    """Save events and recall cases as deterministic JSONL files."""

    _write_jsonl(Path(events_path), [_event_to_record(event) for event in events])
    _write_jsonl(Path(cases_path), [_case_to_record(case) for case in cases])


def save_dialogue_benchmark(
    events: list[Event],
    recall_cases: list[EpisodicCase],
    evidence_cases: list[EvidenceCase],
    *,
    events_path: str | Path,
    recall_cases_path: str | Path,
    evidence_cases_path: str | Path,
) -> None:
    """Save events, positive recall cases, and answerability cases."""

    _write_jsonl(Path(events_path), [_event_to_record(event) for event in events])
    _write_jsonl(
        Path(recall_cases_path),
        [_case_to_record(case) for case in recall_cases],
    )
    _write_jsonl(
        Path(evidence_cases_path),
        [_evidence_case_to_record(case) for case in evidence_cases],
    )


def load_episodic_benchmark(
    *,
    events_path: str | Path,
    cases_path: str | Path,
    validate: bool = True,
) -> tuple[list[Event], list[EpisodicCase]]:
    """Load an episodic benchmark from JSONL files.

    When `validate` is enabled, the loader rejects duplicate event IDs and
    cases whose expected event IDs do not exist in the event file.
    """

    events = [_event_from_record(record) for record in _read_jsonl(Path(events_path))]
    cases = [_case_from_record(record) for record in _read_jsonl(Path(cases_path))]

    if validate:
        errors = validate_episodic_benchmark(events, cases)
        if errors:
            joined = "; ".join(errors)
            raise ValueError(f"Invalid episodic benchmark: {joined}")

    return events, cases


def load_dialogue_benchmark(
    *,
    events_path: str | Path,
    recall_cases_path: str | Path,
    evidence_cases_path: str | Path,
    validate: bool = True,
) -> tuple[list[Event], list[EpisodicCase], list[EvidenceCase]]:
    """Load a dialogue-style memory benchmark from JSONL files."""

    events = [_event_from_record(record) for record in _read_jsonl(Path(events_path))]
    recall_cases = [
        _case_from_record(record)
        for record in _read_jsonl(Path(recall_cases_path))
    ]
    evidence_cases = [
        _evidence_case_from_record(record)
        for record in _read_jsonl(Path(evidence_cases_path))
    ]

    if validate:
        errors = validate_dialogue_benchmark(events, recall_cases, evidence_cases)
        if errors:
            joined = "; ".join(errors)
            raise ValueError(f"Invalid dialogue benchmark: {joined}")

    return events, recall_cases, evidence_cases


def validate_episodic_benchmark(events: list[Event], cases: list[EpisodicCase]) -> list[str]:
    """Return validation errors for an episodic benchmark fixture."""

    errors, event_ids = _validate_events(events)
    errors.extend(_validate_recall_cases(cases, event_ids))
    return errors


def validate_evidence_benchmark(events: list[Event], cases: list[EvidenceCase]) -> list[str]:
    """Return validation errors for answerability/evidence cases."""

    errors, event_ids = _validate_events(events)
    errors.extend(_validate_evidence_cases(cases, event_ids))
    return errors


def validate_dialogue_benchmark(
    events: list[Event],
    recall_cases: list[EpisodicCase],
    evidence_cases: list[EvidenceCase],
) -> list[str]:
    """Return validation errors for a full dialogue benchmark fixture."""

    errors, event_ids = _validate_events(events)
    errors.extend(_validate_recall_cases(recall_cases, event_ids))
    errors.extend(_validate_evidence_cases(evidence_cases, event_ids))
    return errors


def _validate_recall_cases(cases: list[EpisodicCase], event_ids: set[str]) -> list[str]:
    errors: list[str] = []
    for index, case in enumerate(cases):
        if not case.query.strip():
            errors.append(f"case {index} has empty query")
        if not case.expected_event_id:
            errors.append(f"case {index} has empty expected_event_id")
        elif case.expected_event_id not in event_ids:
            errors.append(
                f"case {index} expects missing event_id {case.expected_event_id!r}"
            )
        if not case.category.strip():
            errors.append(f"case {index} has empty category")

    return errors


def _validate_evidence_cases(cases: list[EvidenceCase], event_ids: set[str]) -> list[str]:
    errors: list[str] = []
    for index, case in enumerate(cases):
        if not case.query.strip():
            errors.append(f"evidence case {index} has empty query")
        if case.expected_event_id is not None and case.expected_event_id not in event_ids:
            errors.append(
                f"evidence case {index} expects missing event_id {case.expected_event_id!r}"
            )
        if not case.category.strip():
            errors.append(f"evidence case {index} has empty category")

    return errors


def _event_to_record(event: Event) -> dict[str, Any]:
    record = asdict(event)
    record["schema"] = "melm.event.v1"
    return record


def _case_to_record(case: EpisodicCase) -> dict[str, Any]:
    return {
        "schema": "melm.episodic_case.v1",
        "query": case.query,
        "expected_event_id": case.expected_event_id,
        "category": case.category,
    }


def _evidence_case_to_record(case: EvidenceCase) -> dict[str, Any]:
    return {
        "schema": "melm.evidence_case.v1",
        "query": case.query,
        "expected_event_id": case.expected_event_id,
        "category": case.category,
        "story_id": case.story_id,
    }


def _event_from_record(record: dict[str, Any]) -> Event:
    record = dict(record)
    record.pop("schema", None)
    for field in EVENT_TUPLE_FIELDS:
        record[field] = tuple(record.get(field) or ())
    metadata = record.get("metadata") or {}
    record["metadata"] = {str(key): str(value) for key, value in metadata.items()}
    return Event(**record)


def _case_from_record(record: dict[str, Any]) -> EpisodicCase:
    return EpisodicCase(
        query=str(record["query"]),
        expected_event_id=str(record["expected_event_id"]),
        category=str(record.get("category") or "overall"),
    )


def _evidence_case_from_record(record: dict[str, Any]) -> EvidenceCase:
    expected_event_id = record.get("expected_event_id")
    return EvidenceCase(
        query=str(record["query"]),
        expected_event_id=str(expected_event_id) if expected_event_id else None,
        category=str(record.get("category") or "overall"),
        story_id=str(record.get("story_id") or ""),
    )


def _validate_events(events: list[Event]) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    event_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    event_by_id: dict[str, Event] = {}

    for event in events:
        if not event.event_id:
            errors.append("event with empty event_id")
            continue
        if event.event_id in event_ids:
            duplicate_ids.add(event.event_id)
        event_ids.add(event.event_id)
        event_by_id[event.event_id] = event

    if duplicate_ids:
        errors.append(f"duplicate event_id values: {', '.join(sorted(duplicate_ids))}")

    if not duplicate_ids:
        errors.extend(_validate_event_links(event_by_id))

    return errors, event_ids


def _validate_event_links(event_by_id: dict[str, Event]) -> list[str]:
    errors: list[str] = []

    for event in event_by_id.values():
        previous_id = event.previous_event_id
        if previous_id:
            previous = event_by_id.get(previous_id)
            if previous is None:
                errors.append(
                    f"event {event.event_id!r} references missing previous_event_id {previous_id!r}"
                )
            else:
                if previous.time_index > event.time_index:
                    errors.append(
                        f"event {event.event_id!r} previous_event_id {previous_id!r} is not earlier"
                    )
                if previous.next_event_id and previous.next_event_id != event.event_id:
                    errors.append(
                        f"event {event.event_id!r} previous_event_id {previous_id!r} "
                        f"does not reciprocate next_event_id"
                    )

        next_id = event.next_event_id
        if next_id:
            next_event = event_by_id.get(next_id)
            if next_event is None:
                errors.append(
                    f"event {event.event_id!r} references missing next_event_id {next_id!r}"
                )
            else:
                if next_event.time_index < event.time_index:
                    errors.append(
                        f"event {event.event_id!r} next_event_id {next_id!r} is not later"
                    )
                if next_event.previous_event_id and next_event.previous_event_id != event.event_id:
                    errors.append(
                        f"event {event.event_id!r} next_event_id {next_id!r} "
                        f"does not reciprocate previous_event_id"
                    )

        for causal_id in event.causal_links:
            cause = event_by_id.get(causal_id)
            if cause is None:
                errors.append(
                    f"event {event.event_id!r} references missing causal_link {causal_id!r}"
                )
            elif cause.time_index > event.time_index:
                errors.append(
                    f"event {event.event_id!r} causal_link {causal_id!r} is not earlier"
                )

    return errors


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
            records.append(record)
    return records
