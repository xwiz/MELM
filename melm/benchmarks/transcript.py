"""Annotated transcript benchmark compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from melm.memory import Event

from .evidence import EvidenceCase
from .io import validate_dialogue_benchmark
from .state_resolution import StateResolutionCase
from .synthetic_episodic import EpisodicCase


@dataclass(frozen=True)
class TranscriptTurn:
    transcript_id: str
    turn_id: str
    speaker: str
    text: str
    time_index: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotatedTranscriptBenchmark:
    turns: list[TranscriptTurn]
    events: list[Event]
    recall_cases: list[EpisodicCase]
    evidence_cases: list[EvidenceCase]
    state_cases: list[StateResolutionCase] = field(default_factory=list)


def load_annotated_transcript_benchmark(
    path: str | Path,
    *,
    validate: bool = True,
) -> AnnotatedTranscriptBenchmark:
    """Load a single JSONL file containing transcript turns and annotations."""

    turns: list[TranscriptTurn] = []
    events: list[Event] = []
    recall_cases: list[EpisodicCase] = []
    evidence_cases: list[EvidenceCase] = []
    state_cases: list[StateResolutionCase] = []

    for record in _read_jsonl(Path(path)):
        schema = record.get("schema") or record.get("type")
        if schema == "melm.transcript_turn.v1":
            turns.append(_turn_from_record(record))
        elif schema == "melm.event.v1":
            events.append(_event_from_record(record))
        elif schema == "melm.episodic_case.v1":
            recall_cases.append(_recall_case_from_record(record))
        elif schema == "melm.evidence_case.v1":
            evidence_cases.append(_evidence_case_from_record(record))
        elif schema == "melm.state_resolution_case.v1":
            state_cases.append(_state_case_from_record(record))
        else:
            raise ValueError(f"Unsupported annotated transcript schema: {schema!r}")

    benchmark = AnnotatedTranscriptBenchmark(
        turns=turns,
        events=events,
        recall_cases=recall_cases,
        evidence_cases=evidence_cases,
        state_cases=state_cases,
    )

    if validate:
        errors = validate_annotated_transcript_benchmark(benchmark)
        if errors:
            raise ValueError("Invalid annotated transcript benchmark: " + "; ".join(errors))

    return benchmark


def validate_annotated_transcript_benchmark(
    benchmark: AnnotatedTranscriptBenchmark,
) -> list[str]:
    """Return validation errors for annotated transcript benchmark input."""

    errors = validate_dialogue_benchmark(
        benchmark.events,
        benchmark.recall_cases,
        benchmark.evidence_cases,
    )

    turn_ids: set[str] = set()
    duplicate_turn_ids: set[str] = set()
    for turn in benchmark.turns:
        if not turn.turn_id.strip():
            errors.append("transcript turn with empty turn_id")
            continue
        if turn.turn_id in turn_ids:
            duplicate_turn_ids.add(turn.turn_id)
        turn_ids.add(turn.turn_id)
        if not turn.text.strip():
            errors.append(f"transcript turn {turn.turn_id!r} has empty text")
        if not turn.speaker.strip():
            errors.append(f"transcript turn {turn.turn_id!r} has empty speaker")

    if duplicate_turn_ids:
        errors.append(f"duplicate turn_id values: {', '.join(sorted(duplicate_turn_ids))}")

    if turn_ids:
        for event in benchmark.events:
            source_turn_id = event.metadata.get("source_turn_id")
            if not source_turn_id:
                errors.append(f"event {event.event_id!r} missing metadata.source_turn_id")
            elif source_turn_id not in turn_ids:
                errors.append(
                    f"event {event.event_id!r} references missing source_turn_id {source_turn_id!r}"
                )

    event_ids = {event.event_id for event in benchmark.events}
    for state_case in benchmark.state_cases:
        if not state_case.object_name.strip():
            errors.append(f"state case {state_case.query!r} has empty object_name")
        if state_case.before_event_id and state_case.before_event_id not in event_ids:
            errors.append(
                f"state case {state_case.query!r} references missing before_event_id "
                f"{state_case.before_event_id!r}"
            )
        if state_case.at_or_before_event_id and state_case.at_or_before_event_id not in event_ids:
            errors.append(
                f"state case {state_case.query!r} references missing at_or_before_event_id "
                f"{state_case.at_or_before_event_id!r}"
            )

    return errors


def _turn_from_record(record: dict[str, Any]) -> TranscriptTurn:
    metadata = record.get("metadata") or {}
    return TranscriptTurn(
        transcript_id=str(record["transcript_id"]),
        turn_id=str(record["turn_id"]),
        speaker=str(record["speaker"]),
        text=str(record["text"]),
        time_index=int(record["time_index"]),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )


def _event_from_record(record: dict[str, Any]) -> Event:
    record = dict(record)
    record.pop("schema", None)
    for field_name in ("actors", "objects", "causal_links"):
        record[field_name] = tuple(record.get(field_name) or ())
    metadata = record.get("metadata") or {}
    record["metadata"] = {str(key): str(value) for key, value in metadata.items()}
    return Event(**record)


def _recall_case_from_record(record: dict[str, Any]) -> EpisodicCase:
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


def _state_case_from_record(record: dict[str, Any]) -> StateResolutionCase:
    expected_location = record.get("expected_location")
    return StateResolutionCase(
        query=str(record["query"]),
        object_name=str(record["object_name"]),
        expected_location=str(expected_location) if expected_location else None,
        category=str(record.get("category") or "overall"),
        before_event_id=_optional_str(record.get("before_event_id")),
        at_or_before_event_id=_optional_str(record.get("at_or_before_event_id")),
        story_id=str(record.get("story_id") or ""),
    )


def _optional_str(value: Any) -> str | None:
    return str(value) if value else None


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
