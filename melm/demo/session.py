"""JSONL-backed persistent dialogue session."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

from melm.memory import Event

from .persistent_dialogue import DialogueDemoResponse, PersistentDialogueDemo


EVENT_TUPLE_FIELDS = ("actors", "objects", "causal_links")


class PersistentDialogueSession:
    """Small reloadable event log for the persistent dialogue demo."""

    def __init__(
        self,
        path: str | Path,
        *,
        threshold: float = 1.25,
        k: int = 2,
        confidence_method: str = "score_with_evidence_veto",
    ) -> None:
        self.path = Path(path)
        self.threshold = threshold
        self.k = k
        self.confidence_method = confidence_method
        self._events = tuple(load_session_events(self.path))
        self._demo = self._build_demo()

    def events(self) -> tuple[Event, ...]:
        return self._events

    def replace_events(self, events: list[Event] | tuple[Event, ...]) -> None:
        self._events = tuple(events)
        save_session_events(self.path, self._events)
        self._demo = self._build_demo()

    def remember(self, event: Event) -> Event:
        if self._event_ids() & {event.event_id}:
            raise ValueError(f"duplicate event_id {event.event_id!r}")

        events = list(self._events)
        if events:
            previous = events[-1]
            if event.previous_event_id is None:
                event = replace(event, previous_event_id=previous.event_id)
            if previous.next_event_id is None:
                events[-1] = replace(previous, next_event_id=event.event_id)

        events.append(event)
        self._events = tuple(events)
        save_session_events(self.path, self._events)
        self._demo = self._build_demo()
        return event

    def remember_observation(
        self,
        source_span: str,
        *,
        actors: tuple[str, ...] = (),
        action_or_state: str = "",
        objects: tuple[str, ...] = (),
        location: str | None = None,
        causal_links: tuple[str, ...] = (),
        salience: float = 1.0,
        surprise_score: float = 0.0,
        metadata: dict[str, str] | None = None,
    ) -> Event:
        event = Event(
            event_id=self._next_event_id(),
            source_span=source_span,
            time_index=self._next_time_index(),
            actors=actors,
            action_or_state=action_or_state,
            objects=objects,
            location=location,
            causal_links=causal_links,
            salience=salience,
            surprise_score=surprise_score,
            metadata=metadata or {},
        )
        return self.remember(event)

    def ask(self, query: str) -> DialogueDemoResponse:
        return self._demo.ask(query)

    def _build_demo(self) -> PersistentDialogueDemo:
        return PersistentDialogueDemo(
            self._events,
            threshold=self.threshold,
            k=self.k,
            confidence_method=self.confidence_method,
        )

    def _event_ids(self) -> set[str]:
        return {event.event_id for event in self._events}

    def _next_event_id(self) -> str:
        stem = self.path.stem or "session"
        return f"{stem}_e{len(self._events) + 1:04d}"

    def _next_time_index(self) -> int:
        if not self._events:
            return 1
        return max(event.time_index for event in self._events) + 1


def save_session_events(path: str | Path, events: tuple[Event, ...] | list[Event]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(_event_to_record(event), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def load_session_events(path: str | Path) -> list[Event]:
    source = Path(path)
    if not source.exists():
        return []

    events: list[Event] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSONL record") from exc
            event = _event_from_record(record)
            if event.event_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate event_id {event.event_id!r}")
            seen_ids.add(event.event_id)
            events.append(event)
    return events


def _event_to_record(event: Event) -> dict[str, Any]:
    record = asdict(event)
    record["schema"] = "melm.demo.event.v1"
    return record


def _event_from_record(record: dict[str, Any]) -> Event:
    record = dict(record)
    record.pop("schema", None)
    for field in EVENT_TUPLE_FIELDS:
        record[field] = tuple(record.get(field) or ())
    metadata = record.get("metadata") or {}
    record["metadata"] = {str(key): str(value) for key, value in metadata.items()}
    return Event(**record)
