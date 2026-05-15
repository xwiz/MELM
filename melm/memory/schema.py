"""Event schema for MELM's external memory prototype."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    event_id: str
    source_span: str
    time_index: int
    actors: tuple[str, ...] = ()
    action_or_state: str = ""
    objects: tuple[str, ...] = ()
    location: str | None = None
    causal_links: tuple[str, ...] = ()
    salience: float = 1.0
    surprise_score: float = 0.0
    previous_event_id: str | None = None
    next_event_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [
            self.source_span,
            " ".join(self.actors),
            self.action_or_state,
            " ".join(self.objects),
            self.location or "",
            " ".join(self.causal_links),
        ]
        return " ".join(part for part in parts if part).lower()

    @property
    def entities(self) -> set[str]:
        raw_entities = set(self.actors)
        raw_entities.update(self.objects)
        if self.location:
            raw_entities.add(self.location)

        entities: set[str] = set()
        for entity in raw_entities:
            normalized = entity.lower().strip()
            if not normalized:
                continue
            entities.add(normalized)
            entities.update(part for part in normalized.replace("-", " ").split() if part)
        return entities
