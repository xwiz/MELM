"""Small controlled episodic recall fixture."""

from __future__ import annotations

from melm.memory import Event


def episodic_memory_fixture() -> tuple[list[Event], list[tuple[str, str]]]:
    """Return events and `(query, expected_event_id)` cases.

    The fixture is intentionally tiny and deterministic. It exists to validate
    retrieval mechanics before larger generated benchmarks are introduced.
    """

    events = [
        Event(
            event_id="e1",
            source_span="Maya put the red cup on the kitchen table.",
            time_index=1,
            actors=("maya",),
            action_or_state="put",
            objects=("red cup", "table"),
            location="kitchen",
            next_event_id="e2",
        ),
        Event(
            event_id="e2",
            source_span="Leo moved the blue book from the table to the shelf.",
            time_index=2,
            actors=("leo",),
            action_or_state="moved",
            objects=("blue book", "table", "shelf"),
            location="kitchen",
            previous_event_id="e1",
            next_event_id="e3",
        ),
        Event(
            event_id="e3",
            source_span="Maya looked for the red cup before snack time.",
            time_index=3,
            actors=("maya",),
            action_or_state="looked",
            objects=("red cup",),
            location="kitchen",
            previous_event_id="e2",
            next_event_id="e4",
        ),
        Event(
            event_id="e4",
            source_span="Nora told Maya that the red cup was still on the table.",
            time_index=4,
            actors=("nora", "maya"),
            action_or_state="told",
            objects=("red cup", "table"),
            location="kitchen",
            previous_event_id="e3",
        ),
    ]

    cases = [
        ("Where did Maya put the red cup?", "e1"),
        ("What happened right after Maya put the red cup down?", "e2"),
        ("Who told Maya where the red cup was?", "e4"),
        ("What did Maya look for before snack time?", "e3"),
    ]
    return events, cases
