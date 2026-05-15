"""State-resolution benchmark cases derived from synthetic event chains."""

from __future__ import annotations

from dataclasses import dataclass

from .synthetic_episodic import (
    ALTERNATE_SURFACES,
    LOCATIONS,
    OBJECTS,
    PEOPLE,
    _surface_for,
    generate_synthetic_episodic_benchmark,
)


@dataclass(frozen=True)
class StateResolutionCase:
    query: str
    object_name: str
    expected_location: str | None
    category: str
    before_event_id: str | None = None
    at_or_before_event_id: str | None = None
    story_id: str = ""


def synthetic_state_resolution_fixture(
    stories: int = 10,
    *,
    seed: int = 13,
    distractors_per_story: int = 1,
):
    """Generate object-location state cases from the episodic benchmark."""

    events, _recall_cases = generate_synthetic_episodic_benchmark(
        stories=stories,
        seed=seed,
        distractors_per_story=distractors_per_story,
    )
    cases: list[StateResolutionCase] = []

    for index in range(stories):
        protagonist = PEOPLE[index % len(PEOPLE)]
        helper = PEOPLE[(index + 1) % len(PEOPLE)]
        obj = f"{OBJECTS[index % len(OBJECTS)]} {index}"
        location = LOCATIONS[index % len(LOCATIONS)]
        surface = _surface_for(location)
        alternate_surface = ALTERNATE_SURFACES[index % len(ALTERNATE_SURFACES)]
        story_id = f"s{index:03d}"
        put_event_id = f"{story_id}_e1"
        moved_event_id = f"{story_id}_e5"

        cases.extend(
            [
                StateResolutionCase(
                    query=f"Where is the {obj} now?",
                    object_name=obj,
                    expected_location=alternate_surface,
                    category="latest_after_move",
                    story_id=story_id,
                ),
                StateResolutionCase(
                    query=f"Where was the {obj} before {helper} moved it later?",
                    object_name=obj,
                    expected_location=surface,
                    category="before_move",
                    before_event_id=moved_event_id,
                    story_id=story_id,
                ),
                StateResolutionCase(
                    query=f"Where was the {obj} right after {protagonist} put it down?",
                    object_name=obj,
                    expected_location=surface,
                    category="at_initial_put",
                    at_or_before_event_id=put_event_id,
                    story_id=story_id,
                ),
                StateResolutionCase(
                    query=f"Where is the purple violin {index} now?",
                    object_name=f"purple violin {index}",
                    expected_location=None,
                    category="unknown_object",
                    story_id=story_id,
                ),
            ]
        )

    return events, cases
