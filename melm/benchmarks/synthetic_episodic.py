"""Deterministic synthetic episodic benchmark generator."""

from __future__ import annotations

from dataclasses import dataclass
import random

from melm.memory import Event


@dataclass(frozen=True)
class EpisodicCase:
    query: str
    expected_event_id: str
    category: str


PEOPLE = ("Maya", "Leo", "Nora", "Sam", "Iris", "Owen")
OBJECTS = ("red cup", "blue book", "green key", "yellow ball", "silver spoon", "paper map")
LOCATIONS = ("kitchen", "garden", "bedroom", "hallway", "porch", "classroom")
ALTERNATE_SURFACES = ("counter", "box", "basket", "drawer", "stool", "cabinet")
FOLLOWUP_ACTIONS = (
    "closed the window",
    "washed their hands",
    "fed the plant",
    "drew a star",
    "rang the bell",
    "folded the blanket",
)


def generate_synthetic_episodic_benchmark(
    stories: int = 10,
    *,
    seed: int = 13,
    distractors_per_story: int = 1,
) -> tuple[list[Event], list[EpisodicCase]]:
    """Generate controlled event chains and recall cases.

    Each story has four core events:

    1. protagonist places an object somewhere;
    2. another actor performs a lexically unrelated follow-up action;
    3. protagonist searches for the object;
    4. witness tells protagonist the object location.

    The "after" cases are intentionally designed to favor structured temporal
    retrieval over plain lexical RAG: the query overlaps strongly with event 1,
    while the expected answer is event 2.

    Optional distractors reuse the same actor/object/location without being the
    target event. They make direct recall less forgiving.
    """

    rng = random.Random(seed)
    events: list[Event] = []
    cases: list[EpisodicCase] = []

    for index in range(stories):
        protagonist = PEOPLE[index % len(PEOPLE)]
        helper = PEOPLE[(index + 1) % len(PEOPLE)]
        witness = PEOPLE[(index + 2) % len(PEOPLE)]
        obj = f"{OBJECTS[index % len(OBJECTS)]} {index}"
        location = LOCATIONS[index % len(LOCATIONS)]
        surface = _surface_for(location)
        followup = rng.choice(FOLLOWUP_ACTIONS)

        base = f"s{index:03d}"
        e1 = f"{base}_e1"
        e2 = f"{base}_e2"
        e3 = f"{base}_e3"
        e4 = f"{base}_e4"
        e5 = f"{base}_e5"
        alternate_surface = ALTERNATE_SURFACES[index % len(ALTERNATE_SURFACES)]

        story_events: list[Event] = []
        for distractor_index in range(distractors_per_story):
            distractor_id = f"{base}_d{distractor_index + 1}"
            story_events.append(
                Event(
                    event_id=distractor_id,
                    source_span=(
                        f"{protagonist} noticed the {obj} near the {surface} "
                        f"while walking through the {location}."
                    ),
                    time_index=index * 10 + distractor_index,
                    actors=(protagonist.lower(),),
                    action_or_state="noticed",
                    objects=(obj, surface),
                    location=location,
                    salience=0.35,
                )
            )

        story_events.extend(
            [
                Event(
                    event_id=e1,
                    source_span=f"{protagonist} put the {obj} on the {surface} in the {location}.",
                    time_index=index * 10 + 1,
                    actors=(protagonist.lower(),),
                    action_or_state="put",
                    objects=(obj, surface),
                    location=location,
                    next_event_id=e2,
                ),
                Event(
                    event_id=e2,
                    source_span=f"{helper} {followup} in the {location}.",
                    time_index=index * 10 + 2,
                    actors=(helper.lower(),),
                    action_or_state=followup.split()[0],
                    objects=tuple(followup.split()[1:]),
                    location=location,
                    previous_event_id=e1,
                    next_event_id=e3,
                ),
                Event(
                    event_id=e3,
                    source_span=f"{protagonist} looked for the {obj} before snack time.",
                    time_index=index * 10 + 3,
                    actors=(protagonist.lower(),),
                    action_or_state="looked",
                    objects=(obj,),
                    location=location,
                    previous_event_id=e2,
                    next_event_id=e4,
                ),
                Event(
                    event_id=e4,
                    source_span=f"{witness} told {protagonist} that the {obj} was still on the {surface}.",
                    time_index=index * 10 + 4,
                    actors=(witness.lower(), protagonist.lower()),
                    action_or_state="told",
                    objects=(obj, surface),
                    location=location,
                    previous_event_id=e3,
                    causal_links=(e1,),
                ),
                Event(
                    event_id=e5,
                    source_span=(
                        f"Later, {helper} moved the {obj} from the {surface} "
                        f"to the {alternate_surface}."
                    ),
                    time_index=index * 10 + 5,
                    actors=(helper.lower(),),
                    action_or_state="moved",
                    objects=(obj, surface, alternate_surface),
                    location=location,
                    previous_event_id=e4,
                    causal_links=(e1,),
                ),
            ]
        )
        events.extend(story_events)

        cases.extend(
            [
                EpisodicCase(
                    query=f"Where did {protagonist} put the {obj}?",
                    expected_event_id=e1,
                    category="direct",
                ),
                EpisodicCase(
                    query=f"What happened right after {protagonist} put the {obj} down?",
                    expected_event_id=e2,
                    category="temporal_after",
                ),
                EpisodicCase(
                    query=f"What did {protagonist} look for before snack time?",
                    expected_event_id=e3,
                    category="direct",
                ),
                EpisodicCase(
                    query=f"Who told {protagonist} where the {obj} was?",
                    expected_event_id=e4,
                    category="witness",
                ),
                EpisodicCase(
                    query=f"What happened right before {witness} told {protagonist} where the {obj} was?",
                    expected_event_id=e3,
                    category="temporal_before",
                ),
                EpisodicCase(
                    query=f"What earlier event explains why {witness} knew where the {obj} was?",
                    expected_event_id=e1,
                    category="causal_source",
                ),
                EpisodicCase(
                    query=f"What changed where the {obj} was later?",
                    expected_event_id=e5,
                    category="entity_conflict",
                ),
                EpisodicCase(
                    query=f"Where was the {obj} before {helper} moved it later?",
                    expected_event_id=e1,
                    category="entity_conflict_before_move",
                ),
            ]
        )

    return events, cases


def _surface_for(location: str) -> str:
    return {
        "kitchen": "table",
        "garden": "bench",
        "bedroom": "dresser",
        "hallway": "mat",
        "porch": "chair",
        "classroom": "desk",
    }.get(location, "table")
