"""Synthetic answerability cases for event-memory calibration."""

from __future__ import annotations

from dataclasses import dataclass

from melm.memory import Event

from .synthetic_episodic import (
    OBJECTS,
    PEOPLE,
    generate_synthetic_episodic_benchmark,
)


@dataclass(frozen=True)
class EvidenceCase:
    query: str
    expected_event_id: str | None
    category: str
    story_id: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.expected_event_id is not None


def generate_synthetic_evidence_benchmark(
    stories: int = 10,
    *,
    seed: int = 13,
    distractors_per_story: int = 1,
) -> tuple[list[Event], list[EvidenceCase]]:
    """Generate positive recall cases plus controlled no-answer cases."""

    events, recall_cases = generate_synthetic_episodic_benchmark(
        stories=stories,
        seed=seed,
        distractors_per_story=distractors_per_story,
    )
    evidence_cases = [
        EvidenceCase(
            query=case.query,
            expected_event_id=case.expected_event_id,
            category=f"positive_{case.category}",
            story_id=case.expected_event_id.split("_", 1)[0],
        )
        for case in recall_cases
    ]

    for index in range(stories):
        protagonist = PEOPLE[index % len(PEOPLE)]
        helper = PEOPLE[(index + 1) % len(PEOPLE)]
        witness = PEOPLE[(index + 2) % len(PEOPLE)]
        real_object = f"{OBJECTS[index % len(OBJECTS)]} {index}"
        absent_object = f"purple violin {index}"
        absent_event = f"opened the spaceship {index}"
        underspecified_object = "thing"
        story_id = f"s{index:03d}"

        evidence_cases.extend(
            [
                EvidenceCase(
                    query=f"Where did {protagonist} put the {absent_object}?",
                    expected_event_id=None,
                    category="negative_unknown_object",
                    story_id=story_id,
                ),
                EvidenceCase(
                    query=f"Who told {helper} where the {absent_object} was?",
                    expected_event_id=None,
                    category="negative_unknown_object_witness",
                    story_id=story_id,
                ),
                EvidenceCase(
                    query=f"What happened right after {witness} {absent_event}?",
                    expected_event_id=None,
                    category="negative_unknown_event",
                    story_id=story_id,
                ),
                EvidenceCase(
                    query=f"Where was the {real_object} after {witness} moved it later?",
                    expected_event_id=None,
                    category="negative_wrong_actor",
                    story_id=story_id,
                ),
                EvidenceCase(
                    query=f"Where did {protagonist} put the {underspecified_object}?",
                    expected_event_id=None,
                    category="negative_underspecified",
                    story_id=story_id,
                ),
            ]
        )

    return events, evidence_cases
