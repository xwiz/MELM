"""Hand-authored child-dialogue memory fixture."""

from __future__ import annotations

from melm.memory import Event

from .evidence import EvidenceCase
from .synthetic_episodic import EpisodicCase


def authored_child_dialogue_fixture() -> tuple[list[Event], list[EpisodicCase], list[EvidenceCase]]:
    """Return hand-authored dialogue-like events and answerability cases.

    This fixture is still small, but unlike the generated benchmark it avoids a
    single repeated template. It is meant as a bridge between synthetic mechanics
    checks and real transcript/dialogue evaluation.
    """

    events = _events()
    recall_cases = _recall_cases()
    evidence_cases = [
        EvidenceCase(
            query=case.query,
            expected_event_id=case.expected_event_id,
            category=f"positive_{case.category}",
            story_id=case.expected_event_id.split("_", 1)[0],
        )
        for case in recall_cases
    ]
    evidence_cases.extend(_negative_cases())
    return events, recall_cases, evidence_cases


def _events() -> list[Event]:
    return [
        Event(
            event_id="dialogue1_e1",
            source_span=(
                "Maya whispered that she tucked the moon sticker inside the purple "
                "lunchbox before art class."
            ),
            time_index=1,
            actors=("maya",),
            action_or_state="tucked",
            objects=("moon sticker", "purple lunchbox"),
            location="art table",
            next_event_id="dialogue1_e2",
        ),
        Event(
            event_id="dialogue1_e2",
            source_span="Leo carried the purple lunchbox from the art table to the reading rug.",
            time_index=2,
            actors=("leo",),
            action_or_state="carried",
            objects=("purple lunchbox", "art table", "reading rug"),
            location="classroom",
            previous_event_id="dialogue1_e1",
            next_event_id="dialogue1_e3",
        ),
        Event(
            event_id="dialogue1_e3",
            source_span="Maya checked the art table and asked where the moon sticker had gone.",
            time_index=3,
            actors=("maya",),
            action_or_state="asked",
            objects=("moon sticker", "art table"),
            location="classroom",
            previous_event_id="dialogue1_e2",
            next_event_id="dialogue1_e4",
        ),
        Event(
            event_id="dialogue1_e4",
            source_span="Sam said he saw Leo place the purple lunchbox beside the reading rug.",
            time_index=4,
            actors=("sam", "leo"),
            action_or_state="said",
            objects=("purple lunchbox", "reading rug"),
            location="classroom",
            previous_event_id="dialogue1_e3",
            causal_links=("dialogue1_e2",),
        ),
        Event(
            event_id="dialogue2_e1",
            source_span="Nora built a green block tower under the sunny window.",
            time_index=10,
            actors=("nora",),
            action_or_state="built",
            objects=("green block tower", "sunny window"),
            location="play corner",
            next_event_id="dialogue2_e2",
        ),
        Event(
            event_id="dialogue2_e2",
            source_span="Iris bumped the tower while reaching for the green cape.",
            time_index=11,
            actors=("iris",),
            action_or_state="bumped",
            objects=("green block tower", "green cape"),
            location="play corner",
            previous_event_id="dialogue2_e1",
            next_event_id="dialogue2_e3",
        ),
        Event(
            event_id="dialogue2_e3",
            source_span="Nora rebuilt the green blocks on the blue mat.",
            time_index=12,
            actors=("nora",),
            action_or_state="rebuilt",
            objects=("green blocks", "blue mat"),
            location="play corner",
            previous_event_id="dialogue2_e2",
            next_event_id="dialogue2_e4",
        ),
        Event(
            event_id="dialogue2_e4",
            source_span="Owen reminded Nora that the first tower had been under the window.",
            time_index=13,
            actors=("owen", "nora"),
            action_or_state="reminded",
            objects=("green block tower", "window"),
            location="play corner",
            previous_event_id="dialogue2_e3",
            causal_links=("dialogue2_e1",),
        ),
        Event(
            event_id="dialogue3_e1",
            source_span="Owen put the tiny shell in the blue bucket near the sandbox.",
            time_index=20,
            actors=("owen",),
            action_or_state="put",
            objects=("tiny shell", "blue bucket"),
            location="sandbox",
            next_event_id="dialogue3_e2",
        ),
        Event(
            event_id="dialogue3_e2",
            source_span="Maya poured sand into the red pail while Owen washed his hands.",
            time_index=21,
            actors=("maya", "owen"),
            action_or_state="poured",
            objects=("sand", "red pail"),
            location="sandbox",
            previous_event_id="dialogue3_e1",
            next_event_id="dialogue3_e3",
        ),
        Event(
            event_id="dialogue3_e3",
            source_span="Later, Iris moved the tiny shell from the blue bucket to the red pail.",
            time_index=22,
            actors=("iris",),
            action_or_state="moved",
            objects=("tiny shell", "blue bucket", "red pail"),
            location="sandbox",
            previous_event_id="dialogue3_e2",
            causal_links=("dialogue3_e1",),
        ),
        Event(
            event_id="dialogue3_e4",
            source_span="Owen looked in the blue bucket and frowned because the shell was gone.",
            time_index=23,
            actors=("owen",),
            action_or_state="looked",
            objects=("tiny shell", "blue bucket"),
            location="sandbox",
            previous_event_id="dialogue3_e3",
        ),
    ]


def _recall_cases() -> list[EpisodicCase]:
    return [
        EpisodicCase(
            query="Where did Maya tuck the moon sticker?",
            expected_event_id="dialogue1_e1",
            category="direct",
        ),
        EpisodicCase(
            query="What happened right after Maya tucked away the moon sticker?",
            expected_event_id="dialogue1_e2",
            category="temporal_after",
        ),
        EpisodicCase(
            query="Who said where the purple lunchbox was?",
            expected_event_id="dialogue1_e4",
            category="witness",
        ),
        EpisodicCase(
            query="What earlier event explains why Sam knew about the purple lunchbox?",
            expected_event_id="dialogue1_e2",
            category="causal_source",
        ),
        EpisodicCase(
            query="Where did Nora first build the green block tower?",
            expected_event_id="dialogue2_e1",
            category="direct",
        ),
        EpisodicCase(
            query="What happened right before Nora rebuilt the green blocks?",
            expected_event_id="dialogue2_e2",
            category="temporal_before",
        ),
        EpisodicCase(
            query="Who reminded Nora where the first tower had been?",
            expected_event_id="dialogue2_e4",
            category="witness",
        ),
        EpisodicCase(
            query="Where did Owen put the tiny shell at first?",
            expected_event_id="dialogue3_e1",
            category="direct",
        ),
        EpisodicCase(
            query="What changed where the tiny shell was later?",
            expected_event_id="dialogue3_e3",
            category="entity_conflict",
        ),
        EpisodicCase(
            query="Where was the tiny shell before Iris moved it?",
            expected_event_id="dialogue3_e1",
            category="entity_conflict_before_move",
        ),
    ]


def _negative_cases() -> list[EvidenceCase]:
    return [
        EvidenceCase(
            query="Where did Maya hide the silver train?",
            expected_event_id=None,
            category="negative_unknown_object",
            story_id="dialogue1",
        ),
        EvidenceCase(
            query="Where was the moon sticker after Sam moved it?",
            expected_event_id=None,
            category="negative_wrong_actor",
            story_id="dialogue1",
        ),
        EvidenceCase(
            query="Where did Nora put the thing?",
            expected_event_id=None,
            category="negative_underspecified",
            story_id="dialogue2",
        ),
        EvidenceCase(
            query="Who explained the missing telescope?",
            expected_event_id=None,
            category="negative_unknown_event",
            story_id="dialogue2",
        ),
        EvidenceCase(
            query="Where was the tiny shell after Owen moved it to the shelf?",
            expected_event_id=None,
            category="negative_false_presupposition",
            story_id="dialogue3",
        ),
    ]
