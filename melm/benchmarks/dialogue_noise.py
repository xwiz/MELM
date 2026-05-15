"""Paraphrased dialogue evidence cases for transcript smoke tests."""

from __future__ import annotations

from melm.memory import Event

from .evidence import EvidenceCase


def sample_transcript_distractor_events() -> list[Event]:
    """Return conflicting but irrelevant events for the sample transcript."""

    return [
        Event(
            event_id="sample_d1",
            source_span="Jules put a silver tag inside the red bin during cleanup.",
            time_index=50,
            actors=("jules",),
            action_or_state="put",
            objects=("silver tag", "red bin"),
            location="classroom",
            metadata={"transcript_id": "sample_child_dialogue_001", "distractor": "true"},
        ),
        Event(
            event_id="sample_d2",
            source_span="Rafi moved a green bead from the tray to the shelf.",
            time_index=51,
            actors=("rafi",),
            action_or_state="moved",
            objects=("green bead", "tray", "shelf"),
            location="classroom",
            metadata={"transcript_id": "sample_child_dialogue_001", "distractor": "true"},
        ),
        Event(
            event_id="sample_d3",
            source_span="Mina stacked blue cups beside the window after snack.",
            time_index=52,
            actors=("mina",),
            action_or_state="stacked",
            objects=("blue cups", "window"),
            location="classroom",
            metadata={"transcript_id": "sample_child_dialogue_001", "distractor": "true"},
        ),
        Event(
            event_id="sample_d4",
            source_span="Ben slid a yellow card under the puzzle shelf.",
            time_index=53,
            actors=("ben",),
            action_or_state="slid",
            objects=("yellow card", "puzzle shelf"),
            location="classroom",
            metadata={"transcript_id": "sample_child_dialogue_001", "distractor": "true"},
        ),
    ]


def sample_transcript_noisy_evidence_cases() -> list[EvidenceCase]:
    """Return paraphrased answerability cases for the sample transcript."""

    story_id = "sample_child_dialogue_001"
    return [
        EvidenceCase(
            query="Do you remember where Lila left the rainbow bead?",
            expected_event_id="sample_e1",
            category="noisy_positive_direct",
            story_id=story_id,
        ),
        EvidenceCase(
            query="After Lila put away the rainbow bead, what came next?",
            expected_event_id="sample_e2",
            category="noisy_positive_temporal_after",
            story_id=story_id,
        ),
        EvidenceCase(
            query="Which child noticed where the yellow box went?",
            expected_event_id="sample_e4",
            category="noisy_positive_witness",
            story_id=story_id,
        ),
        EvidenceCase(
            query="Why could Rafi say where the yellow box was?",
            expected_event_id="sample_e2",
            category="noisy_positive_causal_source",
            story_id=story_id,
        ),
        EvidenceCase(
            query="At first, where were Mina's star blocks?",
            expected_event_id="sample_e5",
            category="noisy_positive_direct",
            story_id=story_id,
        ),
        EvidenceCase(
            query="Just before Mina rebuilt the star blocks, what happened?",
            expected_event_id="sample_e6",
            category="noisy_positive_temporal_before",
            story_id=story_id,
        ),
        EvidenceCase(
            query="Did Lila hide a silver button somewhere?",
            expected_event_id=None,
            category="noisy_negative_unknown_object",
            story_id=story_id,
        ),
        EvidenceCase(
            query="After Rafi moved the rainbow bead, where was it?",
            expected_event_id=None,
            category="noisy_negative_wrong_actor",
            story_id=story_id,
        ),
        EvidenceCase(
            query="Where was Mina's thing?",
            expected_event_id=None,
            category="noisy_negative_underspecified",
            story_id=story_id,
        ),
        EvidenceCase(
            query="Who explained where the missing robot went?",
            expected_event_id=None,
            category="noisy_negative_unknown_event",
            story_id=story_id,
        ),
    ]
