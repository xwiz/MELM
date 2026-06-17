"""T2 personal_experience entity writer — records conversation turns as entities.

Every conversation turn processed by the kernel becomes an entity of kind
``personal_experience`` in the entity store.  The entity carries slots defined
in ``seed_class_schemas()``:

    outcome          — resolved / unresolved / escalated / abandoned
    polarity         — aggregate sentiment -1.0 to +1.0 (default 0.0)
    learned_fact_ids — JSON array of fact entity IDs (initially [])
    follow_up        — next-action hint (initially None)
    intent_achieved  — yes / partial / no

This is T2 (conversation meaning) in the three-timescale meaning model.
T1 (utterance meaning) is the UOL parse.  T3 (historical meaning) is the
accumulated lexicon + entity store.
"""

import uuid
from typing import Any

from .assistant_os_store import AssistantOSStore
from .assistant_synthesis import BoundedSynthesisResult
from .local_assistant_router import AssistantDecision


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_conversation_experience(
    store: AssistantOSStore,
    decision: AssistantDecision,
    synthesis: BoundedSynthesisResult | None,
) -> str | None:
    """Write a ``personal_experience`` entity for a single conversation turn.

    Parameters
    ----------
    store:
        The active entity store.
    decision:
        The ``AssistantDecision`` produced by the router.
    synthesis:
        The ``BoundedSynthesisResult`` from the synthesizer, or *None* when
        synthesis was bypassed (e.g. privacy control with no applied answer).

    Returns
    -------
    str | None
        The entity_id of the created entity, or *None* if no store was given.
    """
    outcome = _compute_outcome(synthesis, decision)
    polarity = _compute_polarity(synthesis, decision)
    intent_achieved = _compute_intent_achieved(synthesis)
    entity_id = f"pe_{uuid.uuid4().hex[:12]}"

    store.add_entity(
        entity_id=entity_id,
        kind="personal_experience",
        label=f"turn: {decision.intent}",
        semantic_class_id="personal_experience",
        canonical_lemma=decision.utterance[:80],
    )
    store.set_entity_slot(
        entity_id, "outcome", outcome,
        provenance="experience_writer",
    )
    store.set_entity_slot(
        entity_id, "polarity", polarity,
        provenance="experience_writer",
    )
    learned_fact_ids = [
        key.split(".", 1)[1]
        for key in decision.evidence_keys
        if key.startswith("learned_fact.")
    ]
    store.set_entity_slot(
        entity_id, "learned_fact_ids", learned_fact_ids,
        provenance="experience_writer",
    )
    store.set_entity_slot(
        entity_id, "follow_up", None,
        provenance="experience_writer",
    )
    store.set_entity_slot(
        entity_id, "intent_achieved", intent_achieved,
        provenance="experience_writer",
    )
    return entity_id


# ---------------------------------------------------------------------------
# Slot-value helpers  (kept as module-level functions so they can be tested
# independently and swapped without touching the caller)
# ---------------------------------------------------------------------------

def _compute_outcome(
    synthesis: BoundedSynthesisResult | None,
    decision: AssistantDecision,
) -> str:
    """Derive the ``outcome`` slot from the synthesis result.

    Priority:
        1. ``boundary_crossed`` is truthy         → ``"escalated"``
        2. ``refused`` is True                     → ``"abandoned"``
        3. ``applied`` is True                     → ``"resolved"``
        4. everything else (no synthesis, etc.)    → ``"unresolved"``
    """
    if synthesis is None:
        return "unresolved"
    if synthesis.boundary_crossed and synthesis.boundary_crossed not in ("none", ""):
        return "escalated"
    if synthesis.refused:
        return "abandoned"
    if synthesis.applied:
        return "resolved"
    return "unresolved"


def _compute_intent_achieved(
    synthesis: BoundedSynthesisResult | None,
) -> str:
    """Derive the ``intent_achieved`` slot from the synthesis result.

    Only ``applied=True`` with no refusal counts as a definite ``"yes"``.
    Everything else defaults to ``"no"`` — the assistant did not produce a
    useful answer for this turn.
    """
    if synthesis is not None and synthesis.applied and not synthesis.refused:
        return "yes"
    return "no"


def _compute_polarity(
    synthesis: BoundedSynthesisResult | None,
    decision: AssistantDecision,
) -> float:
    """Derive the ``polarity`` slot from the synthesis result and decision.

    Current implementation is deliberately simple — polarity defaults to 0.0
    (neutral).  A future enhancement could inspect evidence quality, the
    presence of positive/negative UOL frame roles, or user feedback.
    """
    return 0.0
