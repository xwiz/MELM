"""Deterministic reasoning layer (merged-plan slices 5+).

Pure, stdlib-only solvers gated behind the ``reasoning`` capability family and
dispatched from the router BEFORE closed-intent handlers, so reasoning task
signatures outrank keyword routing (e.g. quantity arithmetic outranks
meal_suggestion). No ML; faithful copy-slot rendering; refuse on missing facts.
"""

from .task_router import detect_reasoning_task
from .solvers import solve
from .implications import MoralContext, derive_moral_context, record_verb_candidate, flush_verb_candidates

__all__ = ["detect_reasoning_task", "solve", "MoralContext", "derive_moral_context",
           "record_verb_candidate", "flush_verb_candidates"]
