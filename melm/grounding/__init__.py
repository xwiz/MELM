"""Deterministic grounding primitives inspired by nameless_vector."""

from .state import (
    DEFAULT_CONTRADICTIONS,
    StateSet,
    TransitionCheck,
    check_transition,
)

__all__ = [
    "DEFAULT_CONTRADICTIONS",
    "StateSet",
    "TransitionCheck",
    "check_transition",
]
