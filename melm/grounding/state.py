"""Small state algebra for grounding checks.

This is a Python-first prototype inspired by `nameless_vector` concepts. It is
not a port of the Rust code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_CONTRADICTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("open", "closed"),
        ("awake", "asleep"),
        ("wet", "dry"),
        ("hot", "cold"),
        ("present", "absent"),
        ("known", "unknown"),
        ("inside", "outside"),
        ("before", "after"),
    }
)


@dataclass(frozen=True)
class StateSet:
    physical: frozenset[str] = field(default_factory=frozenset)
    emotional: frozenset[str] = field(default_factory=frozenset)
    positional: frozenset[str] = field(default_factory=frozenset)
    mental: frozenset[str] = field(default_factory=frozenset)

    def satisfies(self, required: "StateSet") -> bool:
        return (
            required.physical <= self.physical
            and required.emotional <= self.emotional
            and required.positional <= self.positional
            and required.mental <= self.mental
        )

    def missing(self, required: "StateSet") -> dict[str, set[str]]:
        missing = {
            "physical": set(required.physical - self.physical),
            "emotional": set(required.emotional - self.emotional),
            "positional": set(required.positional - self.positional),
            "mental": set(required.mental - self.mental),
        }
        return {key: value for key, value in missing.items() if value}

    def conflicts(
        self,
        other: "StateSet | None" = None,
        contradictions: frozenset[tuple[str, str]] = DEFAULT_CONTRADICTIONS,
    ) -> list[tuple[str, str, str]]:
        combined = self if other is None else self.merge(other)
        conflicts: list[tuple[str, str, str]] = []
        for dimension in ("physical", "emotional", "positional", "mental"):
            states = getattr(combined, dimension)
            for left, right in contradictions:
                if left in states and right in states:
                    conflicts.append((dimension, left, right))
        return conflicts

    def merge(self, other: "StateSet") -> "StateSet":
        return StateSet(
            physical=self.physical | other.physical,
            emotional=self.emotional | other.emotional,
            positional=self.positional | other.positional,
            mental=self.mental | other.mental,
        )

    @classmethod
    def from_mapping(cls, value: dict[str, list[str] | tuple[str, ...] | set[str]]) -> "StateSet":
        return cls(
            physical=frozenset(value.get("physical", ())),
            emotional=frozenset(value.get("emotional", ())),
            positional=frozenset(value.get("positional", ())),
            mental=frozenset(value.get("mental", ())),
        )


@dataclass(frozen=True)
class TransitionCheck:
    valid: bool
    missing: dict[str, set[str]]
    conflicts: list[tuple[str, str, str]]


def check_transition(current: StateSet, required: StateSet, effects: StateSet) -> TransitionCheck:
    missing = current.missing(required)
    conflicts = current.conflicts(effects)
    return TransitionCheck(
        valid=not missing and not conflicts,
        missing=missing,
        conflicts=conflicts,
    )
