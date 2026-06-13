"""Data-driven frame linker for UOL-based intent classification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from ..contracts import CONTRACT_ROOT, validate_frame_templates


_WEIGHT_REQUIRED = 0.40
_WEIGHT_OPTIONAL = 0.15
_WEIGHT_ACTION = 0.15
_WEIGHT_STRUCTURE = 0.15
_WEIGHT_EXCLUDE_PENALTY = 0.25

# Slot state constants — track fill state for entity slots.
SLOT_STATE_FILLED = "filled"
SLOT_STATE_ASKED_BUT_EMPTY = "asked_but_empty"
SLOT_STATE_UNKNOWN_ENTITY = "unknown_entity"
SLOT_STATE_UNKNOWN = "unknown"
SLOT_STATE_INFERRED = "inferred"


def enrich_candidate_slot_states(
    candidate: FrameCandidate,
    slot_states: dict[str, str],
) -> FrameCandidate:
    return replace(candidate, slot_states=slot_states)


@dataclass(frozen=True)
class FrameCandidate:
    frame_id: str
    intent: str
    score: float
    score_components: dict[str, float]
    threshold: float
    slot_states: dict[str, str] = field(default_factory=dict)  # slot_name → state constant


class FrameLinker:
    """Scores frame templates against input tokens and the runtime lexicon."""

    def __init__(self) -> None:
        self._templates: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        path = CONTRACT_ROOT / "frame_templates.v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_frame_templates(payload)
        self._templates = payload["templates"]

    def score(
        self,
        tokens: tuple[str, ...],
        lexicon: dict[str, frozenset[str]],
        is_question_like: bool = False,
        is_request_like: bool = False,
    ) -> list[FrameCandidate]:
        token_set = set(tokens)
        candidates: list[FrameCandidate] = []

        for fid, tmpl in self._templates.items():
            act = tmpl["activation"]
            score, components = self._score_template(
                token_set, act, lexicon, is_question_like, is_request_like
            )
            threshold = float(tmpl["threshold"])
            if score >= threshold:
                candidates.append(
                    FrameCandidate(
                        frame_id=fid,
                        intent=str(tmpl["intent"]),
                        score=round(score, 4),
                        score_components=components,
                        threshold=threshold,
                    )
                )

        candidates.sort(key=lambda c: (-c.score, c.frame_id))
        return candidates

    def _score_template(
        self,
        token_set: set[str],
        act: dict[str, Any],
        lexicon: dict[str, frozenset[str]],
        is_question_like: bool,
        is_request_like: bool,
    ) -> tuple[float, dict[str, float]]:
        components: dict[str, float] = {}

        required_classes: list[str] = [str(c) for c in act.get("required_classes", [])]
        required_all: list[str] = [str(c) for c in act.get("required_all_classes", [])]
        optional_classes: list[str] = [str(c) for c in act.get("optional_classes", [])]
        exclude_classes: list[str] = [str(c) for c in act.get("exclude_classes", [])]
        action_tokens: list[str] = [str(t) for t in act.get("action_tokens", [])]

        required = self._match_required_classes(token_set, required_classes, lexicon)
        required_and = self._match_required_all_classes(token_set, required_all, lexicon)
        # Use MAX of OR-gate and AND-gate
        required = max(required, required_and)
        components["required"] = round(required, 4)

        if exclude_classes:
            exclude_penalty = self._compute_exclude_penalty(
                token_set, exclude_classes, lexicon
            )
            components["exclude_penalty"] = round(exclude_penalty, 4)
        else:
            exclude_penalty = 0.0

        optional = self._match_optional_classes(token_set, optional_classes, lexicon)
        components["optional"] = round(optional, 4)

        action = self._match_action_tokens(token_set, action_tokens)
        components["action"] = round(action, 4)

        structure = self._match_structure(is_question_like, is_request_like)
        components["structure"] = round(structure, 4)

        score = required + optional + action + structure - exclude_penalty
        score = max(0.0, min(1.0, score))
        return score, components

    def _match_required_all_classes(
        self,
        token_set: set[str],
        required_all: list[str],
        lexicon: dict[str, frozenset[str]],
    ) -> float:
        """Return _WEIGHT_REQUIRED only when EVERY listed class is present (AND gate)."""
        if not required_all:
            return 0.0
        for cls in required_all:
            found = False
            for token in token_set:
                if token in lexicon and cls in lexicon[token]:
                    found = True
                    break
            if not found:
                return 0.0
        return _WEIGHT_REQUIRED

    def _match_required_classes(
        self,
        token_set: set[str],
        required_classes: list[str],
        lexicon: dict[str, frozenset[str]],
    ) -> float:
        if not required_classes:
            return 0.0

        required_set = set(required_classes)
        for token in token_set:
            if token not in lexicon:
                continue
            if required_set & lexicon[token]:
                return _WEIGHT_REQUIRED

        return 0.0

    def _match_optional_classes(
        self,
        token_set: set[str],
        optional_classes: list[str],
        lexicon: dict[str, frozenset[str]],
    ) -> float:
        if not optional_classes:
            return 0.0

        optional_set = set(optional_classes)
        matched: set[str] = set()
        for token in token_set:
            if token not in lexicon:
                continue
            matched.update(optional_set & lexicon[token])

        fraction = len(matched) / len(optional_set)
        return fraction * _WEIGHT_OPTIONAL

    def _compute_exclude_penalty(
        self,
        token_set: set[str],
        exclude_classes: list[str],
        lexicon: dict[str, frozenset[str]],
    ) -> float:
        exclude_set = set(exclude_classes)
        for token in token_set:
            if token in lexicon and (exclude_set & lexicon[token]):
                return _WEIGHT_EXCLUDE_PENALTY
        return 0.0

    def _match_action_tokens(
        self,
        token_set: set[str],
        action_tokens: list[str],
    ) -> float:
        if not action_tokens:
            return 0.0
        if token_set & set(action_tokens):
            return _WEIGHT_ACTION
        return 0.0

    def _match_structure(
        self,
        is_question_like: bool,
        is_request_like: bool,
    ) -> float:
        if is_question_like or is_request_like:
            return _WEIGHT_STRUCTURE
        return 0.0
