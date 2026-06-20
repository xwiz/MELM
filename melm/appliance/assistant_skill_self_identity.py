"""Self-identity derivation module.

Derives the system's sense of self from usage patterns stored as
personal_experience entities in the entity store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from melm.contracts import load_self_identity

_SELF_IDENTITY_CONTRACT: dict[str, Any] | None = None


def _get_self_identity_contract() -> dict[str, Any]:
    global _SELF_IDENTITY_CONTRACT
    if _SELF_IDENTITY_CONTRACT is None:
        _SELF_IDENTITY_CONTRACT = load_self_identity()
    return _SELF_IDENTITY_CONTRACT


@dataclass
class DerivedIdentity:
    user_id: str
    highest_meaning_intent: str
    highest_meaning_polarity: float
    top_intent: str
    top_intent_count: int
    top_intent_mean_polarity: float
    total_turns: int
    per_intent_counts: dict[str, int] = field(default_factory=dict)
    per_intent_mean_polarities: dict[str, float] = field(default_factory=dict)
    has_name: bool = False
    given_name: str | None = None
    derived_at: str = ""


def analyze_user_identity(
    store: Any,
    user_id: str,
    days: int = 30,
) -> DerivedIdentity | None:
    contract = _get_self_identity_contract()
    min_data_points = contract.get("min_data_points", 3)

    entities = store.find_entities(kind="personal_experience")

    polarity_by_intent: dict[str, list[float]] = {}
    count_by_intent: dict[str, int] = {}

    for entity in entities:
        uid_slot = store.get_entity_slot(entity.entity_id, "user_id")
        if uid_slot is None:
            continue
        uid = json.loads(uid_slot.value_json)
        if uid != user_id:
            continue

        if "turn: " not in entity.label:
            continue
        intent = entity.label.split("turn: ", 1)[1]

        pol_slot = store.get_entity_slot(entity.entity_id, "polarity")
        if pol_slot is not None:
            polarity = json.loads(pol_slot.value_json)
            if isinstance(polarity, (int, float)):
                polarity_by_intent.setdefault(intent, []).append(float(polarity))

        count_by_intent[intent] = count_by_intent.get(intent, 0) + 1

    total_turns = sum(count_by_intent.values())
    if total_turns < min_data_points:
        return None

    per_intent_counts = dict(count_by_intent)
    per_intent_mean_polarities: dict[str, float] = {}
    for intent, vals in polarity_by_intent.items():
        per_intent_mean_polarities[intent] = sum(vals) / len(vals) if vals else 0.0

    intents_with_polarity = [
        (intent, per_intent_mean_polarities.get(intent, 0.0))
        for intent in per_intent_counts
    ]
    intents_with_polarity.sort(key=lambda x: (-x[1], -per_intent_counts[x[0]], x[0]))
    highest_meaning_intent = intents_with_polarity[0][0]
    highest_meaning_polarity = intents_with_polarity[0][1]

    intents_with_count = [
        (intent, per_intent_counts[intent])
        for intent in per_intent_counts
    ]
    intents_with_count.sort(
        key=lambda x: (
            -x[1],
            -per_intent_mean_polarities.get(x[0], 0.0),
            x[0],
        )
    )
    top_intent = intents_with_count[0][0]
    top_intent_count = intents_with_count[0][1]
    top_intent_mean_polarity = per_intent_mean_polarities.get(top_intent, 0.0)

    state = store.load_self_state()
    has_name = bool(state.get("has_name", False))
    given_name = state.get("given_name", None)

    return DerivedIdentity(
        user_id=user_id,
        highest_meaning_intent=highest_meaning_intent,
        highest_meaning_polarity=highest_meaning_polarity,
        top_intent=top_intent,
        top_intent_count=top_intent_count,
        top_intent_mean_polarity=top_intent_mean_polarity,
        total_turns=total_turns,
        per_intent_counts=per_intent_counts,
        per_intent_mean_polarities=per_intent_mean_polarities,
        has_name=has_name,
        given_name=given_name,
        derived_at=datetime.now(timezone.utc).isoformat(),
    )


def derive_identity_narrative(
    identity: DerivedIdentity,
    mood_id: str,
) -> str | None:
    contract = _get_self_identity_contract()

    identity_labels = contract.get("identity_labels", {})
    entry = identity_labels.get(identity.highest_meaning_intent)
    if entry is None:
        return None
    label = entry.get("label", "")

    narratives = contract.get("identity_narratives", {})
    template = narratives.get(mood_id, narratives.get("neutral"))
    if template is None:
        return None

    return template.format(label=label)


def derive_identity_explanation(
    identity: DerivedIdentity,
) -> str | None:
    contract = _get_self_identity_contract()

    identity_labels = contract.get("identity_labels", {})
    entry = identity_labels.get(identity.highest_meaning_intent)
    if entry is None:
        return None
    frame = entry.get("frame", "")

    templates = contract.get("name_awareness_templates", {})
    template = templates.get("why")
    if template is None:
        return None

    count = identity.top_intent_count
    polarity = identity.top_intent_mean_polarity
    formatted_polarity = f"{polarity:+0.1f}"

    return template.format(frame=frame, count=count, polarity=formatted_polarity)


def get_name_awareness_template(
    identity: DerivedIdentity,
    template_key: str,
) -> str | None:
    contract = _get_self_identity_contract()

    templates = contract.get("name_awareness_templates", {})
    template = templates.get(template_key)
    if template is None:
        return None

    identity_labels = contract.get("identity_labels", {})
    entry = identity_labels.get(identity.highest_meaning_intent)
    if entry is None:
        return None
    label = entry.get("label", "")

    return template.format(label=label)
