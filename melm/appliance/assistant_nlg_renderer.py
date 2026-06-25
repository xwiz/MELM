"""Contract-backed deterministic NLG renderer consuming SemanticAttentionPacket.

Reads nlg_atomic_renderers.v1.json to select the best matching renderer family
and renders from packet slots. No domain knowledge outside contracts.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from melm.contracts import load_nlg_atomic_renderers
from melm.appliance.assistant_semantic_attention import SemanticAttentionPacket

__all__ = [
    "render_from_packet",
    "score_renderer_family",
    "render_template",
]


_RendererCache: dict[str, Any] | None = None


def _get_renderers() -> dict[str, Any]:
    global _RendererCache
    if _RendererCache is not None:
        return _RendererCache
    try:
        _RendererCache = load_nlg_atomic_renderers()
    except Exception:
        _RendererCache = {}
    return _RendererCache


def score_renderer_family(
    family_id: str,
    family: dict[str, Any],
    packet: SemanticAttentionPacket,
) -> float:
    """Return a match score (0.0 = no match, higher = better match)."""
    packet_dict = asdict(packet)
    required = family.get("required_conditions", {})
    forbidden = family.get("forbidden_conditions", {})
    predicate_hints = family.get("predicate_hints", [])
    extra = family.get("extra_conditions", {})

    for slot_name, condition in required.items():
        present = condition.get("present", False)
        equals = condition.get("equals", None)
        actual = packet_dict.get(slot_name)

        if present and not actual:
            return 0.0
        if present and isinstance(actual, (list, tuple)) and not actual:
            return 0.0
        if equals is not None:
            if isinstance(equals, bool) and isinstance(actual, bool):
                if actual != equals:
                    return 0.0
            elif str(actual).lower() != str(equals).lower():
                return 0.0

    for slot_name, condition in forbidden.items():
        present = condition.get("present", False)
        equals = condition.get("equals", None)
        actual = packet_dict.get(slot_name)

        if present and actual:
            return 0.0
        if present and isinstance(actual, (list, tuple)) and actual:
            return 0.0
        if equals is not None:
            if isinstance(equals, bool) and isinstance(actual, bool):
                if actual == equals:
                    return 0.0
            elif str(actual).lower() == str(equals).lower():
                return 0.0

    if extra.get("normalization_alerts_present") and not packet.normalization_alerts:
        return 0.0
    if extra.get("skill_absent") and packet.capability.installed:
        return 0.0

    base_score = float(family.get("priority", 50))
    bonus = 0.0
    if predicate_hints and packet.predicate:
        for hint in predicate_hints:
            if hint in packet.predicate.lower():
                bonus += 5.0
                break

    return base_score + bonus


def render_template(template: str, packet: SemanticAttentionPacket) -> str:
    """Render a template string with packet slot values."""
    packet_dict = asdict(packet)
    result = template

    if "{topic_if_entity}" in result:
        if packet.content_entities:
            replacement = packet.content_entities[0].get("lemma", packet.task_topic)
        else:
            replacement = packet.task_topic
        result = result.replace("{topic_if_entity}", replacement)

    for key, value in packet_dict.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            if isinstance(value, str):
                result = result.replace(placeholder, value)
            elif isinstance(value, bool):
                result = result.replace(placeholder, str(value).lower())
            elif isinstance(value, (list, tuple)):
                result = result.replace(placeholder, ", ".join(str(v) for v in value))
            elif isinstance(value, dict):
                continue
            else:
                result = result.replace(placeholder, str(value))

    for key in ["task_topic", "predicate", "speech_act", "output_type"]:
        placeholder = "{" + key + "}"
        if placeholder in result:
            val = packet_dict.get(key, "")
            result = result.replace(placeholder, str(val))

    return result


def render_from_packet(packet: SemanticAttentionPacket) -> str | None:
    """Render an answer from the packet, or return None if no renderer matches."""
    renderers = _get_renderers()
    families = renderers.get("renderer_families", {})

    if not families:
        return None

    best_family_id: str | None = None
    best_family: dict[str, Any] | None = None
    best_score = -1.0

    for family_id, family in families.items():
        score = score_renderer_family(family_id, family, packet)
        if score > best_score:
            best_score = score
            best_family_id = family_id
            best_family = family

    if best_family is None or best_score <= 0:
        return None

    templates = best_family.get("templates", [])
    if not templates:
        return None

    parts: list[str] = []

    if not packet.capability.installed and packet.task_topic:
        parts.append(
            f"I do not have a dedicated {packet.task_topic} skill installed. "
            f"I can use {packet.capability.fallback}."
        )

    if packet.task_topic_class:
        parts.append(
            f"{packet.task_topic} is understood as {packet.task_topic_class} "
            f"from {packet.task_topic_source}."
        )

    if packet.learned_summary:
        parts.append(f"local learned fact: {packet.learned_summary}")

    template = templates[0]
    rendered = render_template(template, packet)
    parts.append(rendered)

    if packet.normalization_alerts:
        tech_tokens = ", ".join(packet.normalization_alerts)
        parts.append(
            f"Note: some technical terms ({tech_tokens}) may appear simplified."
        )

    return " ".join(parts)
