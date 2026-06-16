"""Memory formatting skill module — radial consumer of knowledge contracts."""

from __future__ import annotations

from typing import Any

from melm.contracts import load_memory_insights
from .assistant_skill_base import SkillManifest, register_skill

MANIFEST = SkillManifest(
    family="memory",
    frames=("personal_memory", "autobiographical_memory"),
    knowledge_refs=("memory_insights.v1.json",),
    template_refs={},
)

register_skill(MANIFEST)


def personal_memory_summary(evidence: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for item in evidence:
        kind = getattr(item, "kind", "")
        key = getattr(item, "key", "")
        value = getattr(item, "value", "")
        if kind == "profile":
            label = key.split(".", 1)[1].replace("_", " ")
            if label == "age":
                parts.append(f"your age is {value}")
            elif label == "location":
                parts.append(f"you are in {value}")
            elif label == "culture":
                parts.append(f"your culture hint is {value}")
            elif label == "user name":
                parts.append(f"your name is {value}")
        elif kind == "user_fact":
            label = key.split(".", 1)[1].replace("_", " ")
            parts.append(f"your {label} is {value}")
        elif kind == "preference":
            label = key.split(".", 1)[1].replace("_", " ")
            parts.append(f"your {label} preference is {value}")
    return "; ".join(dict.fromkeys(parts))


def autobiographical_memory_summary(evidence: tuple[Any, ...]) -> str:
    events = [_event_memory_parts(item) for item in evidence if getattr(item, "kind", "") == "event_memory"]
    if not events:
        return ""
    parts: list[str] = []
    for index, event in enumerate(events[:5], start=1):
        label = _event_label(event)
        parts.append(f'{index}. {label} - you said "{event["utterance"]}"')
    insight = _event_memory_insight_text(events)
    if insight:
        parts.append(insight)
    return " ".join(parts)


def autobiographical_session_summary(evidence: tuple[Any, ...]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in evidence:
        if getattr(item, "kind", "") != "event_memory":
            continue
        event = _event_memory_parts(item)
        session_id = event["session_id"]
        grouped.setdefault(session_id, []).append(event)
    if not grouped:
        return ""
    parts: list[str] = []
    all_events: list[dict[str, str]] = []
    for session_index, (session_id, events) in enumerate(grouped.items(), start=1):
        utterances: list[str] = []
        intents: list[str] = []
        for event in events[:4]:
            intents.append(event["intent"].replace("_", " "))
            utterances.append(event["utterance"])
            all_events.append(event)
        intent_text = ", ".join(dict.fromkeys(intents))
        quoted = "; ".join(f'"{utterance}"' for utterance in utterances)
        parts.append(f"session {session_index} ({session_id}) covered {intent_text}: {quoted}")
    insight = _event_memory_insight_text(all_events)
    if insight:
        parts.append(insight)
    return " ".join(parts)


def autobiographical_digest_summary(evidence: tuple[Any, ...]) -> str:
    digests = [item.value for item in evidence if getattr(item, "kind", "") == "memory_digest"]
    if not digests:
        return ""
    return f"local long-horizon memory digest: {digests[0]}"


def _event_memory_parts(item: Any) -> dict[str, str]:
    value = item.value
    label, detail = value.split(": ", 1) if ": " in value else ("event via local_answer", value)
    intent, route = label.split(" via ", 1) if " via " in label else (label, "")
    utterance = detail
    reason = ""
    if " (" in detail and detail.endswith(")"):
        utterance, reason = detail.rsplit(" (", 1)
        reason = reason[:-1]
    session_id = item.source.split(":", 1)[1] if ":" in item.source else "session"
    return {
        "session_id": session_id,
        "intent": intent,
        "route": route,
        "utterance": utterance,
        "reason": reason,
    }


def _event_label(event: dict[str, str]) -> str:
    intent = event.get("intent", "event").replace("_", " ")
    route = event.get("route", "")
    return f"{intent} via {route}" if route else intent


def _event_memory_insight_text(events: list[dict[str, str]]) -> str:
    insights = load_memory_insights()
    rules = insights.get("rules", [])
    consented_text = insights.get("consented_stored_text", "")

    buckets: dict[str, list[str]] = {
        "transitions": [],
        "open_loops": [],
        "action_state": [],
        "boundary_controls": [],
    }
    for event in events:
        intent = event["intent"]
        route = event["route"]
        reason = event["reason"]
        matched = False
        for rule in rules:
            if "intent" in rule and rule["intent"] != intent:
                continue
            if "reason" in rule and rule["reason"] != reason:
                continue
            if "route" in rule and rule["route"] != route:
                continue
            cat = rule["category"]
            if cat in buckets:
                buckets[cat].append(rule["text"])
            matched = True
            break
        if not matched and reason.startswith("consented_") and reason.endswith("_stored") and consented_text:
            buckets["transitions"].append(consented_text)
    sections: list[str] = []
    if buckets["transitions"]:
        sections.append(f"Capability transitions: {_join_short_list(tuple(dict.fromkeys(buckets['transitions'])), fallback='none')}.")
    if buckets["open_loops"]:
        sections.append(f"Open local gaps: {_join_short_list(tuple(dict.fromkeys(buckets['open_loops'])), fallback='none')}.")
    if buckets["action_state"]:
        sections.append(f"Action state: {_join_short_list(tuple(dict.fromkeys(buckets['action_state'])), fallback='none')}.")
    if buckets["boundary_controls"]:
        sections.append(
            f"Boundary controls: {_join_short_list(tuple(dict.fromkeys(buckets['boundary_controls'])), fallback='none')}."
        )
    return " ".join(sections)


def _join_short_list(items: tuple[str, ...], *, fallback: str) -> str:
    if not items:
        return fallback
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"
