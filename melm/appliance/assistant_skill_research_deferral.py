"""Research deferral skill for the Local Assistant OS.

Decides whether to defer auto-research and manages deferred research tasks.
"""

from __future__ import annotations

from typing import Any

from .assistant_skill_deferred_tasks import queue_deferred_task

_RESEARCH_DEFERRAL = None


def _load_research_triggers():
    global _RESEARCH_DEFERRAL
    if _RESEARCH_DEFERRAL is None:
        try:
            from melm.contracts import load_research_deferral_triggers
            _RESEARCH_DEFERRAL = load_research_deferral_triggers()
        except Exception:
            _RESEARCH_DEFERRAL = {}
    return _RESEARCH_DEFERRAL


def should_defer_research(decision: Any, profile: Any, store: Any) -> bool:
    """Check if research should be deferred instead of done immediately."""
    triggers = _load_research_triggers()
    defer_when = triggers.get("defer_when", {})
    
    if defer_when.get("provider_unavailable", True):
        return True
    
    utterance = getattr(decision, "utterance", "").lower()
    defer_keywords = triggers.get("defer_keywords", [])
    if any(kw in utterance for kw in defer_keywords):
        return True
    
    return False


def queue_research_task(
    store: Any,
    topic: str,
    action: str = "auto_research",
    provider_hint: str = "",
    session_id: str = "",
) -> str | None:
    """Queue a deferred research task in the entity store."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return queue_deferred_task(
        store=store,
        topic=topic,
        action=action,
        scheduled_at=now,
        session_id=session_id,
        engagement_prompt=f"I was looking into {topic} while you were away.",
    )


def run_deferred_research(store: Any, provider: Any) -> list[dict[str, Any]]:
    """Execute queued deferred research tasks. Returns results list."""
    from .assistant_skill_deferred_tasks import find_due_tasks, complete_deferred_task
    results = []
    tasks = find_due_tasks(store)
    for task in tasks:
        topic = task.get("topic", "")
        entity_id = task.get("entity_id", "")
        try:
            if provider is not None and hasattr(provider, "research"):
                research_result = provider.research(topic)
                summary = getattr(research_result, "summary", "") or str(research_result)[:200]
            else:
                summary = ""
            complete_deferred_task(store, entity_id, result_summary=summary)
            results.append({"topic": topic, "status": "completed"})
        except Exception as exc:
            complete_deferred_task(store, entity_id, result_summary=f"Failed: {exc}")
            results.append({"topic": topic, "status": "failed"})
    return results
