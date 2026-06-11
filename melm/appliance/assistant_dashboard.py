"""Ledger dashboard for the Local Assistant OS v0.1 MVP."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .assistant_inventory import MIN_STORY_METADATA_QUALITY
from .assistant_synthesis import SYNTHESIS_QUALITY_FLOOR
from .assistant_os_store import AssistantOSStore


@dataclass(frozen=True)
class AssistantOSDashboard:
    """Query-only summary of the assistant OS SQLite ledger."""

    counts: dict[str, int]
    route_counts: dict[str, int]
    intent_counts: dict[str, int]
    membrane: dict[str, Any]
    homeostasis: dict[str, Any]
    memory: dict[str, Any]
    synthesis: dict[str, Any]
    response_integrity: dict[str, Any]
    jobs: dict[str, Any]
    inventories: dict[str, Any]
    pending_actions: dict[str, int]
    safety_flags: dict[str, int | bool]
    db_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "route_counts": self.route_counts,
            "intent_counts": self.intent_counts,
            "membrane": self.membrane,
            "homeostasis": self.homeostasis,
            "memory": self.memory,
            "synthesis": self.synthesis,
            "response_integrity": self.response_integrity,
            "jobs": self.jobs,
            "inventories": self.inventories,
            "pending_actions": self.pending_actions,
            "safety_flags": self.safety_flags,
            "db_bytes": self.db_bytes,
        }


def build_assistant_os_dashboard(store: AssistantOSStore) -> AssistantOSDashboard:
    """Build a compact health/evidence dashboard from persisted OS tables."""

    counts = store.table_counts()
    route_counts = _group_counts(store, "events", "route")
    intent_counts = _group_counts(store, "events", "intent")
    membrane = _membrane_summary(store)
    homeostasis = _homeostasis_summary(store)
    memory = _memory_link_summary(store)
    synthesis = _synthesis_quality_summary(store)
    response_integrity = _response_integrity_summary(store)
    jobs = {
        "by_status": _group_counts(store, "jobs", "status"),
        "by_kind": _group_counts(store, "jobs", "kind"),
        "queued": len(store.load_jobs(status="queued")),
        "completed": len(store.load_jobs(status="completed")),
        "failed": len(store.load_jobs(status="failed")),
        "priority": _job_priority_summary(store),
        "importer_health": _importer_health_summary(store),
        "importer_trends": _importer_trend_summary(store),
    }
    inventories = {
        "by_kind": _group_counts(store, "inventories", "kind"),
        "by_source": _group_counts(store, "inventories", "source"),
        "by_license": _group_counts(store, "inventories", "license"),
        "story_quality": _story_inventory_quality_summary(store),
    }
    pending_actions = _pending_actions_summary(store)
    safety_flags = _safety_flags(store, counts)
    return AssistantOSDashboard(
        counts=counts,
        route_counts=route_counts,
        intent_counts=intent_counts,
        membrane=membrane,
        homeostasis=homeostasis,
        memory=memory,
        synthesis=synthesis,
        response_integrity=response_integrity,
        jobs=jobs,
        inventories=inventories,
        pending_actions=pending_actions,
        safety_flags=safety_flags,
        db_bytes=_sqlite_size(store.path),
    )


def _group_counts(store: AssistantOSStore, table: str, column: str) -> dict[str, int]:
    if table not in {"events", "jobs", "inventories"}:
        raise ValueError(f"unsupported dashboard table: {table}")
    allowed_columns = {
        "events": {"route", "intent"},
        "jobs": {"status", "kind"},
        "inventories": {"kind", "source", "license"},
    }
    if column not in allowed_columns[table]:
        raise ValueError(f"unsupported dashboard column: {table}.{column}")
    rows = store.connection.execute(
        f"SELECT {column} AS key, COUNT(*) AS count FROM {table} GROUP BY {column} ORDER BY {column}"
    )
    return {str(row["key"]): int(row["count"]) for row in rows}


def _membrane_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT allowed, boundary_crossed, confirmation_required,
               personal_facts_included_json, personal_facts_excluded_json, reason
        FROM membrane_decisions
        """
    ).fetchall()
    boundaries = Counter(str(row["boundary_crossed"]) for row in rows)
    reasons = Counter(str(row["reason"]) for row in rows)
    fact_privacy = store.load_user_fact_privacy_index()
    cloud_private_inclusions = 0
    excluded_private_facts = 0
    for row in rows:
        included = _loads(row["personal_facts_included_json"], default=[])
        excluded = _loads(row["personal_facts_excluded_json"], default=[])
        if row["boundary_crossed"] == "cloud" and _included_cloud_private_facts(included, fact_privacy):
            cloud_private_inclusions += 1
        excluded_private_facts += len(excluded)
    return {
        "total": len(rows),
        "allowed": sum(1 for row in rows if int(row["allowed"])),
        "blocked": sum(1 for row in rows if not int(row["allowed"])),
        "boundary_counts": dict(sorted(boundaries.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "confirmations_required": sum(1 for row in rows if int(row["confirmation_required"])),
        "cloud_private_inclusions": cloud_private_inclusions,
        "excluded_private_fact_count": excluded_private_facts,
    }


def _included_cloud_private_facts(included: Any, fact_privacy: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    if not isinstance(included, list):
        return ()
    unsafe: list[str] = []
    for raw_key in included:
        key = str(raw_key)
        if key.startswith("facts."):
            policy = fact_privacy.get(key, {})
            if (
                bool(policy.get("consent", False))
                and not bool(policy.get("local_only", True))
                and bool(policy.get("cloud_eligible", False))
            ):
                continue
        unsafe.append(key)
    return tuple(unsafe)


def _homeostasis_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT privacy_risk, cloud_dependence, local_capability, uncertainty,
               cache_freshness, action_risk, user_trust, inventory_coverage
        FROM homeostatic_snapshots
        """
    ).fetchall()
    fields = (
        "privacy_risk",
        "cloud_dependence",
        "local_capability",
        "uncertainty",
        "cache_freshness",
        "action_risk",
        "user_trust",
        "inventory_coverage",
    )
    if not rows:
        return {
            "samples": 0,
            **{f"avg_{field}": 0.0 for field in fields},
            **{f"max_{field}": 0.0 for field in fields},
        }
    summary: dict[str, Any] = {"samples": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        summary[f"avg_{field}"] = round(sum(values) / len(values), 3)
        summary[f"max_{field}"] = round(max(values), 3)
    return summary


def _pending_actions_summary(store: AssistantOSStore) -> dict[str, int]:
    rows = store.connection.execute(
        """
        SELECT confirmation_state, executed, COUNT(*) AS count
        FROM pending_actions
        GROUP BY confirmation_state, executed
        """
    ).fetchall()
    summary = {"pending": 0, "confirmed": 0, "cancelled": 0, "executed": 0, "total": 0}
    for row in rows:
        count = int(row["count"])
        summary["total"] += count
        if int(row["executed"]):
            summary["executed"] += count
        if str(row["confirmation_state"]) == "pending":
            summary["pending"] += count
        if str(row["confirmation_state"]) == "confirmed":
            summary["confirmed"] += count
        if str(row["confirmation_state"]) == "cancelled":
            summary["cancelled"] += count
    return summary


def _memory_link_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT event_id, session_id, previous_event_id, next_event_id
        FROM events
        ORDER BY rowid
        """
    ).fetchall()
    if not rows:
        memory_digests = store.load_inventory("memory_digest")
        latest_digest = memory_digests.get("long_horizon_latest", {})
        return {
            "sessions": 0,
            "events": 0,
            "linked_previous": 0,
            "linked_next": 0,
            "dangling_previous": 0,
            "dangling_next": 0,
            "first_event_id": "",
            "last_event_id": "",
            "last_session_id": "",
            "recent_sessions": [],
            "memory_digests": len(memory_digests),
            "latest_memory_digest": _memory_digest_dashboard_summary(latest_digest),
        }
    event_ids = {str(row["event_id"]) for row in rows}
    previous_links = [str(row["previous_event_id"]) for row in rows if row["previous_event_id"]]
    next_links = [str(row["next_event_id"]) for row in rows if row["next_event_id"]]
    memory_digests = store.load_inventory("memory_digest")
    latest_digest = memory_digests.get("long_horizon_latest", {})
    return {
        "sessions": len({str(row["session_id"]) for row in rows}),
        "events": len(rows),
        "linked_previous": len(previous_links),
        "linked_next": len(next_links),
        "dangling_previous": sum(1 for item in previous_links if item not in event_ids),
        "dangling_next": sum(1 for item in next_links if item not in event_ids),
        "first_event_id": str(rows[0]["event_id"]),
        "last_event_id": str(rows[-1]["event_id"]),
        "last_session_id": str(rows[-1]["session_id"]),
        "recent_sessions": list(store.memory_session_summaries(limit=3)),
        "memory_digests": len(memory_digests),
        "latest_memory_digest": _memory_digest_dashboard_summary(latest_digest),
    }


def _memory_digest_dashboard_summary(digest: dict[str, Any]) -> dict[str, Any]:
    quality = dict(digest.get("quality", {}) or {})
    return {
        "digest_id": digest.get("digest_id", ""),
        "session_count": int(digest.get("session_count", 0) or 0),
        "event_count": int(digest.get("event_count", 0) or 0),
        "first_event_id": str(digest.get("first_event_id", "")),
        "last_event_id": str(digest.get("last_event_id", "")),
        "quality_score": float(quality.get("score", 0.0) or 0.0),
        "quality_floor": float(quality.get("floor", 0.0) or 0.0),
        "quality_passed": bool(quality.get("passed", False)),
        "quality_warnings": list(quality.get("warnings", []) or []),
    }


def _job_priority_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT kind, status, priority, attempts, max_attempts
        FROM jobs
        """
    ).fetchall()
    if not rows:
        return {
            "avg_queued_priority": 0.0,
            "max_queued_priority": 0.0,
            "retryable_queued": 0,
            "avg_priority_by_kind": {},
        }
    queued = [float(row["priority"]) for row in rows if row["status"] == "queued"]
    by_kind: dict[str, list[float]] = {}
    retryable_queued = 0
    for row in rows:
        by_kind.setdefault(str(row["kind"]), []).append(float(row["priority"]))
        if (
            row["status"] == "queued"
            and int(row["attempts"]) > 0
            and int(row["attempts"]) < int(row["max_attempts"])
        ):
            retryable_queued += 1
    return {
        "avg_queued_priority": _avg(queued),
        "max_queued_priority": round(max(queued), 3) if queued else 0.0,
        "retryable_queued": retryable_queued,
        "avg_priority_by_kind": {
            kind: _avg(values)
            for kind, values in sorted(by_kind.items())
        },
    }


def _synthesis_quality_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT applied, refused, quality_score, citation_count, evidence_count, warnings_json
        FROM synthesis_traces
        """
    ).fetchall()
    if not rows:
        return {
            "samples": 0,
            "applied": 0,
            "refused": 0,
            "quality_floor": SYNTHESIS_QUALITY_FLOOR,
            "avg_quality_score": 0.0,
            "min_quality_score": 0.0,
            "low_quality_applied": 0,
            "avg_citation_count": 0.0,
            "avg_evidence_count": 0.0,
            "warning_counts": {},
        }
    quality_scores = [float(row["quality_score"]) for row in rows]
    warning_counts: Counter[str] = Counter()
    for row in rows:
        for warning in _loads(row["warnings_json"], default=[]):
            warning_counts[str(warning)] += 1
    return {
        "samples": len(rows),
        "applied": sum(1 for row in rows if int(row["applied"])),
        "refused": sum(1 for row in rows if int(row["refused"])),
        "quality_floor": SYNTHESIS_QUALITY_FLOOR,
        "avg_quality_score": _avg(quality_scores),
        "min_quality_score": round(min(quality_scores), 3),
        "low_quality_applied": sum(
            1
            for row in rows
            if int(row["applied"]) and float(row["quality_score"]) < SYNTHESIS_QUALITY_FLOOR
        ),
        "avg_citation_count": _avg([float(row["citation_count"]) for row in rows]),
        "avg_evidence_count": _avg([float(row["evidence_count"]) for row in rows]),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def _response_integrity_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT understanding_score, response_integrity_score, overall_score,
               band, research_recommended, flags_json
        FROM response_integrity
        """
    ).fetchall()
    queue = store.improvement_queue(limit=1)
    if not rows:
        return {
            "samples": 0,
            "avg_understanding_score": 0.0,
            "avg_response_integrity_score": 0.0,
            "avg_overall_score": 0.0,
            "low_or_review_turns": 0,
            "research_recommended": 0,
            "band_counts": {},
            "flag_counts": {},
            "improvement_queue": queue,
        }
    band_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for row in rows:
        band_counts[str(row["band"])] += 1
        for flag in _loads(row["flags_json"], default=[]):
            flag_counts[str(flag)] += 1
    return {
        "samples": len(rows),
        "avg_understanding_score": _avg([float(row["understanding_score"]) for row in rows]),
        "avg_response_integrity_score": _avg(
            [float(row["response_integrity_score"]) for row in rows]
        ),
        "avg_overall_score": _avg([float(row["overall_score"]) for row in rows]),
        "low_or_review_turns": sum(1 for row in rows if str(row["band"]) != "reliable"),
        "research_recommended": sum(1 for row in rows if bool(row["research_recommended"])),
        "band_counts": dict(sorted(band_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "improvement_queue": queue,
    }


def _importer_health_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT kind, status, attempts, result_json, error
        FROM jobs
        WHERE kind='import_story_metadata'
        ORDER BY updated_at
        """
    ).fetchall()
    if not rows:
        return {
            "completed_import_jobs": 0,
            "failed_import_jobs": 0,
            "imported_items": 0,
            "selected_items": 0,
            "raw_rejected_items": 0,
            "quality_rejected_items": 0,
            "duplicate_rejected_items": 0,
            "network_used_results": 0,
            "pages_fetched": 0,
            "fetch_attempts_total": 0,
            "rate_limit_sleep_count": 0,
            "rate_limit_delay_total_seconds": 0.0,
            "max_pages_requested": 0,
            "byte_budget_exhausted_results": 0,
            "last_next_cursor": "",
            "sources": {},
            "last_error": "",
        }
    sources: Counter[str] = Counter()
    imported_items = 0
    selected_items = 0
    raw_rejected_items = 0
    quality_rejected_items = 0
    duplicate_rejected_items = 0
    network_used_results = 0
    pages_fetched = 0
    fetch_attempts_total = 0
    rate_limit_sleep_count = 0
    rate_limit_delay_total_seconds = 0.0
    max_pages_requested = 0
    byte_budget_exhausted_results = 0
    last_next_cursor = ""
    last_error = ""
    for row in rows:
        if row["error"]:
            last_error = str(row["error"])
        result = _loads(row["result_json"], default={})
        imported_items += int(result.get("imported_items", 0)) if isinstance(result, dict) else 0
        for source_result in result.get("results", []) if isinstance(result, dict) else []:
            source = str(source_result.get("source", "unknown"))
            sources[source] += 1
            selected_items += int(source_result.get("selected_count", 0))
            raw_rejected_items += int(source_result.get("rejected_count", 0))
            if source_result.get("network_used"):
                network_used_results += 1
            observability = dict(source_result.get("observability", {}))
            quality_rejected_items += int(observability.get("quality_rejected_count", 0))
            duplicate_rejected_items += int(observability.get("duplicate_rejected_count", 0))
            pages_fetched += int(observability.get("page_count", 0))
            fetch_attempts_total += int(
                observability.get("fetch_attempts_total", observability.get("fetch_attempts", 0))
            )
            rate_limit_sleep_count += int(observability.get("rate_limit_sleep_count", 0))
            rate_limit_delay_total_seconds += float(
                observability.get("rate_limit_delay_total_seconds", 0.0)
            )
            max_pages_requested = max(max_pages_requested, int(observability.get("max_pages", 0)))
            if observability.get("byte_budget_exhausted"):
                byte_budget_exhausted_results += 1
            if "next_cursor" in observability:
                last_next_cursor = str(observability.get("next_cursor", ""))
    return {
        "completed_import_jobs": sum(1 for row in rows if row["status"] == "completed"),
        "failed_import_jobs": sum(1 for row in rows if row["status"] == "failed"),
        "imported_items": imported_items,
        "selected_items": selected_items,
        "raw_rejected_items": raw_rejected_items,
        "quality_rejected_items": quality_rejected_items,
        "duplicate_rejected_items": duplicate_rejected_items,
        "network_used_results": network_used_results,
        "pages_fetched": pages_fetched,
        "fetch_attempts_total": fetch_attempts_total,
        "rate_limit_sleep_count": rate_limit_sleep_count,
        "rate_limit_delay_total_seconds": round(rate_limit_delay_total_seconds, 3),
        "max_pages_requested": max_pages_requested,
        "byte_budget_exhausted_results": byte_budget_exhausted_results,
        "last_next_cursor": last_next_cursor,
        "sources": dict(sorted(sources.items())),
        "last_error": last_error,
    }


def _importer_trend_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT job_id, status, attempts, payload_json, resource_budget_json,
               result_json, error, updated_at
        FROM jobs
        WHERE kind='import_story_metadata'
        ORDER BY updated_at
        """
    ).fetchall()
    cycles = [_importer_cycle_summary(row) for row in rows]
    completed = [cycle for cycle in cycles if cycle["status"] == "completed"]
    quality_values = [
        float(cycle["avg_metadata_quality"])
        for cycle in completed
        if float(cycle["avg_metadata_quality"]) > 0.0
    ]
    latest = completed[-1] if completed else {}
    quality_delta = 0.0
    if len(quality_values) >= 2:
        quality_delta = round(quality_values[-1] - quality_values[0], 3)
    return {
        "cycles": len(cycles),
        "completed_cycles": len(completed),
        "failed_cycles": sum(1 for cycle in cycles if cycle["status"] == "failed"),
        "running_or_queued_cycles": sum(
            1 for cycle in cycles if cycle["status"] in {"queued", "running"}
        ),
        "recent_cycles": cycles[-5:],
        "imported_items_total": sum(int(cycle["imported_items"]) for cycle in completed),
        "selected_items_total": sum(int(cycle["selected_items"]) for cycle in completed),
        "avg_imported_items": _avg([float(cycle["imported_items"]) for cycle in completed]),
        "avg_selected_items": _avg([float(cycle["selected_items"]) for cycle in completed]),
        "avg_metadata_quality": _avg(quality_values),
        "min_metadata_quality": min(quality_values) if quality_values else 0.0,
        "quality_delta": quality_delta,
        "pages_fetched_total": sum(int(cycle["pages_fetched"]) for cycle in completed),
        "fetch_attempts_total": sum(int(cycle["fetch_attempts_total"]) for cycle in completed),
        "rate_limit_sleep_count": sum(int(cycle["rate_limit_sleep_count"]) for cycle in completed),
        "byte_budget_exhausted_cycles": sum(
            1 for cycle in completed if bool(cycle["byte_budget_exhausted"])
        ),
        "network_used_cycles": sum(
            1 for cycle in completed if int(cycle["network_used_results"]) > 0
        ),
        "latest_completed_cycle": latest,
    }


def _importer_cycle_summary(row: Any) -> dict[str, Any]:
    payload = dict(_loads(row["payload_json"], default={}))
    resource_budget = dict(_loads(row["resource_budget_json"], default={}))
    result = dict(_loads(row["result_json"], default={}))
    sources: Counter[str] = Counter()
    selected_items = 0
    raw_rejected_items = 0
    quality_rejected_items = 0
    duplicate_rejected_items = 0
    network_used_results = 0
    pages_fetched = 0
    fetch_attempts_total = 0
    rate_limit_sleep_count = 0
    metadata_quality_total = 0.0
    metadata_quality_weight = 0
    byte_budget_exhausted = False
    for source_result in result.get("results", []) if isinstance(result, dict) else []:
        source = str(source_result.get("source", "unknown"))
        sources[source] += 1
        selected_count = int(source_result.get("selected_count", 0))
        selected_items += selected_count
        raw_rejected_items += int(source_result.get("rejected_count", 0))
        if source_result.get("network_used"):
            network_used_results += 1
        observability = dict(source_result.get("observability", {}))
        quality_rejected_items += int(observability.get("quality_rejected_count", 0))
        duplicate_rejected_items += int(observability.get("duplicate_rejected_count", 0))
        pages_fetched += int(observability.get("page_count", 0))
        fetch_attempts_total += int(
            observability.get("fetch_attempts_total", observability.get("fetch_attempts", 0))
        )
        rate_limit_sleep_count += int(observability.get("rate_limit_sleep_count", 0))
        if observability.get("byte_budget_exhausted"):
            byte_budget_exhausted = True
        metadata_quality = float(observability.get("selected_avg_metadata_quality", 0.0))
        if metadata_quality > 0 and selected_count > 0:
            metadata_quality_total += metadata_quality * selected_count
            metadata_quality_weight += selected_count
    return {
        "job_id": str(row["job_id"]),
        "status": str(row["status"]),
        "attempts": int(row["attempts"]),
        "refresh_cycle": int(payload.get("refresh_cycle", resource_budget.get("refresh_cycle", 1))),
        "imported_items": int(result.get("imported_items", 0)) if isinstance(result, dict) else 0,
        "selected_items": selected_items,
        "raw_rejected_items": raw_rejected_items,
        "quality_rejected_items": quality_rejected_items,
        "duplicate_rejected_items": duplicate_rejected_items,
        "network_used_results": network_used_results,
        "pages_fetched": pages_fetched,
        "fetch_attempts_total": fetch_attempts_total,
        "rate_limit_sleep_count": rate_limit_sleep_count,
        "byte_budget_exhausted": byte_budget_exhausted,
        "avg_metadata_quality": round(metadata_quality_total / metadata_quality_weight, 3)
        if metadata_quality_weight
        else 0.0,
        "sources": dict(sorted(sources.items())),
        "error": str(row["error"]),
        "updated_at": str(row["updated_at"]),
    }


def _story_inventory_quality_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT payload_json
        FROM inventories
        WHERE kind='story_model'
        """
    ).fetchall()
    quality_scores: list[float] = []
    local_fit_scores: list[float] = []
    metadata_scores: list[float] = []
    missing_quality_fields = 0
    for row in rows:
        payload = dict(_loads(row["payload_json"], default={}))
        if not {
            "quality_score",
            "local_fit_score",
            "metadata_quality",
        }.issubset(payload):
            missing_quality_fields += 1
            continue
        quality_scores.append(float(payload["quality_score"]))
        local_fit_scores.append(float(payload["local_fit_score"]))
        metadata_scores.append(float(payload["metadata_quality"]))
    return {
        "count": len(rows),
        "with_quality_scores": len(metadata_scores),
        "missing_quality_fields": missing_quality_fields,
        "metadata_quality_floor": MIN_STORY_METADATA_QUALITY,
        "below_metadata_quality_floor": sum(
            1 for value in metadata_scores if value < MIN_STORY_METADATA_QUALITY
        ),
        "avg_quality_score": _avg(quality_scores),
        "avg_local_fit_score": _avg(local_fit_scores),
        "avg_metadata_quality": _avg(metadata_scores),
        "min_metadata_quality": round(min(metadata_scores), 3) if metadata_scores else 0.0,
        "max_quality_score": round(max(quality_scores), 3) if quality_scores else 0.0,
    }


def _safety_flags(store: AssistantOSStore, counts: dict[str, int]) -> dict[str, int | bool]:
    cloud_private_inclusions = _membrane_summary(store)["cloud_private_inclusions"]
    memory_links = _memory_link_summary(store)
    synthesis = _synthesis_quality_summary(store)
    unconfirmed_executed_actions = int(
        store.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM pending_actions
            WHERE executed=1 AND confirmation_state!='confirmed'
            """
        ).fetchone()["count"]
    )
    action_without_gate = int(
        store.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM events e
            JOIN membrane_decisions m ON e.event_id=m.event_id
            WHERE e.device_action=1
              AND m.confirmation_required=0
              AND e.reason!='confirmed_device_action'
            """
        ).fetchone()["count"]
    )
    fake_latest_news_local = int(
        store.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM events
            WHERE LOWER(utterance) LIKE '%latest news%'
              AND route IN ('local_answer', 'cached_tool')
            """
        ).fetchone()["count"]
    )
    action_replay_blocks = int(
        store.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM events
            WHERE reason='no_pending_action_to_confirm'
            """
        ).fetchone()["count"]
    )
    cancelled_pending_actions = int(
        store.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM pending_actions
            WHERE confirmation_state='cancelled'
            """
        ).fetchone()["count"]
    )
    confirmation_target_mismatches = int(
        store.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM events
            WHERE reason='confirmation_target_mismatch'
            """
        ).fetchone()["count"]
    )
    consent_revocations = int(
        store.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM events
            WHERE reason='consent_revoked_user_fact'
            """
        ).fetchone()["count"]
    )
    ledger_complete = (
        counts["events"] == counts["membrane_decisions"] == counts["homeostatic_snapshots"]
    )
    return {
        "ledger_complete": ledger_complete,
        "missing_membrane_or_homeostasis": int(not ledger_complete),
        "dangling_memory_links": int(
            memory_links["dangling_previous"] + memory_links["dangling_next"]
        ),
        "low_quality_applied_synthesis": int(synthesis["low_quality_applied"]),
        "cloud_private_inclusions": cloud_private_inclusions,
        "unconfirmed_executed_actions": unconfirmed_executed_actions,
        "action_without_confirmation_gate": action_without_gate,
        "fake_latest_news_local_answers": fake_latest_news_local,
        "action_replay_blocks": action_replay_blocks,
        "cancelled_pending_actions": cancelled_pending_actions,
        "confirmation_target_mismatches": confirmation_target_mismatches,
        "consent_revocations": consent_revocations,
    }


def _sqlite_size(path: Path) -> int:
    if str(path) == ":memory:":
        return 0
    total = 0
    for suffix in ("", "-wal", "-shm"):
        item = Path(f"{path}{suffix}")
        if item.exists():
            total += item.stat().st_size
    return total


def _loads(value: str, *, default: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)
