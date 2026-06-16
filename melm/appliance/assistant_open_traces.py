"""Open-ended transcript runner for the Local Assistant OS MVP.

The deterministic eval and lifecycle probes prove fixed expected cases. This
module runs messier, fixture-authored chat traces through the same kernel,
SQLite store, scheduler, weather cache, action gate, and debug parser so MVP
direction changes have a harder end-to-end evidence gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
from typing import Any

from .assistant_actions import LocalDeviceActionExecutor
from .assistant_dashboard import build_assistant_os_dashboard
from .assistant_inventory import (
    PublicDomainStoryMetadataAdapter,
    schedule_inventory_refreshes,
    story_items_to_inventory_rows,
)
from .assistant_os_kernel import (
    AssistantOSKernel,
    Opportunity,
    persist_self_observation,
    self_model_from_profile,
)
from .assistant_os_store import AssistantOSStore
from .assistant_weather import OpenMeteoWeatherAdapter, weather_items_to_inventory_rows
from .assistant_transcript_import import (
    AUTHORED_TRANSCRIPT_SOURCE_TYPE,
    EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
    IMPORTED_TRANSCRIPT_SOURCE_TYPE,
    STATIC_TRANSCRIPT_EXPECTATION_KEYS,
    SUPPORTED_TRANSCRIPT_REPLAY_SOURCE_TYPES,
    TRANSCRIPT_REPLAY_SCHEMA,
)
from .local_assistant_router import (
    AssistantDecision,
    AssistantRoute,
    AssistantStrategyReport,
    LocalAssistantProfile,
    compare_assistant_strategy_reports_for_utterances,
    parse_assistant_debug_frame,
)


DEFAULT_OPEN_TRACE_FIXTURE = Path("benchmarks/local_assistant_open_traces.json")
DEFAULT_TRANSCRIPT_REPLAY_FIXTURE = Path("benchmarks/local_assistant_transcript_replay.jsonl")
DEFAULT_OPEN_TRACE_WEATHER = Path("benchmarks/sample_open_meteo_forecast.json")
LOCAL_OR_DEVICE_ROUTES = {"local_answer", "cached_tool", "device_action"}
CANDIDATE_USER_CAPTURE_SOURCES = frozenset(
    {
        "interactive_cli",
        "browser_ui",
        "target_device_browser",
        "target_device_cli",
    }
)
SCRIPTED_CAPTURE_SOURCES = frozenset(
    {
        "single_cli_ask",
        "scripted_cli_turn",
        "scripted_api_smoke",
        "scripted_ui_smoke",
    }
)


@dataclass(frozen=True)
class OpenTraceTurn:
    day: int
    label: str
    utterance: str
    network_available: bool = True
    run_reflection: bool = True
    new_session: bool = False
    schedule_refreshes: bool = False
    execute_jobs: bool = False
    min_story_models: int = 3
    execute_opportunities: tuple[str, ...] = ()
    capture_surface: str = ""
    capture_source: str = ""


@dataclass(frozen=True)
class OpenTraceScenario:
    name: str
    description: str
    profile: LocalAssistantProfile
    turns: tuple[OpenTraceTurn, ...]
    expectations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenTraceTurnResult:
    day: int
    label: str
    utterance: str
    intent: str
    route: str
    reason: str
    answer: str
    cloud_needed: bool
    external_fetch_needed: bool
    confirmation_required: bool
    action_executed: bool
    blocked_offline: bool
    debug_parse: dict[str, Any]
    opportunities: tuple[dict[str, Any], ...] = ()
    executed_opportunities: tuple[str, ...] = ()
    scheduled_jobs: tuple[dict[str, Any], ...] = ()
    executed_jobs: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "label": self.label,
            "utterance": self.utterance,
            "intent": self.intent,
            "route": self.route,
            "reason": self.reason,
            "answer": self.answer,
            "cloud_needed": self.cloud_needed,
            "external_fetch_needed": self.external_fetch_needed,
            "confirmation_required": self.confirmation_required,
            "action_executed": self.action_executed,
            "blocked_offline": self.blocked_offline,
            "debug_parse": self.debug_parse,
            "opportunities": list(self.opportunities),
            "executed_opportunities": list(self.executed_opportunities),
            "scheduled_jobs": list(self.scheduled_jobs),
            "executed_jobs": list(self.executed_jobs),
        }


@dataclass(frozen=True)
class OpenTraceScenarioReport:
    name: str
    description: str
    db_path: str
    turns: tuple[OpenTraceTurnResult, ...]
    counts: dict[str, int]
    route_counts: dict[str, int]
    intent_counts: dict[str, int]
    reason_counts: dict[str, int]
    safety_flags: dict[str, Any]
    checks: dict[str, bool]
    priority_signal_samples: tuple[dict[str, Any], ...]
    self_observation: dict[str, Any]
    inventory: dict[str, int]
    expectation_failures: tuple[str, ...]
    capture_provenance: dict[str, Any] = field(default_factory=dict)
    memory_digest: dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def local_resolution_rate(self) -> float:
        if not self.turns:
            return 0.0
        local = sum(turn.route in LOCAL_OR_DEVICE_ROUTES for turn in self.turns)
        return round(local / len(self.turns), 3)

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and not self.expectation_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "db": self.db_path,
            "passed": self.passed,
            "turns": self.turn_count,
            "local_resolution_rate": self.local_resolution_rate,
            "route_counts": self.route_counts,
            "intent_counts": self.intent_counts,
            "reason_counts": self.reason_counts,
            "counts": self.counts,
            "safety_flags": self.safety_flags,
            "checks": self.checks,
            "expectation_failures": list(self.expectation_failures),
            "priority_signal_samples": list(self.priority_signal_samples),
            "capture_provenance": self.capture_provenance,
            "self_observation": self.self_observation,
            "inventory": self.inventory,
            "memory_digest": self.memory_digest,
            "routes": [turn.to_dict() for turn in self.turns],
        }


@dataclass(frozen=True)
class OpenTraceSuiteReport:
    scenarios: tuple[OpenTraceScenarioReport, ...]

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def turns(self) -> int:
        return sum(item.turn_count for item in self.scenarios)

    @property
    def local_resolution_rate(self) -> float:
        total = self.turns
        if not total:
            return 0.0
        local = sum(
            sum(turn.route in LOCAL_OR_DEVICE_ROUTES for turn in scenario.turns)
            for scenario in self.scenarios
        )
        return round(local / total, 3)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        safety_totals = Counter()
        route_counts = Counter()
        intent_counts = Counter()
        reason_counts = Counter()
        for scenario in self.scenarios:
            route_counts.update(scenario.route_counts)
            intent_counts.update(scenario.intent_counts)
            reason_counts.update(scenario.reason_counts)
            for key, value in scenario.safety_flags.items():
                if isinstance(value, bool):
                    safety_totals[key] += 0 if value else 1
                elif isinstance(value, int):
                    safety_totals[key] += value
        priority_samples = [
            sample
            for scenario in self.scenarios
            for sample in scenario.priority_signal_samples
        ]
        return {
            "schema": "melm.local_assistant_open_trace_report.v1",
            "passed": self.passed,
            "scenarios": self.scenario_count,
            "turns": self.turns,
            "local_resolution_rate": self.local_resolution_rate,
            "route_counts": dict(sorted(route_counts.items())),
            "intent_counts": dict(sorted(intent_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "safety_totals": dict(sorted(safety_totals.items())),
            "priority_signal_samples": priority_samples,
            "capture_provenance": _merge_capture_provenance_summaries(
                tuple(scenario.capture_provenance for scenario in self.scenarios)
            ),
            "scenario_reports": [item.to_dict() for item in self.scenarios],
        }


@dataclass(frozen=True)
class TranscriptReplaySuiteReport:
    fixture_path: str
    fixture_schema: str
    source_type: str
    source_note: str
    open_trace_report: OpenTraceSuiteReport
    fixture_checks: dict[str, bool]
    fixture_failures: tuple[str, ...]
    baseline_comparison: dict[str, Any]
    skipped_non_user_rows: int = 0

    @property
    def passed(self) -> bool:
        baseline_required = bool(self.baseline_comparison.get("required", True))
        return (
            self.open_trace_report.passed
            and all(self.fixture_checks.values())
            and not self.fixture_failures
            and ((not baseline_required) or bool(self.baseline_comparison.get("passed", False)))
        )

    def to_dict(self) -> dict[str, Any]:
        trace_payload = self.open_trace_report.to_dict()
        return {
            "schema": "melm.local_assistant_transcript_replay_report.v1",
            "fixture_path": self.fixture_path,
            "fixture_schema": self.fixture_schema,
            "source_type": self.source_type,
            "source_note": self.source_note,
            "passed": self.passed,
            "fixture_checks": self.fixture_checks,
            "fixture_failures": list(self.fixture_failures),
            "skipped_non_user_rows": self.skipped_non_user_rows,
            "scenarios": trace_payload["scenarios"],
            "turns": trace_payload["turns"],
            "local_resolution_rate": trace_payload["local_resolution_rate"],
            "route_counts": trace_payload["route_counts"],
            "intent_counts": trace_payload["intent_counts"],
            "reason_counts": trace_payload["reason_counts"],
            "safety_totals": trace_payload["safety_totals"],
            "priority_signal_samples": trace_payload["priority_signal_samples"],
            "capture_provenance": trace_payload["capture_provenance"],
            "complexity": _trace_complexity_summary(self.open_trace_report),
            "debug_mapping": _trace_debug_mapping_summary(self.open_trace_report),
            "baseline_comparison": self.baseline_comparison,
            "scenario_reports": trace_payload["scenario_reports"],
        }


def load_open_trace_scenarios(trace_path: str | Path = DEFAULT_OPEN_TRACE_FIXTURE) -> tuple[OpenTraceScenario, ...]:
    payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "melm.local_assistant_open_traces.v1":
        raise ValueError(f"unsupported open trace schema: {payload.get('schema')!r}")
    return tuple(_scenario_from_payload(item) for item in payload.get("scenarios", []))


def run_open_trace_suite(
    *,
    trace_path: str | Path = DEFAULT_OPEN_TRACE_FIXTURE,
    db_dir: str | Path | None = None,
    reset: bool = False,
    weather_offline_json: str | Path = DEFAULT_OPEN_TRACE_WEATHER,
) -> OpenTraceSuiteReport:
    scenarios = load_open_trace_scenarios(trace_path)
    reports: list[OpenTraceScenarioReport] = []
    for index, scenario in enumerate(scenarios, start=1):
        db_path = _scenario_db_path(scenario.name, index=index, db_dir=db_dir)
        if reset and str(db_path) != ":memory:":
            _remove_sqlite_files(Path(db_path))
        store = AssistantOSStore(db_path)
        try:
            reports.append(
                run_open_trace_scenario(
                    scenario,
                    store=store,
                    db_path=str(db_path),
                    weather_offline_json=weather_offline_json,
                )
            )
        finally:
            store.close()
    return OpenTraceSuiteReport(tuple(reports))


def load_transcript_replay_scenarios(
    transcript_path: str | Path = DEFAULT_TRANSCRIPT_REPLAY_FIXTURE,
) -> tuple[OpenTraceScenario, ...]:
    fixture = _load_transcript_replay_fixture(transcript_path)
    if fixture["schema"] != TRANSCRIPT_REPLAY_SCHEMA:
        raise ValueError(f"unsupported transcript replay schema: {fixture['schema']!r}")
    if fixture["static_expectation_rows"]:
        rows = ", ".join(str(item) for item in fixture["static_expectation_rows"])
        raise ValueError(f"transcript replay fixture contains static expectation fields on rows: {rows}")
    return tuple(fixture["scenarios"])


def run_transcript_replay_suite(
    *,
    transcript_path: str | Path = DEFAULT_TRANSCRIPT_REPLAY_FIXTURE,
    db_dir: str | Path | None = None,
    reset: bool = False,
    weather_offline_json: str | Path = DEFAULT_OPEN_TRACE_WEATHER,
    auto_lifecycle: bool = False,
) -> TranscriptReplaySuiteReport:
    fixture = _load_transcript_replay_fixture(transcript_path)
    reports: list[OpenTraceScenarioReport] = []
    for index, scenario in enumerate(fixture["scenarios"], start=1):
        db_path = _scenario_db_path(scenario.name, index=index, db_dir=db_dir)
        if reset and str(db_path) != ":memory:":
            _remove_sqlite_files(Path(db_path))
        store = AssistantOSStore(db_path)
        try:
            reports.append(
                run_open_trace_scenario(
                    scenario,
                    store=store,
                    db_path=str(db_path),
                    weather_offline_json=weather_offline_json,
                    auto_lifecycle=auto_lifecycle,
                )
            )
        finally:
            store.close()
    trace_report = OpenTraceSuiteReport(tuple(reports))
    fixture_checks = _transcript_replay_fixture_checks(fixture, trace_report)
    fixture_failures = tuple(key for key, passed in fixture_checks.items() if not passed)
    baseline_comparison = _transcript_baseline_comparison(fixture, trace_report)
    return TranscriptReplaySuiteReport(
        fixture_path=str(transcript_path),
        fixture_schema=str(fixture["schema"]),
        source_type=str(fixture["source_type"]),
        source_note=str(fixture["source_note"]),
        open_trace_report=trace_report,
        fixture_checks=fixture_checks,
        fixture_failures=fixture_failures,
        baseline_comparison=baseline_comparison,
        skipped_non_user_rows=int(fixture["skipped_non_user_rows"]),
    )


def run_open_trace_scenario(
    scenario: OpenTraceScenario,
    *,
    store: AssistantOSStore,
    db_path: str = ":memory:",
    weather_offline_json: str | Path = DEFAULT_OPEN_TRACE_WEATHER,
    auto_lifecycle: bool = False,
) -> OpenTraceScenarioReport:
    results: list[OpenTraceTurnResult] = []
    priority_samples: list[dict[str, Any]] = []
    for turn in scenario.turns:
        if turn.new_session:
            store.start_new_session()
        profile = store.load_profile(scenario.profile)
        kernel = AssistantOSKernel(
            profile=profile,
            store=store,
            action_executor=LocalDeviceActionExecutor(mode="dry-run"),
            capture_surface=turn.capture_surface,
            capture_source=turn.capture_source,
        )
        decision = kernel.decide(turn.utterance)
        blocked_offline = False
        if not turn.network_available:
            offline_decision = _offline_blocked_decision(decision)
            if offline_decision is not decision:
                decision = offline_decision
                blocked_offline = True
        kernel.remember(decision)
        event_id = kernel.events[-1].event_id if kernel.events else ""
        opportunities = kernel.reflect() if turn.run_reflection else ()
        priority_samples.extend(
            _opportunity_priority_samples(
                turn,
                opportunities,
                source="kernel_reflection",
            )
        )
        executed_opportunities = _execute_requested_opportunities(kernel, opportunities, turn.execute_opportunities)
        schedule_report = None
        if turn.schedule_refreshes or auto_lifecycle:
            schedule_report = schedule_inventory_refreshes(
                store,
                store.load_profile(scenario.profile),
                min_story_models=max(0, turn.min_story_models),
                story_limit=max(3, turn.min_story_models),
                source="both",
                use_offline_samples=True,
                gutenberg_csv=Path("benchmarks/sample_gutenberg_catalog.csv"),
                internet_archive_json=Path("benchmarks/sample_internet_archive_search.json"),
            )
            priority_samples.extend(_scheduled_priority_samples(turn, schedule_report.to_dict()))
        executed_jobs = ()
        if turn.execute_jobs or (auto_lifecycle and schedule_report is not None):
            executed_jobs = _execute_trace_jobs(
                store,
                scenario.profile,
                weather_offline_json=weather_offline_json,
                limit=6,
            )
        if auto_lifecycle:
            store.build_memory_digest()
        self_model = self_model_from_profile(store.load_profile(scenario.profile))
        self_observation = persist_self_observation(store, self_model)
        debug_parse = parse_assistant_debug_frame(turn.utterance, decision).to_dict()
        membrane = _membrane_for_event(store, event_id)
        results.append(
            OpenTraceTurnResult(
                day=turn.day,
                label=turn.label,
                utterance=turn.utterance,
                intent=decision.intent,
                route=decision.route,
                reason=decision.reason,
                answer=decision.answer,
                cloud_needed=decision.cloud_needed,
                external_fetch_needed=decision.external_fetch_needed,
                confirmation_required=bool(membrane.get("confirmation_required", False)),
                action_executed=decision.reason == "confirmed_device_action" and _latest_action_executed(store),
                blocked_offline=blocked_offline,
                debug_parse=debug_parse,
                opportunities=tuple(_opportunity_dict(item) for item in opportunities),
                executed_opportunities=tuple(executed_opportunities),
                scheduled_jobs=tuple(schedule_report.to_dict()["recommendations"]) if schedule_report is not None else (),
                executed_jobs=executed_jobs,
            )
        )
        _ = self_observation
    dashboard = build_assistant_os_dashboard(store).to_dict()
    final_profile = store.load_profile(scenario.profile)
    self_observation = persist_self_observation(store, self_model_from_profile(final_profile))
    route_counts = Counter(result.route for result in results)
    intent_counts = Counter(result.intent for result in results)
    reason_counts = Counter(result.reason for result in results)
    safety_flags = dict(dashboard["safety_flags"])
    checks, failures = _scenario_checks(
        scenario,
        results,
        safety_flags=safety_flags,
        priority_samples=priority_samples,
    )
    return OpenTraceScenarioReport(
        name=scenario.name,
        description=scenario.description,
        db_path=db_path,
        turns=tuple(results),
        counts=store.table_counts(),
        route_counts=dict(sorted(route_counts.items())),
        intent_counts=dict(sorted(intent_counts.items())),
        reason_counts=dict(sorted(reason_counts.items())),
        safety_flags=safety_flags,
        checks=checks,
        priority_signal_samples=tuple(priority_samples),
        capture_provenance=_store_capture_provenance_summary(store),
        self_observation=self_observation,
        inventory={
            "story_models": len(store.load_inventory("story_model")),
            "weather_days": len(store.load_inventory("weather")),
            "contacts": len(store.load_inventory("contact")),
            "media_items": len(store.load_inventory("media")),
        },
        expectation_failures=tuple(failures),
        memory_digest=dashboard.get("memory", {}).get("latest_memory_digest", {}),
    )


def _scenario_from_payload(payload: dict[str, Any]) -> OpenTraceScenario:
    return OpenTraceScenario(
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        profile=_profile_from_payload(dict(payload.get("profile", {}))),
        expectations=dict(payload.get("expectations", {})),
        turns=tuple(_turn_from_payload(item) for item in payload.get("turns", [])),
    )


def _turn_from_payload(payload: dict[str, Any]) -> OpenTraceTurn:
    return OpenTraceTurn(
        day=int(payload.get("day", 0)),
        label=str(payload.get("label") or f"turn_{payload.get('day', 0)}"),
        utterance=str(payload["utterance"]),
        network_available=bool(payload.get("network_available", True)),
        run_reflection=bool(payload.get("run_reflection", True)),
        new_session=bool(payload.get("new_session", False)),
        schedule_refreshes=bool(payload.get("schedule_refreshes", False)),
        execute_jobs=bool(payload.get("execute_jobs", False)),
        min_story_models=int(payload.get("min_story_models", 3)),
        execute_opportunities=tuple(str(item) for item in payload.get("execute_opportunities", [])),
        capture_surface=str(payload.get("capture_surface", "")),
        capture_source=str(payload.get("capture_source", "")),
    )


def _load_transcript_replay_fixture(transcript_path: str | Path) -> dict[str, Any]:
    path = Path(transcript_path)
    records = _read_jsonl_records(path)
    meta = next((record for record in records if str(record.get("type", "turn")) == "meta"), {})
    source_type = str(meta.get("source_type", ""))
    source_note = str(meta.get("source_note", meta.get("description", "")))
    turn_records = [
        record
        for record in records
        if str(record.get("type", "turn")) == "turn"
        and str(record.get("speaker", "user")).lower() == "user"
    ]
    skipped_non_user_rows = sum(
        1
        for record in records
        if str(record.get("type", "turn")) == "turn"
        and str(record.get("speaker", "user")).lower() != "user"
    )
    scenario = OpenTraceScenario(
        name=str(meta.get("scenario", "transcript_replay")),
        description=str(meta.get("description", "Transcript replay over the real local assistant OS path.")),
        profile=_profile_from_payload(dict(meta.get("profile", {}))),
        expectations=dict(meta.get("expectations", {})),
        turns=tuple(_turn_from_payload(_turn_with_capture_defaults(record, source_type)) for record in turn_records),
    )
    return {
        "schema": str(meta.get("schema", "")),
        "source_type": source_type,
        "source_note": source_note,
        "records": records,
        "turn_records": turn_records,
        "scenarios": (scenario,),
        "expectations": dict(meta.get("expectations", {})),
        "static_expectation_rows": _static_expectation_rows(records),
        "skipped_non_user_rows": skipped_non_user_rows,
    }


def _read_jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL record on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"transcript JSONL line {line_number} must be an object")
        record = dict(record)
        record["_line"] = line_number
        records.append(record)
    return tuple(records)


def _static_expectation_rows(records: tuple[dict[str, Any], ...]) -> tuple[int, ...]:
    rows: list[int] = []
    for record in records:
        keys = {str(key) for key in record}
        if keys & STATIC_TRANSCRIPT_EXPECTATION_KEYS:
            rows.append(int(record.get("_line", 0) or 0))
    return tuple(rows)


def _turn_with_capture_defaults(record: dict[str, Any], source_type: str) -> dict[str, Any]:
    payload = dict(record)
    if source_type == IMPORTED_TRANSCRIPT_SOURCE_TYPE:
        payload.setdefault("capture_surface", "imported_redacted_transcript")
        payload.setdefault("capture_source", IMPORTED_TRANSCRIPT_SOURCE_TYPE)
    elif source_type == EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE:
        if payload.get("capture_surface") and not payload.get("capture_source"):
            payload["capture_source"] = EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE
    return payload


def _store_capture_provenance_summary(store: AssistantOSStore) -> dict[str, Any]:
    rows = [
        {
            "capture_surface": event.capture_surface,
            "capture_source": event.capture_source,
        }
        for event in store.load_events()
    ]
    return _capture_provenance_summary(rows)


def _capture_provenance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    surfaces = Counter(str(row.get("capture_surface", "") or "") for row in rows)
    sources = Counter(str(row.get("capture_source", "") or "") for row in rows)
    missing = int(surfaces.pop("", 0) or 0)
    missing += int(sources.pop("", 0) or 0)
    return _capture_provenance_from_counts(
        turn_count=len(rows),
        surface_counts=dict(sorted(surfaces.items())),
        source_counts=dict(sorted(sources.items())),
        missing_field_count=missing,
    )


def _merge_capture_provenance_summaries(summaries: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    surfaces: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    turn_count = 0
    missing = 0
    for summary in summaries:
        turn_count += int(summary.get("turn_count", 0) or 0)
        missing += int(summary.get("missing_field_count", 0) or 0)
        surfaces.update(
            {str(key): int(value or 0) for key, value in dict(summary.get("capture_surface_counts", {})).items()}
        )
        sources.update(
            {str(key): int(value or 0) for key, value in dict(summary.get("capture_source_counts", {})).items()}
        )
    return _capture_provenance_from_counts(
        turn_count=turn_count,
        surface_counts=dict(sorted(surfaces.items())),
        source_counts=dict(sorted(sources.items())),
        missing_field_count=missing,
    )


def _capture_provenance_from_counts(
    *,
    turn_count: int,
    surface_counts: dict[str, int],
    source_counts: dict[str, int],
    missing_field_count: int,
) -> dict[str, Any]:
    scripted_count = sum(
        count
        for source, count in source_counts.items()
        if source in SCRIPTED_CAPTURE_SOURCES or "scripted" in source
    )
    candidate_source_counts = {
        source: int(count)
        for source, count in sorted(source_counts.items())
        if source in CANDIDATE_USER_CAPTURE_SOURCES
    }
    scripted_source_counts = {
        source: int(count)
        for source, count in sorted(source_counts.items())
        if source in SCRIPTED_CAPTURE_SOURCES or "scripted" in source
    }
    ambiguous_source_counts = {
        source: int(count)
        for source, count in sorted(source_counts.items())
        if source
        and source not in CANDIDATE_USER_CAPTURE_SOURCES
        and source not in scripted_source_counts
        and source != EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE
    }
    candidate_capture_source_count = sum(candidate_source_counts.values())
    interactive_or_browser_count = sum(
        count
        for source, count in candidate_source_counts.items()
        if source in {"interactive_cli", "browser_ui", "target_device_browser"}
    )
    imported_count = int(surface_counts.get("imported_redacted_transcript", 0) or 0)
    return {
        "turn_count": int(turn_count),
        "capture_surface_counts": surface_counts,
        "capture_source_counts": source_counts,
        "missing_field_count": int(missing_field_count),
        "scripted_turn_count": scripted_count,
        "scripted_capture_source_counts": scripted_source_counts,
        "candidate_capture_source_counts": candidate_source_counts,
        "candidate_capture_source_count": candidate_capture_source_count,
        "ambiguous_capture_source_counts": ambiguous_source_counts,
        "interactive_or_browser_turn_count": interactive_or_browser_count,
        "imported_turn_count": imported_count,
        "has_capture_provenance": bool(turn_count) and int(missing_field_count) == 0,
        "all_turns_scripted": bool(turn_count) and scripted_count == int(turn_count),
    }


def _profile_from_payload(payload: dict[str, Any]) -> LocalAssistantProfile:
    base = LocalAssistantProfile()
    return LocalAssistantProfile(
        user_name=str(payload.get("user_name", base.user_name)),
        age=int(payload.get("age", base.age)),
        location=str(payload.get("location", base.location)),
        culture=str(payload.get("culture", base.culture)),
        facts={str(key): str(value) for key, value in dict(payload.get("facts", base.facts)).items()},
        preferences={
            str(key): str(value)
            for key, value in dict(payload.get("preferences", base.preferences)).items()
        },
        health_goals=tuple(str(item) for item in payload.get("health_goals", base.health_goals)),
        contacts={str(key): str(value) for key, value in dict(payload.get("contacts", base.contacts)).items()},
        weekly_weather={
            str(key): str(value)
            for key, value in dict(payload.get("weekly_weather", base.weekly_weather)).items()
        },
        story_models={
            str(key): str(value)
            for key, value in dict(payload.get("story_models", base.story_models)).items()
        },
        media_library=tuple(str(item) for item in payload.get("media_library", base.media_library)),
        food_inventory=tuple(str(item) for item in payload.get("food_inventory", base.food_inventory)),
    )


def _transcript_replay_fixture_checks(
    fixture: dict[str, Any],
    trace_report: OpenTraceSuiteReport,
) -> dict[str, bool]:
    expectations = dict(fixture.get("expectations", {}))
    min_turns = int(expectations.get("min_turns", 12) or 12)
    min_routes = int(expectations.get("min_route_kinds", 4) or 4)
    require_digest_quality = bool(expectations.get("required_memory_digest_quality", False))
    require_unknown_tokens = bool(
        expectations.get(
            "require_unknown_tokens",
            fixture.get("source_type") == AUTHORED_TRANSCRIPT_SOURCE_TYPE,
        )
    )
    route_counts = Counter()
    for scenario in trace_report.scenarios:
        route_counts.update(scenario.route_counts)
    digest_quality_passed = all(
        bool(scenario.memory_digest.get("quality_passed", False))
        for scenario in trace_report.scenarios
    )
    return {
        "schema_valid": fixture.get("schema") == TRANSCRIPT_REPLAY_SCHEMA,
        "source_type_supported": fixture.get("source_type") in SUPPORTED_TRANSCRIPT_REPLAY_SOURCE_TYPES,
        "user_turns_present": bool(fixture.get("turn_records")),
        "min_turns_met": trace_report.turns >= min_turns,
        "no_static_answer_or_route_expectations": not bool(fixture.get("static_expectation_rows")),
        "scenario_expectations_passed": trace_report.passed,
        "route_diversity_observed": len([key for key, value in route_counts.items() if value]) >= min_routes,
        "debug_mapping_present": all(
            bool(scenario.checks.get("debug_maps_present", False))
            for scenario in trace_report.scenarios
        ),
        "complexity_scored": _trace_complexity_summary(trace_report)["turns_scored"] == trace_report.turns,
        "messy_unknown_tokens_observed": (
            (not require_unknown_tokens)
            or _trace_complexity_summary(trace_report)["unknown_tokens_total"] > 0
        ),
        "real_kernel_ledgers_written": all(
            int(scenario.counts.get("events", 0) or 0) >= scenario.turn_count
            and int(scenario.counts.get("membrane_decisions", 0) or 0) >= scenario.turn_count
            and int(scenario.counts.get("homeostatic_snapshots", 0) or 0) >= scenario.turn_count
            for scenario in trace_report.scenarios
        ),
        "memory_digest_quality_passed": (not require_digest_quality) or digest_quality_passed,
        "skipped_non_user_rows_not_used_as_expected_answers": int(fixture.get("skipped_non_user_rows", 0) or 0) >= 0,
    }


def _trace_complexity_summary(trace_report: OpenTraceSuiteReport) -> dict[str, Any]:
    scores: list[float] = []
    unknown_total = 0
    high_complexity: list[dict[str, Any]] = []
    for scenario in trace_report.scenarios:
        for turn in scenario.turns:
            chat_frame = dict(turn.debug_parse.get("chat_frame", {}) or {})
            nlp = dict(turn.debug_parse.get("nlp", {}) or {})
            score = float(chat_frame.get("complexity_score", 0.0) or 0.0)
            scores.append(score)
            unknown_total += int(nlp.get("unknown_token_count", 0) or 0)
            if score >= 0.45:
                high_complexity.append(
                    {
                        "scenario": scenario.name,
                        "label": turn.label,
                        "intent": turn.intent,
                        "route": turn.route,
                        "reason": turn.reason,
                        "complexity_score": score,
                        "unknown_token_count": int(nlp.get("unknown_token_count", 0) or 0),
                    }
                )
    return {
        "turns_scored": len(scores),
        "avg_complexity_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "max_complexity_score": round(max(scores), 3) if scores else 0.0,
        "unknown_tokens_total": unknown_total,
        "high_complexity_turns": high_complexity,
    }


def _trace_debug_mapping_summary(trace_report: OpenTraceSuiteReport, *, limit: int = 5) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for scenario in trace_report.scenarios:
        for turn in scenario.turns:
            if len(samples) >= limit:
                break
            debug_parse = turn.debug_parse
            samples.append(
                {
                    "scenario": scenario.name,
                    "label": turn.label,
                    "utterance": turn.utterance,
                    "stages": [str(stage.get("stage", "")) for stage in debug_parse.get("mapping", [])],
                    "basic_nlp": debug_parse.get("nlp", {}),
                    "uol": debug_parse.get("uol", {}),
                    "chat_frame": debug_parse.get("chat_frame", {}),
                }
            )
        if len(samples) >= limit:
            break
    return {
        "stages": ["basic_nlp", "uol_parse", "chat_frame"],
        "sample_count": len(samples),
        "samples": samples,
    }


def _transcript_baseline_comparison(
    fixture: dict[str, Any],
    trace_report: OpenTraceSuiteReport,
) -> dict[str, Any]:
    required = _transcript_baseline_required(fixture)
    current = _current_transcript_strategy_metrics(trace_report)
    baseline_reports = _transcript_static_baseline_reports(fixture)
    baselines = tuple(_baseline_strategy_metrics(report) for report in baseline_reports)
    best_baseline = max(
        baselines,
        key=lambda item: (float(item["local_resolution_rate"]), -int(item["privacy_exposures"])),
        default={},
    )
    max_baseline = {
        "profile_updates": max((int(item["profile_updates"]) for item in baselines), default=0),
        "private_cloud_blocks": max((int(item["private_cloud_blocks"]) for item in baselines), default=0),
        "autobiographical_local_answers": max(
            (int(item["autobiographical_local_answers"]) for item in baselines),
            default=0,
        ),
        "identity_local_answers": max((int(item["identity_local_answers"]) for item in baselines), default=0),
        "status_local_answers": max((int(item["status_local_answers"]) for item in baselines), default=0),
        "privacy_exposures": max((int(item["privacy_exposures"]) for item in baselines), default=0),
    }
    local_gain = round(
        float(current["local_resolution_rate"]) - float(best_baseline.get("local_resolution_rate", 0.0)),
        3,
    )
    capability_advantages = {
        "profile_updates_vs_best_baseline": int(current["profile_updates"]) - max_baseline["profile_updates"],
        "private_cloud_blocks_vs_best_baseline": int(current["private_cloud_blocks"])
        - max_baseline["private_cloud_blocks"],
        "autobiographical_local_answers_vs_best_baseline": int(current["autobiographical_local_answers"])
        - max_baseline["autobiographical_local_answers"],
        "identity_local_answers_vs_best_baseline": int(current["identity_local_answers"])
        - max_baseline["identity_local_answers"],
        "status_local_answers_vs_best_baseline": int(current["status_local_answers"])
        - max_baseline["status_local_answers"],
        "privacy_exposures_reduced_vs_worst_baseline": max_baseline["privacy_exposures"]
        - int(current["privacy_exposures"]),
    }
    transition_checks = _current_transcript_transition_checks(current)
    checks = {
        "same_user_turns_compared": all(int(item["cases"]) == int(current["cases"]) for item in baselines),
        "current_beats_best_baseline_local_resolution": local_gain >= 0.2,
        "current_has_zero_private_cloud_exposure": int(current["privacy_exposures"]) == 0,
        "current_blocks_private_cloud_request": int(current["private_cloud_blocks"]) >= 1,
        "current_stores_profile_updates": int(current["profile_updates"]) >= 2,
        "current_has_long_horizon_memory_digest": int(current["autobiographical_local_answers"]) >= 1,
        "dynamic_capability_transitions_observed": all(transition_checks.values()),
    }
    strict_passed = all(checks.values())
    return {
        "schema": "melm.local_assistant_transcript_baseline_comparison.v1",
        "required": required,
        "passed": strict_passed if required else bool(checks["same_user_turns_compared"]),
        "strict_passed": strict_passed,
        "comparison_type": "static_structural_baselines_over_same_user_turns",
        "turns": int(current["cases"]),
        "current": current,
        "baselines": list(baselines),
        "best_baseline": best_baseline,
        "wins": {
            "local_resolution_rate_gain_vs_best_baseline": local_gain,
            "cloud_handoff_reduction_vs_best_baseline": int(best_baseline.get("cloud_handoffs", 0))
            - int(current["cloud_handoffs"]),
            "clarification_reduction_vs_best_baseline": int(best_baseline.get("clarifications", 0))
            - int(current["clarifications"]),
            "capability_advantages": capability_advantages,
            "dynamic_transitions": transition_checks,
        },
        "checks": checks,
        "limitations": [
            "Baselines are static structural routes over the same user utterances, not alternate learned agents.",
            "The current architecture is scored through the real kernel/store/job/action/debug path.",
            "Authored transcript coverage is evidence for architecture direction; imported redacted transcripts are calibration unless required_baseline_win is true.",
        ],
    }


def _transcript_baseline_required(fixture: dict[str, Any]) -> bool:
    expectations = dict(fixture.get("expectations", {}))
    if "required_baseline_win" in expectations:
        return bool(expectations.get("required_baseline_win"))
    return fixture.get("source_type") == AUTHORED_TRANSCRIPT_SOURCE_TYPE


def _transcript_static_baseline_reports(fixture: dict[str, Any]) -> tuple[AssistantStrategyReport, ...]:
    by_strategy: dict[str, list[AssistantDecision]] = {}
    for scenario in fixture.get("scenarios", ()):
        utterances = tuple(turn.utterance for turn in scenario.turns)
        for report in compare_assistant_strategy_reports_for_utterances(
            utterances,
            profile=scenario.profile,
        ):
            by_strategy.setdefault(report.strategy, []).extend(report.decisions)
    reports: list[AssistantStrategyReport] = []
    for strategy, decisions in by_strategy.items():
        reports.append(_strategy_report_from_decisions(strategy, tuple(decisions)))
    return tuple(reports)


def _strategy_report_from_decisions(
    strategy: str,
    decisions: tuple[AssistantDecision, ...],
) -> AssistantStrategyReport:
    return AssistantStrategyReport(
        strategy=strategy,
        cases=len(decisions),
        local_or_device_resolved=sum(decision.route in LOCAL_OR_DEVICE_ROUTES for decision in decisions),
        cloud_handoffs=sum(decision.route == "cloud_handoff" for decision in decisions),
        external_fetches=sum(decision.route == "external_fetch" for decision in decisions),
        clarifications=sum(decision.route == "clarify" for decision in decisions),
        privacy_exposures=sum(decision.privacy_exposure for decision in decisions),
        memory_uses=sum(decision.local_memory_used for decision in decisions),
        decisions=decisions,
    )


def _current_transcript_strategy_metrics(trace_report: OpenTraceSuiteReport) -> dict[str, Any]:
    turns = tuple(turn for scenario in trace_report.scenarios for turn in scenario.turns)
    route_counts = Counter(turn.route for turn in turns)
    intent_counts = Counter(turn.intent for turn in turns)
    reason_counts = Counter(turn.reason for turn in turns)
    safety_totals = Counter()
    for scenario in trace_report.scenarios:
        for key, value in scenario.safety_flags.items():
            if isinstance(value, bool):
                safety_totals[key] += 0 if value else 1
            elif isinstance(value, int):
                safety_totals[key] += value
    cases = len(turns)
    local_or_device = sum(turn.route in LOCAL_OR_DEVICE_ROUTES for turn in turns)
    memory_or_store_uses = sum(
        bool(turn.debug_parse.get("chat_frame", {}).get("local_memory_candidate"))
        and turn.route in LOCAL_OR_DEVICE_ROUTES
        for turn in turns
    )
    return {
        "strategy": "memory_os_kernel_with_learning",
        "cases": cases,
        "local_or_device_resolved": local_or_device,
        "local_resolution_rate": round(local_or_device / cases, 3) if cases else 0.0,
        "cloud_handoffs": int(route_counts.get("cloud_handoff", 0)),
        "external_fetches": int(route_counts.get("external_fetch", 0)),
        "clarifications": int(route_counts.get("clarify", 0)),
        "rejections": int(route_counts.get("reject", 0)),
        "privacy_exposures": int(safety_totals.get("cloud_private_inclusions", 0)),
        "private_cloud_blocks": int(reason_counts.get("blocked_private_facts_to_cloud", 0)),
        "memory_uses": memory_or_store_uses,
        "device_actions": int(route_counts.get("device_action", 0)),
        "confirmations_required": sum(turn.confirmation_required for turn in turns),
        "profile_updates": int(reason_counts.get("profile_update", 0)),
        "identity_local_answers": sum(
            turn.intent == "assistant_identity" and turn.route in LOCAL_OR_DEVICE_ROUTES
            for turn in turns
        ),
        "status_local_answers": sum(
            turn.intent == "assistant_status" and turn.route in LOCAL_OR_DEVICE_ROUTES
            for turn in turns
        ),
        "autobiographical_local_answers": sum(
            turn.intent == "autobiographical_memory" and turn.route in LOCAL_OR_DEVICE_ROUTES
            for turn in turns
        ),
        "route_counts": dict(sorted(route_counts.items())),
        "intent_counts": dict(sorted(intent_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _baseline_strategy_metrics(report: AssistantStrategyReport) -> dict[str, Any]:
    route_counts = Counter(decision.route for decision in report.decisions)
    intent_counts = Counter(decision.intent for decision in report.decisions)
    reason_counts = Counter(decision.reason for decision in report.decisions)
    return {
        "strategy": report.strategy,
        "cases": report.cases,
        "local_or_device_resolved": report.local_or_device_resolved,
        "local_resolution_rate": report.local_resolution_rate,
        "cloud_handoffs": report.cloud_handoffs,
        "external_fetches": report.external_fetches,
        "clarifications": report.clarifications,
        "rejections": int(route_counts.get("reject", 0)),
        "privacy_exposures": report.privacy_exposures,
        "private_cloud_blocks": int(reason_counts.get("blocked_private_facts_to_cloud", 0)),
        "memory_uses": report.memory_uses,
        "device_actions": int(route_counts.get("device_action", 0)),
        "confirmations_required": 0,
        "profile_updates": int(reason_counts.get("profile_update", 0)),
        "identity_local_answers": sum(
            decision.intent == "assistant_identity" and decision.route in LOCAL_OR_DEVICE_ROUTES
            for decision in report.decisions
        ),
        "status_local_answers": sum(
            decision.intent == "assistant_status" and decision.route in LOCAL_OR_DEVICE_ROUTES
            for decision in report.decisions
        ),
        "autobiographical_local_answers": sum(
            decision.intent == "autobiographical_memory" and decision.route in LOCAL_OR_DEVICE_ROUTES
            for decision in report.decisions
        ),
        "route_counts": dict(sorted(route_counts.items())),
        "intent_counts": dict(sorted(intent_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "sample_decisions": [
            {
                "utterance": decision.utterance,
                "intent": decision.intent,
                "route": decision.route,
                "reason": decision.reason,
                "privacy_exposure": decision.privacy_exposure,
            }
            for decision in report.decisions[:6]
        ],
    }


def _current_transcript_transition_checks(current: dict[str, Any]) -> dict[str, bool]:
    reasons = dict(current.get("reason_counts", {}))
    return {
        "weather_miss_to_cached_hit": bool(reasons.get("weather_cache_miss")) and bool(reasons.get("weather_cache_hit")),
        "story_cloud_to_local_inventory": bool(reasons.get("missing_story_model"))
        and bool(reasons.get("local_story_inventory")),
        "media_gap_to_local_action": bool(reasons.get("empty_media_library"))
        and bool(reasons.get("local_media_action"))
        and bool(reasons.get("confirmed_device_action")),
        "contact_gap_to_trusted_action": bool(reasons.get("missing_contact"))
        and bool(reasons.get("consented_trusted_contact_stored"))
        and bool(reasons.get("trusted_contact_action")),
        "profile_facts_stored_locally": int(reasons.get("profile_update", 0) or 0) >= 2,
        "private_cloud_blocked": bool(reasons.get("blocked_private_facts_to_cloud")),
        "long_horizon_digest_recalled": bool(reasons.get("autobiographical_memory_digest")),
    }


def _offline_blocked_decision(decision: AssistantDecision) -> AssistantDecision:
    if decision.cloud_needed:
        return replace(
            decision,
            route="clarify",
            answer="I cannot use the larger model while offline.",
            cloud_needed=False,
            reason="cloud_unavailable",
            confidence=min(decision.confidence, 0.72),
        )
    if decision.external_fetch_needed:
        return replace(
            decision,
            route="clarify",
            answer="I cannot fetch that while offline.",
            external_fetch_needed=False,
            reason="tool_unavailable",
            confidence=min(decision.confidence, 0.72),
        )
    return decision


def _execute_requested_opportunities(
    kernel: AssistantOSKernel,
    opportunities: tuple[Opportunity, ...],
    requested_kinds: tuple[str, ...],
) -> list[str]:
    requested = set(requested_kinds)
    executed: list[str] = []
    if not requested:
        return executed
    for opportunity in opportunities:
        if opportunity.kind not in requested:
            continue
        before = len(kernel.executed_jobs)
        kernel.execute(opportunity)
        executed.extend(kernel.executed_jobs[before:])
    return executed


def _execute_trace_jobs(
    store: AssistantOSStore,
    base_profile: LocalAssistantProfile,
    *,
    weather_offline_json: str | Path,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    executed: list[dict[str, Any]] = []
    for _ in range(max(0, limit)):
        job = store.start_next_job(kinds=("import_story_metadata", "refresh_weather_cache"))
        if job is None:
            break
        try:
            if job.kind == "refresh_weather_cache":
                profile = store.load_profile(base_profile)
                result = OpenMeteoWeatherAdapter().refresh(
                    profile,
                    offline_json=weather_offline_json,
                    live=False,
                )
                for row in weather_items_to_inventory_rows(result):
                    store.upsert_inventory(
                        str(row["kind"]),
                        str(row["item_id"]),
                        dict(row["payload"]),
                        source=str(row["source"]),
                        license=str(row["license"]),
                        tags=tuple(str(tag) for tag in row["tags"]),
                    )
                store.connection.commit()
                payload = result.to_dict()
                store.complete_job(job.job_id, result=payload)
                _mark_opportunity_for_job(store, job)
                executed.append(
                    {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "weather_days": result.weather_days,
                        "network_used": result.network_used,
                        "priority": job.priority,
                    }
                )
            elif job.kind == "import_story_metadata":
                profile = store.load_profile(base_profile)
                limit_value = int(job.payload.get("limit", 3) or 3)
                result = PublicDomainStoryMetadataAdapter().build_story_inventory(
                    profile,
                    limit=max(1, limit_value),
                )
                for row in story_items_to_inventory_rows(result.selected_items, profile=profile):
                    store.upsert_inventory(
                        str(row["kind"]),
                        str(row["item_id"]),
                        dict(row["payload"]),
                        source=str(row["source"]),
                        license=str(row["license"]),
                        tags=tuple(str(tag) for tag in row["tags"]),
                    )
                store.connection.commit()
                payload = {
                    "source": result.source_path,
                    "imported_items": len(result.selected_items),
                    "network_used": False,
                }
                store.complete_job(job.job_id, result=payload)
                _mark_opportunity_for_job(store, job)
                executed.append(
                    {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "imported_items": len(result.selected_items),
                        "network_used": False,
                        "priority": job.priority,
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive trace runner path
            store.fail_job(job.job_id, error=str(exc))
            executed.append({"job_id": job.job_id, "kind": job.kind, "error": str(exc)})
    profile = store.load_profile(base_profile)
    persist_self_observation(store, self_model_from_profile(profile))
    return tuple(executed)


def _mark_opportunity_for_job(store: AssistantOSStore, job: Any) -> None:
    opportunity_id = str(job.payload.get("opportunity_id", ""))
    if opportunity_id:
        store.mark_opportunity_executed_by_id(opportunity_id)


def _scenario_checks(
    scenario: OpenTraceScenario,
    results: list[OpenTraceTurnResult],
    *,
    safety_flags: dict[str, Any],
    priority_samples: list[dict[str, Any]],
) -> tuple[dict[str, bool], list[str]]:
    expectations = scenario.expectations
    failures: list[str] = []
    route_values = [result.route for result in results]
    intent_values = [result.intent for result in results]
    reason_values = [result.reason for result in results]
    min_rate = float(expectations.get("min_local_resolution_rate", 0.0) or 0.0)
    local_rate = round(sum(route in LOCAL_OR_DEVICE_ROUTES for route in route_values) / len(results), 3) if results else 0.0
    required_intents = tuple(str(item) for item in expectations.get("required_intents", ()))
    required_routes = tuple(str(item) for item in expectations.get("required_routes", ()))
    required_reasons = tuple(str(item) for item in expectations.get("required_reasons", ()))
    required_transitions = tuple(tuple(str(part) for part in item) for item in expectations.get("required_reason_transitions", ()))
    require_priority_signals = bool(expectations.get("required_priority_signals", True))
    checks = {
        "min_local_resolution_rate": local_rate >= min_rate,
        "required_intents_seen": all(item in intent_values for item in required_intents),
        "required_routes_seen": all(item in route_values for item in required_routes),
        "required_reasons_seen": all(item in reason_values for item in required_reasons),
        "required_reason_transitions_seen": all(
            _reason_transition_seen(reason_values, before, after)
            for before, after in required_transitions
        ),
        "debug_maps_present": all(bool(result.debug_parse.get("uol")) and bool(result.debug_parse.get("chat_frame")) for result in results),
        "primary_uol_chatframe_not_secondary_phrase_route": _primary_uol_debug_maps_are_not_secondary_phrase_routes(results),
        "identity_maps_to_self_model": _identity_debug_maps_are_local(results),
        "status_maps_to_runtime_or_next_steps": _status_debug_maps_are_local(results),
        "safety_flags_clean": _safety_flags_clean(safety_flags),
        "priority_signals_present": (not require_priority_signals) or bool(priority_samples),
    }
    for key, passed in checks.items():
        if not passed:
            failures.append(key)
    return checks, failures


def _reason_transition_seen(reason_values: list[str], before: str, after: str) -> bool:
    try:
        before_index = reason_values.index(before)
    except ValueError:
        return False
    return after in reason_values[before_index + 1 :]


def _identity_debug_maps_are_local(results: list[OpenTraceTurnResult]) -> bool:
    identity_results = [item for item in results if item.intent == "assistant_identity"]
    if not identity_results:
        return True
    return all(item.debug_parse.get("uol", {}).get("object") == "self_model" for item in identity_results)


def _primary_uol_debug_maps_are_not_secondary_phrase_routes(results: list[OpenTraceTurnResult]) -> bool:
    for result in results:
        debug = result.debug_parse
        nlp = dict(debug.get("nlp", {}))
        chat_frame = dict(debug.get("chat_frame", {}))
        primary = dict(nlp.get("primary_domain_evidence", {}))
        primary_basis = [str(part) for part in chat_frame.get("primary_routing_basis", [])]
        if nlp.get("primary_parse_basis") != "uol_chat_frame":
            return False
        if nlp.get("secondary_hint_policy") != "debug_only_never_primary_route":
            return False
        if chat_frame.get("secondary_hint_policy") != "debug_only_never_primary_route":
            return False
        if any(
            part.startswith("secondary_meaning_hints:") or part.startswith("vocabulary_hits:")
            for part in primary_basis
        ):
            return False
        frame_registry = str(primary.get("frame_registry", ""))
        frame_id = str(primary.get("frame_id", ""))
        source_policy = str(primary.get("source_policy", ""))
        if not frame_registry or not frame_id:
            if result.route in LOCAL_OR_DEVICE_ROUTES:
                return False
            continue
        if (
            frame_registry != "melm.assistant_frame_registry.v1"
            or source_policy != "primary_uol_chatframe_only"
            or chat_frame.get("frame_registry") != frame_registry
            or chat_frame.get("frame_id") != frame_id
            or chat_frame.get("frame_source_policy") != source_policy
        ):
            return False
    return True


def _status_debug_maps_are_local(results: list[OpenTraceTurnResult]) -> bool:
    status_results = [item for item in results if item.intent == "assistant_status"]
    if not status_results:
        return True
    return all(
        item.debug_parse.get("uol", {}).get("object") in {"runtime_status", "next_steps"}
        for item in status_results
    )


def _safety_flags_clean(safety_flags: dict[str, Any]) -> bool:
    blocking_flags = (
        "cloud_private_inclusions",
        "unconfirmed_executed_actions",
        "action_without_confirmation_gate",
        "fake_latest_news_local_answers",
        "low_quality_applied_synthesis",
        "dangling_memory_links",
    )
    return all(int(safety_flags.get(key, 0) or 0) == 0 for key in blocking_flags)


def _opportunity_priority_samples(
    turn: OpenTraceTurn,
    opportunities: tuple[Opportunity, ...],
    *,
    source: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "source": source,
            "turn": turn.label,
            "kind": item.kind,
            "priority": item.priority,
            "reason": item.reason,
            "signals": dict(item.priority_signals),
        }
        for item in opportunities
        if item.priority_signals
    )


def _scheduled_priority_samples(turn: OpenTraceTurn, schedule_payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    samples = []
    for item in schedule_payload.get("recommendations", []):
        samples.append(
            {
                "source": "inventory_scheduler",
                "turn": turn.label,
                "kind": item.get("kind", ""),
                "priority": item.get("priority", 0.0),
                "reason": item.get("reason", ""),
                "signals": dict(item.get("priority_signals", {})),
            }
        )
    return tuple(samples)


def _opportunity_dict(opportunity: Opportunity) -> dict[str, Any]:
    return {
        "kind": opportunity.kind,
        "priority": opportunity.priority,
        "reason": opportunity.reason,
        "expected_cloud_reduction": opportunity.expected_cloud_reduction,
        "priority_signals": dict(opportunity.priority_signals),
    }


def _membrane_for_event(store: AssistantOSStore, event_id: str) -> dict[str, Any]:
    row = store.connection.execute(
        """
        SELECT boundary_crossed, confirmation_required, reason
        FROM membrane_decisions
        WHERE event_id=?
        """,
        (event_id,),
    ).fetchone()
    return dict(row) if row is not None else {}


def _latest_action_executed(store: AssistantOSStore) -> bool:
    row = store.connection.execute(
        """
        SELECT executed, confirmation_state
        FROM pending_actions
        ORDER BY rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return False
    return bool(row["executed"]) and str(row["confirmation_state"]) == "confirmed"


def _scenario_db_path(name: str, *, index: int, db_dir: str | Path | None) -> str | Path:
    if db_dir is None:
        return ":memory:"
    root = Path(db_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or f"scenario_{index}"
    return root / f"{index:02d}_{safe_name}.sqlite"


def _remove_sqlite_files(db: Path) -> None:
    for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
        if path.exists():
            path.unlink()
