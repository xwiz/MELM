"""Transcript import helpers for Local Assistant OS replay fixtures."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from .local_assistant_router import LocalAssistantProfile


TRANSCRIPT_REPLAY_SCHEMA = "melm.local_assistant_transcript_replay.v1"
TRANSCRIPT_IMPORT_REPORT_SCHEMA = "melm.local_assistant_transcript_import_report.v1"
AUTHORED_TRANSCRIPT_SOURCE_TYPE = "authored_transcript_fixture"
IMPORTED_TRANSCRIPT_SOURCE_TYPE = "redacted_user_transcript_import"
EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE = "event_ledger_transcript_export"
SUPPORTED_TRANSCRIPT_REPLAY_SOURCE_TYPES = frozenset(
    {
        AUTHORED_TRANSCRIPT_SOURCE_TYPE,
        IMPORTED_TRANSCRIPT_SOURCE_TYPE,
        EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
    }
)
STATIC_TRANSCRIPT_EXPECTATION_KEYS = frozenset(
    {
        "answer",
        "assistant_answer",
        "assistant_response",
        "expected_answer",
        "expected_intent",
        "expected_reason",
        "expected_route",
        "expected_response",
        "expected_text",
    }
)
SAFE_TRANSCRIPT_TURN_CONTROL_KEYS = frozenset(
    {
        "network_available",
        "run_reflection",
        "new_session",
        "schedule_refreshes",
        "execute_jobs",
        "min_story_models",
        "execute_opportunities",
    }
)
SAFE_TRANSCRIPT_EXPECTATION_CONTROL_KEYS = frozenset(
    {
        "min_turns",
        "min_route_kinds",
        "min_local_resolution_rate",
        "required_priority_signals",
        "required_memory_digest_quality",
        "required_baseline_win",
        "require_unknown_tokens",
    }
)

_TEXT_FIELDS = ("utterance", "text", "content", "message")
_ROLE_FIELDS = ("speaker", "role", "author", "from")
_SESSION_FIELDS = ("session_id", "conversation_id", "thread_id", "chat_id")
_USER_ROLES = {"", "user", "human", "person", "client", "customer", "me", "child"}
_ASSISTANT_ROLES = {"assistant", "ai", "bot", "agent", "system", "tool", "developer"}
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)")
_LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")


@dataclass(frozen=True)
class TranscriptReplayImportReport:
    input_path: str
    output_path: str
    records_read: int
    turns_written: int
    assistant_rows_skipped: int
    non_user_rows_skipped: int
    empty_text_rows_skipped: int
    redaction_counts: dict[str, int]
    static_expectation_fields_dropped: dict[str, int]
    control_fields_applied: dict[str, int]
    source_type: str
    source_note: str
    output_fixture_schema: str = TRANSCRIPT_REPLAY_SCHEMA
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.turns_written > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRANSCRIPT_IMPORT_REPORT_SCHEMA,
            "passed": self.passed,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "records_read": self.records_read,
            "turns_written": self.turns_written,
            "assistant_rows_skipped": self.assistant_rows_skipped,
            "non_user_rows_skipped": self.non_user_rows_skipped,
            "empty_text_rows_skipped": self.empty_text_rows_skipped,
            "redaction_counts": dict(sorted(self.redaction_counts.items())),
            "static_expectation_fields_dropped": dict(
                sorted(self.static_expectation_fields_dropped.items())
            ),
            "control_fields_applied": dict(sorted(self.control_fields_applied.items())),
            "source_type": self.source_type,
            "source_note": self.source_note,
            "output_fixture_schema": self.output_fixture_schema,
            "warnings": list(self.warnings),
        }


def import_transcript_replay_fixture(
    *,
    input_path: str | Path,
    output_path: str | Path,
    profile: LocalAssistantProfile | dict[str, Any] | None = None,
    scenario: str = "imported_local_assistant_transcript",
    description: str = "Redacted local chat transcript replay imported for Local Assistant OS calibration.",
    source_type: str = IMPORTED_TRANSCRIPT_SOURCE_TYPE,
    source_note: str = (
        "Redacted local transcript import; assistant/system rows are skipped and no per-turn "
        "expected answers, routes, reasons, or response text are retained."
    ),
    replacements: tuple[tuple[str, str], ...] = (),
    min_turns: int = 1,
    min_route_kinds: int = 1,
    controls: dict[str, Any] | None = None,
) -> TranscriptReplayImportReport:
    """Convert raw JSONL chat rows into the replay fixture shape.

    The importer is intentionally conservative: only user turns are retained,
    obvious private tokens are redacted, and static expectation fields are
    counted but never copied into the replay rows.
    """

    in_path = Path(input_path)
    out_path = Path(output_path)
    records = _read_jsonl_records(in_path)
    redaction_counts: Counter[str] = Counter()
    static_drops: Counter[str] = Counter()
    control_counts: Counter[str] = Counter()
    warnings: list[str] = []
    turns: list[dict[str, Any]] = []
    assistant_rows_skipped = 0
    non_user_rows_skipped = 0
    empty_text_rows_skipped = 0
    current_day = 0
    previous_session = None
    control_defaults, control_turns, control_expectations = _normalize_import_controls(controls)

    for record in records:
        for key in set(record) & STATIC_TRANSCRIPT_EXPECTATION_KEYS:
            static_drops[str(key)] += 1
        role = _record_role(record)
        if role in _ASSISTANT_ROLES:
            assistant_rows_skipped += 1
            continue
        if role not in _USER_ROLES:
            non_user_rows_skipped += 1
            continue
        text = _record_text(record)
        if not text:
            empty_text_rows_skipped += 1
            continue
        session = _record_session(record)
        new_session = bool(record.get("new_session", False))
        if session and previous_session is not None and session != previous_session:
            current_day += 1
            new_session = True
        if session:
            previous_session = session
        if "day" in record:
            try:
                current_day = int(record.get("day", current_day) or current_day)
            except (TypeError, ValueError):
                warnings.append(f"invalid_day_on_line_{record.get('_line', 0)}")
        redacted = _redact_text(text, replacements=replacements, counts=redaction_counts)
        label = str(record.get("label") or f"imported_user_{len(turns) + 1:03d}")
        turn_index = len(turns) + 1
        turn_controls = _merged_turn_controls(
            control_defaults,
            _raw_record_turn_controls(record, warnings=warnings),
            _lookup_turn_controls(control_turns, label=label, turn_index=turn_index),
        )
        turn = {
            "type": "turn",
            "day": current_day,
            "speaker": "user",
            "label": label,
            "utterance": redacted,
            "capture_surface": "imported_redacted_transcript",
            "capture_source": source_type,
        }
        if new_session or bool(turn_controls.get("new_session", False)):
            turn["new_session"] = True
            control_counts["new_session"] += int("new_session" in turn_controls)
        for key, value in turn_controls.items():
            if key == "new_session":
                continue
            turn[key] = value
            control_counts[key] += 1
        turns.append(turn)

    if not turns:
        warnings.append("no_user_turns_written")

    expectations_payload = {
        "min_turns": max(1, int(min_turns or 1)),
        "min_route_kinds": max(1, int(min_route_kinds or 1)),
        "min_local_resolution_rate": 0.0,
        "required_priority_signals": False,
        "required_memory_digest_quality": False,
        "required_baseline_win": False,
        "require_unknown_tokens": False,
    }
    expectations_payload.update(control_expectations)
    expectations_payload["min_turns"] = max(
        max(1, int(min_turns or 1)),
        int(expectations_payload.get("min_turns", 1) or 1),
    )
    expectations_payload["min_route_kinds"] = max(
        max(1, int(min_route_kinds or 1)),
        int(expectations_payload.get("min_route_kinds", 1) or 1),
    )

    meta = {
        "type": "meta",
        "schema": TRANSCRIPT_REPLAY_SCHEMA,
        "source_type": source_type,
        "source_note": source_note,
        "scenario": scenario,
        "description": description,
        "profile": _profile_payload(profile),
        "expectations": expectations_payload,
        "import_report": {
            "source_records_read": len(records),
            "turns_written": len(turns),
            "assistant_rows_skipped": assistant_rows_skipped,
            "non_user_rows_skipped": non_user_rows_skipped,
            "empty_text_rows_skipped": empty_text_rows_skipped,
            "redaction_counts": dict(sorted(redaction_counts.items())),
            "static_expectation_fields_dropped": dict(sorted(static_drops.items())),
            "control_fields_applied": dict(sorted(control_counts.items())),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_path, (meta, *turns))
    return TranscriptReplayImportReport(
        input_path=str(in_path),
        output_path=str(out_path),
        records_read=len(records),
        turns_written=len(turns),
        assistant_rows_skipped=assistant_rows_skipped,
        non_user_rows_skipped=non_user_rows_skipped,
        empty_text_rows_skipped=empty_text_rows_skipped,
        redaction_counts=dict(redaction_counts),
        static_expectation_fields_dropped=dict(static_drops),
        control_fields_applied=dict(control_counts),
        source_type=source_type,
        source_note=source_note,
        warnings=tuple(warnings),
    )


def _normalize_import_controls(
    controls: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    if controls is None:
        return {}, {}, {}
    if not isinstance(controls, dict):
        raise ValueError("transcript controls must be a JSON object")
    allowed_root = {"defaults", "turns", "expectations"}
    unknown_root = sorted(str(key) for key in controls if key not in allowed_root)
    if unknown_root:
        raise ValueError(f"unsupported transcript controls keys: {', '.join(unknown_root)}")
    defaults = _coerce_turn_controls(controls.get("defaults", {}) or {}, context="controls.defaults")
    raw_turns = controls.get("turns", {}) or {}
    if not isinstance(raw_turns, dict):
        raise ValueError("controls.turns must be an object keyed by turn label or 1-based index")
    turn_controls = {
        str(key): _coerce_turn_controls(value, context=f"controls.turns.{key}")
        for key, value in raw_turns.items()
    }
    expectations = _coerce_expectation_controls(
        controls.get("expectations", {}) or {},
        context="controls.expectations",
    )
    return defaults, turn_controls, expectations


def _raw_record_turn_controls(record: dict[str, Any], *, warnings: list[str]) -> dict[str, Any]:
    raw = {key: record[key] for key in SAFE_TRANSCRIPT_TURN_CONTROL_KEYS if key in record}
    if not raw:
        return {}
    try:
        return _coerce_turn_controls(raw, context=f"raw_line_{record.get('_line', 0)}")
    except ValueError as exc:
        warnings.append(str(exc).replace(" ", "_"))
        return {}


def _lookup_turn_controls(
    turn_controls: dict[str, dict[str, Any]],
    *,
    label: str,
    turn_index: int,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in (str(turn_index), f"#{turn_index}", label):
        merged.update(turn_controls.get(key, {}))
    return merged


def _merged_turn_controls(*controls: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in controls:
        merged.update(item)
    return merged


def _coerce_turn_controls(payload: dict[str, Any], *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    forbidden = sorted(str(key) for key in set(payload) & STATIC_TRANSCRIPT_EXPECTATION_KEYS)
    if forbidden:
        raise ValueError(f"{context} cannot contain static expectation fields: {', '.join(forbidden)}")
    unknown = sorted(str(key) for key in set(payload) - SAFE_TRANSCRIPT_TURN_CONTROL_KEYS)
    if unknown:
        raise ValueError(f"{context} contains unsupported control fields: {', '.join(unknown)}")
    coerced: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {
            "network_available",
            "run_reflection",
            "new_session",
            "schedule_refreshes",
            "execute_jobs",
        }:
            if not isinstance(value, bool):
                raise ValueError(f"{context}.{key} must be boolean")
            coerced[key] = value
        elif key == "min_story_models":
            try:
                coerced[key] = max(0, int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}.min_story_models must be an integer") from exc
        elif key == "execute_opportunities":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{context}.execute_opportunities must be a list of strings")
            coerced[key] = list(value)
    return coerced


def _coerce_expectation_controls(payload: dict[str, Any], *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    forbidden = sorted(str(key) for key in set(payload) & STATIC_TRANSCRIPT_EXPECTATION_KEYS)
    if forbidden:
        raise ValueError(f"{context} cannot contain static expectation fields: {', '.join(forbidden)}")
    unknown = sorted(str(key) for key in set(payload) - SAFE_TRANSCRIPT_EXPECTATION_CONTROL_KEYS)
    if unknown:
        raise ValueError(f"{context} contains unsupported expectation fields: {', '.join(unknown)}")
    coerced: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"min_turns", "min_route_kinds"}:
            try:
                coerced[key] = max(1, int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}.{key} must be an integer") from exc
        elif key == "min_local_resolution_rate":
            try:
                coerced[key] = max(0.0, float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}.{key} must be numeric") from exc
        elif key in {
            "required_priority_signals",
            "required_memory_digest_quality",
            "required_baseline_win",
            "require_unknown_tokens",
        }:
            if not isinstance(value, bool):
                raise ValueError(f"{context}.{key} must be boolean")
            coerced[key] = value
    return coerced


def _read_jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL record on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"transcript JSONL line {line_number} must be an object")
        item = dict(record)
        item["_line"] = line_number
        records.append(item)
    return tuple(records)


def _write_jsonl(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _record_role(record: dict[str, Any]) -> str:
    for field in _ROLE_FIELDS:
        value = record.get(field)
        if isinstance(value, dict):
            value = value.get("role") or value.get("name") or value.get("type")
        if value is not None:
            return str(value).strip().lower()
    return "user"


def _record_text(record: dict[str, Any]) -> str:
    for field in _TEXT_FIELDS:
        if field not in record:
            continue
        text = _coerce_text(record[field])
        if text:
            return _compact_text(text)
    return ""


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "value", "content", "message"):
            text = _coerce_text(value.get(key))
            if text:
                return text
        return ""
    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return str(value)


def _record_session(record: dict[str, Any]) -> str:
    for field in _SESSION_FIELDS:
        value = record.get(field)
        if value:
            return str(value)
    return ""


def _redact_text(
    text: str,
    *,
    replacements: tuple[tuple[str, str], ...],
    counts: Counter[str],
) -> str:
    redacted = _compact_text(text)
    redacted, count = _EMAIL_RE.subn("<email>", redacted)
    counts["email"] += count
    redacted, count = _URL_RE.subn("<url>", redacted)
    counts["url"] += count
    redacted = _PHONE_CANDIDATE_RE.sub(lambda match: _phone_replacement(match, counts), redacted)
    redacted, count = _LONG_NUMBER_RE.subn("<number>", redacted)
    counts["long_number"] += count
    for index, (source, replacement) in enumerate(replacements, start=1):
        if not source:
            continue
        pattern = re.compile(re.escape(source), re.IGNORECASE)
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            counts[f"manual_rule_{index}"] += count
    return redacted


def _phone_replacement(match: re.Match[str], counts: Counter[str]) -> str:
    text = match.group(0)
    digits = re.sub(r"\D", "", text)
    if len(digits) < 7:
        return text
    counts["phone"] += 1
    return "<phone>"


def _compact_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def _profile_payload(profile: LocalAssistantProfile | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(profile, LocalAssistantProfile):
        return asdict(profile)
    if isinstance(profile, dict):
        return dict(profile)
    return asdict(
        LocalAssistantProfile(
            user_name="local_user",
            age=0,
            location="unknown",
            culture="unknown",
            facts={},
            preferences={},
            health_goals=(),
            contacts={},
            weekly_weather={},
            story_models={},
            media_library=(),
            food_inventory=(),
        )
    )
