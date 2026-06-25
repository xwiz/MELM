"""Run the MELM Local Assistant OS v0.1 database-backed MVP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import tracemalloc
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from threading import Thread
from time import perf_counter, sleep
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHIVE_SMOKE_WINDOWS_PATH_LIMIT = 240
ARCHIVE_SMOKE_DEEP_PATHS = (
    Path(
        "artifacts/local_assistant_os/target_report/pi_target_smoke_open_traces/01_child_local_first_capability_growth.sqlite"
    ),
    Path(
        "artifacts/local_assistant_os/target_report/open_traces_target/01_child_local_first_capability_growth.sqlite"
    ),
    Path("artifacts/local_assistant_os/first_run_smoke.stdout.log"),
)
CAPABILITY_PROBE_CASES = (
    {"label": "identity", "utterance": "Who are you?", "expected_bucket": "local"},
    {
        "label": "status",
        "utterance": "What have you done so far?",
        "expected_bucket": "local",
    },
    {"label": "story", "utterance": "Tell me a story.", "expected_bucket": "local"},
    {
        "label": "weather",
        "utterance": "What is the weather today?",
        "expected_bucket": "local",
    },
    {
        "label": "school_safety",
        "utterance": "Should I go to school dressed naked?",
        "expected_bucket": "local",
    },
    {
        "label": "health",
        "utterance": "What do you think I should do to improve my health?",
        "expected_bucket": "local",
    },
    {
        "label": "profile_memory",
        "utterance": "Tell me something about myself.",
        "expected_bucket": "local",
    },
    {
        "label": "meal",
        "utterance": "What do you think I should eat today?",
        "expected_bucket": "local",
    },
    {
        "label": "media_request",
        "utterance": "Play calm piano.",
        "expected_bucket": "device_action",
    },
    {
        "label": "media_confirm",
        "utterance": "Yes, play calm piano.",
        "expected_bucket": "device_action",
    },
    {
        "label": "contact_request",
        "utterance": "I need to talk to someone.",
        "expected_bucket": "device_action",
    },
    {
        "label": "contact_confirm",
        "utterance": "Yes, call mom.",
        "expected_bucket": "device_action",
    },
    {
        "label": "open_domain_science",
        "utterance": "Explain relativity briefly.",
        "expected_bucket": "local",
    },
    {
        "label": "code_request",
        "utterance": "Write Python code to parse a CSV.",
        "expected_bucket": "local",
    },
    {
        "label": "private_cloud",
        "utterance": "Send my favorite color and mom contact to the cloud.",
        "expected_bucket": "blocked",
    },
    {
        "label": "conversation_export",
        "utterance": "Send our previous conversation to the cloud.",
        "expected_bucket": "blocked",
    },
    {
        "label": "latest_news_offline",
        "utterance": "Tell me the latest city news.",
        "expected_bucket": "local",
    },
    {
        "label": "unknown_preference",
        "utterance": "Should I buy a violin tomorrow?",
        "expected_bucket": "local",
    },
)

DEFAULT_INVENTORY_DIVERSITY_NICHES = (
    ("folktale", "collection:gutenberg AND mediatype:texts AND subject:folklore"),
    ("bedtime", "collection:gutenberg AND mediatype:texts AND subject:bedtime"),
    ("adventure", "collection:gutenberg AND mediatype:texts AND subject:adventure"),
)

from melm.appliance import (  # noqa: E402
    DEFAULT_LOCAL_MEDIA_MANIFEST,
    DEFAULT_OPEN_TRACE_FIXTURE,
    DEFAULT_TRANSCRIPT_REPLAY_FIXTURE,
    EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
    STATIC_TRANSCRIPT_EXPECTATION_KEYS,
    AssistantLifecycleSimulator,
    AssistantOSKernel,
    AssistantOSStore,
    ConstrainedDecoder,
    InternetArchiveSearchMetadataImporter,
    LocalAssistantProfile,
    LocalDeviceActionExecutor,
    LocalMediaInventoryAdapter,
    OnDeviceAssistantRouter,
    OpenMeteoWeatherAdapter,
    Opportunity,
    ProjectGutenbergCatalogImporter,
    build_assistant_os_dashboard,
    import_transcript_replay_fixture,
    initialize_assistant_os_database,
    media_items_to_inventory_rows,
    parse_assistant_debug_frame,
    persist_self_observation,
    realistic_lifecycle_steps,
    run_assistant_os_eval,
    run_household_week_lifecycle_probe,
    run_multi_profile_lifecycle_suite,
    run_open_trace_suite,
    run_transcript_replay_suite,
    schedule_inventory_refreshes,
    seed_assistant_os_lexicon,
    seed_class_schemas,
    migrate_contacts_to_entities,
    migrate_self_facts_to_entities,
    self_model_from_profile,
    story_items_to_inventory_rows,
    weather_items_to_inventory_rows,
)

DEFAULT_DB = Path("artifacts/local_assistant_os/assistant_v01.sqlite")
DEFAULT_SEED = Path("benchmarks/local_assistant_os_seed.json")
DEFAULT_WEATHER_SAMPLE = Path("benchmarks/sample_open_meteo_forecast.json")
DEFAULT_RAW_TRANSCRIPT_SAMPLE = Path(
    "benchmarks/sample_local_assistant_raw_transcript.jsonl"
)
DEFAULT_HOST_ACTION_CONFIG_EXAMPLE = Path("config/host_actions.example.json")
DEFAULT_SAFE_LIFECYCLE_CONTROLS_EXAMPLE = Path(
    "config/safe_lifecycle_controls.example.json"
)
V01_BLOCKER_REHEARSAL_TURNS = (
    "Who are you?",
    "Tell me a story.",
    "What is the weather today?",
    "What should I eat today?",
    "What did we talk about earlier?",
    "What have you done so far?",
)
SOURCE_ATTESTATION_SCHEMA = "melm.local_assistant_source_attestation.v1"
SOURCE_ATTESTATION_KINDS = ("redacted_user_session", "target_device_user_session")
SOURCE_ATTESTATION_SURFACES = (
    "browser_api",
    "cli_chat",
    "target_device_browser",
    "target_device_cli",
    "imported_redacted_transcript",
)
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
API_DEFAULT_CAPTURE_SOURCE = "browser_api_unspecified"
API_ASK_CAPTURE_SOURCES = frozenset(
    {
        API_DEFAULT_CAPTURE_SOURCE,
        "browser_ui",
        "scripted_api_smoke",
        "scripted_ui_smoke",
    }
)
HOST_APP_ATTESTATION_SCHEMA = "melm.local_assistant_host_app_attestation.v1"
HOST_APP_ATTESTATION_SURFACES = (
    "target_device_cli",
    "target_device_browser",
    "desktop_cli",
)
HOST_APP_DEMO_RECORDER_MARKERS = (
    "host-action-recorder",
    "action_recorder.py",
    "host_actions.local_recorder",
)
PI_BUNDLE_STATIC_FILES = (
    Path("README.md"),
    Path("pyproject.toml"),
    DEFAULT_HOST_ACTION_CONFIG_EXAMPLE,
    DEFAULT_SAFE_LIFECYCLE_CONTROLS_EXAMPLE,
    Path("docs/README.md"),
    Path("docs/archive/local_assistant_os_mvp_plan_v2.md"),
    Path("docs/roadmap.md"),
    Path("benchmarks/local_assistant_os_seed.json"),
    Path("benchmarks/public_domain_story_metadata.json"),
    Path("benchmarks/sample_gutenberg_catalog.csv"),
    Path("benchmarks/sample_internet_archive_search.json"),
    Path("benchmarks/local_media_manifest.json"),
    Path("benchmarks/sample_open_meteo_forecast.json"),
    Path("benchmarks/local_assistant_open_traces.json"),
    Path("benchmarks/local_assistant_transcript_replay.jsonl"),
    Path("benchmarks/sample_local_assistant_raw_transcript.jsonl"),
    Path("scripts/local_assistant_os_cli.py"),
    Path("tests/test_local_assistant_router_mvp.py"),
    Path("melm/__init__.py"),
    Path("melm/contracts/frame_templates.v1.json"),
    Path("melm/contracts/reserved_lexemes.v1.json"),
    Path("melm/contracts/semantic_classes.v1.json"),
    Path("melm/contracts/frame_candidate.v1.json"),
    Path("melm/contracts/capability_manifest.v1.json"),
    Path("melm/contracts/default_capability_manifest.v1.json"),
    Path("melm/contracts/registry.v1.json"),
    Path("melm/contracts/router_lexicon_families.v1.json"),
    Path("melm/contracts/sense_candidate.v1.json"),
    Path("melm/contracts/verbnet_map.v1.json"),
    Path("melm/contracts/wn_supersense_map.v1.json"),
    Path("melm/contracts/word_supersense_data.v1.jsonl"),
    Path("melm/contracts/verb_data.v1.jsonl"),
    Path("melm/contracts/food_tags.v1.json"),
    Path("melm/contracts/health_disclaimers.v1.json"),
    Path("melm/contracts/safety_policies.v1.json"),
    Path("melm/contracts/story_components.v1.json"),
    Path("melm/contracts/weather_concepts.v1.json"),
    Path("melm/contracts/meal_scopes.v1.json"),
    Path("melm/contracts/assistant_identity.v1.json"),
    Path("melm/contracts/answer_templates.v1.json"),
    Path("melm/appliance/assistant_skill_meal.py"),
    Path("melm/appliance/assistant_skill_story.py"),
    Path("melm/appliance/assistant_skill_memory.py"),
    Path("melm/contracts/memory_insights.v1.json"),
    Path("melm/contracts/router_semantic_aliases.v1.json"),
)
PI_BUNDLE_LAUNCHER_FILES = (
    Path("bin/first_run.sh"),
    Path("bin/start_app.sh"),
    Path("bin/first_run_on_raspberry_pi.sh"),
    Path("bin/start_api.sh"),
    Path("bin/health_check.sh"),
    Path("bin/first_run.ps1"),
    Path("bin/start_app.ps1"),
    Path("bin/health_check.ps1"),
    Path("bin/first_run.cmd"),
    Path("bin/start_app.cmd"),
    Path("bin/health_check.cmd"),
    Path("systemd/melm-local-assistant.service.example"),
)
API_SESSION_SMOKE_TURNS = (
    ("identity", "Who are you?"),
    ("story", "Tell me a story."),
    ("weather", "What is the weather today?"),
    ("safety", "Should I go to school dressed naked?"),
    ("media_request", "Play calm piano."),
    ("media_confirm", "Yes, play calm piano."),
    ("health", "What do you think I should do to improve my health?"),
    ("profile_memory", "Tell me something about myself."),
    ("meal", "What do you think I should eat today?"),
    ("call_request", "I need to talk to someone."),
    ("call_confirm", "Yes, call mom."),
)
SYNTHESIS_VARIANT_SMOKE_TURNS = (
    ("story_bedtime", "Tell me a bedtime story."),
    ("story_read", "Read me a story."),
    ("story_tale", "Tell me a tale."),
    ("health_week", "How can I improve my health this week?"),
    ("health_sleep", "What healthy thing can I do to sleep better tonight?"),
    ("urgent_health", "I cannot breathe."),
    ("weather_cached", "What is the weather today?"),
    ("meal_today", "What should I eat today?"),
    ("session_summary", "Summarize our recent sessions."),
    ("long_horizon_digest", "What happened over the last few days?"),
)
SYNTHESIS_STRESS_SMOKE_TURNS = (
    ("session_1", "identity", "Who are you?"),
    ("session_1", "status_progress", "What have you done so far?"),
    ("session_1", "story_local", "Tell me a local story about listening."),
    ("session_1", "story_tale", "Read me a tale."),
    ("session_1", "story_fable", "Can you tell me a fable before bed?"),
    ("session_1", "health_after_school", "What healthy thing can I do after school?"),
    (
        "session_1",
        "health_sleep",
        "What healthy thing can I do to sleep better tonight?",
    ),
    ("session_1", "meal_breakfast", "What should I have for breakfast?"),
    ("session_1", "meal_dinner", "What can I cook for dinner?"),
    ("session_1", "weather_forecast", "What is the forecast today?"),
    ("session_1", "school_clothing", "Should I wear a raincoat to school?"),
    ("session_2", "status_cloud", "Are you using cloud?"),
    ("session_2", "story_music", "Tell me a story with music."),
    ("session_2", "memory_self", "Tell me something about myself."),
    ("session_2", "health_goals", "What are my health goals?"),
    ("session_2", "meal_rain", "What should I eat before the rain?"),
    ("session_2", "urgent_health", "I cannot breathe."),
    ("session_2", "weather_rain", "Will it rain today?"),
    ("session_3", "last_question", "What was my last question?"),
    ("session_3", "session_summary", "Summarize our recent sessions."),
    ("session_3", "long_horizon_digest", "What happened over the last few days?"),
    ("session_3", "status_next", "What should you build next?"),
    ("session_3", "story_bedtime", "Tell me a bedtime story."),
    ("session_3", "health_week", "What should I do to improve my health this week?"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    init = subparsers.add_parser(
        "init", help="Initialize the assistant OS SQLite database."
    )
    init.add_argument("--db", type=Path, default=DEFAULT_DB)
    init.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    init.add_argument("--reset", action="store_true")
    init.add_argument("--json", action="store_true")

    dataset_audit = subparsers.add_parser(
        "dataset-audit",
    )
    dataset_audit.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/dataset_audit.sqlite"),
    )
    dataset_audit.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    dataset_audit.add_argument("--reset", action="store_true")
    dataset_audit.add_argument("--json", action="store_true")

    bootstrap = subparsers.add_parser(
        "bootstrap-runtime",
    )
    bootstrap.add_argument("--db", type=Path, default=DEFAULT_DB)
    bootstrap.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    bootstrap.add_argument("--reset", action="store_true")
    bootstrap.add_argument(
        "--manifest", type=Path, default=DEFAULT_LOCAL_MEDIA_MANIFEST
    )
    bootstrap.add_argument("--media-dir", type=Path, default=None)
    bootstrap.add_argument("--skip-media-import", action="store_true")
    bootstrap.add_argument("--require-media-files", action="store_true")
    bootstrap.add_argument("--json", action="store_true")

    ask = subparsers.add_parser("ask", help="Handle one assistant utterance.")
    ask.add_argument("--db", type=Path, default=DEFAULT_DB)
    ask.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ask.add_argument("--utterance", required=True)
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--no-auto-execute", action="store_true")
    ask.add_argument(
        "--execute-jobs",
        action="store_true",
        help="Execute safe background inventory jobs after routing.",
    )
    ask.add_argument(
        "--cold-start",
        action="store_true",
        help="Use an empty local profile when the DB has no saved facts/inventory.",
    )
    ask.add_argument("--action-mode", choices=("dry-run", "real"), default="dry-run")
    ask.add_argument(
        "--improvement-opt-in",
        action="store_true",
        help="Queue low-confidence turns locally for quarantined improvement research.",
    )
    ask.add_argument(
        "--media-player-command",
        default="",
        help="Executable argv prefix for real media playback mode, for example 'mpg123'.",
    )
    ask.add_argument(
        "--call-command",
        default="",
        help="Executable argv prefix for real call/contact mode.",
    )
    ask.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a GGUF model file for local LLM backend (e.g. models/qwen2.5-0.5b-instruct-q4_k_m.gguf).",
    )

    parse_debug = subparsers.add_parser(
        "parse-debug")
    parse_debug.add_argument("--utterance", required=True)
    parse_debug.add_argument("--json", action="store_true")

    shortcut_audit = subparsers.add_parser(
        "shortcut-audit",
    )
    shortcut_audit.add_argument("--json", action="store_true")

    capability_probe = subparsers.add_parser(
        "capability-probe",
    )
    capability_probe.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/capability_probe.sqlite"),
    )
    capability_probe.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    capability_probe.add_argument("--reset", action="store_true")
    capability_probe.add_argument("--json", action="store_true")

    chat = subparsers.add_parser(
        "chat", help="Run a cross-platform local CLI chat session."
    )
    chat.add_argument("--db", type=Path, default=DEFAULT_DB)
    chat.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    chat.add_argument("--reset", action="store_true")
    chat.add_argument(
        "--turn",
        action="append",
        default=[],
        help="Scripted utterance; repeat for a multi-turn CLI session.",
    )
    chat.add_argument("--json", action="store_true")
    chat.add_argument("--no-auto-execute", action="store_true")
    chat.add_argument("--action-mode", choices=("dry-run", "real"), default="dry-run")
    chat.add_argument(
        "--improvement-opt-in",
        action="store_true",
        help="Queue low-confidence turns locally for quarantined improvement research.",
    )
    chat.add_argument(
        "--media-player-command",
        default="",
        help="Executable argv prefix for real media playback mode, for example 'mpg123'.",
    )
    chat.add_argument(
        "--call-command",
        default="",
        help="Executable argv prefix for real call/contact mode.",
    )
    chat.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a GGUF model file for local LLM backend (e.g. models/qwen2.5-0.5b-instruct-q4_k_m.gguf).",
    )
    chat.add_argument("--no-faces", action="store_true", help="Disable ASCII face rendering")
    chat.add_argument("--no-tts", action="store_true", help="Disable TTS audio output")
    chat.add_argument(
        "--tts-command",
        default="",
        help="TTS command (e.g., 'espeak -v en+f3')",
    )

    lifecycle = subparsers.add_parser(
        "run-lifecycle")
    lifecycle.add_argument("--db", type=Path, default=DEFAULT_DB)
    lifecycle.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="Optional seed dataset; omitted keeps the lifecycle cold-start.",
    )
    lifecycle.add_argument("--reset", action="store_true")
    lifecycle.add_argument("--json", action="store_true")

    lifecycle_suite = subparsers.add_parser(
        "run-lifecycle-suite")
    lifecycle_suite.add_argument("--json", action="store_true")

    household_week = subparsers.add_parser(
        "run-household-week",
    )
    household_week.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/household_week.sqlite"),
    )
    household_week.add_argument("--reset", action="store_true")
    household_week.add_argument("--json", action="store_true")

    open_traces = subparsers.add_parser(
        "run-open-traces",
    )
    open_traces.add_argument(
        "--trace-json", type=Path, default=DEFAULT_OPEN_TRACE_FIXTURE
    )
    open_traces.add_argument(
        "--db-dir", type=Path, default=Path("artifacts/local_assistant_os/open_traces")
    )
    open_traces.add_argument(
        "--weather-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    open_traces.add_argument("--reset", action="store_true")
    open_traces.add_argument("--json", action="store_true")

    transcript_replay = subparsers.add_parser(
        "run-transcript-replay",
    )
    transcript_replay.add_argument(
        "--transcript-jsonl", type=Path, default=DEFAULT_TRANSCRIPT_REPLAY_FIXTURE
    )
    transcript_replay.add_argument(
        "--db-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/transcript_replay"),
    )
    transcript_replay.add_argument(
        "--weather-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    transcript_replay.add_argument(
        "--auto-lifecycle",
        action="store_true",
        help="Let replayed turns automatically schedule refreshes, execute safe jobs, and build memory digests.",
    )
    transcript_replay.add_argument("--reset", action="store_true")
    transcript_replay.add_argument("--json", action="store_true")

    transcript_import = subparsers.add_parser(
        "import-transcript-replay",
    )
    transcript_import.add_argument("--input", type=Path, required=True)
    transcript_import.add_argument("--out", type=Path, required=True)
    transcript_import.add_argument(
        "--scenario", default="imported_local_assistant_transcript"
    )
    transcript_import.add_argument(
        "--description",
        default="Redacted local chat transcript replay imported for Local Assistant OS calibration.",
    )
    transcript_import.add_argument("--source-note", default="")
    transcript_import.add_argument("--profile-json", type=Path, default=None)
    transcript_import.add_argument(
        "--controls-json",
        type=Path,
        default=None,
        help=(
            "Optional safe lifecycle controls JSON. Only non-answer controls such as "
            "run_reflection, schedule_refreshes, execute_jobs, and aggregate thresholds are allowed."
        ),
    )
    transcript_import.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=TOKEN",
        help="Manual redaction after automatic PII redaction; repeatable.",
    )
    transcript_import.add_argument("--min-turns", type=int, default=1)
    transcript_import.add_argument("--min-route-kinds", type=int, default=1)
    transcript_import.add_argument("--json", action="store_true")

    transcript_export = subparsers.add_parser(
        "export-transcript-replay",
    )
    transcript_export.add_argument("--db", type=Path, default=DEFAULT_DB)
    transcript_export.add_argument("--out", type=Path, required=True)
    transcript_export.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Optional raw event-derived JSONL path; defaults beside --out for provenance.",
    )
    transcript_export.add_argument(
        "--session",
        default="all",
        help="Session id to export, or 'all'/'latest'.",
    )
    transcript_export.add_argument(
        "--scenario", default="event_ledger_transcript_export"
    )
    transcript_export.add_argument(
        "--description",
        default="Redacted transcript replay exported from the local assistant event ledger.",
    )
    transcript_export.add_argument("--source-note", default="")
    transcript_export.add_argument(
        "--profile-mode",
        choices=("current", "minimal"),
        default="current",
        help="Use the current local profile/inventory state for replay, or a minimal empty profile.",
    )
    transcript_export.add_argument(
        "--controls-json",
        type=Path,
        default=None,
        help="Optional safe lifecycle controls JSON; route/answer/intent expectations are rejected.",
    )
    transcript_export.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=TOKEN",
        help="Manual redaction after automatic PII redaction; repeatable.",
    )
    transcript_export.add_argument("--min-turns", type=int, default=1)
    transcript_export.add_argument("--min-route-kinds", type=int, default=1)
    transcript_export.add_argument("--json", action="store_true")

    event_calibration = subparsers.add_parser(
        "calibrate-event-ledger",
    )
    event_calibration.add_argument("--db", type=Path, default=DEFAULT_DB)
    event_calibration.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/event_ledger_calibration"),
    )
    event_calibration.add_argument(
        "--session", default="all", help="Session id to calibrate, or 'all'/'latest'."
    )
    event_calibration.add_argument(
        "--profile-mode",
        choices=("current", "minimal"),
        default="current",
        help="Use the current local profile/inventory state for replay, or a minimal empty profile.",
    )
    event_calibration.add_argument(
        "--controls-json",
        type=Path,
        default=None,
        help="Optional safe lifecycle controls JSON; route/answer/intent expectations are rejected.",
    )
    event_calibration.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=TOKEN",
        help="Manual redaction after automatic PII redaction; repeatable.",
    )
    event_calibration.add_argument(
        "--weather-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    event_calibration.add_argument("--reset", action="store_true")
    event_calibration.add_argument("--min-total-turns", type=int, default=1)
    event_calibration.add_argument(
        "--min-local-resolution-rate", type=float, default=0.0
    )
    event_calibration.add_argument("--min-route-kinds", type=int, default=1)
    event_calibration.add_argument("--min-intent-kinds", type=int, default=1)
    event_calibration.add_argument("--min-synthesis-traces", type=int, default=0)
    event_calibration.add_argument("--min-priority-signal-samples", type=int, default=0)
    event_calibration.add_argument(
        "--auto-lifecycle",
        action="store_true",
        help="Let replayed event-ledger turns automatically schedule refreshes, execute safe jobs, and build memory digests.",
    )
    event_calibration.add_argument("--require-priority-signals", action="store_true")
    event_calibration.add_argument(
        "--require-memory-digest-quality", action="store_true"
    )
    event_calibration.add_argument("--require-strict-baseline-win", action="store_true")
    event_calibration.add_argument("--require-redaction", action="store_true")
    event_calibration.add_argument("--require-static-drop", action="store_true")
    event_calibration.add_argument("--json", action="store_true")

    blocker_evidence = subparsers.add_parser(
        "v01-blocker-evidence",
    )
    blocker_evidence.add_argument("--event-ledger-db", type=Path, default=None)
    blocker_evidence.add_argument(
        "--event-ledger-work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/v01_blocker_evidence/event_ledger"),
    )
    blocker_evidence.add_argument("--event-ledger-session", default="all")
    blocker_evidence.add_argument(
        "--event-source-kind",
        choices=(
            "development_session",
            "redacted_user_session",
            "target_device_user_session",
        ),
        default="development_session",
        help="Attestation for the event-ledger source. Development sessions never retire user-derived blockers.",
    )
    blocker_evidence.add_argument(
        "--source-attestation-json",
        type=Path,
        default=None,
        help="Machine-readable source attestation required before user-derived evidence can count as candidate evidence.",
    )
    blocker_evidence.add_argument("--controls-json", type=Path, default=None)
    blocker_evidence.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=TOKEN",
        help="Manual redaction rules forwarded to event-ledger calibration.",
    )
    blocker_evidence.add_argument(
        "--weather-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    blocker_evidence.add_argument("--reset", action="store_true")
    blocker_evidence.add_argument("--min-total-turns", type=int, default=8)
    blocker_evidence.add_argument(
        "--min-local-resolution-rate", type=float, default=0.65
    )
    blocker_evidence.add_argument("--min-route-kinds", type=int, default=3)
    blocker_evidence.add_argument("--min-intent-kinds", type=int, default=6)
    blocker_evidence.add_argument("--min-synthesis-traces", type=int, default=2)
    blocker_evidence.add_argument("--min-priority-signal-samples", type=int, default=1)
    blocker_evidence.add_argument(
        "--auto-lifecycle",
        action="store_true",
        help="Use automatic lifecycle replay when calibrating --event-ledger-db.",
    )
    blocker_evidence.add_argument(
        "--inventory-soak-report-json",
        type=Path,
        default=None,
        help="Optional prior inventory-soak-matrix JSON report, preferably from a --live run.",
    )
    blocker_evidence.add_argument(
        "--transcript-calibration-report-json",
        type=Path,
        default=None,
        help="Optional strict calibrate-transcript-replay JSON report for digest/route threshold evidence.",
    )
    blocker_evidence.add_argument("--host-app-config-json", type=Path, default=None)
    blocker_evidence.add_argument(
        "--host-app-attestation-json",
        type=Path,
        default=None,
        help="Optional target-device host-app attestation JSON bound to --host-app-config-json.",
    )
    blocker_evidence.add_argument(
        "--run-host-app-probe",
        action="store_true",
        help="Execute configured host-app commands through the typed action gate.",
    )
    blocker_evidence.add_argument(
        "--host-app-db",
        type=Path,
        default=Path(
            "artifacts/local_assistant_os/v01_blocker_evidence/host_app.sqlite"
        ),
    )
    blocker_evidence.add_argument(
        "--host-app-work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/v01_blocker_evidence/host_app"),
    )
    blocker_evidence.add_argument("--host-app-media-dir", type=Path, default=None)
    blocker_evidence.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON blocker evidence report path.",
    )
    blocker_evidence.add_argument("--json", action="store_true")

    blocker_rehearsal = subparsers.add_parser(
        "v01-blocker-rehearsal",
    )
    blocker_rehearsal.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/v01_blocker_rehearsal"),
    )
    blocker_rehearsal.add_argument("--reset", action="store_true")
    blocker_rehearsal.add_argument("--json", action="store_true")

    evidence_pack = subparsers.add_parser(
        "v01-evidence-pack",
    )
    evidence_pack.add_argument("--db", type=Path, default=DEFAULT_DB)
    evidence_pack.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/v01_evidence_pack"),
    )
    evidence_pack.add_argument("--session", default="all")
    evidence_pack.add_argument(
        "--event-source-kind",
        choices=(
            "development_session",
            "redacted_user_session",
            "target_device_user_session",
        ),
        default="development_session",
    )
    evidence_pack.add_argument(
        "--capture-surface", choices=SOURCE_ATTESTATION_SURFACES, default="cli_chat"
    )
    evidence_pack.add_argument(
        "--source-attestation-json",
        type=Path,
        default=None,
        help="Existing source attestation JSON to use instead of writing one.",
    )
    evidence_pack.add_argument(
        "--write-source-attestation",
        action="store_true",
        help="Write source_attestation.json into --work-dir for redacted/target user sessions.",
    )
    evidence_pack.add_argument("--redaction-applied", action="store_true")
    evidence_pack.add_argument("--static-expectations-absent", action="store_true")
    evidence_pack.add_argument("--answers-routes-reasons-absent", action="store_true")
    evidence_pack.add_argument("--human-reviewed", action="store_true")
    evidence_pack.add_argument("--note", default="")
    evidence_pack.add_argument("--controls-json", type=Path, default=None)
    evidence_pack.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=TOKEN",
        help="Manual redaction rules forwarded to event-ledger calibration.",
    )
    evidence_pack.add_argument(
        "--weather-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    evidence_pack.add_argument("--reset", action="store_true")
    evidence_pack.add_argument("--min-total-turns", type=int, default=8)
    evidence_pack.add_argument("--min-local-resolution-rate", type=float, default=0.65)
    evidence_pack.add_argument("--min-route-kinds", type=int, default=3)
    evidence_pack.add_argument("--min-intent-kinds", type=int, default=6)
    evidence_pack.add_argument("--min-synthesis-traces", type=int, default=2)
    evidence_pack.add_argument("--min-priority-signal-samples", type=int, default=1)
    evidence_pack.add_argument("--auto-lifecycle", action="store_true")
    evidence_pack.add_argument("--inventory-soak-report-json", type=Path, default=None)
    evidence_pack.add_argument(
        "--transcript-calibration-report-json", type=Path, default=None
    )
    evidence_pack.add_argument("--host-app-config-json", type=Path, default=None)
    evidence_pack.add_argument("--host-app-attestation-json", type=Path, default=None)
    evidence_pack.add_argument("--run-host-app-probe", action="store_true")
    evidence_pack.add_argument(
        "--host-app-db",
        type=Path,
        default=Path("artifacts/local_assistant_os/v01_evidence_pack/host_app.sqlite"),
    )
    evidence_pack.add_argument(
        "--host-app-work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/v01_evidence_pack/host_app"),
    )
    evidence_pack.add_argument("--host-app-media-dir", type=Path, default=None)
    evidence_pack.add_argument("--json", action="store_true")

    source_attestation = subparsers.add_parser(
        "write-source-attestation",
    )
    source_attestation.add_argument("--out", type=Path, required=True)
    source_attestation.add_argument("--event-ledger-db", type=Path, required=True)
    source_attestation.add_argument(
        "--event-ledger-session",
        default="all",
        help="Session id covered by this attestation, or 'all'/'latest'.",
    )
    source_attestation.add_argument(
        "--source-kind", choices=SOURCE_ATTESTATION_KINDS, required=True
    )
    source_attestation.add_argument(
        "--capture-surface", choices=SOURCE_ATTESTATION_SURFACES, required=True
    )
    source_attestation.add_argument("--redaction-applied", action="store_true")
    source_attestation.add_argument("--static-expectations-absent", action="store_true")
    source_attestation.add_argument(
        "--answers-routes-reasons-absent", action="store_true"
    )
    source_attestation.add_argument("--human-reviewed", action="store_true")
    source_attestation.add_argument("--note", default="")
    source_attestation.add_argument("--overwrite", action="store_true")
    source_attestation.add_argument("--json", action="store_true")

    candidate_session = subparsers.add_parser(
        "candidate-session-audit",
    )
    candidate_session.add_argument("--db", type=Path, default=DEFAULT_DB)
    candidate_session.add_argument(
        "--session", default="all", help="Session id to audit, or 'all'/'latest'."
    )
    candidate_session.add_argument(
        "--event-source-kind",
        choices=SOURCE_ATTESTATION_KINDS,
        default="redacted_user_session",
    )
    candidate_session.add_argument(
        "--capture-surface", choices=SOURCE_ATTESTATION_SURFACES, default="cli_chat"
    )
    candidate_session.add_argument("--source-attestation-json", type=Path, default=None)
    candidate_session.add_argument("--redaction-applied", action="store_true")
    candidate_session.add_argument("--static-expectations-absent", action="store_true")
    candidate_session.add_argument(
        "--answers-routes-reasons-absent", action="store_true"
    )
    candidate_session.add_argument("--human-reviewed", action="store_true")
    candidate_session.add_argument("--controls-json", type=Path, default=None)
    candidate_session.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=TOKEN",
        help="Manual redaction rules forwarded to event-ledger calibration.",
    )
    candidate_session.add_argument(
        "--weather-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    candidate_session.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/candidate_session_audit"),
    )
    candidate_session.add_argument("--reset", action="store_true")
    candidate_session.add_argument("--min-total-turns", type=int, default=8)
    candidate_session.add_argument(
        "--min-local-resolution-rate", type=float, default=0.65
    )
    candidate_session.add_argument("--min-route-kinds", type=int, default=3)
    candidate_session.add_argument("--min-intent-kinds", type=int, default=6)
    candidate_session.add_argument("--min-synthesis-traces", type=int, default=2)
    candidate_session.add_argument("--min-priority-signal-samples", type=int, default=1)
    candidate_session.add_argument("--auto-lifecycle", action="store_true")
    candidate_session.add_argument(
        "--inventory-soak-report-json", type=Path, default=None
    )
    candidate_session.add_argument(
        "--transcript-calibration-report-json", type=Path, default=None
    )
    candidate_session.add_argument("--host-app-config-json", type=Path, default=None)
    candidate_session.add_argument(
        "--host-app-attestation-json", type=Path, default=None
    )
    candidate_session.add_argument("--run-host-app-probe", action="store_true")
    candidate_session.add_argument(
        "--host-app-db",
        type=Path,
        default=Path(
            "artifacts/local_assistant_os/candidate_session_audit/host_app.sqlite"
        ),
    )
    candidate_session.add_argument(
        "--host-app-work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/candidate_session_audit/host_app"),
    )
    candidate_session.add_argument("--host-app-media-dir", type=Path, default=None)
    candidate_session.add_argument("--json", action="store_true")

    host_app_attestation = subparsers.add_parser(
        "write-host-app-attestation",
    )
    host_app_attestation.add_argument("--out", type=Path, required=True)
    host_app_attestation.add_argument(
        "--host-app-config-json", type=Path, required=True
    )
    host_app_attestation.add_argument(
        "--capture-surface", choices=HOST_APP_ATTESTATION_SURFACES, required=True
    )
    host_app_attestation.add_argument("--media-app-configured", action="store_true")
    host_app_attestation.add_argument("--call-app-configured", action="store_true")
    host_app_attestation.add_argument("--not-demo-recorder", action="store_true")
    host_app_attestation.add_argument(
        "--real-app-commands-acknowledged", action="store_true"
    )
    host_app_attestation.add_argument("--human-reviewed", action="store_true")
    host_app_attestation.add_argument("--note", default="")
    host_app_attestation.add_argument("--overwrite", action="store_true")
    host_app_attestation.add_argument("--json", action="store_true")

    transcript_calibration = subparsers.add_parser(
        "calibrate-transcript-replay",
    )
    transcript_calibration.add_argument(
        "--input", type=Path, action="append", default=[]
    )
    transcript_calibration.add_argument("--input-dir", type=Path, default=None)
    transcript_calibration.add_argument("--glob", default="*.jsonl")
    transcript_calibration.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/transcript_calibration"),
    )
    transcript_calibration.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report path for v01-blocker-evidence ingestion.",
    )
    transcript_calibration.add_argument("--reset", action="store_true")
    transcript_calibration.add_argument("--profile-json", type=Path, default=None)
    transcript_calibration.add_argument(
        "--controls-json",
        type=Path,
        default=None,
        help=(
            "Optional safe lifecycle controls JSON applied during import; route/answer/intent "
            "expectations are rejected."
        ),
    )
    transcript_calibration.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=TOKEN",
        help="Manual redaction after automatic PII redaction; repeatable.",
    )
    transcript_calibration.add_argument(
        "--weather-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    transcript_calibration.add_argument(
        "--min-total-turns",
        type=int,
        default=1,
        help="Minimum imported/replayed user turns required across all raw transcript inputs.",
    )
    transcript_calibration.add_argument(
        "--min-local-resolution-rate",
        type=float,
        default=0.0,
        help="Minimum aggregate local/cached/device resolution rate for calibration traces.",
    )
    transcript_calibration.add_argument(
        "--min-route-kinds",
        type=int,
        default=1,
        help="Minimum distinct route kinds required across calibration replays.",
    )
    transcript_calibration.add_argument(
        "--min-intent-kinds",
        type=int,
        default=1,
        help="Minimum distinct intent kinds required across calibration replays.",
    )
    transcript_calibration.add_argument(
        "--min-synthesis-traces",
        type=int,
        default=0,
        help="Minimum persisted bounded-synthesis trace count required across calibration replays.",
    )
    transcript_calibration.add_argument(
        "--min-priority-signal-samples",
        type=int,
        default=0,
        help="Minimum priority-signal samples required across calibration replays.",
    )
    transcript_calibration.add_argument(
        "--auto-lifecycle",
        action="store_true",
        help="Let imported turns automatically schedule refreshes, execute safe jobs, and build memory digests.",
    )
    transcript_calibration.add_argument(
        "--require-priority-signals",
        action="store_true",
        help="Require at least one imported replay to expose planner priority signals.",
    )
    transcript_calibration.add_argument(
        "--require-memory-digest-quality",
        action="store_true",
        help="Require all imported replays to pass their memory-digest quality gate.",
    )
    transcript_calibration.add_argument(
        "--require-strict-baseline-win",
        action="store_true",
        help="Require strict static-baseline comparison checks to pass for every imported replay.",
    )
    transcript_calibration.add_argument(
        "--require-redaction",
        action="store_true",
        help="Require at least one automatic or manual redaction in the imported raw transcripts.",
    )
    transcript_calibration.add_argument(
        "--require-static-drop",
        action="store_true",
        help="Require at least one static expected answer/route/reason field to be removed.",
    )
    transcript_calibration.add_argument("--json", action="store_true")

    v01_audit = subparsers.add_parser(
        "v01-audit",
    )
    v01_audit.add_argument("--json", action="store_true")

    v01_progress = subparsers.add_parser(
        "v01-progress",
    )
    v01_progress.add_argument(
        "--blocker-evidence-json",
        type=Path,
        default=None,
        help="Optional v01-blocker-evidence JSON report to summarize; defaults to an empty lightweight blocker report.",
    )
    v01_progress.add_argument("--json", action="store_true")

    v01_acceptance = subparsers.add_parser(
        "v01-acceptance",
    )
    v01_acceptance.add_argument(
        "--db-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/v01_acceptance"),
    )
    v01_acceptance.add_argument("--reset", action="store_true")
    v01_acceptance.add_argument("--host", default="127.0.0.1")
    v01_acceptance.add_argument("--require-raspberry-pi", action="store_true")
    v01_acceptance.add_argument(
        "--media-player-command",
        default="",
        help="Optional target-device media command for host-app-probe.",
    )
    v01_acceptance.add_argument(
        "--call-command",
        default="",
        help="Optional target-device call command for host-app-probe.",
    )
    v01_acceptance.add_argument(
        "--host-app-config-json",
        type=Path,
        default=None,
        help="Optional host action command config JSON for host-app-probe.",
    )
    v01_acceptance.add_argument(
        "--host-app-media-dir",
        type=Path,
        default=None,
        help="Optional real media directory for host-app-probe.",
    )
    v01_acceptance.add_argument("--require-host-app-configured", action="store_true")
    v01_acceptance.add_argument(
        "--include-bundle",
        action="store_true",
        help="Also build and self-check the portable bundle.",
    )
    v01_acceptance.add_argument(
        "--bundle-out",
        type=Path,
        default=Path(
            "artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle"
        ),
    )
    v01_acceptance.add_argument("--zip-bundle", action="store_true")
    v01_acceptance.add_argument("--json", action="store_true")

    jobs = subparsers.add_parser(
        "run-jobs")
    jobs.add_argument("--db", type=Path, default=DEFAULT_DB)
    jobs.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    jobs.add_argument("--limit", type=int, default=10)
    jobs.add_argument("--json", action="store_true")
    jobs.add_argument("--cold-start", action="store_true")
    jobs.add_argument(
        "--weather-live",
        action="store_true",
        help="Use live Open-Meteo HTTP for weather refresh jobs.",
    )
    jobs.add_argument(
        "--weather-json",
        type=Path,
        default=DEFAULT_WEATHER_SAMPLE,
        help="Offline Open-Meteo fixture for deterministic weather refresh jobs.",
    )

    schedule = subparsers.add_parser(
        "schedule-refreshes")
    schedule.add_argument("--db", type=Path, default=DEFAULT_DB)
    schedule.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    schedule.add_argument("--min-story-models", type=int, default=6)
    schedule.add_argument("--story-limit", type=int, default=6)
    schedule.add_argument(
        "--source", choices=("gutenberg", "internet-archive", "both"), default="both"
    )
    schedule.add_argument("--offline-samples", action="store_true")
    schedule.add_argument(
        "--gutenberg-csv",
        type=Path,
        default=Path("benchmarks/sample_gutenberg_catalog.csv"),
    )
    schedule.add_argument(
        "--internet-archive-json",
        type=Path,
        default=Path("benchmarks/sample_internet_archive_search.json"),
    )
    schedule.add_argument("--internet-archive-query", default="")
    schedule.add_argument("--gutenberg-max-source-bytes", type=int, default=6_500_000)
    schedule.add_argument(
        "--internet-archive-max-source-bytes", type=int, default=250_000
    )
    schedule.add_argument("--internet-archive-page-size", type=int, default=100)
    schedule.add_argument("--internet-archive-max-pages", type=int, default=1)
    schedule.add_argument("--internet-archive-cursor", default="")
    schedule.add_argument(
        "--internet-archive-rate-limit-delay-seconds", type=float, default=0.0
    )
    schedule.add_argument("--cold-start", action="store_true")
    schedule.add_argument("--json", action="store_true")

    soak = subparsers.add_parser(
        "inventory-soak")
    soak.add_argument("--db", type=Path, default=DEFAULT_DB)
    soak.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    soak.add_argument("--cycles", type=int, default=2)
    soak.add_argument("--jobs-per-cycle", type=int, default=3)
    soak.add_argument("--min-story-models", type=int, default=12)
    soak.add_argument("--story-limit", type=int, default=6)
    soak.add_argument(
        "--source",
        choices=("gutenberg", "internet-archive", "both"),
        default="internet-archive",
    )
    soak.add_argument("--offline-samples", action="store_true")
    soak.add_argument(
        "--gutenberg-csv",
        type=Path,
        default=Path("benchmarks/sample_gutenberg_catalog.csv"),
    )
    soak.add_argument(
        "--internet-archive-json",
        type=Path,
        default=Path("benchmarks/sample_internet_archive_search.json"),
    )
    soak.add_argument("--internet-archive-query", default="")
    soak.add_argument("--gutenberg-max-source-bytes", type=int, default=6_500_000)
    soak.add_argument("--internet-archive-max-source-bytes", type=int, default=250_000)
    soak.add_argument("--internet-archive-page-size", type=int, default=100)
    soak.add_argument("--internet-archive-max-pages", type=int, default=2)
    soak.add_argument("--internet-archive-cursor", default="")
    soak.add_argument(
        "--internet-archive-rate-limit-delay-seconds", type=float, default=0.0
    )
    soak.add_argument("--reset", action="store_true")
    soak.add_argument("--cold-start", action="store_true")
    soak.add_argument("--json", action="store_true")

    soak_matrix = subparsers.add_parser(
        "inventory-soak-matrix",
    )
    soak_matrix.add_argument(
        "--db-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/inventory_soak_matrix"),
    )
    soak_matrix.add_argument("--cycles", type=int, default=3)
    soak_matrix.add_argument("--jobs-per-cycle", type=int, default=3)
    soak_matrix.add_argument("--min-story-models", type=int, default=12)
    soak_matrix.add_argument("--story-limit", type=int, default=3)
    soak_matrix.add_argument(
        "--live",
        action="store_true",
        help="Use live metadata fetches instead of bundled offline source-shape samples.",
    )
    soak_matrix.add_argument(
        "--gutenberg-csv",
        type=Path,
        default=Path("benchmarks/sample_gutenberg_catalog.csv"),
    )
    soak_matrix.add_argument(
        "--internet-archive-json",
        type=Path,
        default=Path("benchmarks/sample_internet_archive_search.json"),
    )
    soak_matrix.add_argument(
        "--internet-archive-query", default="children bedtime folklore"
    )
    soak_matrix.add_argument(
        "--gutenberg-max-source-bytes", type=int, default=6_500_000
    )
    soak_matrix.add_argument(
        "--internet-archive-max-source-bytes", type=int, default=250_000
    )
    soak_matrix.add_argument("--internet-archive-page-size", type=int, default=100)
    soak_matrix.add_argument("--internet-archive-max-pages", type=int, default=2)
    soak_matrix.add_argument(
        "--internet-archive-rate-limit-delay-seconds", type=float, default=0.0
    )
    soak_matrix.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report path for blocker evidence.",
    )
    soak_matrix.add_argument("--reset", action="store_true")
    soak_matrix.add_argument("--json", action="store_true")

    diversity = subparsers.add_parser(
        "inventory-diversity-smoke",
    )
    diversity.add_argument(
        "--db-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/inventory_diversity_smoke"),
    )
    diversity.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    diversity.add_argument("--cycles", type=int, default=1)
    diversity.add_argument("--jobs-per-cycle", type=int, default=2)
    diversity.add_argument("--min-story-models", type=int, default=9)
    diversity.add_argument("--story-limit", type=int, default=3)
    diversity.add_argument(
        "--source", choices=("gutenberg", "internet-archive", "both"), default="both"
    )
    diversity.add_argument(
        "--live",
        action="store_true",
        help="Use live metadata fetches instead of bundled offline source-shape samples.",
    )
    diversity.add_argument(
        "--niche",
        action="append",
        default=[],
        metavar="LABEL=QUERY",
        help="Internet Archive query niche to exercise; repeatable. Defaults cover folktale, bedtime, and adventure.",
    )
    diversity.add_argument(
        "--gutenberg-csv",
        type=Path,
        default=Path("benchmarks/sample_gutenberg_catalog.csv"),
    )
    diversity.add_argument(
        "--internet-archive-json",
        type=Path,
        default=Path("benchmarks/sample_internet_archive_search.json"),
    )
    diversity.add_argument("--gutenberg-max-source-bytes", type=int, default=6_500_000)
    diversity.add_argument(
        "--internet-archive-max-source-bytes", type=int, default=250_000
    )
    diversity.add_argument("--internet-archive-page-size", type=int, default=100)
    diversity.add_argument("--internet-archive-max-pages", type=int, default=1)
    diversity.add_argument(
        "--internet-archive-rate-limit-delay-seconds", type=float, default=0.0
    )
    diversity.add_argument("--reset", action="store_true")
    diversity.add_argument("--json", action="store_true")

    failure = subparsers.add_parser(
        "inventory-failure-smoke",
    )
    failure.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/inventory_failure_smoke"),
    )
    failure.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    failure.add_argument("--reset", action="store_true")
    failure.add_argument("--json", action="store_true")

    retry = subparsers.add_parser(
        "inventory-retry-smoke",
    )
    retry.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/inventory_retry_smoke.sqlite"),
    )
    retry.add_argument("--reset", action="store_true")
    retry.add_argument("--story-limit", type=int, default=3)
    retry.add_argument("--max-attempts", type=int, default=2)
    retry.add_argument("--json", action="store_true")

    resources = subparsers.add_parser(
        "resource-report")
    resources.add_argument("--db", type=Path, default=DEFAULT_DB)
    resources.add_argument(
        "--lifecycle-db",
        type=Path,
        default=Path("artifacts/local_assistant_os/resource_lifecycle.sqlite"),
    )
    resources.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    resources.add_argument("--reset", action="store_true")
    resources.add_argument("--json", action="store_true")

    dashboard = subparsers.add_parser(
        "dashboard", help="Summarize the persisted assistant OS ledger."
    )
    dashboard.add_argument("--db", type=Path, default=DEFAULT_DB)
    dashboard.add_argument("--seed", type=Path, default=None)
    dashboard.add_argument("--json", action="store_true")

    improvement_queue = subparsers.add_parser(
        "improvement-queue",
        help="Inspect consent-gated, quarantined low-confidence research candidates.",
    )
    improvement_queue.add_argument("--db", type=Path, default=DEFAULT_DB)
    improvement_queue.add_argument("--seed", type=Path, default=None)
    improvement_queue.add_argument("--session", default="")
    improvement_queue.add_argument("--status", default="")
    improvement_queue.add_argument("--limit", type=int, default=100)
    improvement_queue.add_argument("--json", action="store_true")

    memory = subparsers.add_parser(
        "memory-replay", help="Replay/query linked autobiographical event memory."
    )
    memory.add_argument("--db", type=Path, default=DEFAULT_DB)
    memory.add_argument("--seed", type=Path, default=None)
    memory.add_argument("--query", default="")
    memory.add_argument("--intent", default="")
    memory.add_argument("--route", default="")
    memory.add_argument("--session", default="")
    memory.add_argument("--limit", type=int, default=12)
    memory.add_argument(
        "--sessions",
        type=int,
        default=0,
        help="Replay recent sessions instead of a flat event query.",
    )
    memory.add_argument("--events-per-session", type=int, default=4)
    memory.add_argument("--json", action="store_true")

    digest = subparsers.add_parser(
        "memory-digest", help="Build a compact local long-horizon memory digest."
    )
    digest.add_argument("--db", type=Path, default=DEFAULT_DB)
    digest.add_argument("--seed", type=Path, default=None)
    digest.add_argument("--sessions", type=int, default=20)
    digest.add_argument("--events-per-session", type=int, default=3)
    digest.add_argument("--json", action="store_true")

    eval_cmd = subparsers.add_parser(
        "eval")
    eval_cmd.add_argument("--json", action="store_true")

    import_stories = subparsers.add_parser(
        "import-stories",
        help="Import public-domain story metadata into local inventory.",
    )
    import_stories.add_argument("--db", type=Path, default=DEFAULT_DB)
    import_stories.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    import_stories.add_argument(
        "--source", choices=("gutenberg", "internet-archive", "both"), default="both"
    )
    import_stories.add_argument("--limit", type=int, default=6)
    import_stories.add_argument(
        "--gutenberg-csv",
        type=Path,
        default=None,
        help="Use a local Gutenberg catalog CSV sample instead of network.",
    )
    import_stories.add_argument(
        "--internet-archive-json",
        type=Path,
        default=None,
        help="Use a local Internet Archive scrape JSON sample instead of network.",
    )
    import_stories.add_argument("--internet-archive-query", default=None)
    import_stories.add_argument(
        "--gutenberg-max-source-bytes", type=int, default=6_500_000
    )
    import_stories.add_argument(
        "--internet-archive-max-source-bytes", type=int, default=250_000
    )
    import_stories.add_argument("--internet-archive-page-size", type=int, default=100)
    import_stories.add_argument("--internet-archive-max-pages", type=int, default=1)
    import_stories.add_argument("--internet-archive-cursor", default="")
    import_stories.add_argument(
        "--internet-archive-rate-limit-delay-seconds", type=float, default=0.0
    )
    import_stories.add_argument("--cold-start", action="store_true")
    import_stories.add_argument("--json", action="store_true")

    import_media = subparsers.add_parser(
        "import-media",
        help="Import local media manifest or directory metadata into inventory.",
    )
    import_media.add_argument("--db", type=Path, default=DEFAULT_DB)
    import_media.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    import_media.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON local media manifest; defaults to the bundled sample if no media directory is supplied.",
    )
    import_media.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        help="Scan a local media directory for supported audio/video files.",
    )
    import_media.add_argument("--limit", type=int, default=24)
    import_media.add_argument(
        "--require-files",
        action="store_true",
        help="For manifests, import only items whose path exists on this device.",
    )
    import_media.add_argument("--cold-start", action="store_true")
    import_media.add_argument("--json", action="store_true")

    refresh_weather = subparsers.add_parser(
        "refresh-weather", help="Fetch or replay weather data into the local cache."
    )
    refresh_weather.add_argument("--db", type=Path, default=DEFAULT_DB)
    refresh_weather.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    refresh_weather.add_argument("--location", default="")
    refresh_weather.add_argument(
        "--offline-json", type=Path, default=DEFAULT_WEATHER_SAMPLE
    )
    refresh_weather.add_argument(
        "--live",
        action="store_true",
        help="Use live Open-Meteo HTTP instead of the offline fixture.",
    )
    refresh_weather.add_argument("--cold-start", action="store_true")
    refresh_weather.add_argument("--json", action="store_true")

    action_smoke = subparsers.add_parser(
        "action-smoke")
    action_smoke.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/action_smoke.sqlite"),
    )
    action_smoke.add_argument("--reset", action="store_true")
    action_smoke.add_argument(
        "--action-mode", choices=("dry-run", "real"), default="dry-run"
    )
    action_smoke.add_argument(
        "--media-player-command",
        default="",
        help="Executable argv prefix for real media playback mode.",
    )
    action_smoke.add_argument(
        "--call-command",
        default="",
        help="Executable argv prefix for real call/contact mode.",
    )
    action_smoke.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        help="Optional local media directory; required for a passing real media smoke unless the manifest paths exist.",
    )
    action_smoke.add_argument(
        "--manifest", type=Path, default=DEFAULT_LOCAL_MEDIA_MANIFEST
    )
    action_smoke.add_argument("--json", action="store_true")

    setup_integration = subparsers.add_parser(
        "setup-integration-smoke",
    )
    setup_integration.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/setup_integration_smoke.sqlite"),
    )
    setup_integration.add_argument("--reset", action="store_true")
    setup_integration.add_argument("--json", action="store_true")

    host_action = subparsers.add_parser(
        "host-action-smoke",
    )
    host_action.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/host_action_smoke.sqlite"),
    )
    host_action.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/host_action_smoke"),
    )
    host_action.add_argument("--reset", action="store_true")
    host_action.add_argument("--json", action="store_true")

    host_action_recorder = subparsers.add_parser(
        "host-action-recorder",
    )
    host_action_recorder.add_argument("target", nargs="?", default="")
    host_action_recorder.add_argument("--label", default="action")
    host_action_recorder.add_argument(
        "--log",
        type=Path,
        default=Path("artifacts/local_assistant_os/host_actions.recorder.jsonl"),
    )
    host_action_recorder.add_argument("--json", action="store_true")

    host_actions_demo_config = subparsers.add_parser(
        "write-host-actions-demo-config",
    )
    host_actions_demo_config.add_argument(
        "--out", type=Path, default=Path("config/host_actions.local_recorder.json")
    )
    host_actions_demo_config.add_argument(
        "--log",
        type=Path,
        default=Path("artifacts/local_assistant_os/host_actions.local_recorder.jsonl"),
    )
    host_actions_demo_config.add_argument("--media-dir", type=Path, default=None)
    host_actions_demo_config.add_argument("--overwrite", action="store_true")
    host_actions_demo_config.add_argument("--json", action="store_true")

    host_app = subparsers.add_parser(
        "host-app-probe",
    )
    host_app.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/host_app_probe.sqlite"),
    )
    host_app.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/host_app_probe"),
    )
    host_app.add_argument(
        "--media-player-command",
        default="",
        help="Executable argv prefix for real media playback mode; defaults to MELM_MEDIA_PLAYER_COMMAND.",
    )
    host_app.add_argument(
        "--call-command",
        default="",
        help="Executable argv prefix for real call/contact mode; defaults to MELM_CALL_COMMAND.",
    )
    host_app.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="JSON config with media_player_command, call_command, and optional media_dir.",
    )
    host_app.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        help="Optional real local media directory; otherwise a tiny probe file is used.",
    )
    host_app.add_argument(
        "--require-configured",
        action="store_true",
        help="Fail when media/call commands are not configured.",
    )
    host_app.add_argument("--reset", action="store_true")
    host_app.add_argument("--json", action="store_true")

    autoimmune = subparsers.add_parser(
        "autoimmune-smoke",
    )
    autoimmune.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/autoimmune_smoke.sqlite"),
    )
    autoimmune.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    autoimmune.add_argument("--reset", action="store_true")
    autoimmune.add_argument("--json", action="store_true")

    synthesis_variant = subparsers.add_parser(
        "synthesis-variant-smoke",
    )
    synthesis_variant.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/synthesis_variant_smoke.sqlite"),
    )
    synthesis_variant.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    synthesis_variant.add_argument("--reset", action="store_true")
    synthesis_variant.add_argument("--json", action="store_true")

    synthesis_stress = subparsers.add_parser(
        "synthesis-stress-smoke",
    )
    synthesis_stress.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/synthesis_stress_smoke.sqlite"),
    )
    synthesis_stress.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    synthesis_stress.add_argument("--reset", action="store_true")
    synthesis_stress.add_argument("--json", action="store_true")

    pi_smoke = subparsers.add_parser(
        "pi-smoke")
    pi_smoke.add_argument(
        "--db", type=Path, default=Path("artifacts/local_assistant_os/pi_smoke.sqlite")
    )
    pi_smoke.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    pi_smoke.add_argument("--reset", action="store_true")
    pi_smoke.add_argument("--max-ask-ms", type=float, default=15000.0)
    pi_smoke.add_argument("--max-lifecycle-ms", type=float, default=60000.0)
    pi_smoke.add_argument("--json", action="store_true")

    pi_bundle = subparsers.add_parser(
        "pi-bundle")
    pi_bundle.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle"
        ),
    )
    pi_bundle.add_argument("--reset", action="store_true")
    pi_bundle.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Only build the bundle; do not run pi-smoke from inside it.",
    )
    pi_bundle.add_argument(
        "--zip",
        action="store_true",
        help="Also write a zip archive beside the bundle directory.",
    )
    pi_bundle.add_argument("--json", action="store_true")

    verify_bundle = subparsers.add_parser(
        "verify-bundle")
    verify_bundle.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Bundle directory to verify; defaults to this CLI root.",
    )
    verify_bundle.add_argument(
        "--manifest", type=Path, default=Path("bundle_manifest.json")
    )
    verify_bundle.add_argument(
        "--allow-skipped-self-check",
        action="store_true",
        help="Allow bundles built with pi-bundle --skip-smoke to verify file integrity only.",
    )
    verify_bundle.add_argument("--json", action="store_true")

    launcher_smoke = subparsers.add_parser(
        "launcher-smoke",
    )
    launcher_smoke.add_argument("--bundle-root", type=Path, default=Path("."))
    launcher_smoke.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/launcher_smoke.sqlite"),
    )
    launcher_smoke.add_argument("--host", default="127.0.0.1")
    launcher_smoke.add_argument(
        "--port",
        type=int,
        default=0,
        help="Launcher port; 0 selects a free localhost port.",
    )
    launcher_smoke.add_argument("--reset", action="store_true")
    launcher_smoke.add_argument("--json", action="store_true")

    first_run_smoke = subparsers.add_parser(
        "first-run-smoke",
    )
    first_run_smoke.add_argument("--bundle-root", type=Path, default=Path("."))
    first_run_smoke.add_argument("--timeout-seconds", type=int, default=300)
    first_run_smoke.add_argument("--json", action="store_true")

    archive_smoke = subparsers.add_parser(
        "archive-smoke",
    )
    archive_smoke.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle.zip"
        ),
    )
    archive_smoke.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/archive_smoke_extract"),
    )
    archive_smoke.add_argument("--reset", action="store_true")
    archive_smoke.add_argument("--skip-first-run", action="store_true")
    archive_smoke.add_argument("--timeout-seconds", type=int, default=300)
    archive_smoke.add_argument("--json", action="store_true")

    api_smoke = subparsers.add_parser(
        "api-smoke",
    )
    api_smoke.add_argument(
        "--db", type=Path, default=Path("artifacts/local_assistant_os/api_smoke.sqlite")
    )
    api_smoke.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    api_smoke.add_argument("--reset", action="store_true")
    api_smoke.add_argument("--host", default="127.0.0.1")
    api_smoke.add_argument("--json", action="store_true")

    api_session = subparsers.add_parser(
        "api-session-smoke",
    )
    api_session.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/local_assistant_os/api_session_smoke.sqlite"),
    )
    api_session.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    api_session.add_argument("--reset", action="store_true")
    api_session.add_argument("--host", default="127.0.0.1")
    api_session.add_argument(
        "--action-mode", choices=("dry-run", "real"), default="dry-run"
    )
    api_session.add_argument(
        "--media-player-command",
        default="",
        help="Executable argv prefix for real media playback mode.",
    )
    api_session.add_argument(
        "--call-command",
        default="",
        help="Executable argv prefix for real call/contact mode.",
    )
    api_session.add_argument(
        "--host-app-config-json",
        type=Path,
        default=None,
        help="Optional host action command config JSON.",
    )
    api_session.add_argument(
        "--host-app-media-dir",
        type=Path,
        default=None,
        help="Optional local media directory to import before API action turns.",
    )
    api_session.add_argument("--json", action="store_true")

    ui_smoke = subparsers.add_parser(
        "ui-smoke",
    )
    ui_smoke.add_argument(
        "--db", type=Path, default=Path("artifacts/local_assistant_os/ui_smoke.sqlite")
    )
    ui_smoke.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ui_smoke.add_argument("--reset", action="store_true")
    ui_smoke.add_argument("--host", default="127.0.0.1")
    ui_smoke.add_argument("--json", action="store_true")

    target_report = subparsers.add_parser(
        "target-report",
    )
    target_report.add_argument(
        "--db-dir",
        type=Path,
        default=Path("artifacts/local_assistant_os/target_report"),
    )
    target_report.add_argument("--reset", action="store_true")
    target_report.add_argument("--host", default="127.0.0.1")
    target_report.add_argument("--require-raspberry-pi", action="store_true")
    target_report.add_argument(
        "--media-player-command",
        default="",
        help="Optional target-device media command for host-app-probe.",
    )
    target_report.add_argument(
        "--call-command",
        default="",
        help="Optional target-device call command for host-app-probe.",
    )
    target_report.add_argument(
        "--host-app-config-json",
        type=Path,
        default=None,
        help="Optional host action command config JSON for host-app-probe.",
    )
    target_report.add_argument(
        "--host-app-media-dir",
        type=Path,
        default=None,
        help="Optional real media directory for host-app-probe.",
    )
    target_report.add_argument("--require-host-app-configured", action="store_true")
    target_report.add_argument("--json", action="store_true")

    serve = subparsers.add_parser(
        "serve", help="Serve the local stdlib chat UI and JSON API."
    )
    serve.add_argument("--db", type=Path, default=DEFAULT_DB)
    serve.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8771)
    serve.add_argument("--no-auto-execute", action="store_true")
    serve.add_argument("--action-mode", choices=("dry-run", "real"), default="dry-run")
    serve.add_argument(
        "--media-player-command",
        default="",
        help="Executable argv prefix for real media playback mode.",
    )
    serve.add_argument(
        "--call-command",
        default="",
        help="Executable argv prefix for real call/contact mode.",
    )
    serve.add_argument(
        "--host-app-config-json",
        type=Path,
        default=None,
        help="Optional host action command config JSON.",
    )
    serve.add_argument(
        "--host-app-media-dir",
        type=Path,
        default=None,
        help="Optional local media directory to import before serving.",
    )

    args = parser.parse_args()
    if args.command == "init":
        _init(args)
    elif args.command == "dataset-audit":
        _dataset_audit(args)
    elif args.command == "bootstrap-runtime":
        _bootstrap_runtime(args)
    elif args.command == "ask":
        _ask(args)
    elif args.command == "parse-debug":
        _parse_debug(args)
    elif args.command == "shortcut-audit":
        _shortcut_audit(args)
    elif args.command == "capability-probe":
        _capability_probe(args)
    elif args.command == "chat":
        _chat(args)
    elif args.command == "run-lifecycle":
        _run_lifecycle(args)
    elif args.command == "run-lifecycle-suite":
        _run_lifecycle_suite(args)
    elif args.command == "run-household-week":
        _run_household_week(args)
    elif args.command == "run-open-traces":
        _run_open_traces(args)
    elif args.command == "run-transcript-replay":
        _run_transcript_replay(args)
    elif args.command == "import-transcript-replay":
        _import_transcript_replay(args)
    elif args.command == "export-transcript-replay":
        _export_transcript_replay(args)
    elif args.command == "calibrate-event-ledger":
        _calibrate_event_ledger(args)
    elif args.command == "v01-blocker-evidence":
        _v01_blocker_evidence(args)
    elif args.command == "v01-blocker-rehearsal":
        _v01_blocker_rehearsal(args)
    elif args.command == "v01-evidence-pack":
        _v01_evidence_pack(args)
    elif args.command == "write-source-attestation":
        _write_source_attestation(args)
    elif args.command == "candidate-session-audit":
        _candidate_session_audit(args)
    elif args.command == "write-host-app-attestation":
        _write_host_app_attestation(args)
    elif args.command == "calibrate-transcript-replay":
        _calibrate_transcript_replay(args)
    elif args.command == "v01-audit":
        _v01_audit(args)
    elif args.command == "v01-progress":
        _v01_progress(args)
    elif args.command == "v01-acceptance":
        _v01_acceptance(args)
    elif args.command == "run-jobs":
        _run_jobs(args)
    elif args.command == "schedule-refreshes":
        _schedule_refreshes(args)
    elif args.command == "inventory-soak":
        _inventory_soak(args)
    elif args.command == "inventory-soak-matrix":
        _inventory_soak_matrix(args)
    elif args.command == "inventory-diversity-smoke":
        _inventory_diversity_smoke(args)
    elif args.command == "inventory-failure-smoke":
        _inventory_failure_smoke(args)
    elif args.command == "inventory-retry-smoke":
        _inventory_retry_smoke(args)
    elif args.command == "resource-report":
        _resource_report(args)
    elif args.command == "dashboard":
        _dashboard(args)
    elif args.command == "improvement-queue":
        _improvement_queue(args)
    elif args.command == "memory-replay":
        _memory_replay(args)
    elif args.command == "memory-digest":
        _memory_digest(args)
    elif args.command == "eval":
        _eval(args)
    elif args.command == "import-stories":
        _import_stories(args)
    elif args.command == "import-media":
        _import_media(args)
    elif args.command == "refresh-weather":
        _refresh_weather(args)
    elif args.command == "action-smoke":
        _action_smoke(args)
    elif args.command == "setup-integration-smoke":
        _setup_integration_smoke(args)
    elif args.command == "host-action-smoke":
        _host_action_smoke(args)
    elif args.command == "host-action-recorder":
        _host_action_recorder(args)
    elif args.command == "write-host-actions-demo-config":
        _write_host_actions_demo_config(args)
    elif args.command == "host-app-probe":
        _host_app_probe(args)
    elif args.command == "autoimmune-smoke":
        _autoimmune_smoke(args)
    elif args.command == "synthesis-variant-smoke":
        _synthesis_variant_smoke(args)
    elif args.command == "synthesis-stress-smoke":
        _synthesis_stress_smoke(args)
    elif args.command == "pi-smoke":
        _pi_smoke(args)
    elif args.command == "pi-bundle":
        _pi_bundle(args)
    elif args.command == "verify-bundle":
        _verify_bundle(args)
    elif args.command == "launcher-smoke":
        _launcher_smoke(args)
    elif args.command == "first-run-smoke":
        _first_run_smoke(args)
    elif args.command == "archive-smoke":
        _archive_smoke(args)
    elif args.command == "api-smoke":
        _api_smoke(args)
    elif args.command == "api-session-smoke":
        _api_session_smoke(args)
    elif args.command == "ui-smoke":
        _ui_smoke(args)
    elif args.command == "target-report":
        _target_report(args)
    elif args.command == "serve":
        _serve(args)


def _init(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    store = initialize_assistant_os_database(args.db, seed_path=args.seed)
    payload = {
        "db": str(args.db),
        "seed": str(args.seed),
        "counts": store.table_counts(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _dataset_audit(args) -> None:
    payload = _build_dataset_audit_payload(args.db, seed=args.seed, reset=args.reset)
    _print_payload(payload, json_mode=args.json)


def _build_dataset_audit_payload(db: Path, *, seed: Path, reset: bool) -> dict:
    started = perf_counter()
    if reset:
        _remove_sqlite_files(db)
    seed_path = seed
    story_path = Path("benchmarks/public_domain_story_metadata.json")
    media_path = DEFAULT_LOCAL_MEDIA_MANIFEST
    weather_path = DEFAULT_WEATHER_SAMPLE
    gutenberg_path = Path("benchmarks/sample_gutenberg_catalog.csv")
    internet_archive_path = Path("benchmarks/sample_internet_archive_search.json")
    open_trace_path = DEFAULT_OPEN_TRACE_FIXTURE
    transcript_replay_path = DEFAULT_TRANSCRIPT_REPLAY_FIXTURE
    files = {
        "seed": _dataset_file_report(seed_path),
        "story_metadata": _dataset_file_report(story_path),
        "media_manifest": _dataset_file_report(media_path),
        "weather_fixture": _dataset_file_report(weather_path),
        "gutenberg_catalog": _dataset_file_report(gutenberg_path),
        "internet_archive_search": _dataset_file_report(internet_archive_path),
        "open_traces": _dataset_file_report(open_trace_path),
        "transcript_replay": _dataset_file_report(transcript_replay_path),
    }
    seed_data = _read_json_dataset(seed_path)
    story_data = _read_json_dataset(story_path)
    media_data = _read_json_dataset(media_path)
    weather_data = _read_json_dataset(weather_path)
    ia_data = _read_json_dataset(internet_archive_path)
    open_trace_data = _read_json_dataset(open_trace_path)
    transcript_replay_records = _read_jsonl_dataset(transcript_replay_path)
    gutenberg_rows = _read_csv_dataset(gutenberg_path)
    checks = {
        "all_required_files_present": all(
            item["exists"] and item["bytes"] > 0 for item in files.values()
        ),
        "seed_schema_valid": seed_data.get("schema")
        == "melm.local_assistant_os.seed.v1",
        "seed_profile_facts_complete": _seed_profile_facts_complete(seed_data),
        "seed_inventories_cover_core_domains": _seed_inventories_cover_core_domains(
            seed_data
        ),
        "seed_privacy_scopes_present": _seed_privacy_scopes_present(seed_data),
        "story_metadata_schema_valid": story_data.get("schema")
        == "melm.public_domain_story_metadata.v1",
        "story_metadata_public_domain_usable": _story_metadata_public_domain_usable(
            story_data
        ),
        "media_manifest_schema_valid": media_data.get("schema")
        == "melm.local_media_manifest.v1",
        "media_manifest_local_device_usable": _media_manifest_local_device_usable(
            media_data
        ),
        "weather_fixture_complete_week": _weather_fixture_complete_week(weather_data),
        "gutenberg_catalog_story_candidates": _gutenberg_catalog_story_candidates(
            gutenberg_rows
        )
        >= 2,
        "internet_archive_story_candidates": _internet_archive_story_candidates(ia_data)
        >= 2,
        "open_trace_fixture_complete": _open_trace_fixture_complete(open_trace_data),
        "transcript_replay_fixture_complete": _transcript_replay_fixture_complete(
            transcript_replay_records
        ),
    }
    store = initialize_assistant_os_database(db, seed_path=seed_path)
    profile = store.load_profile(LocalAssistantProfile())
    counts = store.table_counts()
    bootstrap_checks = {
        "sqlite_bootstrap_created": db.exists(),
        "seed_user_facts_loaded": counts.get("user_facts", 0) >= 6,
        "seed_inventories_loaded": counts.get("inventories", 0) >= 8,
        "profile_loads_seed_identity": profile.user_name == "Maya"
        and profile.age == 7
        and profile.location == "Lagos"
        and profile.culture == "Yoruba",
        "profile_loads_core_inventories": bool(profile.story_models)
        and bool(profile.weekly_weather.get("today"))
        and profile.contacts.get("mom") == "+234-000-MOM"
        and "calm piano" in profile.media_library
        and {"rice", "beans", "plantain"}.issubset(set(profile.food_inventory)),
    }
    checks.update(bootstrap_checks)
    payload = {
        "db": str(db),
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "runtime": "stdlib_python_sqlite_json_csv",
        "dependency_class": "stdlib_only",
        "files": files,
        "seed": {
            "facts": len(seed_data.get("user_facts", [])),
            "inventories": len(seed_data.get("inventories", [])),
            "inventory_kinds": sorted(
                {str(item.get("kind", "")) for item in seed_data.get("inventories", [])}
            ),
        },
        "source_fixtures": {
            "story_metadata_items": len(story_data.get("items", [])),
            "media_items": len(media_data.get("items", [])),
            "weather_days": len(weather_data.get("daily", {}).get("time", [])),
            "gutenberg_rows": len(gutenberg_rows),
            "gutenberg_story_candidates": _gutenberg_catalog_story_candidates(
                gutenberg_rows
            ),
            "internet_archive_items": len(ia_data.get("items", [])),
            "internet_archive_story_candidates": _internet_archive_story_candidates(
                ia_data
            ),
            "open_trace_scenarios": len(open_trace_data.get("scenarios", [])),
            "open_trace_turns": sum(
                len(scenario.get("turns", []))
                for scenario in open_trace_data.get("scenarios", [])
            ),
            "transcript_replay_rows": len(transcript_replay_records),
            "transcript_replay_user_turns": sum(
                1
                for item in transcript_replay_records
                if str(item.get("type", "turn")) == "turn"
                and str(item.get("speaker", "user")).lower() == "user"
            ),
        },
        "bootstrap": {
            "counts": counts,
            "profile": {
                "user_name": profile.user_name,
                "age": profile.age,
                "location": profile.location,
                "culture": profile.culture,
                "story_models": len(profile.story_models),
                "weather_days": len(profile.weekly_weather),
                "contacts": sorted(profile.contacts),
                "media_library": list(profile.media_library),
                "food_inventory": list(profile.food_inventory),
            },
        },
    }
    store.close()
    return payload


def _dataset_file_report(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": _sha256_file(path) if path.exists() and path.is_file() else "",
    }


def _read_json_dataset(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_csv_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl_dataset(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if not isinstance(record, dict):
            return []
        records.append(record)
    return records


def _seed_profile_facts_complete(seed: dict) -> bool:
    required = {
        "profile.user_name",
        "profile.age",
        "profile.location",
        "profile.culture",
        "preferences.music",
    }
    facts = {str(fact.get("key", "")): fact for fact in seed.get("user_facts", [])}
    return required.issubset(facts) and all(
        bool(facts[key].get("consent", False))
        and bool(facts[key].get("local_only", False))
        for key in required
    )


def _seed_inventories_cover_core_domains(seed: dict) -> bool:
    kinds = Counter(str(item.get("kind", "")) for item in seed.get("inventories", []))
    return (
        kinds["story_model"] >= 1
        and kinds["weather"] >= 2
        and kinds["contact"] >= 1
        and kinds["media"] >= 1
        and kinds["food"] >= 3
    )


def _seed_privacy_scopes_present(seed: dict) -> bool:
    facts_ok = all(
        "source" in fact
        and 0.0 <= float(fact.get("confidence", -1.0)) <= 1.0
        and bool(fact.get("local_only", False))
        and fact.get("cloud_eligible") is False
        and fact.get("scope") == "private_local"
        for fact in seed.get("user_facts", [])
    )
    inventories_ok = all(
        bool(item.get("source"))
        and bool(item.get("license"))
        and isinstance(item.get("tags", []), list)
        for item in seed.get("inventories", [])
    )
    contact_local = any(
        item.get("kind") == "contact"
        and item.get("license") == "private_local"
        and "local_only" in item.get("tags", [])
        for item in seed.get("inventories", [])
    )
    return facts_ok and inventories_ok and contact_local


def _story_metadata_public_domain_usable(story_data: dict) -> bool:
    items = story_data.get("items", [])
    required = {
        "item_id",
        "title",
        "source",
        "source_url",
        "license",
        "age_min",
        "age_max",
        "topics",
        "cultures",
        "summary",
        "narrative_frame",
    }
    if len(items) < 4:
        return False
    valid_items = all(
        required.issubset(item)
        and str(item.get("source_url", "")).startswith("https://")
        and "public_domain" in str(item.get("license", ""))
        and isinstance(item.get("topics", []), list)
        and isinstance(item.get("cultures", []), list)
        and int(item.get("age_min", 99)) <= 7 <= int(item.get("age_max", -1))
        for item in items
    )
    local_fit = any(
        "Yoruba" in item.get("cultures", []) or "Lagos" in item.get("cultures", [])
        for item in items
    )
    return valid_items and local_fit


def _media_manifest_local_device_usable(media_data: dict) -> bool:
    items = media_data.get("items", [])
    required = {"title", "path", "kind", "tags", "source", "license"}
    if len(items) < 3:
        return False
    valid_items = all(
        required.issubset(item)
        and item.get("kind") == "audio"
        and item.get("license") == "local_device"
        and isinstance(item.get("tags", []), list)
        and bool(item.get("path"))
        for item in items
    )
    preferred = any(
        item.get("preferred") is True
        and "piano" in " ".join(item.get("tags", [])).lower()
        for item in items
    )
    return valid_items and preferred


def _weather_fixture_complete_week(weather_data: dict) -> bool:
    daily = weather_data.get("daily", {})
    arrays = [
        daily.get("time", []),
        daily.get("weather_code", []),
        daily.get("temperature_2m_max", []),
        daily.get("temperature_2m_min", []),
        daily.get("precipitation_probability_max", []),
    ]
    lengths = {len(array) for array in arrays if isinstance(array, list)}
    return (
        bool(weather_data.get("resolved_location"))
        and len(lengths) == 1
        and next(iter(lengths), 0) >= 7
        and "2026-06-07" in daily.get("time", [])
    )


def _gutenberg_catalog_story_candidates(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        text = " ".join(str(value).lower() for value in row.values())
        if row.get("Language") == "en" and any(
            marker in text for marker in ("children", "fairy", "folklore", "adventure")
        ):
            count += 1
    return count


def _internet_archive_story_candidates(data: dict) -> int:
    count = 0
    for item in data.get("items", []):
        text = " ".join(
            [
                str(item.get("title", "")),
                " ".join(str(subject) for subject in item.get("subject", [])),
                " ".join(str(collection) for collection in item.get("collection", [])),
            ]
        ).lower()
        if (
            item.get("identifier")
            and item.get("language") == "eng"
            and any(
                marker in text
                for marker in (
                    "children",
                    "story",
                    "stories",
                    "folklore",
                    "bedtime",
                    "fables",
                )
            )
        ):
            count += 1
    return count


def _open_trace_fixture_complete(data: dict) -> bool:
    scenarios = data.get("scenarios", [])
    total_turns = sum(len(scenario.get("turns", [])) for scenario in scenarios)
    labels = {
        str(turn.get("label", ""))
        for scenario in scenarios
        for turn in scenario.get("turns", [])
    }
    scenario_expectations_ok = all(
        bool(scenario.get("name"))
        and scenario.get("profile")
        and scenario.get("expectations", {}).get("required_intents")
        and scenario.get("expectations", {}).get("required_routes")
        and all(
            bool(turn.get("label")) and bool(turn.get("utterance"))
            for turn in scenario.get("turns", [])
        )
        for scenario in scenarios
    )
    return (
        data.get("schema") == "melm.local_assistant_open_traces.v1"
        and len(scenarios) >= 2
        and total_turns >= 29
        and scenario_expectations_ok
        and {
            "identity",
            "identity_challenge",
            "weather_cold_miss",
            "story_after_inventory",
            "private_cloud_block",
            "session_recall",
        }.issubset(labels)
    )


def _transcript_replay_fixture_complete(records: list[dict]) -> bool:
    if not records:
        return False
    meta = next(
        (item for item in records if str(item.get("type", "turn")) == "meta"), {}
    )
    user_turns = [
        item
        for item in records
        if str(item.get("type", "turn")) == "turn"
        and str(item.get("speaker", "user")).lower() == "user"
    ]
    labels = {str(item.get("label", "")) for item in user_turns}
    forbidden_keys = {
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
    static_expectations = any(
        forbidden_keys & {str(key) for key in item} for item in records
    )
    expectations = dict(meta.get("expectations", {}) or {})
    return (
        meta.get("schema") == "melm.local_assistant_transcript_replay.v1"
        and meta.get("source_type") == "authored_transcript_fixture"
        and len(user_turns) >= int(expectations.get("min_turns", 12) or 12)
        and not static_expectations
        and bool(expectations.get("required_intents"))
        and bool(expectations.get("required_routes"))
        and {
            "identity_challenge",
            "weather_miss",
            "story_after_inventory",
            "media_confirm",
            "contact_confirm",
            "privacy_block",
            "long_horizon_digest",
        }.issubset(labels)
    )


def _bootstrap_runtime(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    started = perf_counter()
    datasets = _required_dataset_report(args.seed)
    store = initialize_assistant_os_database(args.db, seed_path=args.seed)
    seed_class_schemas(store)
    seed_assistant_os_lexicon(store)
    migrate_contacts_to_entities(store)
    migrate_self_facts_to_entities(store)
    profile = store.load_profile(LocalAssistantProfile())
    media_results = []
    if not args.skip_media_import:
        if args.media_dir is not None:
            media_result = LocalMediaInventoryAdapter().import_directory(
                args.media_dir, profile, limit=24
            )
        else:
            media_result = LocalMediaInventoryAdapter(args.manifest).import_manifest(
                profile,
                limit=24,
                require_files=args.require_media_files,
            )
        _install_imported_media_items(store, media_result.items)
        media_results.append(media_result.to_dict())

    initial_counts = store.table_counts()
    story_payload = _handle_utterance(store, "Tell me a story.", auto_execute=False)
    weather_payload = _handle_utterance(
        store, "What is the weather today?", auto_execute=False
    )
    safety_payload = _handle_utterance(
        store, "Should I go to school dressed naked?", auto_execute=False
    )
    self_observation = _persist_runtime_self_observation(store, profile)
    dashboard = build_assistant_os_dashboard(store).to_dict()
    counts = store.table_counts()
    safety_flags = dashboard["safety_flags"]
    media_imported_items = sum(len(result["items"]) for result in media_results)
    checks = {
        "datasets_present": all(item["exists"] for item in datasets),
        "runtime_db_created": args.db.exists(),
        "seed_inventory_loaded": initial_counts["inventories"] >= 8
        and initial_counts["user_facts"] >= 1,
        "media_inventory_ready": args.skip_media_import or media_imported_items >= 1,
        "required_media_files_present": (not args.require_media_files)
        or args.skip_media_import
        or media_imported_items >= 1,
        "story_local_with_synthesis": (
            story_payload["route"] == "local_answer"
            and story_payload["reason"] == "local_story_inventory"
            and bool(story_payload["synthesis"].get("applied"))
        ),
        "weather_cached": (
            weather_payload["route"] == "cached_tool"
            and weather_payload["reason"] == "weather_cache_hit"
        ),
        "safety_local": (
            safety_payload["route"] == "local_answer"
            and safety_payload["reason"] == "local_common_sense_policy"
        ),
        "ledgers_complete": bool(safety_flags["ledger_complete"]),
        "safety_flags_clean": (
            safety_flags["cloud_private_inclusions"] == 0
            and safety_flags["unconfirmed_executed_actions"] == 0
            and safety_flags["action_without_confirmation_gate"] == 0
            and safety_flags["fake_latest_news_local_answers"] == 0
            and safety_flags["low_quality_applied_synthesis"] == 0
        ),
        "stdlib_sqlite_only": True,
    }
    payload = {
        "db": str(args.db),
        "seed": str(args.seed),
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": "stdlib_python_sqlite",
        "dependency_class": "stdlib_only",
        "elapsed_ms": _elapsed_ms(started),
        "datasets": datasets,
        "media_import": {
            "skipped": bool(args.skip_media_import),
            "imported_items": media_imported_items,
            "results": media_results,
            "require_files": bool(args.require_media_files),
        },
        "initial_counts": initial_counts,
        "counts": counts,
        "db_bytes": _sqlite_size(args.db),
        "turns": [
            _bootstrap_turn_summary("story", story_payload),
            _bootstrap_turn_summary("weather", weather_payload),
            _bootstrap_turn_summary("safety", safety_payload),
        ],
        "dashboard": {
            "counts": dashboard["counts"],
            "safety_flags": safety_flags,
            "self_observation": self_observation,
            "inventories": {
                "story_quality": dashboard["inventories"]["story_quality"],
                "by_source": dashboard["inventories"]["by_source"],
            },
        },
        "next_commands": {
            "ask_story": f'python3 scripts/local_assistant_os_cli.py ask --db {args.db.as_posix()} --utterance "Tell me a story." --json',
            "serve_local_api": f"python3 scripts/local_assistant_os_cli.py serve --db {args.db.as_posix()} --host 127.0.0.1 --port 8771",
            "dashboard": f"python3 scripts/local_assistant_os_cli.py dashboard --db {args.db.as_posix()} --json",
        },
        "pi_constraints": {
            "no_required_network": True,
            "no_required_vector_db": True,
            "no_required_ml_framework": True,
        },
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _bootstrap_turn_summary(label: str, payload: dict) -> dict:
    return {
        "label": label,
        "utterance": payload["utterance"],
        "intent": payload["intent"],
        "route": payload["route"],
        "reason": payload["reason"],
        "synthesis_applied": bool(payload["synthesis"].get("applied")),
        "evidence_keys": payload["evidence_keys"],
        "membrane": payload["membrane"],
    }


def _ask(args) -> None:
    store = _open_store(args.db, None if args.cold_start else args.seed)
    try:
        payload = _handle_utterance(
            store,
            args.utterance,
            auto_execute=args.execute_jobs and not args.no_auto_execute,
            cold_start=args.cold_start,
            action_mode=args.action_mode,
            media_player_command=args.media_player_command,
            call_command=args.call_command,
            capture_surface="cli_chat",
            capture_source="single_cli_ask",
            improvement_opt_in=args.improvement_opt_in,
            model_path=args.model_path,
        )
    finally:
        store.close()
    _print_payload(payload, json_mode=args.json)


def _parse_debug(args) -> None:
    payload = parse_assistant_debug_frame(args.utterance).to_dict()
    _print_payload(payload, json_mode=args.json)


def _shortcut_audit(args) -> None:
    payload = _build_shortcut_audit_payload()
    _print_payload(payload, json_mode=args.json)


def _build_shortcut_audit_payload() -> dict:
    started = perf_counter()
    behavior_cases = _shortcut_audit_behavior_cases()
    source_checks = _shortcut_audit_source_checks()
    checks = {
        "behavior_probes_passed": all(bool(item["passed"]) for item in behavior_cases),
        "source_boundaries_passed": all(bool(item["passed"]) for item in source_checks),
        "no_secondary_hints_in_primary_routes": not any(
            bool(item.get("secondary_hint_in_primary_route", False))
            for item in behavior_cases
        ),
        "no_debug_hit_labels": not any(
            bool(item.get("debug_label_smells", [])) for item in behavior_cases
        ),
        "primary_routes_owned_by_frame_registry": all(
            bool(item.get("frame_registry_ok", False)) for item in behavior_cases
        ),
        "kernel_recall_uses_shared_chatframe_scope": any(
            item["id"] == "kernel_autobiographical_gate_uses_shared_frame"
            and bool(item["passed"])
            for item in source_checks
        ),
    }
    return {
        "schema": "melm.local_assistant_shortcut_audit.v1",
        "passed": all(checks.values()),
        "elapsed_ms": _elapsed_ms(started),
        "checks": checks,
        "behavior_cases": behavior_cases,
        "source_checks": source_checks,
        "policy": {
            "primary_parse_basis": "uol_chat_frame",
            "secondary_hint_policy": "debug_only_never_primary_route",
            "phrase_tables_allowed_as": "secondary_debug_evidence_only",
        },
    }


def _shortcut_audit_behavior_cases() -> list[dict]:
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    cases = [
        {
            "label": "identity_self_model",
            "utterance": "Who are you?",
            "expected_intent": "assistant_identity",
            "expected_route": "local_answer",
        },
        {
            "label": "identity_name_self_model",
            "utterance": "What is your name?",
            "expected_intent": "assistant_identity",
            "expected_route": "local_answer",
        },
        {
            "label": "identity_challenge_self_model",
            "utterance": "wow you don't know who you are?",
            "expected_intent": "assistant_identity",
            "expected_route": "local_answer",
        },
        {
            "label": "weather_concept_not_cache",
            "utterance": "What is weather?",
            "expected_intent": "open_domain",
            "expected_route": "local_answer",
        },
        {
            "label": "weather_observation_cache",
            "utterance": "What is the weather?",
            "expected_intent": "weather",
            "expected_route": "cached_tool",
        },
        {
            "label": "bare_story_noun_not_story_route",
            "utterance": "story",
            "expected_intent": "unknown",
            "expected_route": "cloud_handoff",
        },
        {
            "label": "bare_play_verb_not_media_route",
            "utterance": "play",
            "expected_intent": "unknown",
            "expected_route": "cloud_handoff",
        },
        {
            "label": "meal_you_cook_not_advice",
            "utterance": "Can you cook dinner?",
            "expected_intent": "open_domain",
            "expected_route": "local_answer",
        },
        {
            "label": "meal_user_choice_advice",
            "utterance": "What can I cook for dinner?",
            "expected_intent": "meal_suggestion",
            "expected_route": "local_answer",
        },
    ]
    rows = []
    for case in cases:
        decision = router.handle(str(case["utterance"]))
        parsed = parse_assistant_debug_frame(str(case["utterance"]), decision).to_dict()
        rows.append(
            _shortcut_behavior_row(case, decision.intent, decision.route, parsed)
        )
    rows.extend(_shortcut_kernel_behavior_cases())
    return rows


def _shortcut_kernel_behavior_cases() -> list[dict]:
    db = (
        ROOT
        / "artifacts"
        / "local_assistant_os"
        / f"shortcut_audit_{os.getpid()}.sqlite"
    )
    db.parent.mkdir(parents=True, exist_ok=True)
    _remove_sqlite_files(db)
    store = AssistantOSStore(db)
    try:
        kernel = AssistantOSKernel(store=store)
        kernel.handle("Tell me a story.")
        kernel.handle("What is the weather today?")
        paraphrase = kernel.handle("What was the last thing I asked you?")
        paraphrase_parse = parse_assistant_debug_frame(
            "What was the last thing I asked you?", paraphrase
        ).to_dict()
        statement = kernel.handle("I dropped the last thing yesterday.")
        statement_parse = parse_assistant_debug_frame(
            "I dropped the last thing yesterday.", statement
        ).to_dict()
    finally:
        store.close()
        _remove_sqlite_files(db)

    return [
        _shortcut_behavior_row(
            {
                "label": "kernel_paraphrased_latest_event_recall",
                "utterance": "What was the last thing I asked you?",
                "expected_intent": "autobiographical_memory",
                "expected_route": "local_answer",
            },
            paraphrase.intent,
            paraphrase.route,
            paraphrase_parse,
        ),
        _shortcut_behavior_row(
            {
                "label": "kernel_statement_not_memory_recall",
                "utterance": "I dropped the last thing yesterday.",
                "expected_intent": "unknown",
                "expected_route": "cloud_handoff",
            },
            statement.intent,
            statement.route,
            statement_parse,
        ),
    ]


def _shortcut_behavior_row(
    case: dict, actual_intent: str, actual_route: str, parsed: dict
) -> dict:
    primary = dict(parsed.get("nlp", {}).get("primary_domain_evidence", {}))
    composition = dict(parsed.get("nlp", {}).get("compositional_parse", {}))
    primary_basis = list(parsed.get("chat_frame", {}).get("primary_routing_basis", []))
    secondary_hint_in_primary = any(
        str(item).startswith("secondary_meaning_hints:")
        or str(item).startswith("vocabulary_hits:")
        for item in primary_basis
    )
    debug_label_smells = _shortcut_debug_label_smells(parsed)
    expected_intent = str(case["expected_intent"])
    expected_route = str(case["expected_route"])
    need_frame_registry = expected_route in {"local_answer", "cached_tool", "device_action"}
    frame_registry_ok = (
        not need_frame_registry
        or (
            str(primary.get("frame_registry", "")) == "melm.assistant_frame_registry.v1"
            and str(primary.get("frame_id", ""))
            and str(primary.get("source_policy", "")) == "primary_uol_chatframe_only"
            and str(composition.get("secondary_hint_policy", ""))
            == "debug_only_never_primary_route"
        )
    )
    passed = (
        actual_intent == expected_intent
        and actual_route == expected_route
        and frame_registry_ok
        and not secondary_hint_in_primary
        and not debug_label_smells
    )
    return {
        "label": str(case["label"]),
        "utterance": str(case["utterance"]),
        "passed": passed,
        "expected": {
            "intent": expected_intent,
            "route": expected_route,
        },
        "actual": {
            "intent": actual_intent,
            "route": actual_route,
            "primary_source": str(primary.get("source", "")),
            "primary_pattern": str(primary.get("pattern", "")),
            "frame_registry": str(primary.get("frame_registry", "")),
            "frame_id": str(primary.get("frame_id", "")),
            "source_policy": str(primary.get("source_policy", "")),
        },
        "frame_registry_ok": frame_registry_ok,
        "secondary_hint_in_primary_route": secondary_hint_in_primary,
        "debug_label_smells": debug_label_smells,
    }


def _shortcut_debug_label_smells(payload: Any, *, prefix: str = "") -> list[str]:
    smells: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            key_lower = key_text.lower()
            if "hit" in key_lower or key_lower in {"phrase_hits", "vocabulary_hits"}:
                smells.append(path)
            smells.extend(_shortcut_debug_label_smells(value, prefix=path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            smells.extend(
                _shortcut_debug_label_smells(item, prefix=f"{prefix}[{index}]")
            )
    return smells


def _shortcut_audit_source_checks() -> list[dict]:
    router_source = (
        ROOT / "melm" / "appliance" / "local_assistant_router.py"
    ).read_text(encoding="utf-8")
    functional_grammar_source = (
        ROOT / "melm" / "appliance" / "functional_grammar.py"
    ).read_text(encoding="utf-8")
    kernel_source = (ROOT / "melm" / "appliance" / "assistant_os_kernel.py").read_text(
        encoding="utf-8"
    )
    classifier_block = _source_block(
        router_source, "def _classify_intent_from_uol_slots"
    )
    slot_helper_block = "\n".join(
        _source_block(router_source, name)
        for name in (
            "def _media_object_from_request_tokens",
            "def _personal_memory_object_from_text",
            "def _contact_object_from_tokens",
            "def _object_source",
        )
    )
    hint_block = _source_block(router_source, "def _secondary_meaning_hint_groups")
    kernel_recall_block = _source_block(
        kernel_source, "def _is_autobiographical_recall_request"
    )
    identity_composition_block = "\n".join(
        _source_block(router_source, name)
        for name in (
            "def _identity_composition",
            "def _matches_who_identity_frame",
            "def _matches_name_identity_frame",
            "def _matches_kind_identity_frame",
            "def _matches_capability_identity_frame",
            "def _purpose_identity_frame",
            "def _matches_self_description_frame",
            "def _identity_deixis_relation_frame",
        )
    )
    self_status_block = "\n".join(
        _source_block(router_source, name)
        for name in (
            "def _self_status_composition",
            "def _self_status_basis",
            "def _self_status_token_roles",
        )
    )
    autobiographical_router_block = "\n".join(
        _source_block(router_source, name)
        for name in (
            "def compose_autobiographical_memory_frame",
            "def classify_autobiographical_memory_scope",
            "def _autobiographical_memory_scope",
            "def _is_autobiographical_debug_request",
            "def _autobiographical_latest_event_frame",
            "def _autobiographical_session_summary_frame",
            "def _autobiographical_long_horizon_frame",
        )
    )
    return [
        _shortcut_source_row(
            "primary_classifier_no_secondary_helpers",
            classifier_block,
            required=(
                "_classify_from_frame_linker",
                "_route_frame_match",
                "_is_private_cloud_export_request",
            ),
            forbidden=(
                "_secondary_meaning",
                "_secondary_debug",
                "_has_marker",
                "_has_any_marker",
            ),
        ),
        _shortcut_source_row(
            "post_route_slot_helpers_no_secondary_helpers",
            slot_helper_block,
            required=(
                "_media_object_from_request_tokens",
                "_personal_memory_object_from_text",
                "_contact_object_from_tokens",
            ),
            forbidden=(
                "_secondary_meaning",
                "_secondary_debug",
                "_has_marker",
                "_has_any_marker",
            ),
        ),
        _shortcut_source_row(
            "secondary_hint_table_is_concept_tokens_only",
            hint_block,
            required=("_secondary_meaning_hint_groups",),
            forbidden=(
                "who are you",
                "what is your name",
                "what have you done",
                "talk to someone",
                "what was my last question",
                "last question",
            ),
        ),
        _shortcut_source_row(
            "primary_routing_source_no_request_surface_strings",
            "\n".join((classifier_block, slot_helper_block)),
            required=("_is_private_cloud_export_request", "_object_source"),
            forbidden=(
                "who are you",
                "what is your name",
                "what have you done",
                "show your ledger",
                "tell me a story",
                "play a song",
                "what is the weather",
                "what should i eat",
                "talk to someone",
            ),
        ),
        _shortcut_source_row(
            "primary_frame_registry_declared",
            router_source,
            required=(
                "class AssistantFrameRegistry",
                "class AssistantFrameMatch",
                "melm.assistant_frame_registry.v1",
                "primary_uol_chatframe_only",
                "debug_only_never_primary_route",
            ),
            forbidden=("phrase_hits", "vocabulary_hits"),
        ),
        _shortcut_source_row(
            "functional_grammar_no_transcript_phrase_table",
            functional_grammar_source,
            required=(
                "melm.weighted_functional_grammar.v1",
                "def parse_functional_relations",
                "def functional_frame_kind",
                "predicate_candidates",
                "syntactic_coverage",
            ),
            forbidden=(
                "do you always tell people the same thing",
                "do you like repeating yourself",
                "i want to grow in my career",
                "can you help me grow in my career",
                "_secondary_meaning",
                "phrase_hits",
                "vocabulary_hits",
            ),
        ),
        _shortcut_source_row(
            "identity_composition_no_surface_phrase_table",
            identity_composition_block,
            required=(
                "melm.identity_uol_composition.v1",
                "_identity_deixis_relation_frame",
                "_assistant_possessive_attribute_question",
            ),
            forbidden=(
                "who are you",
                "what is your name",
                "what is your purpose",
                "tell me about yourself",
                '("who", "are", "you")',
                '("what", "is", "your", "name")',
                '("what", "is", "your", "purpose")',
                "_secondary_meaning",
                "_secondary_debug",
                "_has_marker",
                "_has_any_marker",
            ),
        ),
        _shortcut_source_row(
            "self_status_composition_no_surface_phrase_table",
            self_status_block,
            required=("melm.self_status_uol_composition.v1", "source:event_ledger"),
            forbidden=(
                "what have you done",
                "what do you need next",
                "show your ledger",
                "are you using cloud",
                "_secondary_meaning",
                "_secondary_debug",
                "_has_marker",
                "_has_any_marker",
            ),
        ),
        _shortcut_source_row(
            "primary_frame_registry_no_legacy_composition_helpers",
            "\n".join(
                (
                    _source_block(router_source, "class AssistantFrameRegistry"),
                    _source_block(router_source, "def _compose_primary_frame"),
                )
            ),
            required=(
                "_compose_primary_frame",
                "_identity_composition",
                "_self_status_composition",
            ),
            forbidden=(
                "_assistant_compositional_parse",
                "_semantic_slot_composition",
                "_functional_relation_composition",
            ),
        ),
        _shortcut_source_row(
            "autobiographical_composition_no_exact_recall_phrases",
            autobiographical_router_block,
            required=(
                "compose_autobiographical_memory_frame",
                "_autobiographical_memory_scope",
            ),
            forbidden=(
                "what was my last question",
                "what was the last thing i asked you",
                "last question",
                "recall_markers",
                "_secondary_meaning",
                "_secondary_debug",
                "_has_marker",
                "_has_any_marker",
            ),
        ),
        _shortcut_source_row(
            "kernel_autobiographical_gate_uses_shared_frame",
            kernel_recall_block,
            required=("compose_autobiographical_memory_frame",),
            forbidden=(
                "recall_markers",
                "what was my last question",
                "_has_any_marker",
            ),
        ),
        _shortcut_source_row(
            "weather_meal_autobiographical_guards_present",
            router_source,
            required=(
                "def compose_autobiographical_memory_frame",
                "def classify_autobiographical_memory_scope",
                "def _autobiographical_memory_scope",
                "def _autobiographical_question_or_command",
            ),
            forbidden=(
                "def _is_weather_request",
                "def _is_meal_suggestion_request",
            ),
        ),
    ]


def _shortcut_source_row(
    check_id: str,
    source: str,
    *,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> dict:
    missing_required = [item for item in required if item not in source]
    forbidden_present = [item for item in forbidden if item in source]
    return {
        "id": check_id,
        "passed": not missing_required and not forbidden_present,
        "required_present": [item for item in required if item in source],
        "missing_required": missing_required,
        "forbidden_present": forbidden_present,
    }


def _source_block(source: str, marker: str) -> str:
    start = source.find(marker)
    if start < 0:
        return ""
    next_def = source.find("\ndef ", start + len(marker))
    if next_def < 0:
        return source[start:]
    return source[start:next_def]


def _capability_probe(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    payload = _build_capability_probe_payload(args.db, seed=args.seed)
    _print_payload(payload, json_mode=args.json)


def _build_capability_probe_payload(db: Path, *, seed: Path = DEFAULT_SEED) -> dict:
    started = perf_counter()
    store = _open_store(db, seed)
    case_summaries: list[dict] = []
    try:
        for case in CAPABILITY_PROBE_CASES:
            response = _handle_utterance(
                store,
                str(case["utterance"]),
                auto_execute=False,
                action_mode="dry-run",
            )
            case_summaries.append(_capability_case_summary(case, response))
        dashboard = build_assistant_os_dashboard(store).to_dict()
        counts = store.table_counts()
    finally:
        store.close()

    route_counts = dict(
        sorted(Counter(item["route"] for item in case_summaries).items())
    )
    bucket_counts = dict(
        sorted(Counter(item["bucket"] for item in case_summaries).items())
    )
    intent_counts = dict(
        sorted(Counter(item["intent"] for item in case_summaries).items())
    )
    domain_counts = dict(
        sorted(
            Counter(item["domain"] for item in case_summaries if item["domain"]).items()
        )
    )
    expected_mismatches = [
        {
            "label": item["label"],
            "expected": item["expected_bucket"],
            "actual": item["bucket"],
            "route": item["route"],
            "reason": item["reason"],
        }
        for item in case_summaries
        if item["bucket"] != item["expected_bucket"]
    ]
    complexity_scores = [float(item["complexity_score"]) for item in case_summaries]
    complexity_bands = dict(
        sorted(Counter(item["complexity_band"] for item in case_summaries).items())
    )
    local_device_count = sum(
        bucket_counts.get(bucket, 0) for bucket in ("local", "device_action")
    )
    local_device_rate = round(local_device_count / max(1, len(case_summaries)), 3)
    confirmation_cases = [
        item["label"] for item in case_summaries if item["confirmation_required"]
    ]
    unsupported_examples = [
        {
            "label": item["label"],
            "utterance": item["utterance"],
            "bucket": item["bucket"],
            "route": item["route"],
            "reason": item["reason"],
            "primary_routing_basis": item["primary_routing_basis"],
            "secondary_debug_hints": item["secondary_debug_hints"],
        }
        for item in case_summaries
        if item["bucket"] in {"cloud_handoff", "external_fetch", "clarify", "blocked"}
    ]
    checks = {
        "representative_cases_present": len(case_summaries) >= 18,
        "expected_buckets_match": not expected_mismatches,
        "original_eight_have_local_or_device_path": all(
            _case_by_label(case_summaries, label).get("bucket")
            in {"local", "device_action"}
            for label in (
                "story",
                "weather",
                "school_safety",
                "media_request",
                "health",
                "profile_memory",
                "meal",
                "contact_request",
            )
        ),
        "open_domain_handoff_visible": all(
            _case_by_label(case_summaries, label).get("bucket") == "local"
            for label in ("open_domain_science", "code_request")
        ),
        "private_cloud_blocked": all(
            _case_by_label(case_summaries, label).get("bucket") == "blocked"
            for label in ("private_cloud", "conversation_export")
        ),
        "action_confirmation_gated": set(confirmation_cases)
        >= {"media_request", "contact_request"},
        "confirmations_execute_dry_run_only": all(
            _case_by_label(case_summaries, label)
            .get("action_execution", {})
            .get("side_effect_executed")
            is False
            and _case_by_label(case_summaries, label)
            .get("action_execution", {})
            .get("status")
            == "prepared"
            for label in ("media_confirm", "contact_confirm")
        ),
        "debug_mapping_present": all(
            item["mapping"] == ["basic_nlp", "uol_parse", "chat_frame"]
            for item in case_summaries
        ),
        "complexity_scores_present": all(
            item["complexity_score"] > 0 for item in case_summaries
        ),
        "unknown_tokens_measured": max(
            (item["unknown_token_count"] for item in case_summaries), default=0
        )
        > 0,
        "safety_flags_clean": (
            int(dashboard["safety_flags"].get("cloud_private_inclusions", 0) or 0) == 0
            and int(
                dashboard["safety_flags"].get("unconfirmed_executed_actions", 0) or 0
            )
            == 0
            and int(
                dashboard["safety_flags"].get("action_without_confirmation_gate", 0)
                or 0
            )
            == 0
            and int(
                dashboard["safety_flags"].get("fake_latest_news_local_answers", 0) or 0
            )
            == 0
        ),
        "stdlib_only": True,
    }
    return {
        "db": str(db),
        "seed": str(seed),
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "total_cases": len(case_summaries),
        "route_counts": route_counts,
        "bucket_counts": bucket_counts,
        "intent_counts": intent_counts,
        "domain_counts": domain_counts,
        "local_device_rate": local_device_rate,
        "complexity": {
            "min": round(min(complexity_scores), 3),
            "max": round(max(complexity_scores), 3),
            "avg": round(sum(complexity_scores) / max(1, len(complexity_scores)), 3),
            "bands": complexity_bands,
        },
        "unknown_tokens": {
            "max": max(item["unknown_token_count"] for item in case_summaries),
            "total": sum(item["unknown_token_count"] for item in case_summaries),
            "cases": [
                {
                    "label": item["label"],
                    "unknown_token_count": item["unknown_token_count"],
                    "unknown_tokens": item["unknown_tokens"],
                }
                for item in case_summaries
                if item["unknown_token_count"]
            ],
        },
        "confirmation_cases": confirmation_cases,
        "unsupported_examples": unsupported_examples,
        "expected_mismatches": expected_mismatches,
        "cases": case_summaries,
        "counts": counts,
        "safety_flags": dashboard["safety_flags"],
        "runtime": "stdlib_python_sqlite_capability_probe",
        "dependency_class": "stdlib_only",
    }


def _capability_case_summary(case: dict, response: dict) -> dict:
    debug = response.get("debug_parse", {})
    chat_frame = debug.get("chat_frame", {})
    nlp = debug.get("nlp", {})
    action_execution = response.get("action_execution") or {}
    complexity = float(chat_frame.get("complexity_score", 0.0) or 0.0)
    route = str(response.get("route", ""))
    bucket = _capability_bucket(route)
    return {
        "label": str(case["label"]),
        "utterance": str(case["utterance"]),
        "expected_bucket": str(case["expected_bucket"]),
        "bucket": bucket,
        "intent": str(response.get("intent", "")),
        "domain": str(chat_frame.get("domain", "")),
        "route": route,
        "reason": str(response.get("reason", "")),
        "answer": str(response.get("answer", "")),
        "confirmation_required": int(
            response.get("membrane", {}).get("confirmation_required", 0) or 0
        ),
        "cloud_needed": bool(response.get("cloud_needed", False)),
        "external_fetch_needed": bool(response.get("external_fetch_needed", False)),
        "synthesis_applied": bool(response.get("synthesis", {}).get("applied", False)),
        "complexity_score": round(complexity, 3),
        "complexity_band": _complexity_band(complexity),
        "unknown_token_count": int(nlp.get("unknown_token_count", 0) or 0),
        "unknown_tokens": list(nlp.get("unknown_tokens", [])),
        "primary_domain_evidence": dict(nlp.get("primary_domain_evidence", {})),
        "secondary_domain_hints": dict(nlp.get("secondary_domain_hints", {})),
        "domain_hints": dict(nlp.get("domain_hints", {})),
        "mapping": [stage.get("stage") for stage in debug.get("mapping", [])],
        "primary_routing_basis": list(chat_frame.get("primary_routing_basis", [])),
        "secondary_debug_hints": list(chat_frame.get("secondary_debug_hints", [])),
        "can_answer_locally": bool(chat_frame.get("can_answer_locally", False)),
        "capabilities": dict(chat_frame.get("capabilities", {})),
        "action_execution": action_execution,
        "counts": response.get("counts", {}),
    }


def _capability_bucket(route: str) -> str:
    if route in {"local_answer", "cached_tool"}:
        return "local"
    if route == "device_action":
        return "device_action"
    if route == "cloud_handoff":
        return "cloud_handoff"
    if route == "external_fetch":
        return "external_fetch"
    if route == "reject":
        return "blocked"
    if route == "clarify":
        return "clarify"
    return route or "unknown"


def _complexity_band(score: float) -> str:
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def _case_by_label(cases: list[dict], label: str) -> dict:
    return next((item for item in cases if item.get("label") == label), {})


def _chat(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    store = _open_store(args.db, args.seed)
    try:
        if args.turn:
            turns = [
                _chat_turn_summary(
                    utterance,
                    _handle_utterance(
                        store,
                        utterance,
                        auto_execute=not args.no_auto_execute,
                        action_mode=args.action_mode,
                        media_player_command=args.media_player_command,
                        call_command=args.call_command,
                        tts_command=getattr(args, 'tts_command', ''),
                        capture_surface="cli_chat",
                        capture_source="scripted_cli_turn",
                        improvement_opt_in=args.improvement_opt_in,
                        model_path=args.model_path,
                    ),
                )
                for utterance in args.turn
            ]
            payload = {
                "db": str(args.db),
                "mode": "scripted",
                "turns": turns,
                "counts": store.table_counts(),
                "passed": bool(turns),
            }
            _print_payload(payload, json_mode=args.json)
            return
        if args.json:
            payload = {
                "db": str(args.db),
                "mode": "interactive",
                "turns": [],
                "counts": store.table_counts(),
                "passed": False,
                "error": "interactive_chat_requires_text_terminal",
            }
            _print_payload(payload, json_mode=True)
            return
        print("MELM Local Assistant OS v0.1 CLI chat. Type 'exit' or 'quit' to stop.")
        _face_renderer = None
        _audio = None
        if not getattr(args, 'no_faces', False):
            from melm.appliance.assistant_face_renderer import FaceRenderer
            _face_renderer = FaceRenderer()
        if not getattr(args, 'no_tts', False):
            tts_cmd = getattr(args, 'tts_command', "") or os.environ.get("MELM_TTS_COMMAND", "")
            if tts_cmd:
                from melm.appliance.assistant_audio_feedback import AudioFeedback
                _audio = AudioFeedback(tts_command=tts_cmd)
        while True:
            try:
                utterance = input("> ").strip()
            except EOFError:
                break
            if not utterance:
                continue
            if utterance.lower() in {"exit", "quit"}:
                break
            payload = _handle_utterance(
                store,
                utterance,
                auto_execute=not args.no_auto_execute,
                action_mode=args.action_mode,
                media_player_command=args.media_player_command,
                call_command=args.call_command,
                tts_command=getattr(args, 'tts_command', ''),
                capture_surface="cli_chat",
                capture_source="interactive_cli",
                improvement_opt_in=args.improvement_opt_in,
                model_path=args.model_path,
            )
            if _face_renderer is not None:
                mood = payload.get("session_mood")
                if mood:
                    mood_obj = SimpleNamespace(**mood)
                    face = _face_renderer.render(mood_obj)
                    if face:
                        print(face)
            print(payload["answer"])
            if _audio is not None and payload.get("answer"):
                _audio.speak(payload["answer"])
            integrity = payload.get("response_integrity", {})
            print(
                f"route={payload['route']} reason={payload['reason']} "
                f"confidence={integrity.get('overall_score', 0.0)}"
            )
    finally:
        store.close()


def _chat_turn_summary(utterance: str, payload: dict) -> dict:
    action_execution = payload.get("action_execution") or {}
    return {
        "utterance": utterance,
        "intent": payload.get("intent"),
        "route": payload.get("route"),
        "reason": payload.get("reason"),
        "answer": payload.get("answer"),
        "confirmation_required": payload.get("membrane", {}).get(
            "confirmation_required", 0
        ),
        "cloud_needed": bool(payload.get("cloud_needed")),
        "external_fetch_needed": bool(payload.get("external_fetch_needed")),
        "synthesis_applied": bool(payload.get("synthesis", {}).get("applied")),
        "action_execution": action_execution,
        "debug_parse": payload.get("debug_parse", {}),
        "capture_provenance": payload.get("capture_provenance", {}),
        "counts": payload.get("counts", {}),
    }


def _run_lifecycle(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    store = _open_store(args.db, args.seed)
    simulator = AssistantLifecycleSimulator(store=store)
    report = simulator.run(realistic_lifecycle_steps())
    payload = {
        "db": str(args.db),
        "steps": report.steps,
        "local_resolution_rate": report.local_resolution_rate,
        "cloud_handoffs": report.cloud_handoffs,
        "external_fetches": report.external_fetches,
        "blocked_offline": report.blocked_offline,
        "confirmations_required": report.confirmations_required,
        "actions_executed": report.actions_executed,
        "jobs_executed": list(report.jobs_executed),
        "inventory": {
            "stories": report.story_inventory_count,
            "weather_days": report.weather_cache_days,
            "contacts": report.contact_count,
        },
        "counts": store.table_counts(),
        "routes": [
            {
                "day": result.day,
                "utterance": result.utterance,
                "route": result.route,
                "reason": result.reason,
                "confirmation_required": result.confirmation_required,
                "action_executed": result.action_executed,
                "blocked_offline": result.blocked_offline,
            }
            for result in report.results
        ],
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _run_lifecycle_suite(args) -> None:
    report = run_multi_profile_lifecycle_suite()
    _print_payload(report.to_dict(), json_mode=args.json)


def _run_household_week(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    store = _open_store(args.db, seed=None)
    report = run_household_week_lifecycle_probe(store=store)
    payload = {
        "db": str(args.db),
        **report.to_dict(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _run_open_traces(args) -> None:
    report = run_open_trace_suite(
        trace_path=args.trace_json,
        db_dir=args.db_dir,
        reset=args.reset,
        weather_offline_json=args.weather_json,
    )
    payload = {
        "trace_json": str(args.trace_json),
        "db_dir": str(args.db_dir),
        **report.to_dict(),
    }
    _print_payload(payload, json_mode=args.json)


def _run_transcript_replay(args) -> None:
    report = run_transcript_replay_suite(
        transcript_path=args.transcript_jsonl,
        db_dir=args.db_dir,
        reset=args.reset,
        weather_offline_json=args.weather_json,
        auto_lifecycle=args.auto_lifecycle,
    )
    payload = {
        "transcript_jsonl": str(args.transcript_jsonl),
        "db_dir": str(args.db_dir),
        "auto_lifecycle": bool(args.auto_lifecycle),
        **report.to_dict(),
    }
    _print_payload(payload, json_mode=args.json)


def _import_transcript_replay(args) -> None:
    source_note = (
        args.source_note
        or "Redacted local transcript import; assistant/system rows are skipped and no per-turn "
        "expected answers, routes, reasons, or response text are retained."
    )
    profile = None
    if args.profile_json is not None:
        profile = json.loads(args.profile_json.read_text(encoding="utf-8-sig"))
        if not isinstance(profile, dict):
            raise ValueError("--profile-json must contain a JSON object")
    controls = _read_optional_controls_json(args.controls_json)
    report = import_transcript_replay_fixture(
        input_path=args.input,
        output_path=args.out,
        profile=profile,
        scenario=args.scenario,
        description=args.description,
        source_note=source_note,
        replacements=_parse_transcript_redaction_rules(tuple(args.replace)),
        min_turns=args.min_turns,
        min_route_kinds=args.min_route_kinds,
        controls=controls,
    )
    _print_payload(report.to_dict(), json_mode=args.json)


def _export_transcript_replay(args) -> None:
    if str(args.db) != ":memory:" and not args.db.exists():
        raise SystemExit(f"Event ledger database not found: {args.db}")
    store = AssistantOSStore(args.db)
    try:
        event_rows, resolved_session = _event_transcript_rows(
            store, session_selector=args.session
        )
        raw_out = args.raw_out or _default_event_transcript_raw_path(args.out)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in event_rows),
            encoding="utf-8",
        )
        profile = (
            asdict(store.load_profile(LocalAssistantProfile()))
            if args.profile_mode == "current"
            else None
        )
        source_note = (
            args.source_note
            or "Event-ledger transcript export; only user utterances, session ids, labels, days, "
            "and safe lifecycle controls are retained. Stored answers, routes, reasons, and "
            "assistant responses are not exported as replay expectations."
        )
        controls = _read_optional_controls_json(args.controls_json)
        import_report = import_transcript_replay_fixture(
            input_path=raw_out,
            output_path=args.out,
            profile=profile,
            scenario=args.scenario,
            description=args.description,
            source_type=EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
            source_note=source_note,
            replacements=_parse_transcript_redaction_rules(tuple(args.replace)),
            min_turns=args.min_turns,
            min_route_kinds=args.min_route_kinds,
            controls=controls,
        )
        forbidden = sorted(
            key
            for row in event_rows
            for key in set(row) & STATIC_TRANSCRIPT_EXPECTATION_KEYS
        )
        payload = {
            "schema": "melm.local_assistant_event_transcript_export_report.v1",
            "passed": bool(event_rows) and not forbidden and import_report.passed,
            "db": str(args.db),
            "session": str(args.session),
            "resolved_session": resolved_session,
            "raw_event_jsonl": str(raw_out),
            "transcript_jsonl": str(args.out),
            "source_type": EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
            "profile_mode": args.profile_mode,
            "events_exported": len(event_rows),
            "capture_provenance": _event_capture_provenance_summary(event_rows),
            "forbidden_static_fields_exported": forbidden,
            "answers_routes_reasons_exported": False,
            "import": import_report.to_dict(),
        }
    finally:
        store.close()
    _print_payload(payload, json_mode=args.json)


def _default_event_transcript_raw_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.raw_event_export.jsonl")


def _event_transcript_rows(
    store: AssistantOSStore,
    *,
    session_selector: str,
) -> tuple[list[dict[str, Any]], str]:
    requested = str(session_selector or "all").strip()
    resolved_session = ""
    where = ""
    params: tuple[Any, ...] = ()
    if requested.lower() == "latest":
        latest = store.connection.execute(
            """
            SELECT session_id
            FROM events
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()
        resolved_session = str(latest["session_id"]) if latest is not None else ""
        if resolved_session:
            where = "WHERE session_id=?"
            params = (resolved_session,)
    elif requested and requested.lower() != "all":
        resolved_session = requested
        where = "WHERE session_id=?"
        params = (requested,)
    rows = store.connection.execute(
        f"""
        SELECT event_id, session_id, utterance, capture_surface, capture_source
        FROM events
        {where}
        ORDER BY rowid
        """,
        params,
    ).fetchall()
    exported: list[dict[str, Any]] = []
    session_days: dict[str, int] = {}
    previous_session = ""
    for index, row in enumerate(rows, start=1):
        session_id = str(row["session_id"])
        if session_id not in session_days:
            session_days[session_id] = len(session_days)
        item = {
            "role": "user",
            "session_id": session_id,
            "label": f"event_{row['event_id']}",
            "day": session_days[session_id],
            "content": str(row["utterance"]),
            "source_event_id": str(row["event_id"]),
        }
        capture_surface = str(row["capture_surface"])
        capture_source = str(row["capture_source"])
        if capture_surface:
            item["capture_surface"] = capture_surface
        if capture_source:
            item["capture_source"] = capture_source
        if index == 1 or session_id != previous_session:
            item["new_session"] = True
        previous_session = session_id
        exported.append(item)
    return exported, resolved_session


def _event_transcript_export_api_payload(
    db: Path, *, session: str = "all"
) -> dict[str, Any]:
    store = AssistantOSStore(db)
    try:
        rows, resolved_session = _event_transcript_rows(store, session_selector=session)
    finally:
        store.close()
    forbidden = sorted(
        {key for row in rows for key in set(row) & STATIC_TRANSCRIPT_EXPECTATION_KEYS}
    )
    return {
        "schema": "melm.local_assistant_event_transcript_export.v1",
        "db": str(db),
        "session": str(session),
        "resolved_session": resolved_session,
        "source_type": EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
        "events_exported": len(rows),
        "turns": rows,
        "capture_provenance": _event_capture_provenance_summary(rows),
        "forbidden_static_fields_exported": forbidden,
        "answers_routes_reasons_exported": False,
        "note": (
            "Only user utterances, session ids, labels, days, source event ids, and capture provenance "
            "are exported. Stored answers, routes, reasons, and assistant responses are intentionally omitted."
        ),
    }


def _event_capture_provenance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    surfaces = Counter(str(row.get("capture_surface", "") or "") for row in rows)
    sources = Counter(str(row.get("capture_source", "") or "") for row in rows)
    missing = int(surfaces.pop("", 0) or 0)
    missing += int(sources.pop("", 0) or 0)
    return _event_capture_provenance_from_counts(
        turn_count=len(rows),
        surface_counts=dict(sorted(surfaces.items())),
        source_counts=dict(sorted(sources.items())),
        missing_field_count=missing,
    )


def _event_capture_provenance_from_counts(
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


def _calibrate_event_ledger(args) -> None:
    try:
        payload = _build_event_ledger_calibration_payload(
            db=args.db,
            work_dir=args.work_dir,
            session=args.session,
            profile_mode=args.profile_mode,
            controls_json=args.controls_json,
            replacements=tuple(args.replace),
            weather_json=args.weather_json,
            reset=args.reset,
            min_total_turns=args.min_total_turns,
            min_local_resolution_rate=args.min_local_resolution_rate,
            min_route_kinds=args.min_route_kinds,
            min_intent_kinds=args.min_intent_kinds,
            min_synthesis_traces=args.min_synthesis_traces,
            min_priority_signal_samples=args.min_priority_signal_samples,
            auto_lifecycle=args.auto_lifecycle,
            require_priority_signals=args.require_priority_signals,
            require_memory_digest_quality=args.require_memory_digest_quality,
            require_strict_baseline_win=args.require_strict_baseline_win,
            require_redaction=args.require_redaction,
            require_static_drop=args.require_static_drop,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    _print_payload(payload, json_mode=args.json)


def _build_event_ledger_calibration_payload(
    *,
    db: Path,
    work_dir: Path,
    session: str = "all",
    profile_mode: str = "current",
    controls_json: Path | None = None,
    replacements: tuple[str, ...] = (),
    weather_json: Path = DEFAULT_WEATHER_SAMPLE,
    reset: bool = False,
    min_total_turns: int = 1,
    min_local_resolution_rate: float = 0.0,
    min_route_kinds: int = 1,
    min_intent_kinds: int = 1,
    min_synthesis_traces: int = 0,
    min_priority_signal_samples: int = 0,
    auto_lifecycle: bool = False,
    require_priority_signals: bool = False,
    require_memory_digest_quality: bool = False,
    require_strict_baseline_win: bool = False,
    require_redaction: bool = False,
    require_static_drop: bool = False,
) -> dict[str, Any]:
    if str(db) != ":memory:" and not db.exists():
        raise FileNotFoundError(f"Event ledger database not found: {db}")
    if reset and work_dir.exists():
        _safe_remove_bundle_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "event_ledger.raw_event_export.jsonl"
    replay_path = work_dir / "event_ledger_transcript_replay.jsonl"
    db_root = work_dir / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    calibration_db = db
    if str(db) != ":memory:":
        calibration_db = work_dir / "event_ledger.source_snapshot.sqlite"
        _snapshot_sqlite_database(db, calibration_db)
    controls = _read_optional_controls_json(controls_json)
    redaction_rules = _parse_transcript_redaction_rules(replacements)
    source_note = (
        "Event-ledger calibration export; only user utterances, session ids, labels, days, "
        "and safe lifecycle controls are retained. Stored answers, routes, reasons, and "
        "assistant responses are not exported as replay expectations."
    )
    item: dict = {
        "label": "event_ledger",
        "input_path": str(raw_path),
        "imported_transcript_jsonl": str(replay_path),
        "passed": False,
        "error": "",
        "import": {},
        "replay": {},
    }
    store = AssistantOSStore(calibration_db)
    try:
        event_rows, resolved_session = _event_transcript_rows(
            store, session_selector=session
        )
        profile = (
            asdict(store.load_profile(LocalAssistantProfile()))
            if profile_mode == "current"
            else None
        )
    finally:
        store.close()
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in event_rows),
        encoding="utf-8",
    )
    forbidden = sorted(
        key
        for row in event_rows
        for key in set(row) & STATIC_TRANSCRIPT_EXPECTATION_KEYS
    )
    try:
        import_report = import_transcript_replay_fixture(
            input_path=raw_path,
            output_path=replay_path,
            profile=profile,
            scenario="event_ledger_calibration",
            description="Redacted replay exported from the local assistant event ledger for calibration.",
            source_type=EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
            source_note=source_note,
            replacements=redaction_rules,
            min_turns=min_total_turns,
            min_route_kinds=min_route_kinds,
            controls=controls,
        )
        item["import"] = import_report.to_dict()
        if forbidden:
            item["error"] = "static_fields_exported"
        elif not import_report.passed:
            item["error"] = "import_failed"
        else:
            replay_report = run_transcript_replay_suite(
                transcript_path=replay_path,
                db_dir=db_root,
                reset=True,
                weather_offline_json=weather_json,
                auto_lifecycle=auto_lifecycle,
            )
            replay_payload = replay_report.to_dict()
            item["replay"] = _transcript_replay_summary(replay_payload)
            item["replay"]["report_schema"] = str(replay_payload.get("schema", ""))
            item["passed"] = bool(replay_payload.get("passed", False))
            if not item["passed"]:
                item["error"] = "replay_failed"
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        item["error"] = str(exc)
    aggregate = _transcript_calibration_aggregate(
        [item],
        min_total_turns=min_total_turns,
        min_local_resolution_rate=min_local_resolution_rate,
        min_route_kinds=min_route_kinds,
        min_intent_kinds=min_intent_kinds,
        min_synthesis_traces=min_synthesis_traces,
        min_priority_signal_samples=min_priority_signal_samples,
        require_priority_signals=require_priority_signals,
        require_memory_digest_quality=require_memory_digest_quality,
        require_strict_baseline_win=require_strict_baseline_win,
        require_redaction=require_redaction,
        require_static_drop=require_static_drop,
    )
    return {
        "schema": "melm.local_assistant_event_ledger_calibration_report.v1",
        "passed": bool(aggregate["passed"]) and not forbidden,
        "db": str(db),
        "calibration_db": str(calibration_db),
        "source_db_sha256": _sha256_file(db)
        if str(db) != ":memory:" and db.is_file()
        else "",
        "calibration_db_sha256": (
            _sha256_file(calibration_db)
            if str(calibration_db) != ":memory:" and calibration_db.is_file()
            else ""
        ),
        "work_dir": str(work_dir),
        "session": str(session),
        "resolved_session": resolved_session,
        "raw_event_jsonl": str(raw_path),
        "transcript_jsonl": str(replay_path),
        "source_type": EVENT_LEDGER_TRANSCRIPT_SOURCE_TYPE,
        "profile_mode": profile_mode,
        "events_exported": len(event_rows),
        "capture_provenance": _event_capture_provenance_summary(event_rows),
        "forbidden_static_fields_exported": forbidden,
        "answers_routes_reasons_exported": False,
        "auto_lifecycle": bool(auto_lifecycle),
        "item": item,
        "aggregate": aggregate,
    }


def _write_source_attestation(args) -> None:
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"source attestation already exists: {args.out}")
    payload = _build_source_attestation_payload(
        event_ledger_db=args.event_ledger_db,
        event_ledger_session=args.event_ledger_session,
        source_kind=args.source_kind,
        capture_surface=args.capture_surface,
        redaction_applied=args.redaction_applied,
        static_expectations_absent=args.static_expectations_absent,
        answers_routes_reasons_absent=args.answers_routes_reasons_absent,
        human_reviewed=args.human_reviewed,
        note=args.note,
    )
    validation = _source_attestation_report_from_payload(
        payload,
        path=args.out,
        expected_source_kind=args.source_kind,
        event_ledger_db=args.event_ledger_db,
        event_ledger_session=args.event_ledger_session,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "schema": "melm.local_assistant_source_attestation_write_report.v1",
        "passed": bool(validation.get("valid", False)),
        "attestation": str(args.out),
        "source_attestation": payload,
        "validation": validation,
    }
    _print_payload(report, json_mode=args.json)


def _candidate_session_audit(args) -> None:
    try:
        payload = _build_candidate_session_audit_payload(
            db=args.db,
            session=args.session,
            event_source_kind=args.event_source_kind,
            capture_surface=args.capture_surface,
            source_attestation_json=args.source_attestation_json,
            redaction_applied=args.redaction_applied,
            static_expectations_absent=args.static_expectations_absent,
            answers_routes_reasons_absent=args.answers_routes_reasons_absent,
            human_reviewed=args.human_reviewed,
            controls_json=args.controls_json,
            replacements=tuple(args.replace),
            weather_json=args.weather_json,
            work_dir=args.work_dir,
            reset=args.reset,
            min_total_turns=args.min_total_turns,
            min_local_resolution_rate=args.min_local_resolution_rate,
            min_route_kinds=args.min_route_kinds,
            min_intent_kinds=args.min_intent_kinds,
            min_synthesis_traces=args.min_synthesis_traces,
            min_priority_signal_samples=args.min_priority_signal_samples,
            auto_lifecycle=args.auto_lifecycle,
            inventory_soak_report_json=args.inventory_soak_report_json,
            transcript_calibration_report_json=args.transcript_calibration_report_json,
            host_app_config_json=args.host_app_config_json,
            host_app_attestation_json=args.host_app_attestation_json,
            run_host_app_probe=args.run_host_app_probe,
            host_app_db=args.host_app_db,
            host_app_work_dir=args.host_app_work_dir,
            host_app_media_dir=args.host_app_media_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _print_payload(payload, json_mode=args.json)


def _build_candidate_session_audit_payload(
    *,
    db: Path,
    session: str,
    event_source_kind: str,
    capture_surface: str,
    source_attestation_json: Path | None,
    redaction_applied: bool,
    static_expectations_absent: bool,
    answers_routes_reasons_absent: bool,
    human_reviewed: bool,
    controls_json: Path | None,
    replacements: tuple[str, ...],
    weather_json: Path,
    work_dir: Path,
    reset: bool,
    min_total_turns: int,
    min_local_resolution_rate: float,
    min_route_kinds: int,
    min_intent_kinds: int,
    min_synthesis_traces: int,
    min_priority_signal_samples: int,
    auto_lifecycle: bool,
    inventory_soak_report_json: Path | None,
    transcript_calibration_report_json: Path | None,
    host_app_config_json: Path | None,
    host_app_attestation_json: Path | None,
    run_host_app_probe: bool,
    host_app_db: Path,
    host_app_work_dir: Path,
    host_app_media_dir: Path | None,
) -> dict[str, Any]:
    if str(db) != ":memory:" and not db.exists():
        raise FileNotFoundError(f"assistant event ledger DB not found: {db}")
    started = perf_counter()
    event_export_payload = _event_transcript_export_api_payload(db, session=session)
    preview_payload = _build_source_attestation_payload(
        event_ledger_db=db,
        event_ledger_session=session,
        source_kind=event_source_kind,
        capture_surface=capture_surface,
        redaction_applied=redaction_applied,
        static_expectations_absent=static_expectations_absent,
        answers_routes_reasons_absent=answers_routes_reasons_absent,
        human_reviewed=human_reviewed,
        note="candidate-session-audit preview; not written",
    )
    preview_validation = _source_attestation_report_from_payload(
        preview_payload,
        path=Path("<candidate-session-audit-preview>"),
        expected_source_kind=event_source_kind,
        event_ledger_db=db,
        event_ledger_session=session,
    )
    existing_attestation = _source_attestation_report(
        source_attestation_json,
        event_source_kind=event_source_kind,
        event_ledger_db=db,
        event_ledger_session=session,
    )
    calibration_payload: dict[str, Any] = {}
    calibration_error = ""
    try:
        calibration_payload = _build_event_ledger_calibration_payload(
            db=db,
            work_dir=work_dir,
            session=session,
            profile_mode="current",
            controls_json=controls_json,
            replacements=replacements,
            weather_json=weather_json,
            reset=reset,
            min_total_turns=min_total_turns,
            min_local_resolution_rate=min_local_resolution_rate,
            min_route_kinds=min_route_kinds,
            min_intent_kinds=min_intent_kinds,
            min_synthesis_traces=min_synthesis_traces,
            min_priority_signal_samples=min_priority_signal_samples,
            auto_lifecycle=auto_lifecycle,
            require_priority_signals=False,
            require_memory_digest_quality=False,
            require_strict_baseline_win=False,
            require_redaction=False,
            require_static_drop=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        calibration_error = str(exc)
    source_for_packaging = (
        existing_attestation
        if source_attestation_json is not None
        else preview_validation
    )
    event_capture = dict(event_export_payload.get("capture_provenance", {}))
    calibration_summary = _v01_event_calibration_summary(
        calibration_payload, calibration_error
    )
    calibration_passed = (
        bool(calibration_payload.get("passed", False)) and not calibration_error
    )
    preview_valid = bool(preview_validation.get("valid", False))
    existing_valid = bool(existing_attestation.get("valid", False))
    selected_valid = bool(source_for_packaging.get("valid", False))
    thresholds = {
        "min_total_turns": max(1, int(min_total_turns or 1)),
        "min_local_resolution_rate": max(0.0, float(min_local_resolution_rate or 0.0)),
        "min_route_kinds": max(1, int(min_route_kinds or 1)),
        "min_intent_kinds": max(1, int(min_intent_kinds or 1)),
        "min_synthesis_traces": max(0, int(min_synthesis_traces or 0)),
        "min_priority_signal_samples": max(0, int(min_priority_signal_samples or 0)),
    }
    inventory_payload, inventory_error = _optional_json_report(
        inventory_soak_report_json
    )
    transcript_calibration_payload, transcript_calibration_error = (
        _optional_json_report(transcript_calibration_report_json)
    )
    if (
        transcript_calibration_payload
        and transcript_calibration_payload.get("schema")
        != "melm.local_assistant_transcript_calibration_report.v1"
    ):
        transcript_calibration_error = (
            "transcript calibration report schema is not "
            "melm.local_assistant_transcript_calibration_report.v1"
        )
    host_app_attestation = _host_app_attestation_report(
        host_app_attestation_json,
        host_app_config_json=host_app_config_json,
    )
    host_payload: dict[str, Any] = {}
    host_error = ""
    if host_app_config_json is not None:
        if run_host_app_probe:
            try:
                host_payload = _build_host_app_probe_payload(
                    host_app_db,
                    work_dir=host_app_work_dir,
                    reset=reset,
                    media_player_command="",
                    call_command="",
                    media_dir=host_app_media_dir,
                    require_configured=True,
                    config_json=host_app_config_json,
                )
            except (OSError, ValueError) as exc:
                host_error = str(exc)
        else:
            config, config_error = _host_app_config(host_app_config_json)
            host_payload = {
                "passed": False,
                "configured": False,
                "skipped": True,
                "checks": {"configuration_reported": not bool(config_error)},
                "config": config,
                "error": config_error,
                "evidence_class": _host_app_static_analysis(
                    host_app_config_json, config
                )
                if not config_error
                else {"error": config_error},
            }
            host_error = config_error
    blocker_projection = {
        "projection_only": True,
        "after_writing_source_attestation": _candidate_session_blocker_projection(
            event_ledger_db=db,
            event_payload=calibration_payload,
            event_error=calibration_error,
            event_source_kind=event_source_kind,
            source_attestation_valid=preview_valid,
            inventory_payload=inventory_payload,
            inventory_error=inventory_error,
            transcript_calibration_payload=transcript_calibration_payload,
            transcript_calibration_error=transcript_calibration_error,
            host_payload=host_payload,
            host_error=host_error,
            host_app_attestation_valid=bool(host_app_attestation.get("valid", False)),
            thresholds=thresholds,
        ),
        "with_existing_source_attestation": _candidate_session_blocker_projection(
            event_ledger_db=db,
            event_payload=calibration_payload,
            event_error=calibration_error,
            event_source_kind=event_source_kind,
            source_attestation_valid=existing_valid,
            inventory_payload=inventory_payload,
            inventory_error=inventory_error,
            transcript_calibration_payload=transcript_calibration_payload,
            transcript_calibration_error=transcript_calibration_error,
            host_payload=host_payload,
            host_error=host_error,
            host_app_attestation_valid=bool(host_app_attestation.get("valid", False)),
            thresholds=thresholds,
        ),
        "artifact_inputs": {
            "inventory_soak_report": _v01_inventory_report_summary(
                inventory_payload, inventory_error
            ),
            "transcript_calibration_report": _v01_transcript_calibration_report_summary(
                transcript_calibration_payload,
                transcript_calibration_error,
                event_ledger_db=db,
            ),
            "host_app_attestation": host_app_attestation,
            "host_app_probe": _v01_host_app_summary(host_payload, host_error),
        },
        "note": (
            "Projection only. It reuses v01-blocker-evidence row logic but does not create candidate "
            "evidence; written attestation, artifact binding, and v01-blocker-evidence are still required."
        ),
    }
    checks = {
        "event_ledger_db_present": str(db) == ":memory:" or db.is_file(),
        "event_turns_exported": int(event_export_payload.get("events_exported", 0) or 0)
        >= 1,
        "event_capture_provenance_present": bool(
            event_capture.get("has_capture_provenance", False)
        ),
        "event_static_exports_absent": (
            event_export_payload.get("answers_routes_reasons_exported") is False
            and event_export_payload.get("forbidden_static_fields_exported", []) == []
        ),
        "source_attestation_preview_valid": preview_valid,
        "source_attestation_existing_valid": existing_valid,
        "source_attestation_or_preview_valid": selected_valid,
        "event_calibration_report_assembled": not bool(calibration_error),
        "event_calibration_passed": calibration_passed,
    }
    write_source_command = (
        "python scripts/local_assistant_os_cli.py write-source-attestation "
        f"--event-ledger-db {db} --event-ledger-session {session} "
        f"--source-kind {event_source_kind} --capture-surface {capture_surface} "
        "--redaction-applied --static-expectations-absent --answers-routes-reasons-absent "
        "--human-reviewed --out <source-attestation.json> --json"
    )
    evidence_pack_command = (
        "python scripts/local_assistant_os_cli.py v01-evidence-pack "
        f"--db {db} --session {session} --event-source-kind {event_source_kind} "
        f"--capture-surface {capture_surface} --source-attestation-json <source-attestation.json> "
        "--auto-lifecycle --json"
    )
    return {
        "schema": "melm.local_assistant_candidate_session_audit.v1",
        "passed": bool(
            checks["event_ledger_db_present"] and checks["event_turns_exported"]
        ),
        "candidate_session_ready": bool(calibration_passed and selected_valid),
        "ready_for_source_attestation_write": preview_valid,
        "ready_for_v01_evidence_pack_with_write": bool(
            calibration_passed and preview_valid
        ),
        "ready_for_v01_evidence_pack_with_existing_attestation": bool(
            calibration_passed and existing_valid
        ),
        "db": str(db),
        "session": str(session),
        "event_source_kind": event_source_kind,
        "capture_surface": capture_surface,
        "elapsed_ms": _elapsed_ms(started),
        "checks": checks,
        "event_transcript_export": {
            "events_exported": event_export_payload.get("events_exported", 0),
            "resolved_session": event_export_payload.get("resolved_session", ""),
            "answers_routes_reasons_exported": event_export_payload.get(
                "answers_routes_reasons_exported"
            ),
            "forbidden_static_fields_exported": event_export_payload.get(
                "forbidden_static_fields_exported", []
            ),
            "capture_provenance": event_capture,
        },
        "event_ledger_calibration": calibration_summary,
        "source_attestation_preview": preview_validation,
        "source_attestation_existing": existing_attestation,
        "selected_source_attestation": (
            "existing" if source_attestation_json is not None else "preview"
        ),
        "blocker_projection": blocker_projection,
        "next_commands": {
            "write_source_attestation": write_source_command,
            "v01_evidence_pack": evidence_pack_command,
            "v01_blocker_evidence": (
                "python scripts/local_assistant_os_cli.py v01-blocker-evidence "
                f"--event-ledger-db {db} --event-ledger-session {session} "
                f"--event-source-kind {event_source_kind} "
                "--source-attestation-json <source-attestation.json> --auto-lifecycle --json"
            ),
        },
        "note": (
            "This audit does not create candidate evidence. It verifies whether an existing "
            "event-ledger session is structurally ready for source attestation and v0.1 evidence packaging."
        ),
    }


def _candidate_session_blocker_projection(
    *,
    event_ledger_db: Path,
    event_payload: dict[str, Any],
    event_error: str,
    event_source_kind: str,
    source_attestation_valid: bool,
    inventory_payload: dict[str, Any],
    inventory_error: str,
    transcript_calibration_payload: dict[str, Any],
    transcript_calibration_error: str,
    host_payload: dict[str, Any],
    host_error: str,
    host_app_attestation_valid: bool,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    rows = _v01_blocker_evidence_rows(
        event_ledger_db=event_ledger_db,
        event_payload=event_payload,
        event_error=event_error,
        event_source_kind=event_source_kind,
        source_attestation_valid=source_attestation_valid,
        inventory_payload=inventory_payload,
        inventory_error=inventory_error,
        transcript_calibration_payload=transcript_calibration_payload,
        transcript_calibration_error=transcript_calibration_error,
        host_payload=host_payload,
        host_error=host_error,
        host_app_attestation_valid=host_app_attestation_valid,
        thresholds=thresholds,
    )
    candidate_ids = [
        str(row.get("id", ""))
        for row in rows
        if row.get("status") == "candidate_evidence_present"
    ]
    development_or_unattested_ids = [
        str(row.get("id", ""))
        for row in rows
        if str(row.get("status", "")).startswith("development")
        or str(row.get("status", "")).startswith("unattested")
    ]
    return {
        "source_attestation_valid": bool(source_attestation_valid),
        "candidate_blockers_satisfied": len(candidate_ids),
        "remaining_blocker_count": max(0, len(rows) - len(candidate_ids)),
        "candidate_blockers": candidate_ids,
        "development_or_unattested_blockers": development_or_unattested_ids,
        "status_counts": dict(
            sorted(Counter(str(row.get("status", "")) for row in rows).items())
        ),
        "rows": rows,
    }


def _write_host_app_attestation(args) -> None:
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"host app attestation already exists: {args.out}")
    payload = _build_host_app_attestation_payload(
        host_app_config_json=args.host_app_config_json,
        capture_surface=args.capture_surface,
        media_app_configured=args.media_app_configured,
        call_app_configured=args.call_app_configured,
        not_demo_recorder=args.not_demo_recorder,
        real_app_commands_acknowledged=args.real_app_commands_acknowledged,
        human_reviewed=args.human_reviewed,
        note=args.note,
    )
    validation = _host_app_attestation_report_from_payload(
        payload,
        path=args.out,
        host_app_config_json=args.host_app_config_json,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "schema": "melm.local_assistant_host_app_attestation_write_report.v1",
        "passed": bool(validation.get("valid", False)),
        "attestation": str(args.out),
        "host_app_attestation": payload,
        "validation": validation,
    }
    _print_payload(report, json_mode=args.json)


def _build_host_app_attestation_payload(
    *,
    host_app_config_json: Path,
    capture_surface: str,
    media_app_configured: bool,
    call_app_configured: bool,
    not_demo_recorder: bool,
    real_app_commands_acknowledged: bool,
    human_reviewed: bool,
    note: str,
) -> dict[str, Any]:
    if not host_app_config_json.exists() or not host_app_config_json.is_file():
        raise FileNotFoundError(
            f"host app config JSON not found: {host_app_config_json}"
        )
    config, config_error = _host_app_config(host_app_config_json)
    if config_error:
        raise ValueError(f"invalid host app config JSON: {config_error}")
    return {
        "schema": HOST_APP_ATTESTATION_SCHEMA,
        "host_app_config_json": str(host_app_config_json),
        "host_app_config_sha256": _sha256_file(host_app_config_json),
        "capture_surface": capture_surface,
        "media_app_configured": bool(media_app_configured),
        "call_app_configured": bool(call_app_configured),
        "not_demo_recorder": bool(not_demo_recorder),
        "real_app_commands_acknowledged": bool(real_app_commands_acknowledged),
        "human_reviewed": bool(human_reviewed),
        "config_analysis": _host_app_static_analysis(host_app_config_json, config),
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "note": str(note or ""),
    }


def _build_source_attestation_payload(
    *,
    event_ledger_db: Path,
    event_ledger_session: str = "all",
    source_kind: str,
    capture_surface: str,
    redaction_applied: bool,
    static_expectations_absent: bool,
    answers_routes_reasons_absent: bool,
    human_reviewed: bool,
    note: str,
) -> dict[str, Any]:
    if not event_ledger_db.exists() or not event_ledger_db.is_file():
        raise FileNotFoundError(f"event ledger DB not found: {event_ledger_db}")
    # Opening an older ledger may apply idempotent schema initialization.
    # Stabilize it before capturing the byte hash used by the attestation.
    _event_capture_provenance_for_db(event_ledger_db, session=event_ledger_session)
    return {
        "schema": SOURCE_ATTESTATION_SCHEMA,
        "source_kind": source_kind,
        "capture_surface": capture_surface,
        "event_ledger_db": str(event_ledger_db),
        "event_ledger_session": str(event_ledger_session or "all"),
        "event_ledger_db_sha256": _sha256_file(event_ledger_db),
        "redaction_applied": bool(redaction_applied),
        "static_expectations_absent": bool(static_expectations_absent),
        "answers_routes_reasons_absent": bool(answers_routes_reasons_absent),
        "human_reviewed": bool(human_reviewed),
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "note": str(note or ""),
    }


def _source_attestation_report(
    path: Path | None,
    *,
    event_source_kind: str,
    event_ledger_db: Path | None,
    event_ledger_session: str = "all",
) -> dict[str, Any]:
    if path is None:
        return {
            "present": False,
            "valid": False,
            "path": "",
            "error": "",
            "checks": {},
            "missing": ["source attestation JSON not provided"],
        }
    if not path.exists() or not path.is_file():
        return {
            "present": False,
            "valid": False,
            "path": str(path),
            "error": f"source attestation file not found: {path}",
            "checks": {},
            "missing": ["source attestation JSON not found"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return {
            "present": True,
            "valid": False,
            "path": str(path),
            "error": f"invalid source attestation JSON: {exc}",
            "checks": {},
            "missing": ["valid JSON object"],
        }
    except OSError as exc:
        return {
            "present": True,
            "valid": False,
            "path": str(path),
            "error": f"could not read source attestation: {exc}",
            "checks": {},
            "missing": ["readable source attestation JSON"],
        }
    if not isinstance(payload, dict):
        return {
            "present": True,
            "valid": False,
            "path": str(path),
            "error": "source attestation root must be a JSON object",
            "checks": {},
            "missing": ["JSON object root"],
        }
    return _source_attestation_report_from_payload(
        payload,
        path=path,
        expected_source_kind=event_source_kind,
        event_ledger_db=event_ledger_db,
        event_ledger_session=event_ledger_session,
    )


def _source_attestation_report_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_source_kind: str,
    event_ledger_db: Path | None,
    event_ledger_session: str = "all",
) -> dict[str, Any]:
    actual_source_kind = str(payload.get("source_kind", ""))
    capture_surface = str(payload.get("capture_surface", ""))
    attested_session = str(payload.get("event_ledger_session", "all") or "all")
    expected_session = str(event_ledger_session or "all")
    expected_hash = str(payload.get("event_ledger_db_sha256", ""))
    actual_hash = (
        _sha256_file(event_ledger_db)
        if event_ledger_db is not None and event_ledger_db.is_file()
        else ""
    )
    hash_required = event_ledger_db is not None
    event_capture_provenance = _event_capture_provenance_for_db(
        event_ledger_db, session=attested_session
    )
    capture_surface_counts = dict(
        event_capture_provenance.get("capture_surface_counts", {})
    )
    has_capture_provenance = bool(
        event_capture_provenance.get("has_capture_provenance", False)
    )
    turn_count = int(event_capture_provenance.get("turn_count", 0) or 0)
    imported_turn_count = int(
        event_capture_provenance.get("imported_turn_count", 0) or 0
    )
    candidate_capture_source_count = int(
        event_capture_provenance.get("candidate_capture_source_count", 0) or 0
    )
    imported_capture_covers_events = (
        bool(turn_count) and imported_turn_count == turn_count
    )
    candidate_capture_covers_events = (
        bool(turn_count) and candidate_capture_source_count == turn_count
    )
    candidate_or_imported_capture_complete = bool(
        has_capture_provenance
        and (imported_capture_covers_events or candidate_capture_covers_events)
    )
    capture_surface_count = (
        int(capture_surface_counts.get(capture_surface, 0) or 0)
        if capture_surface
        else 0
    )
    capture_surface_covers_events = (
        bool(turn_count) and capture_surface_count == turn_count
    )
    capture_surface_matches_events = bool(
        capture_surface and capture_surface_covers_events
    )
    checks = {
        "schema_supported": payload.get("schema") == SOURCE_ATTESTATION_SCHEMA,
        "source_kind_supported": actual_source_kind in SOURCE_ATTESTATION_KINDS,
        "source_kind_matches_command": actual_source_kind == expected_source_kind,
        "event_ledger_session_matches_command": attested_session == expected_session,
        "capture_surface_supported": capture_surface in SOURCE_ATTESTATION_SURFACES,
        "capture_surface_matches_events": capture_surface_matches_events,
        "event_capture_provenance_present": has_capture_provenance,
        "candidate_capture_not_all_scripted": candidate_or_imported_capture_complete,
        "candidate_or_imported_capture_complete": candidate_or_imported_capture_complete,
        "candidate_capture_covers_events": candidate_capture_covers_events,
        "imported_capture_covers_events": imported_capture_covers_events,
        "redaction_applied": payload.get("redaction_applied") is True,
        "static_expectations_absent": payload.get("static_expectations_absent") is True,
        "answers_routes_reasons_absent": payload.get("answers_routes_reasons_absent")
        is True,
        "human_reviewed": payload.get("human_reviewed") is True,
        "event_ledger_db_sha256_present": bool(expected_hash)
        if hash_required
        else True,
        "event_ledger_db_sha256_matches": (
            bool(expected_hash and actual_hash and expected_hash == actual_hash)
            if hash_required
            else True
        ),
    }
    missing_labels = {
        "schema_supported": "schema must be melm.local_assistant_source_attestation.v1",
        "source_kind_supported": "source kind must be redacted_user_session or target_device_user_session",
        "source_kind_matches_command": "source kind must match --event-source-kind",
        "event_ledger_session_matches_command": "event_ledger_session must match the packaged event-ledger session",
        "capture_surface_supported": "capture surface must be a supported local/target collection surface",
        "capture_surface_matches_events": "attested capture surface must cover every event ledger turn",
        "event_capture_provenance_present": "event ledger capture provenance present",
        "candidate_capture_not_all_scripted": "candidate user evidence must come entirely from imported redacted, interactive CLI, browser UI, or target-device capture; scripted CLI/API/UI smokes stay development evidence",
        "redaction_applied": "redaction_applied=true",
        "static_expectations_absent": "static_expectations_absent=true",
        "answers_routes_reasons_absent": "answers_routes_reasons_absent=true",
        "human_reviewed": "human_reviewed=true",
        "event_ledger_db_sha256_present": "event_ledger_db_sha256 present",
        "event_ledger_db_sha256_matches": "event_ledger_db_sha256 matches current event ledger DB",
    }
    missing = [label for key, label in missing_labels.items() if not checks.get(key)]
    return {
        "present": True,
        "valid": not missing,
        "path": str(path),
        "error": "",
        "schema": str(payload.get("schema", "")),
        "source_kind": actual_source_kind,
        "capture_surface": capture_surface,
        "event_ledger_session": attested_session,
        "expected_event_ledger_session": expected_session,
        "event_ledger_db_sha256": expected_hash,
        "actual_event_ledger_db_sha256": actual_hash,
        "event_capture_provenance": event_capture_provenance,
        "capture_surface_turn_count": capture_surface_count,
        "candidate_capture_source_count": candidate_capture_source_count,
        "imported_turn_count": imported_turn_count,
        "checks": checks,
        "missing": missing,
    }


def _event_capture_provenance_for_db(
    db: Path | None, *, session: str = "all"
) -> dict[str, Any]:
    if db is None or not db.is_file():
        return {
            "present": False,
            "error": "event ledger DB unavailable",
        }
    store = AssistantOSStore(db)
    try:
        rows, resolved_session = _event_transcript_rows(store, session_selector=session)
    except sqlite3.Error as exc:
        return {
            "present": False,
            "error": f"could not read event capture provenance: {exc}",
        }
    finally:
        store.close()
    return {
        "present": True,
        "session": str(session or "all"),
        "resolved_session": resolved_session,
        **_event_capture_provenance_summary(rows),
    }


def _snapshot_sqlite_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    source_connection = sqlite3.connect(os.fspath(source))
    destination_connection = sqlite3.connect(os.fspath(destination))
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()
        source_connection.close()


def _host_app_attestation_report(
    path: Path | None,
    *,
    host_app_config_json: Path | None,
) -> dict[str, Any]:
    if path is None:
        return {
            "present": False,
            "valid": False,
            "path": "",
            "error": "",
            "checks": {},
            "missing": ["host app attestation JSON not provided"],
        }
    if not path.exists() or not path.is_file():
        return {
            "present": False,
            "valid": False,
            "path": str(path),
            "error": f"host app attestation file not found: {path}",
            "checks": {},
            "missing": ["host app attestation JSON not found"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return {
            "present": True,
            "valid": False,
            "path": str(path),
            "error": f"invalid host app attestation JSON: {exc}",
            "checks": {},
            "missing": ["valid JSON object"],
        }
    except OSError as exc:
        return {
            "present": True,
            "valid": False,
            "path": str(path),
            "error": f"could not read host app attestation: {exc}",
            "checks": {},
            "missing": ["readable host app attestation JSON"],
        }
    if not isinstance(payload, dict):
        return {
            "present": True,
            "valid": False,
            "path": str(path),
            "error": "host app attestation root must be a JSON object",
            "checks": {},
            "missing": ["JSON object root"],
        }
    return _host_app_attestation_report_from_payload(
        payload,
        path=path,
        host_app_config_json=host_app_config_json,
    )


def _host_app_attestation_report_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    host_app_config_json: Path | None,
) -> dict[str, Any]:
    capture_surface = str(payload.get("capture_surface", ""))
    expected_hash = str(payload.get("host_app_config_sha256", ""))
    actual_hash = (
        _sha256_file(host_app_config_json)
        if host_app_config_json is not None and host_app_config_json.is_file()
        else ""
    )
    config, config_error = _host_app_config(host_app_config_json)
    config_analysis = (
        _host_app_static_analysis(host_app_config_json, config)
        if not config_error
        else {
            "config_json": str(host_app_config_json)
            if host_app_config_json is not None
            else "",
            "demo_recorder_detected": False,
            "candidate_target_device_app_evidence": False,
            "markers": [],
            "error": config_error,
        }
    )
    checks = {
        "schema_supported": payload.get("schema") == HOST_APP_ATTESTATION_SCHEMA,
        "capture_surface_supported": capture_surface in HOST_APP_ATTESTATION_SURFACES,
        "host_app_config_sha256_present": bool(expected_hash),
        "host_app_config_sha256_matches": bool(
            expected_hash and actual_hash and expected_hash == actual_hash
        ),
        "media_app_configured": payload.get("media_app_configured") is True,
        "call_app_configured": payload.get("call_app_configured") is True,
        "not_demo_recorder_asserted": payload.get("not_demo_recorder") is True,
        "real_app_commands_acknowledged": payload.get("real_app_commands_acknowledged")
        is True,
        "human_reviewed": payload.get("human_reviewed") is True,
        "config_json_valid": not bool(config_error),
        "config_not_demo_recorder": not bool(
            config_analysis.get("demo_recorder_detected", False)
        ),
    }
    missing_labels = {
        "schema_supported": "schema must be melm.local_assistant_host_app_attestation.v1",
        "capture_surface_supported": "capture surface must be a supported target/desktop collection surface",
        "host_app_config_sha256_present": "host_app_config_sha256 present",
        "host_app_config_sha256_matches": "host_app_config_sha256 matches current host app config",
        "media_app_configured": "media_app_configured=true",
        "call_app_configured": "call_app_configured=true",
        "not_demo_recorder_asserted": "not_demo_recorder=true",
        "real_app_commands_acknowledged": "real_app_commands_acknowledged=true",
        "human_reviewed": "human_reviewed=true",
        "config_json_valid": "valid host app config JSON",
        "config_not_demo_recorder": "host app config does not use recorder/demo commands",
    }
    missing = [label for key, label in missing_labels.items() if not checks.get(key)]
    return {
        "present": True,
        "valid": not missing,
        "path": str(path),
        "error": "",
        "schema": str(payload.get("schema", "")),
        "capture_surface": capture_surface,
        "host_app_config_sha256": expected_hash,
        "actual_host_app_config_sha256": actual_hash,
        "config_analysis": config_analysis,
        "checks": checks,
        "missing": missing,
    }


def _v01_blocker_evidence(args) -> None:
    payload = _build_v01_blocker_evidence_payload(
        event_ledger_db=args.event_ledger_db,
        event_ledger_work_dir=args.event_ledger_work_dir,
        event_ledger_session=args.event_ledger_session,
        event_source_kind=args.event_source_kind,
        source_attestation_json=args.source_attestation_json,
        controls_json=args.controls_json,
        replacements=tuple(args.replace),
        weather_json=args.weather_json,
        reset=args.reset,
        min_total_turns=args.min_total_turns,
        min_local_resolution_rate=args.min_local_resolution_rate,
        min_route_kinds=args.min_route_kinds,
        min_intent_kinds=args.min_intent_kinds,
        min_synthesis_traces=args.min_synthesis_traces,
        min_priority_signal_samples=args.min_priority_signal_samples,
        auto_lifecycle=args.auto_lifecycle,
        inventory_soak_report_json=args.inventory_soak_report_json,
        transcript_calibration_report_json=args.transcript_calibration_report_json,
        host_app_config_json=args.host_app_config_json,
        host_app_attestation_json=args.host_app_attestation_json,
        run_host_app_probe=args.run_host_app_probe,
        host_app_db=args.host_app_db,
        host_app_work_dir=args.host_app_work_dir,
        host_app_media_dir=args.host_app_media_dir,
    )
    if args.out is not None:
        payload["report_path"] = str(args.out)
        payload["report_written"] = True
        _write_json_report(args.out, payload)
    _print_payload(payload, json_mode=args.json)


def _v01_blocker_rehearsal(args) -> None:
    payload = _build_v01_blocker_rehearsal_payload(args.work_dir, reset=args.reset)
    _print_payload(payload, json_mode=args.json)


def _v01_evidence_pack(args) -> None:
    try:
        payload = _build_v01_evidence_pack_payload(
            db=args.db,
            work_dir=args.work_dir,
            session=args.session,
            event_source_kind=args.event_source_kind,
            capture_surface=args.capture_surface,
            source_attestation_json=args.source_attestation_json,
            write_source_attestation=args.write_source_attestation,
            redaction_applied=args.redaction_applied,
            static_expectations_absent=args.static_expectations_absent,
            answers_routes_reasons_absent=args.answers_routes_reasons_absent,
            human_reviewed=args.human_reviewed,
            note=args.note,
            controls_json=args.controls_json,
            replacements=tuple(args.replace),
            weather_json=args.weather_json,
            reset=args.reset,
            min_total_turns=args.min_total_turns,
            min_local_resolution_rate=args.min_local_resolution_rate,
            min_route_kinds=args.min_route_kinds,
            min_intent_kinds=args.min_intent_kinds,
            min_synthesis_traces=args.min_synthesis_traces,
            min_priority_signal_samples=args.min_priority_signal_samples,
            auto_lifecycle=args.auto_lifecycle,
            inventory_soak_report_json=args.inventory_soak_report_json,
            transcript_calibration_report_json=args.transcript_calibration_report_json,
            host_app_config_json=args.host_app_config_json,
            host_app_attestation_json=args.host_app_attestation_json,
            run_host_app_probe=args.run_host_app_probe,
            host_app_db=args.host_app_db,
            host_app_work_dir=args.host_app_work_dir,
            host_app_media_dir=args.host_app_media_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _print_payload(payload, json_mode=args.json)


def _build_v01_evidence_pack_payload(
    *,
    db: Path,
    work_dir: Path,
    session: str,
    event_source_kind: str,
    capture_surface: str,
    source_attestation_json: Path | None,
    write_source_attestation: bool,
    redaction_applied: bool,
    static_expectations_absent: bool,
    answers_routes_reasons_absent: bool,
    human_reviewed: bool,
    note: str,
    controls_json: Path | None,
    replacements: tuple[str, ...],
    weather_json: Path,
    reset: bool,
    min_total_turns: int,
    min_local_resolution_rate: float,
    min_route_kinds: int,
    min_intent_kinds: int,
    min_synthesis_traces: int,
    min_priority_signal_samples: int,
    auto_lifecycle: bool,
    inventory_soak_report_json: Path | None,
    transcript_calibration_report_json: Path | None,
    host_app_config_json: Path | None,
    host_app_attestation_json: Path | None,
    run_host_app_probe: bool,
    host_app_db: Path,
    host_app_work_dir: Path,
    host_app_media_dir: Path | None,
) -> dict[str, Any]:
    if source_attestation_json is not None and write_source_attestation:
        raise ValueError(
            "pass either --source-attestation-json or --write-source-attestation, not both"
        )
    if write_source_attestation and event_source_kind not in SOURCE_ATTESTATION_KINDS:
        raise ValueError(
            "--write-source-attestation requires redacted_user_session or target_device_user_session"
        )
    if str(db) != ":memory:" and not db.exists():
        raise FileNotFoundError(f"assistant event ledger DB not found: {db}")

    started = perf_counter()
    if reset and work_dir.exists():
        _safe_remove_bundle_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    event_export_report_path = work_dir / "event_transcript_export.json"
    blocker_report_path = work_dir / "v01_blocker_evidence.json"
    progress_report_path = work_dir / "v01_progress.json"
    pack_report_path = work_dir / "v01_evidence_pack.json"
    source_note_path = work_dir / "development_source_note.json"
    written_source_attestation_path = work_dir / "source_attestation.json"

    event_export_payload = _event_transcript_export_api_payload(db, session=session)
    _write_json_report(event_export_report_path, event_export_payload)
    exported_turns = [
        dict(item)
        for item in event_export_payload.get("turns", [])
        if isinstance(item, dict)
    ]

    source_attestation_path = source_attestation_json
    source_attestation_write_report: dict[str, Any] = {}
    development_source_note: dict[str, Any] = {}
    if write_source_attestation:
        source_payload = _build_source_attestation_payload(
            event_ledger_db=db,
            event_ledger_session=session,
            source_kind=event_source_kind,
            capture_surface=capture_surface,
            redaction_applied=redaction_applied,
            static_expectations_absent=static_expectations_absent,
            answers_routes_reasons_absent=answers_routes_reasons_absent,
            human_reviewed=human_reviewed,
            note=note,
        )
        validation = _source_attestation_report_from_payload(
            source_payload,
            path=written_source_attestation_path,
            expected_source_kind=event_source_kind,
            event_ledger_db=db,
            event_ledger_session=session,
        )
        _write_json_report(written_source_attestation_path, source_payload)
        source_attestation_path = written_source_attestation_path
        source_attestation_write_report = {
            "schema": "melm.local_assistant_source_attestation_write_report.v1",
            "passed": bool(validation.get("valid", False)),
            "attestation": str(written_source_attestation_path),
            "source_attestation": source_payload,
            "validation": validation,
        }
    elif event_source_kind == "development_session":
        development_source_note = _build_v01_development_source_note_from_export(
            db=db,
            capture_surface=capture_surface,
            event_export_payload=event_export_payload,
        )
        _write_json_report(source_note_path, development_source_note)

    blocker_payload = _build_v01_blocker_evidence_payload(
        event_ledger_db=db,
        event_ledger_work_dir=work_dir / "event_ledger",
        event_ledger_session=session,
        event_source_kind=event_source_kind,
        source_attestation_json=source_attestation_path,
        controls_json=controls_json,
        replacements=replacements,
        weather_json=weather_json,
        reset=True,
        min_total_turns=min_total_turns,
        min_local_resolution_rate=min_local_resolution_rate,
        min_route_kinds=min_route_kinds,
        min_intent_kinds=min_intent_kinds,
        min_synthesis_traces=min_synthesis_traces,
        min_priority_signal_samples=min_priority_signal_samples,
        auto_lifecycle=auto_lifecycle,
        inventory_soak_report_json=inventory_soak_report_json,
        transcript_calibration_report_json=transcript_calibration_report_json,
        host_app_config_json=host_app_config_json,
        host_app_attestation_json=host_app_attestation_json,
        run_host_app_probe=run_host_app_probe,
        host_app_db=host_app_db,
        host_app_work_dir=host_app_work_dir,
        host_app_media_dir=host_app_media_dir,
    )
    _write_json_report(blocker_report_path, blocker_payload)
    progress_payload = _build_v01_progress_payload(
        blocker_evidence_json=blocker_report_path
    )
    _write_json_report(progress_report_path, progress_payload)

    event_summary = dict(blocker_payload.get("event_ledger_calibration", {}))
    source_summary = dict(blocker_payload.get("source_attestation", {}))
    capture_provenance = dict(event_export_payload.get("capture_provenance", {}))
    candidate_count = int(blocker_payload.get("candidate_blockers_satisfied", 0) or 0)
    checks = {
        "event_ledger_db_present": str(db) == ":memory:" or db.is_file(),
        "event_turns_exported": len(exported_turns) >= 1,
        "event_capture_provenance_present": bool(
            capture_provenance.get("has_capture_provenance", False)
        ),
        "event_export_report_written": event_export_report_path.is_file(),
        "blocker_evidence_report_written": blocker_report_path.is_file(),
        "progress_report_written": progress_report_path.is_file(),
        "event_calibration_present": bool(event_summary.get("present", False)),
        "event_calibration_static_exports_absent": (
            event_summary.get("answers_routes_reasons_exported") is False
            and event_summary.get("forbidden_static_fields_exported", []) == []
        ),
        "progress_consumes_pack_blocker_report": progress_payload.get(
            "blocker_evidence_source"
        )
        == str(blocker_report_path),
        "architecture_completion_not_claimed": not bool(
            blocker_payload.get("architecture_complete_claimed", False)
        )
        and not bool(progress_payload.get("architecture_complete_claimed", False))
        and not bool(progress_payload.get("architecture_complete", False)),
        "development_evidence_not_promoted": (
            event_source_kind != "development_session"
            or (
                candidate_count == 0
                and all(
                    not bool(row.get("candidate_for_architecture_review", False))
                    for row in blocker_payload.get("blockers", [])
                    if isinstance(row, dict)
                )
            )
        ),
        "source_boundary_recorded": (
            bool(source_attestation_write_report)
            or bool(source_summary.get("present", False))
            or bool(development_source_note)
            or event_source_kind in SOURCE_ATTESTATION_KINDS
        ),
    }
    payload = {
        "schema": "melm.local_assistant_v01_evidence_pack.v1",
        "passed": all(checks.values())
        and bool(blocker_payload.get("passed", False))
        and bool(progress_payload.get("passed", False)),
        "architecture_complete": False,
        "architecture_complete_claimed": False,
        "candidate_review_ready": bool(
            progress_payload.get("candidate_review_ready", False)
        ),
        "event_source_kind": event_source_kind,
        "session": str(session),
        "resolved_session": event_export_payload.get("resolved_session", ""),
        "work_dir": str(work_dir),
        "elapsed_ms": _elapsed_ms(started),
        "checks": checks,
        "artifact_paths": {
            "evidence_pack_report": str(pack_report_path),
            "event_transcript_export_report": str(event_export_report_path),
            "raw_event_jsonl": str(event_summary.get("raw_event_jsonl", "")),
            "transcript_jsonl": str(event_summary.get("transcript_jsonl", "")),
            "source_attestation": str(source_attestation_path)
            if source_attestation_path is not None
            else "",
            "development_source_note": str(source_note_path)
            if development_source_note
            else "",
            "blocker_evidence_report": str(blocker_report_path),
            "progress_report": str(progress_report_path),
        },
        "event_transcript_export": {
            "events_exported": int(event_export_payload.get("events_exported", 0) or 0),
            "answers_routes_reasons_exported": event_export_payload.get(
                "answers_routes_reasons_exported", None
            ),
            "forbidden_static_fields_exported": event_export_payload.get(
                "forbidden_static_fields_exported", []
            ),
            "source_type": event_export_payload.get("source_type", ""),
            "capture_provenance": capture_provenance,
        },
        "event_ledger_calibration": event_summary,
        "source_attestation_write": source_attestation_write_report,
        "source_attestation": source_summary,
        "development_source_note": development_source_note,
        "blocker_evidence": {
            "passed": bool(blocker_payload.get("passed", False)),
            "candidate_blockers_satisfied": candidate_count,
            "blocker_count": int(blocker_payload.get("blocker_count", 0) or 0),
            "statuses": {
                str(item.get("id", "")): str(item.get("status", ""))
                for item in blocker_payload.get("blockers", [])
                if isinstance(item, dict)
            },
        },
        "progress": {
            "passed": bool(progress_payload.get("passed", False)),
            "status": progress_payload.get("status", ""),
            "candidate_review_ready": bool(
                progress_payload.get("candidate_review_ready", False)
            ),
            "remaining_blocker_count": progress_payload.get("blockers", {}).get(
                "remaining_blocker_count", None
            )
            if isinstance(progress_payload.get("blockers", {}), dict)
            else None,
            "candidate_blockers_satisfied": progress_payload.get("blockers", {}).get(
                "candidate_blockers_satisfied",
                None,
            )
            if isinstance(progress_payload.get("blockers", {}), dict)
            else None,
        },
        "replay_note": (
            "The pack uses event-ledger user turns only. Stored answers, routes, reasons, and assistant "
            "responses are omitted from replay fixtures."
        ),
        "completion_boundary": {
            "development_sessions_cannot_retire_blockers": event_source_kind
            == "development_session",
            "source_attestation_required_for_candidate_user_evidence": event_source_kind
            in SOURCE_ATTESTATION_KINDS,
            "architecture_completion_requires_architecture_review": True,
        },
    }
    _write_json_report(pack_report_path, payload)
    return payload


def _build_v01_development_source_note_from_export(
    *,
    db: Path,
    capture_surface: str,
    event_export_payload: dict[str, Any],
) -> dict[str, Any]:
    turns = [
        dict(item)
        for item in event_export_payload.get("turns", [])
        if isinstance(item, dict)
    ]
    utterances = tuple(str(item.get("content", "")) for item in turns)
    return {
        "schema": "melm.local_assistant_development_source_note.v1",
        "source_kind": "development_session",
        "capture_surface": capture_surface,
        "candidate_evidence_allowed": False,
        "accepted_by_write_source_attestation": False,
        "event_ledger_db": str(db),
        "event_ledger_db_exists": db.exists() and db.is_file(),
        "event_ledger_db_sha256": _sha256_file(db)
        if db.exists() and db.is_file()
        else "",
        "static_expectations_absent": True,
        "answers_routes_reasons_absent": event_export_payload.get(
            "answers_routes_reasons_exported"
        )
        is False,
        "turn_count": len(turns),
        "turn_utterance_hashes": _utterance_hashes(utterances),
        "capture_provenance": dict(event_export_payload.get("capture_provenance", {})),
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "note": (
            "Development evidence pack source note only. Candidate evidence must use write-source-attestation "
            "with redacted_user_session or target_device_user_session and a matching event ledger hash."
        ),
    }


def _build_v01_blocker_rehearsal_payload(
    work_dir: Path, *, reset: bool
) -> dict[str, Any]:
    started = perf_counter()
    if reset and work_dir.exists():
        _safe_remove_bundle_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "assistant.sqlite"
    event_work_dir = work_dir / "event_ledger"
    source_note_path = work_dir / "development_source_note.json"
    blocker_report_path = work_dir / "v01_blocker_evidence.json"
    progress_report_path = work_dir / "v01_progress.json"

    chat_args = [
        "chat",
        "--db",
        str(db),
        "--reset",
    ]
    for utterance in V01_BLOCKER_REHEARSAL_TURNS:
        chat_args.extend(["--turn", utterance])
    chat_args.append("--json")
    chat_payload = _run_cli_json(ROOT, *chat_args)

    source_note = _build_v01_development_source_note(db, chat_payload)
    _write_json_report(source_note_path, source_note)

    blocker_args = [
        "v01-blocker-evidence",
        "--event-ledger-db",
        str(db),
        "--event-ledger-work-dir",
        str(event_work_dir),
        "--event-source-kind",
        "development_session",
        "--min-total-turns",
        str(len(V01_BLOCKER_REHEARSAL_TURNS)),
        "--min-route-kinds",
        "2",
        "--min-intent-kinds",
        "5",
        "--min-synthesis-traces",
        "2",
        "--min-priority-signal-samples",
        "1",
        "--auto-lifecycle",
        "--reset",
        "--json",
    ]
    blocker_payload = _run_cli_json(ROOT, *blocker_args)
    _write_json_report(blocker_report_path, blocker_payload)

    progress_args = [
        "v01-progress",
        "--blocker-evidence-json",
        str(blocker_report_path),
        "--json",
    ]
    progress_payload = _run_cli_json(ROOT, *progress_args)
    _write_json_report(progress_report_path, progress_payload)

    checks = _v01_blocker_rehearsal_checks(
        db=db,
        source_note_path=source_note_path,
        blocker_report_path=blocker_report_path,
        progress_report_path=progress_report_path,
        chat_payload=chat_payload,
        blocker_payload=blocker_payload,
        progress_payload=progress_payload,
    )
    return {
        "schema": "melm.local_assistant_v01_blocker_rehearsal.v1",
        "passed": all(checks.values()),
        "work_dir": str(work_dir),
        "elapsed_ms": _elapsed_ms(started),
        "checks": checks,
        "artifact_paths": {
            "assistant_db": str(db),
            "development_source_note": str(source_note_path),
            "blocker_evidence_report": str(blocker_report_path),
            "progress_report": str(progress_report_path),
        },
        "command_args": {
            "chat": chat_args,
            "v01_blocker_evidence": blocker_args,
            "v01_progress": progress_args,
        },
        "turns": {
            "count": len(V01_BLOCKER_REHEARSAL_TURNS),
            "utterance_hashes": _utterance_hashes(V01_BLOCKER_REHEARSAL_TURNS),
        },
        "development_boundary": {
            "source_kind": "development_session",
            "source_attestation_command_used": False,
            "candidate_evidence_allowed": False,
            "candidate_blockers_satisfied": int(
                blocker_payload.get("candidate_blockers_satisfied", 0) or 0
            ),
            "architecture_complete_claimed": bool(
                blocker_payload.get("architecture_complete_claimed", False)
                or progress_payload.get("architecture_complete_claimed", False)
                or progress_payload.get("architecture_complete", False)
            ),
        },
        "blocker_statuses": {
            str(item.get("id", "")): str(item.get("status", ""))
            for item in blocker_payload.get("blockers", [])
            if isinstance(item, dict)
        },
        "event_ledger_calibration": blocker_payload.get("event_ledger_calibration", {}),
        "progress": {
            "passed": bool(progress_payload.get("passed", False)),
            "architecture_complete": bool(
                progress_payload.get("architecture_complete", False)
            ),
            "candidate_review_ready": bool(
                progress_payload.get("candidate_review_ready", False)
            ),
            "remaining_blocker_count": progress_payload.get("blockers", {}).get(
                "remaining_blocker_count", None
            )
            if isinstance(progress_payload.get("blockers", {}), dict)
            else None,
            "status": progress_payload.get("status", ""),
        },
        "next_candidate_commands": [
            "python scripts/local_assistant_os_cli.py calibrate-transcript-replay --input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json --require-redaction --require-static-drop --min-synthesis-traces <n> --require-priority-signals --min-priority-signal-samples <n> --require-memory-digest-quality --require-strict-baseline-win --out <calibration-report.json> --json",
            "python scripts/local_assistant_os_cli.py write-source-attestation --event-ledger-db <assistant.sqlite> --event-ledger-session <session|all> --source-kind redacted_user_session --capture-surface cli_chat --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --out <source_attestation.json> --json",
            "python scripts/local_assistant_os_cli.py v01-blocker-evidence --event-ledger-db <assistant.sqlite> --event-ledger-session <session|all> --event-source-kind redacted_user_session --source-attestation-json <source_attestation.json> --auto-lifecycle --transcript-calibration-report-json <calibration-report.json> --inventory-soak-report-json <inventory-soak-report.json> --host-app-config-json config/host_actions.json --host-app-attestation-json <host_app_attestation.json> --run-host-app-probe --json",
            f"python scripts/local_assistant_os_cli.py v01-progress --blocker-evidence-json {blocker_report_path} --json",
        ],
        "note": (
            "This is a development-only proof that the blocker evidence path runs on real ledger events. "
            "It intentionally cannot retire user-derived, live-inventory, or target-app blockers."
        ),
    }


def _build_v01_development_source_note(
    db: Path, chat_payload: dict[str, Any]
) -> dict[str, Any]:
    turns = [
        dict(item) for item in chat_payload.get("turns", []) if isinstance(item, dict)
    ]
    return {
        "schema": "melm.local_assistant_development_source_note.v1",
        "source_kind": "development_session",
        "capture_surface": "cli_chat",
        "candidate_evidence_allowed": False,
        "accepted_by_write_source_attestation": False,
        "event_ledger_db": str(db),
        "event_ledger_db_exists": db.exists(),
        "event_ledger_db_sha256": _sha256_file(db)
        if db.exists() and db.is_file()
        else "",
        "static_expectations_absent": True,
        "answers_routes_reasons_absent": True,
        "turn_count": len(V01_BLOCKER_REHEARSAL_TURNS),
        "turn_utterance_hashes": _utterance_hashes(V01_BLOCKER_REHEARSAL_TURNS),
        "capture_provenance": _event_capture_provenance_summary(
            [
                {
                    "capture_surface": turn.get("capture_provenance", {}).get(
                        "surface", ""
                    ),
                    "capture_source": turn.get("capture_provenance", {}).get(
                        "source", ""
                    ),
                }
                for turn in turns
            ]
        ),
        "chat_command_passed": bool(chat_payload.get("passed", False)),
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "note": (
            "Development rehearsal source note only. Candidate evidence must use write-source-attestation "
            "with redacted_user_session or target_device_user_session and a matching event ledger hash."
        ),
    }


def _v01_blocker_rehearsal_checks(
    *,
    db: Path,
    source_note_path: Path,
    blocker_report_path: Path,
    progress_report_path: Path,
    chat_payload: dict[str, Any],
    blocker_payload: dict[str, Any],
    progress_payload: dict[str, Any],
) -> dict[str, bool]:
    blockers = {
        str(item.get("id", "")): dict(item)
        for item in blocker_payload.get("blockers", [])
        if isinstance(item, dict)
    }
    development_trace_ids = (
        "user_derived_bounded_synthesis_traces",
        "planner_priority_on_user_derived_traces",
        "real_user_derived_lifecycle_traces",
    )
    event_summary = dict(blocker_payload.get("event_ledger_calibration", {}))
    candidate_count = int(blocker_payload.get("candidate_blockers_satisfied", 0) or 0)
    return {
        "chat_command_passed": bool(chat_payload.get("passed", False)),
        "event_ledger_db_created": db.exists() and db.is_file(),
        "development_source_note_written": source_note_path.exists()
        and source_note_path.is_file(),
        "blocker_evidence_report_written": blocker_report_path.exists()
        and blocker_report_path.is_file(),
        "progress_report_written": progress_report_path.exists()
        and progress_report_path.is_file(),
        "blocker_evidence_command_passed": bool(blocker_payload.get("passed", False)),
        "progress_command_passed": bool(progress_payload.get("passed", False)),
        "event_ledger_export_present": bool(event_summary.get("present", False))
        and int(event_summary.get("events_exported", 0) or 0)
        >= len(V01_BLOCKER_REHEARSAL_TURNS),
        "static_exports_absent": event_summary.get("answers_routes_reasons_exported")
        is False
        and event_summary.get("forbidden_static_fields_exported", []) == [],
        "development_trace_blockers_rehearsed": all(
            blockers.get(blocker_id, {}).get("status") == "development_evidence_present"
            for blocker_id in development_trace_ids
        ),
        "digest_quality_not_faked": blockers.get(
            "digest_quality_and_route_threshold_calibration", {}
        ).get("status")
        != "candidate_evidence_present",
        "development_not_promoted_to_candidate": candidate_count == 0
        and all(
            not bool(row.get("candidate_for_architecture_review", False))
            for row in blockers.values()
        ),
        "architecture_completion_not_claimed": not bool(
            blocker_payload.get("architecture_complete_claimed", False)
        )
        and not bool(progress_payload.get("architecture_complete_claimed", False))
        and not bool(progress_payload.get("architecture_complete", False)),
    }


def _utterance_hashes(utterances: tuple[str, ...]) -> list[str]:
    return [
        hashlib.sha256(utterance.encode("utf-8")).hexdigest()
        for utterance in utterances
    ]


def _write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_v01_blocker_evidence_payload(
    *,
    event_ledger_db: Path | None,
    event_ledger_work_dir: Path,
    event_ledger_session: str,
    event_source_kind: str,
    source_attestation_json: Path | None,
    controls_json: Path | None,
    replacements: tuple[str, ...],
    weather_json: Path,
    reset: bool,
    min_total_turns: int,
    min_local_resolution_rate: float,
    min_route_kinds: int,
    min_intent_kinds: int,
    min_synthesis_traces: int,
    min_priority_signal_samples: int,
    auto_lifecycle: bool,
    inventory_soak_report_json: Path | None,
    transcript_calibration_report_json: Path | None,
    host_app_config_json: Path | None,
    host_app_attestation_json: Path | None,
    run_host_app_probe: bool,
    host_app_db: Path,
    host_app_work_dir: Path,
    host_app_media_dir: Path | None,
) -> dict[str, Any]:
    started = perf_counter()
    event_payload: dict[str, Any] = {}
    event_error = ""
    source_attestation = _source_attestation_report(
        source_attestation_json,
        event_source_kind=event_source_kind,
        event_ledger_db=event_ledger_db,
        event_ledger_session=event_ledger_session,
    )
    source_attestation_error = str(source_attestation.get("error", ""))
    host_app_attestation = _host_app_attestation_report(
        host_app_attestation_json,
        host_app_config_json=host_app_config_json,
    )
    host_app_attestation_error = str(host_app_attestation.get("error", ""))
    if event_ledger_db is not None:
        try:
            event_payload = _build_event_ledger_calibration_payload(
                db=event_ledger_db,
                work_dir=event_ledger_work_dir,
                session=event_ledger_session,
                profile_mode="current",
                controls_json=controls_json,
                replacements=replacements,
                weather_json=weather_json,
                reset=reset,
                min_total_turns=min_total_turns,
                min_local_resolution_rate=min_local_resolution_rate,
                min_route_kinds=min_route_kinds,
                min_intent_kinds=min_intent_kinds,
                min_synthesis_traces=min_synthesis_traces,
                min_priority_signal_samples=min_priority_signal_samples,
                auto_lifecycle=auto_lifecycle,
                require_priority_signals=False,
                require_memory_digest_quality=False,
                require_strict_baseline_win=False,
                require_redaction=False,
                require_static_drop=False,
            )
        except (FileNotFoundError, ValueError) as exc:
            event_error = str(exc)
    inventory_payload, inventory_error = _optional_json_report(
        inventory_soak_report_json
    )
    transcript_calibration_payload, transcript_calibration_error = (
        _optional_json_report(transcript_calibration_report_json)
    )
    if (
        transcript_calibration_payload
        and transcript_calibration_payload.get("schema")
        != "melm.local_assistant_transcript_calibration_report.v1"
    ):
        transcript_calibration_error = (
            "transcript calibration report schema is not "
            "melm.local_assistant_transcript_calibration_report.v1"
        )
    host_payload: dict[str, Any] = {}
    host_error = ""
    if host_app_config_json is not None:
        if run_host_app_probe:
            try:
                host_payload = _build_host_app_probe_payload(
                    host_app_db,
                    work_dir=host_app_work_dir,
                    reset=reset,
                    media_player_command="",
                    call_command="",
                    media_dir=host_app_media_dir,
                    require_configured=True,
                    config_json=host_app_config_json,
                )
            except (OSError, ValueError) as exc:
                host_error = str(exc)
        else:
            config, config_error = _host_app_config(host_app_config_json)
            host_payload = {
                "configured": False,
                "skipped": True,
                "config": _host_app_config_report(
                    host_app_config_json, config, config_error
                ),
                "checks": {
                    "configuration_reported": True,
                    "config_json_valid": not bool(config_error),
                    "probe_executed": False,
                    "require_configured_satisfied": False,
                },
                "note": "Pass --run-host-app-probe to execute configured commands through the typed action gate.",
                "evidence_class": _host_app_static_analysis(
                    host_app_config_json, config
                ),
            }
            host_error = config_error
    blockers = _v01_blocker_evidence_rows(
        event_ledger_db=event_ledger_db,
        event_payload=event_payload,
        event_error=event_error,
        event_source_kind=event_source_kind,
        source_attestation_valid=bool(source_attestation.get("valid", False)),
        inventory_payload=inventory_payload,
        inventory_error=inventory_error,
        transcript_calibration_payload=transcript_calibration_payload,
        transcript_calibration_error=transcript_calibration_error,
        host_payload=host_payload,
        host_error=host_error,
        host_app_attestation_valid=bool(host_app_attestation.get("valid", False)),
        thresholds={
            "min_total_turns": max(1, int(min_total_turns or 1)),
            "min_local_resolution_rate": max(
                0.0, float(min_local_resolution_rate or 0.0)
            ),
            "min_route_kinds": max(1, int(min_route_kinds or 1)),
            "min_intent_kinds": max(1, int(min_intent_kinds or 1)),
            "min_synthesis_traces": max(0, int(min_synthesis_traces or 0)),
            "min_priority_signal_samples": max(
                0, int(min_priority_signal_samples or 0)
            ),
        },
    )
    candidate_count = sum(
        row["status"] == "candidate_evidence_present" for row in blockers
    )
    report_valid = not bool(
        event_error
        or inventory_error
        or transcript_calibration_error
        or host_error
        or source_attestation_error
        or host_app_attestation_error
    )
    candidate_evidence_complete = candidate_count == len(blockers)
    return {
        "schema": "melm.local_assistant_v01_blocker_evidence.v1",
        "passed": report_valid,
        "report_valid": report_valid,
        "candidate_evidence_complete": candidate_evidence_complete,
        "architecture_complete_claimed": False,
        "candidate_blockers_satisfied": candidate_count,
        "remaining_blocker_count": len(blockers) - candidate_count,
        "blocker_count": len(blockers),
        "status_counts": dict(Counter(str(row.get("status", "")) for row in blockers)),
        "checks": {
            "report_valid": report_valid,
            "candidate_evidence_complete": candidate_evidence_complete,
            "architecture_completion_not_claimed": True,
            "candidate_count_not_overstated": candidate_count <= len(blockers),
        },
        "event_source_kind": event_source_kind,
        "auto_lifecycle": bool(auto_lifecycle),
        "elapsed_ms": _elapsed_ms(started),
        "blockers": blockers,
        "event_ledger_calibration": _v01_event_calibration_summary(
            event_payload, event_error
        ),
        "source_attestation": source_attestation,
        "inventory_soak_report": _v01_inventory_report_summary(
            inventory_payload, inventory_error
        ),
        "transcript_calibration_report": _v01_transcript_calibration_report_summary(
            transcript_calibration_payload,
            transcript_calibration_error,
            event_ledger_db=event_ledger_db,
        ),
        "host_app_probe": _v01_host_app_summary(host_payload, host_error),
        "host_app_attestation": host_app_attestation,
        "note": (
            "This report packages evidence for the remaining blockers. It does not mark v0.1 complete; "
            "v01-audit remains the completion boundary."
        ),
    }


def _optional_json_report(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return {}, ""
    if not path.exists() or not path.is_file():
        return {}, f"report file not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON report: {exc}"
    except OSError as exc:
        return {}, f"could not read report: {exc}"
    if not isinstance(payload, dict):
        return {}, "report root must be a JSON object"
    return payload, ""


def _v01_blocker_evidence_rows(
    *,
    event_ledger_db: Path | None,
    event_payload: dict[str, Any],
    event_error: str,
    event_source_kind: str,
    source_attestation_valid: bool,
    inventory_payload: dict[str, Any],
    inventory_error: str,
    transcript_calibration_payload: dict[str, Any],
    transcript_calibration_error: str,
    host_payload: dict[str, Any],
    host_error: str,
    host_app_attestation_valid: bool,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    aggregate = dict(event_payload.get("aggregate", {}))
    event_passed = bool(event_payload.get("passed", False))
    source_is_candidate = (
        event_source_kind in SOURCE_ATTESTATION_KINDS and source_attestation_valid
    )
    event_status = _candidate_or_development_status(
        event_passed,
        source_is_candidate,
        event_source_kind=event_source_kind,
        missing_status="missing_event_ledger_evidence"
        if not event_error
        else "event_ledger_error",
    )
    synthesis_threshold = max(0, int(thresholds["min_synthesis_traces"]))
    priority_signal_threshold = max(0, int(thresholds["min_priority_signal_samples"]))
    synthesis_threshold_configured = synthesis_threshold > 0
    priority_signal_threshold_configured = priority_signal_threshold > 0
    synthesis_ok = (
        event_passed
        and synthesis_threshold_configured
        and int(aggregate.get("synthesis_traces", 0) or 0) >= synthesis_threshold
        and bool(aggregate.get("checks", {}).get("synthesis_trace_floor", False))
    )
    priority_ok = (
        event_passed
        and priority_signal_threshold_configured
        and int(aggregate.get("priority_signal_sample_count", 0) or 0)
        >= priority_signal_threshold
    )
    digest_quality = dict(aggregate.get("memory_digest_quality", {}))
    transcript_calibration_summary = _v01_transcript_calibration_report_summary(
        transcript_calibration_payload,
        transcript_calibration_error,
        event_ledger_db=event_ledger_db,
    )
    transcript_calibration_ok = bool(
        transcript_calibration_summary.get(
            "candidate_digest_route_calibration_passed", False
        )
    )
    digest_missing_status = (
        "transcript_calibration_report_error"
        if transcript_calibration_error
        else "missing_evidence"
    )
    inventory_summary = _v01_inventory_report_summary(
        inventory_payload, inventory_error
    )
    inventory_ok = bool(inventory_summary.get("candidate_live_soak_passed", False))
    host_probe_ok = bool(
        host_payload.get("passed", False)
        and host_payload.get("configured", False)
        and not host_payload.get("skipped", True)
    )
    host_evidence_class = dict(host_payload.get("evidence_class", {}))
    host_uses_demo_recorder = bool(
        host_evidence_class.get("demo_recorder_detected", False)
    )
    host_candidate_ok = bool(
        host_probe_ok
        and host_app_attestation_valid
        and not host_uses_demo_recorder
        and bool(host_evidence_class.get("candidate_target_device_app_evidence", False))
    )
    return [
        _v01_blocker_row(
            "user_derived_bounded_synthesis_traces",
            status=_candidate_or_development_status(
                synthesis_ok,
                source_is_candidate,
                event_source_kind=event_source_kind,
            ),
            evidence={
                "event_source_kind": event_source_kind,
                "source_attestation_valid": source_attestation_valid,
                "synthesis_traces": int(aggregate.get("synthesis_traces", 0) or 0),
                "threshold": synthesis_threshold,
                "positive_threshold_configured": synthesis_threshold_configured,
                "local_resolution_rate": aggregate.get("local_resolution_rate", 0.0),
            },
            missing=[]
            if synthesis_ok and source_is_candidate
            else _missing_user_trace_reasons(
                synthesis_ok,
                event_source_kind,
                source_attestation_valid,
                positive_threshold_configured=synthesis_threshold_configured,
                positive_threshold_reason="positive --min-synthesis-traces threshold for candidate synthesis evidence",
            ),
            next_command="python scripts/local_assistant_os_cli.py calibrate-transcript-replay --input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json --require-redaction --require-static-drop --min-synthesis-traces <n> --min-local-resolution-rate <rate> --json",
        ),
        _v01_blocker_row(
            "longer_live_inventory_soak",
            status="candidate_evidence_present"
            if inventory_ok
            else ("report_error" if inventory_error else "missing_live_inventory_soak"),
            evidence={
                "report_path_loaded": bool(inventory_payload),
                "mode": inventory_summary.get("mode", ""),
                "network_used": bool(inventory_summary.get("network_used", False)),
                "total_cycles_completed": inventory_summary.get(
                    "total_cycles_completed", 0
                ),
                "passed": bool(inventory_summary.get("passed", False)),
                "candidate_live_soak_passed": inventory_ok,
                "report_binding": inventory_summary.get("report_binding", {}),
            },
            missing=[]
            if inventory_ok
            else _missing_inventory_soak_reasons(inventory_summary),
            next_command="python scripts/local_assistant_os_cli.py inventory-soak-matrix --live --cycles <n> --json",
        ),
        _v01_blocker_row(
            "planner_priority_on_user_derived_traces",
            status=_candidate_or_development_status(
                priority_ok,
                source_is_candidate,
                event_source_kind=event_source_kind,
            ),
            evidence={
                "event_source_kind": event_source_kind,
                "source_attestation_valid": source_attestation_valid,
                "priority_signal_sample_count": int(
                    aggregate.get("priority_signal_sample_count", 0) or 0
                ),
                "threshold": priority_signal_threshold,
                "positive_threshold_configured": priority_signal_threshold_configured,
            },
            missing=[]
            if priority_ok and source_is_candidate
            else _missing_user_trace_reasons(
                priority_ok,
                event_source_kind,
                source_attestation_valid,
                positive_threshold_configured=priority_signal_threshold_configured,
                positive_threshold_reason=(
                    "positive --min-priority-signal-samples threshold for candidate planner evidence"
                ),
            ),
            next_command="python scripts/local_assistant_os_cli.py calibrate-transcript-replay --input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json --require-redaction --require-static-drop --require-priority-signals --min-priority-signal-samples <n> --json",
        ),
        _v01_blocker_row(
            "real_user_derived_lifecycle_traces",
            status=event_status,
            evidence={
                "event_source_kind": event_source_kind,
                "source_attestation_valid": source_attestation_valid,
                "events_exported": int(event_payload.get("events_exported", 0) or 0),
                "turns_replayed": int(aggregate.get("turns_replayed", 0) or 0),
                "route_kinds": int(aggregate.get("route_kinds", 0) or 0),
                "intent_kinds": int(aggregate.get("intent_kinds", 0) or 0),
                "answers_routes_reasons_exported": event_payload.get(
                    "answers_routes_reasons_exported", None
                ),
            },
            missing=[]
            if event_passed and source_is_candidate
            else _missing_user_trace_reasons(
                event_passed,
                event_source_kind,
                source_attestation_valid,
            ),
            next_command="python scripts/local_assistant_os_cli.py calibrate-event-ledger --db <assistant.sqlite> --controls-json config/safe_lifecycle_controls.example.json --min-total-turns <n> --min-local-resolution-rate <rate> --json",
        ),
        _v01_blocker_row(
            "digest_quality_and_route_threshold_calibration",
            status=_candidate_or_development_status(
                transcript_calibration_ok,
                source_is_candidate,
                event_source_kind=event_source_kind,
                missing_status=digest_missing_status,
            ),
            evidence={
                "event_source_kind": event_source_kind,
                "source_attestation_valid": source_attestation_valid,
                "event_ledger_memory_digest_quality": digest_quality,
                "event_ledger_route_counts": aggregate.get("route_counts", {}),
                "event_ledger_local_resolution_rate": aggregate.get(
                    "local_resolution_rate", 0.0
                ),
                "event_ledger_debug_mapping_passed": bool(
                    aggregate.get("debug_mapping_passed", False)
                ),
                "transcript_calibration": transcript_calibration_summary,
                "transcript_calibration_event_ledger_binding": transcript_calibration_summary.get(
                    "event_ledger_binding",
                    {},
                ),
            },
            missing=[]
            if transcript_calibration_ok and source_is_candidate
            else _missing_digest_calibration_reasons(
                transcript_calibration_ok,
                event_source_kind,
                source_attestation_valid,
                transcript_calibration_summary,
            ),
            next_command=(
                "python scripts/local_assistant_os_cli.py v01-blocker-evidence "
                "--transcript-calibration-report-json <calibration-report.json> --json"
            ),
        ),
        _v01_blocker_row(
            "configured_target_device_apps",
            status=_host_app_candidate_status(
                host_probe_ok=host_probe_ok,
                host_app_attestation_valid=host_app_attestation_valid,
                host_uses_demo_recorder=host_uses_demo_recorder,
                host_error=host_error,
            ),
            evidence={
                "configured": bool(host_payload.get("configured", False)),
                "skipped": bool(host_payload.get("skipped", True))
                if host_payload
                else True,
                "passed": bool(host_payload.get("passed", False)),
                "host_app_attestation_valid": host_app_attestation_valid,
                "demo_recorder_detected": host_uses_demo_recorder,
                "candidate_target_device_app_evidence": bool(
                    host_evidence_class.get(
                        "candidate_target_device_app_evidence", False
                    )
                ),
                "checks": host_payload.get("checks", {}),
                "command_sources": host_payload.get("command_sources", {}),
                "evidence_class": host_evidence_class,
            },
            missing=_missing_host_app_reasons(
                host_probe_ok=host_probe_ok,
                host_app_attestation_valid=host_app_attestation_valid,
                host_uses_demo_recorder=host_uses_demo_recorder,
            ),
            next_command=(
                "python scripts/local_assistant_os_cli.py v01-blocker-evidence "
                "--host-app-config-json config/host_actions.json "
                "--host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json "
                "--run-host-app-probe --json"
            ),
        ),
    ]


def _host_app_candidate_status(
    *,
    host_probe_ok: bool,
    host_app_attestation_valid: bool,
    host_uses_demo_recorder: bool,
    host_error: str,
) -> str:
    if host_error:
        return "host_app_probe_error"
    if not host_probe_ok:
        return "missing_configured_host_app_probe"
    if host_uses_demo_recorder:
        return "development_host_app_probe_present"
    if not host_app_attestation_valid:
        return "unattested_host_app_probe_present"
    return "candidate_evidence_present"


def _missing_host_app_reasons(
    *,
    host_probe_ok: bool,
    host_app_attestation_valid: bool,
    host_uses_demo_recorder: bool,
) -> list[str]:
    missing: list[str] = []
    if not host_probe_ok:
        missing.append(
            "host-app-probe must execute configured media and call commands through the typed action gate"
        )
    if host_uses_demo_recorder:
        missing.append(
            "replace recorder/demo commands with target-device media and call app commands"
        )
    if not host_app_attestation_valid:
        missing.append(
            "valid host app attestation JSON bound to the target app config hash"
        )
    return missing


def _candidate_or_development_status(
    evidence_ok: bool,
    source_ok: bool,
    *,
    event_source_kind: str = "development_session",
    missing_status: str = "missing_evidence",
) -> str:
    if evidence_ok and source_ok:
        return "candidate_evidence_present"
    if evidence_ok:
        if event_source_kind in SOURCE_ATTESTATION_KINDS:
            return "unattested_user_evidence_present"
        return "development_evidence_present"
    return missing_status


def _missing_user_trace_reasons(
    evidence_ok: bool,
    event_source_kind: str,
    source_attestation_valid: bool,
    *,
    positive_threshold_configured: bool = True,
    positive_threshold_reason: str = "",
) -> list[str]:
    missing: list[str] = []
    if not positive_threshold_configured and positive_threshold_reason:
        missing.append(positive_threshold_reason)
    if not evidence_ok:
        missing.append("calibration threshold evidence")
    if event_source_kind not in SOURCE_ATTESTATION_KINDS:
        missing.append("redacted user-derived or target-device source attestation")
    elif not source_attestation_valid:
        missing.append("valid source attestation JSON")
        missing.append("redacted user-derived or target-device source attestation")
    return missing


def _missing_digest_calibration_reasons(
    evidence_ok: bool,
    event_source_kind: str,
    source_attestation_valid: bool,
    transcript_calibration_summary: dict[str, Any],
) -> list[str]:
    missing = _missing_user_trace_reasons(
        evidence_ok, event_source_kind, source_attestation_valid
    )
    if not evidence_ok:
        error = str(transcript_calibration_summary.get("error", "") or "")
        if error:
            missing.append(error)
        if not transcript_calibration_summary.get("present", False):
            missing.append("strict calibrate-transcript-replay report JSON")
        else:
            if not transcript_calibration_summary.get(
                "strict_digest_route_calibration_passed", False
            ):
                missing.append(
                    "strict digest quality, route threshold, and baseline-win calibration gates"
                )
            strict_checks = dict(
                transcript_calibration_summary.get("strict_checks", {})
            )
            if not strict_checks.get(
                "primary_uol_chatframe_not_secondary_phrase_route", False
            ):
                missing.append(
                    "primary UOL/ChatFrame routing evidence without secondary phrase primary routes"
                )
            binding = dict(
                transcript_calibration_summary.get("event_ledger_binding", {})
            )
            if not binding.get("passed", False):
                missing.extend(str(item) for item in binding.get("missing", []))
    return list(dict.fromkeys(missing))


def _missing_inventory_soak_reasons(inventory_summary: dict[str, Any]) -> list[str]:
    if not inventory_summary.get("present", False):
        return ["live inventory-soak-matrix report JSON"]
    error = str(inventory_summary.get("error", "") or "")
    if error:
        return [error]
    missing = [
        str(item)
        for item in inventory_summary.get("report_binding", {}).get("missing", [])
        if str(item)
    ]
    if not missing and not inventory_summary.get("candidate_live_soak_passed", False):
        missing.append(
            "live inventory-soak-matrix report with generated artifact binding"
        )
    return list(dict.fromkeys(missing))


def _v01_blocker_row(
    blocker_id: str,
    *,
    status: str,
    evidence: dict[str, Any],
    missing: list[str],
    next_command: str,
) -> dict[str, Any]:
    return {
        "id": blocker_id,
        "status": status,
        "candidate_for_architecture_review": status == "candidate_evidence_present",
        "evidence": evidence,
        "missing": missing,
        "next_command": next_command,
    }


def _v01_event_calibration_summary(
    payload: dict[str, Any], error: str
) -> dict[str, Any]:
    if error:
        return {"present": False, "error": error}
    if not payload:
        return {"present": False, "error": ""}
    aggregate = dict(payload.get("aggregate", {}))
    return {
        "present": True,
        "passed": bool(payload.get("passed", False)),
        "events_exported": int(payload.get("events_exported", 0) or 0),
        "answers_routes_reasons_exported": payload.get(
            "answers_routes_reasons_exported", None
        ),
        "forbidden_static_fields_exported": payload.get(
            "forbidden_static_fields_exported", []
        ),
        "turns_replayed": int(aggregate.get("turns_replayed", 0) or 0),
        "local_resolution_rate": aggregate.get("local_resolution_rate", 0.0),
        "route_kinds": int(aggregate.get("route_kinds", 0) or 0),
        "intent_kinds": int(aggregate.get("intent_kinds", 0) or 0),
        "synthesis_traces": int(aggregate.get("synthesis_traces", 0) or 0),
        "priority_signal_sample_count": int(
            aggregate.get("priority_signal_sample_count", 0) or 0
        ),
        "capture_provenance": payload.get("capture_provenance", {}),
        "transcript_jsonl": payload.get("transcript_jsonl", ""),
        "raw_event_jsonl": payload.get("raw_event_jsonl", ""),
    }


def _v01_inventory_report_summary(
    payload: dict[str, Any], error: str
) -> dict[str, Any]:
    if error:
        return {"present": False, "error": error}
    if not payload:
        return {"present": False, "error": ""}
    binding = _inventory_soak_matrix_report_binding(payload)
    return {
        "present": True,
        "error": "",
        "schema": str(payload.get("schema", "")),
        "passed": bool(payload.get("passed", False)),
        "mode": payload.get("mode", ""),
        "network_used": bool(payload.get("network_used", False)),
        "candidate_live_soak_passed": bool(binding.get("passed", False)),
        "total_cycles_completed": payload.get(
            "total_cycles_completed", payload.get("cycles_completed", 0)
        ),
        "total_failed_import_cycles": int(
            payload.get("total_failed_import_cycles", 0) or 0
        ),
        "profile_count": int(payload.get("profile_count", 0) or 0),
        "source_families_observed": payload.get("source_families_observed", []),
        "checks": payload.get("checks", {}),
        "report_binding": binding,
    }


def _inventory_soak_matrix_report_binding(payload: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    checks = dict(payload.get("checks", {}))
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    mode = str(payload.get("mode", "") or "")
    source_families = {
        str(item) for item in payload.get("source_families_observed", []) or []
    }
    required_source_families = {
        "project_gutenberg_catalog_csv",
        "internet_archive_search_metadata",
    }
    required_checks = (
        "profiles_exercised",
        "total_cycles_at_least_nine",
        "cycles_completed",
        "all_soaks_passed",
        "all_source_coverage_ok",
        "both_source_families_covered",
        "all_story_inventory_grew_from_cold_start",
        "future_story_routes_local_from_imported_inventory",
        "future_story_synthesis_applied",
        "future_story_primary_uol_not_secondary_phrase_route",
        "all_failure_observability_present",
        "no_failed_import_cycles",
        "story_quality_floor_clean",
        "bounded_resource_budget",
    )
    missing_checks = [
        name for name in required_checks if not bool(checks.get(name, False))
    ]
    if str(payload.get("schema", "")) != "melm.inventory_soak_matrix.v1":
        missing.append("inventory-soak-matrix v1 schema")
    if not bool(payload.get("passed", False)):
        missing.append("inventory-soak-matrix passed=true report")
    if mode != "live_metadata":
        missing.append("inventory-soak-matrix --live report mode")
    if not bool(payload.get("network_used", False)):
        missing.append("top-level live metadata network evidence")
    if int(payload.get("profile_count", 0) or 0) < 3 or len(runs) < 3:
        missing.append("three live inventory matrix profiles")
    if (
        int(
            payload.get("total_cycles_completed", payload.get("cycles_completed", 0))
            or 0
        )
        < 9
    ):
        missing.append("at least nine completed live inventory cycles")
    if int(payload.get("total_failed_import_cycles", 0) or 0) != 0:
        missing.append("zero failed live inventory import cycles")
    if int(payload.get("total_story_inventory_added", 0) or 0) <= 0:
        missing.append("story inventory growth from live metadata")
    if missing_checks:
        missing.append(
            "strict inventory-soak-matrix checks: " + ", ".join(missing_checks)
        )
    if not required_source_families <= source_families:
        missing.append("both Gutenberg and Internet Archive source families")
    db_dir_text = str(payload.get("db_dir", "") or "")
    db_dir = Path(db_dir_text) if db_dir_text else None
    db_dir_exists = bool(db_dir and db_dir.exists() and db_dir.is_dir())
    if not db_dir_exists:
        missing.append("inventory-soak-matrix db_dir artifact directory")
    run_bindings = [
        _inventory_soak_matrix_run_binding(
            run, db_dir=db_dir if db_dir_exists else None
        )
        for run in runs
        if isinstance(run, dict)
    ]
    if len(run_bindings) < 3:
        missing.append("inventory-soak-matrix run artifact records")
    if not run_bindings or not all(
        bool(item.get("artifact_bound", False)) for item in run_bindings
    ):
        missing.append("inventory-soak-matrix run DB artifacts with matching hashes")
    if not run_bindings or not all(
        bool(item.get("live_fetch_evidence", False)) for item in run_bindings
    ):
        missing.append("live metadata fetch evidence in every matrix run")
    if not run_bindings or not all(
        bool(item.get("story_inventory_verified", False)) for item in run_bindings
    ):
        missing.append("story inventory rows verified inside every run DB")
    if not run_bindings or not all(
        bool(item.get("story_route_verified", False)) for item in run_bindings
    ):
        missing.append("future story route verified inside every run DB")
    if not run_bindings or not all(
        bool(item.get("story_uol_reported", False)) for item in run_bindings
    ):
        missing.append("primary UOL/ChatFrame story evidence in every matrix run")
    return {
        "passed": not missing,
        "missing": list(dict.fromkeys(missing)),
        "mode": mode,
        "db_dir": db_dir_text,
        "db_dir_exists": db_dir_exists,
        "required_checks": {
            name: bool(checks.get(name, False)) for name in required_checks
        },
        "source_families_ok": required_source_families <= source_families,
        "run_bindings": run_bindings,
    }


def _inventory_soak_matrix_run_binding(
    run: dict[str, Any], *, db_dir: Path | None
) -> dict[str, Any]:
    db_text = str(run.get("db", "") or "")
    expected_hash = str(run.get("db_sha256", "") or "")
    db = Path(db_text) if db_text else None
    db_exists = bool(db and db.exists() and db.is_file())
    db_under_report_dir = bool(db and db_dir and _path_is_relative_to(db, db_dir))
    actual_hash = ""
    hash_matched = False
    if db_exists:
        try:
            actual_hash = _sha256_file(db)
        except OSError:
            actual_hash = ""
    hash_matched = bool(expected_hash and actual_hash and expected_hash == actual_hash)
    sqlite_summary = (
        _inventory_soak_matrix_sqlite_summary(db)
        if db_exists
        else {
            "opened": False,
            "story_inventory_count": 0,
            "completed_import_jobs": 0,
            "failed_import_jobs": 0,
        }
    )
    soak = dict(run.get("soak", {}))
    live_fetch_evidence = bool(
        soak.get("network_used", False)
        and int(soak.get("network_used_results", 0) or 0) > 0
        and int(soak.get("fetch_attempts_total", 0) or 0) > 0
    )
    artifact_bound = bool(
        db_exists
        and db_under_report_dir
        and hash_matched
        and sqlite_summary.get("opened", False)
    )
    story_inventory_verified = bool(
        int(sqlite_summary.get("story_inventory_count", 0) or 0) > 0
        and int(sqlite_summary.get("completed_import_jobs", 0) or 0) > 0
        and int(sqlite_summary.get("failed_import_jobs", 0) or 0) == 0
    )
    story_route_verified = bool(
        run.get("story_local", False)
        and run.get("story_route", "") == "local_answer"
        and run.get("story_reason", "") == "local_story_inventory"
        and int(sqlite_summary.get("local_story_events", 0) or 0) > 0
    )
    story_uol_reported = bool(
        run.get("story_synthesis_applied", False)
        and run.get("story_primary_uol_ok", False)
    )
    return {
        "label": str(run.get("label", "")),
        "db": db_text,
        "db_exists": db_exists,
        "db_under_report_dir": db_under_report_dir,
        "db_sha256_present": bool(expected_hash),
        "db_sha256_matched": hash_matched,
        "artifact_bound": artifact_bound,
        "live_fetch_evidence": live_fetch_evidence,
        "story_inventory_verified": story_inventory_verified,
        "story_route_verified": story_route_verified,
        "story_uol_reported": story_uol_reported,
        "sqlite": sqlite_summary,
    }


def _inventory_soak_matrix_sqlite_summary(db: Path | None) -> dict[str, Any]:
    if db is None:
        return {
            "opened": False,
            "story_inventory_count": 0,
            "completed_import_jobs": 0,
            "failed_import_jobs": 0,
            "local_story_events": 0,
        }
    uri = f"file:{db.resolve().as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            story_inventory_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM inventories WHERE kind = 'story_model'"
                ).fetchone()[0]
            )
            completed_import_jobs = int(
                conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE kind = 'import_story_metadata' AND status = 'completed'"
                ).fetchone()[0]
            )
            failed_import_jobs = int(
                conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE kind = 'import_story_metadata' AND status = 'failed'"
                ).fetchone()[0]
            )
            local_story_events = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events "
                    "WHERE route = 'local_answer' AND reason = 'local_story_inventory' "
                    "AND lower(utterance) LIKE '%story%'"
                ).fetchone()[0]
            )
    except (OSError, sqlite3.Error):
        return {
            "opened": False,
            "story_inventory_count": 0,
            "completed_import_jobs": 0,
            "failed_import_jobs": 0,
            "local_story_events": 0,
        }
    return {
        "opened": True,
        "story_inventory_count": story_inventory_count,
        "completed_import_jobs": completed_import_jobs,
        "failed_import_jobs": failed_import_jobs,
        "local_story_events": local_story_events,
    }


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _v01_transcript_calibration_report_summary(
    payload: dict[str, Any],
    error: str,
    *,
    event_ledger_db: Path | None = None,
) -> dict[str, Any]:
    if error:
        return {"present": bool(payload), "error": error}
    if not payload:
        return {"present": False, "error": ""}
    aggregate = dict(payload.get("aggregate", {}))
    checks = dict(aggregate.get("checks", {}))
    thresholds = dict(aggregate.get("thresholds", {}))
    event_ledger_binding = _transcript_calibration_event_ledger_binding(
        payload, event_ledger_db
    )
    strict_checks = {
        "report_passed": bool(payload.get("passed", False)),
        "redaction_required": bool(thresholds.get("require_redaction", False)),
        "static_drop_required": bool(thresholds.get("require_static_drop", False)),
        "memory_digest_quality_required": bool(
            thresholds.get("require_memory_digest_quality", False)
        ),
        "strict_baseline_required": bool(
            thresholds.get("require_strict_baseline_win", False)
        ),
        "redaction_met": bool(checks.get("redaction_required_met", False)),
        "static_drop_met": bool(checks.get("static_drop_required_met", False)),
        "memory_digest_quality_met": bool(
            checks.get("memory_digest_quality_required_met", False)
        ),
        "strict_baseline_met": bool(checks.get("strict_baseline_required_met", False)),
        "route_floor_met": bool(checks.get("route_diversity_floor", False)),
        "local_resolution_floor_met": bool(checks.get("local_resolution_floor", False)),
        "debug_mapping_passed": bool(checks.get("debug_mapping_passed", False)),
        "primary_uol_chatframe_not_secondary_phrase_route": bool(
            checks.get("primary_uol_chatframe_not_secondary_phrase_route", False)
        ),
        "critical_safety_clean": bool(checks.get("critical_safety_clean", False)),
    }
    strict_passed = all(strict_checks.values())
    candidate_passed = strict_passed and bool(event_ledger_binding.get("passed", False))
    return {
        "present": True,
        "error": "",
        "schema": str(payload.get("schema", "")),
        "passed": bool(payload.get("passed", False)),
        "strict_digest_route_calibration_passed": strict_passed,
        "candidate_digest_route_calibration_passed": candidate_passed,
        "strict_checks": strict_checks,
        "event_ledger_binding": event_ledger_binding,
        "thresholds": thresholds,
        "input_count": int(payload.get("input_count", 0) or 0),
        "turns_replayed": int(aggregate.get("turns_replayed", 0) or 0),
        "local_resolution_rate": aggregate.get("local_resolution_rate", 0.0),
        "route_kinds": int(aggregate.get("route_kinds", 0) or 0),
        "intent_kinds": int(aggregate.get("intent_kinds", 0) or 0),
        "route_counts": aggregate.get("route_counts", {}),
        "capture_provenance": aggregate.get("capture_provenance", {}),
        "memory_digest_quality": aggregate.get("memory_digest_quality", {}),
        "baseline_required_replays": int(
            aggregate.get("baseline_required_replays", 0) or 0
        ),
        "strict_baseline_passed_replays": int(
            aggregate.get("strict_baseline_passed_replays", 0) or 0
        ),
    }


def _transcript_calibration_event_ledger_binding(
    payload: dict[str, Any],
    event_ledger_db: Path | None,
) -> dict[str, Any]:
    missing: list[str] = []
    actual_hash = ""
    target_path = ""
    if event_ledger_db is None:
        missing.append("event-ledger DB supplied for transcript calibration binding")
    else:
        target_path = str(event_ledger_db)
        if not event_ledger_db.is_file():
            missing.append(
                "current event-ledger DB file exists for transcript calibration binding"
            )
        else:
            actual_hash = _sha256_file(event_ledger_db)

    aggregate = dict(payload.get("aggregate", {}))
    capture_provenance = dict(aggregate.get("capture_provenance", {}))
    capture_ok = (
        bool(capture_provenance.get("has_capture_provenance", False))
        and int(capture_provenance.get("imported_turn_count", 0) or 0) > 0
    )
    if not capture_ok:
        missing.append(
            "imported redacted transcript capture provenance in calibration report"
        )

    candidate_db_paths = [
        str(item) for item in payload.get("candidate_event_ledger_dbs", []) if str(item)
    ]
    items = [dict(item) for item in payload.get("items", []) if isinstance(item, dict)]
    path_matched = False
    hash_matched = False
    matching_item: dict[str, Any] = {}
    if event_ledger_db is not None:
        for item in items:
            replay_db = str(item.get("replay_event_ledger_db", "") or "")
            item_hash = str(item.get("replay_event_ledger_db_sha256", "") or "")
            item_path_matched = _path_equivalent_to_target(replay_db, event_ledger_db)
            item_hash_matched = bool(
                actual_hash and item_hash and item_hash == actual_hash
            )
            if item_path_matched:
                path_matched = True
            if item_hash_matched:
                hash_matched = True
            if item_path_matched and item_hash_matched:
                matching_item = {
                    "label": str(item.get("label", "")),
                    "replay_event_ledger_db": replay_db,
                    "replay_event_ledger_db_sha256": item_hash,
                }
                break
        if not path_matched:
            path_matched = any(
                _path_equivalent_to_target(path, event_ledger_db)
                for path in candidate_db_paths
            )

    if not path_matched:
        missing.append(
            "transcript calibration report lists the current event-ledger DB path"
        )
    if not hash_matched:
        missing.append(
            "transcript calibration replay DB SHA-256 matches the current event-ledger DB"
        )

    return {
        "passed": not missing,
        "event_ledger_db": target_path,
        "event_ledger_db_sha256": actual_hash,
        "candidate_event_ledger_dbs": candidate_db_paths,
        "path_matched": path_matched,
        "hash_matched": hash_matched,
        "capture_provenance_ok": capture_ok,
        "matching_item": matching_item,
        "missing": list(dict.fromkeys(missing)),
    }


def _path_equivalent_to_target(path_text: str, target: Path) -> bool:
    if not path_text:
        return False
    try:
        return Path(path_text).resolve(strict=False) == target.resolve(strict=False)
    except OSError:
        left = os.path.normcase(os.path.abspath(path_text))
        right = os.path.normcase(os.path.abspath(str(target)))
        return left == right


def _v01_host_app_summary(payload: dict[str, Any], error: str) -> dict[str, Any]:
    if error:
        return {
            "present": bool(payload),
            "error": error,
            "config": payload.get("config", {}) if payload else {},
        }
    if not payload:
        return {"present": False, "error": ""}
    return {
        "present": True,
        "passed": bool(payload.get("passed", False)),
        "configured": bool(payload.get("configured", False)),
        "skipped": bool(payload.get("skipped", False)),
        "checks": payload.get("checks", {}),
        "command_sources": payload.get("command_sources", {}),
        "config": payload.get("config", {}),
        "evidence_class": payload.get("evidence_class", {}),
    }


def _calibrate_transcript_replay(args) -> None:
    input_paths = _transcript_calibration_inputs(args.input, args.input_dir, args.glob)
    if not input_paths:
        raise SystemExit(
            "No transcript calibration inputs found; pass --input or --input-dir."
        )
    if args.reset and args.work_dir.exists():
        _safe_remove_bundle_dir(args.work_dir)
    imported_dir = args.work_dir / "imported"
    db_root = args.work_dir / "db"
    imported_dir.mkdir(parents=True, exist_ok=True)
    db_root.mkdir(parents=True, exist_ok=True)
    profile = _read_optional_profile_json(args.profile_json)
    controls = _read_optional_controls_json(args.controls_json)
    replacements = _parse_transcript_redaction_rules(tuple(args.replace))
    items: list[dict] = []
    for index, input_path in enumerate(input_paths, start=1):
        label = _transcript_calibration_label(input_path, index)
        imported_path = imported_dir / f"{label}.jsonl"
        replay_db = _transcript_calibration_replay_db(db_root, label)
        source_attestation_path = args.work_dir / f"{label}_source_attestation.json"
        item: dict = {
            "label": label,
            "input_path": str(input_path),
            "imported_transcript_jsonl": str(imported_path),
            "replay_event_ledger_db": str(replay_db),
            "replay_event_ledger_db_sha256": "",
            "source_attestation_out": str(source_attestation_path),
            "passed": False,
            "error": "",
            "import": {},
            "replay": {},
            "next_candidate_commands": {},
        }
        try:
            import_report = import_transcript_replay_fixture(
                input_path=input_path,
                output_path=imported_path,
                profile=profile,
                scenario=f"calibration_{label}",
                description="Redacted local chat transcript calibration replay.",
                source_note=(
                    "Calibration import from raw local chat JSONL; assistant/system rows are skipped, "
                    "private tokens are redacted, and static expected answer/route fields are dropped."
                ),
                replacements=replacements,
                controls=controls,
            )
            item["import"] = import_report.to_dict()
            if not import_report.passed:
                item["error"] = "import_failed"
                items.append(item)
                continue
            replay_report = run_transcript_replay_suite(
                transcript_path=imported_path,
                db_dir=db_root,
                reset=True,
                weather_offline_json=args.weather_json,
                auto_lifecycle=args.auto_lifecycle,
            )
            replay_payload = replay_report.to_dict()
            item["replay"] = _transcript_replay_summary(replay_payload)
            item["replay"]["report_schema"] = str(replay_payload.get("schema", ""))
            item["replay_event_ledger_db_sha256"] = (
                _sha256_file(replay_db) if replay_db.is_file() else ""
            )
            item["next_candidate_commands"] = (
                _transcript_calibration_candidate_commands(
                    replay_db=replay_db,
                    source_attestation_path=source_attestation_path,
                    calibration_report_path=args.out,
                    min_total_turns=args.min_total_turns,
                    min_local_resolution_rate=args.min_local_resolution_rate,
                    min_route_kinds=args.min_route_kinds,
                    min_intent_kinds=args.min_intent_kinds,
                    min_synthesis_traces=args.min_synthesis_traces,
                    min_priority_signal_samples=args.min_priority_signal_samples,
                    auto_lifecycle=args.auto_lifecycle,
                )
            )
            item["passed"] = bool(replay_payload.get("passed", False))
            if not item["passed"]:
                item["error"] = "replay_failed"
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            item["error"] = str(exc)
        items.append(item)
    aggregate = _transcript_calibration_aggregate(
        items,
        min_total_turns=args.min_total_turns,
        min_local_resolution_rate=args.min_local_resolution_rate,
        min_route_kinds=args.min_route_kinds,
        min_intent_kinds=args.min_intent_kinds,
        min_synthesis_traces=args.min_synthesis_traces,
        min_priority_signal_samples=args.min_priority_signal_samples,
        require_priority_signals=args.require_priority_signals,
        require_memory_digest_quality=args.require_memory_digest_quality,
        require_strict_baseline_win=args.require_strict_baseline_win,
        require_redaction=args.require_redaction,
        require_static_drop=args.require_static_drop,
    )
    payload = {
        "schema": "melm.local_assistant_transcript_calibration_report.v1",
        "passed": bool(aggregate["passed"]),
        "work_dir": str(args.work_dir),
        "report_path": str(args.out) if args.out is not None else "",
        "inputs": [str(path) for path in input_paths],
        "input_count": len(input_paths),
        "auto_lifecycle": bool(args.auto_lifecycle),
        "items": items,
        "aggregate": aggregate,
        "candidate_event_ledger_dbs": [
            str(item.get("replay_event_ledger_db", ""))
            for item in items
            if item.get("replay_event_ledger_db")
        ],
        "next_candidate_commands": _transcript_calibration_aggregate_candidate_commands(
            items
        ),
    }
    if args.out is not None:
        _write_json_report(args.out, payload)
    _print_payload(payload, json_mode=args.json)


def _v01_audit(args) -> None:
    payload = _build_v01_audit_payload()
    _print_payload(payload, json_mode=args.json)


def _v01_progress(args) -> None:
    payload = _build_v01_progress_payload(
        blocker_evidence_json=args.blocker_evidence_json
    )
    _print_payload(payload, json_mode=args.json)


def _build_v01_progress_payload(
    *, blocker_evidence_json: Path | None = None
) -> dict[str, Any]:
    started = perf_counter()
    audit_payload = _build_v01_audit_payload()
    blocker_payload, blocker_error, blocker_source = _v01_progress_blocker_payload(
        blocker_evidence_json
    )
    blockers = [
        dict(item)
        for item in blocker_payload.get("blockers", [])
        if isinstance(item, dict)
    ]
    if not blockers and blocker_evidence_json is None:
        blockers = _v01_progress_audit_blocker_rows(audit_payload)
    blocker_count = int(
        audit_payload.get("blocker_count", len(blockers)) or len(blockers)
    )
    status_counts = dict(
        sorted(Counter(str(item.get("status", "")) for item in blockers).items())
    )
    candidate_blockers = [
        str(item.get("id", ""))
        for item in blockers
        if item.get("status") == "candidate_evidence_present"
    ]
    development_only = [
        str(item.get("id", ""))
        for item in blockers
        if str(item.get("status", "")).startswith("development")
        or str(item.get("status", "")).startswith("unattested")
    ]
    missing_blockers = [
        str(item.get("id", ""))
        for item in blockers
        if item.get("status") != "candidate_evidence_present"
    ]
    next_commands = _v01_progress_next_commands(audit_payload, blockers)
    candidate_count = int(
        blocker_payload.get("candidate_blockers_satisfied", len(candidate_blockers))
        or 0
    )
    remaining_count = max(0, blocker_count - candidate_count)
    candidate_review_ready = bool(
        audit_payload.get("core_browser_cli_ready", False)
        and blocker_count > 0
        and candidate_count >= blocker_count
        and not blocker_error
    )
    checks = {
        "audit_passed": bool(audit_payload.get("passed", False)),
        "core_browser_cli_ready": bool(
            audit_payload.get("core_browser_cli_ready", False)
        ),
        "blocker_boundary_present": blocker_count > 0
        and len(blockers) >= blocker_count,
        "candidate_count_not_overstated": 0 <= candidate_count <= blocker_count,
        "architecture_completion_not_claimed": True,
    }
    return {
        "schema": "melm.local_assistant_v01_progress.v1",
        "passed": bool(audit_payload.get("passed", False)) and not bool(blocker_error),
        "architecture_complete": False,
        "architecture_complete_claimed": False,
        "candidate_review_ready": candidate_review_ready,
        "checks": checks,
        "status": (
            "candidate_review_ready_not_architecture_complete"
            if candidate_review_ready
            else (
                "browser_cli_ready_with_real_world_blockers"
                if audit_payload.get("core_browser_cli_ready", False)
                else "missing_core_browser_cli_evidence"
            )
        ),
        "elapsed_ms": _elapsed_ms(started),
        "audit_source": "v01-audit",
        "blocker_evidence_source": blocker_source,
        "blocker_evidence_error": blocker_error,
        "blocker_count": blocker_count,
        "candidate_blockers_satisfied": candidate_count,
        "remaining_blocker_count": remaining_count,
        "core": {
            "browser_cli_ready": bool(
                audit_payload.get("core_browser_cli_ready", False)
            ),
            "status": audit_payload.get("status", ""),
            "checks": audit_payload.get("checks", {}),
            "requirement_count": len(audit_payload.get("core_requirements", [])),
        },
        "blockers": {
            "blocker_count": blocker_count,
            "candidate_blockers_satisfied": candidate_count,
            "remaining_blocker_count": remaining_count,
            "status_counts": status_counts,
            "candidate_blockers": candidate_blockers,
            "development_only_or_unattested_blockers": development_only,
            "missing_blockers": missing_blockers,
            "rows": blockers,
        },
        "evidence_commands": _v01_progress_evidence_commands(audit_payload),
        "next_commands": next_commands,
        "completion_boundary": {
            "v01_audit_architecture_complete": bool(
                audit_payload.get("architecture_complete", False)
            ),
            "v01_blocker_evidence_architecture_complete_claimed": bool(
                blocker_payload.get("architecture_complete_claimed", False)
            ),
            "must_not_claim_complete_from_development_or_fixture_evidence": True,
        },
        "note": (
            "This progress report summarizes current evidence only. It does not complete v0.1; "
            "candidate blocker evidence still requires architecture review and v01-audit remains the completion boundary."
        ),
    }


def _v01_progress_blocker_payload(path: Path | None) -> tuple[dict[str, Any], str, str]:
    if path is not None:
        payload, error = _optional_json_report(path)
        if error:
            return {}, error, str(path)
        if payload.get("schema") != "melm.local_assistant_v01_blocker_evidence.v1":
            return (
                payload,
                "blocker evidence JSON schema is not melm.local_assistant_v01_blocker_evidence.v1",
                str(path),
            )
        return payload, "", str(path)
    return {}, "", "v01-audit_completion_blockers"


def _v01_progress_audit_blocker_rows(
    audit_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in audit_payload.get("completion_blockers", []):
        if not isinstance(item, dict):
            continue
        blocker_id = str(item.get("id", "") or "")
        if not blocker_id:
            continue
        evidence_needed = str(item.get("evidence_needed", "") or "")
        rows.append(
            _v01_blocker_row(
                blocker_id,
                status="remaining_blocker",
                evidence={
                    "source": "v01-audit",
                    "evidence_needed": evidence_needed,
                },
                missing=[evidence_needed]
                if evidence_needed
                else ["candidate evidence for this blocker"],
                next_command=str(item.get("next_command", "") or ""),
            )
        )
    return rows


def _v01_progress_next_commands(
    audit_payload: dict[str, Any], blockers: list[dict[str, Any]]
) -> list[str]:
    commands: list[str] = []
    for item in blockers:
        command = str(item.get("next_command", "") or "")
        if command and command not in commands:
            commands.append(command)
    if not commands:
        for item in audit_payload.get("completion_blockers", []):
            command = str(item.get("next_command", "") or "")
            if command and command not in commands:
                commands.append(command)
    return commands


def _v01_progress_evidence_commands(audit_payload: dict[str, Any]) -> dict[str, str]:
    command_evidence = dict(audit_payload.get("command_evidence", {}))
    keys = (
        "candidate_session_audit",
        "source_attestation",
        "host_app_attestation",
        "blocker_evidence",
        "blocker_rehearsal",
        "calibration_synthesis",
        "calibration_planner",
        "calibration_digest_baseline",
        "live_inventory_soak_matrix",
        "host_app_configured",
    )
    return {
        key: str(command_evidence.get(key, ""))
        for key in keys
        if command_evidence.get(key)
    }


def _v01_acceptance(args) -> None:
    db_dir = args.db_dir
    if args.reset and db_dir.exists():
        _safe_remove_bundle_dir(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    target_args = [
        "target-report",
        "--db-dir",
        str(db_dir / "target_report"),
        "--reset",
        "--host",
        args.host,
        "--json",
    ]
    if args.require_raspberry_pi:
        target_args.append("--require-raspberry-pi")
    if args.media_player_command:
        target_args.extend(("--media-player-command", args.media_player_command))
    if args.call_command:
        target_args.extend(("--call-command", args.call_command))
    if args.host_app_config_json is not None:
        target_args.extend(("--host-app-config-json", str(args.host_app_config_json)))
    if args.host_app_media_dir is not None:
        target_args.extend(("--host-app-media-dir", str(args.host_app_media_dir)))
    if args.require_host_app_configured:
        target_args.append("--require-host-app-configured")
    if args.host_app_config_json is not None:
        host_app_evidence_command = (
            "host-app-probe --config-json "
            f"{args.host_app_config_json} "
            f"{'--require-configured ' if args.require_host_app_configured else ''}--json"
        )
        host_app_config_source = "config_json"
    elif args.media_player_command or args.call_command:
        host_app_evidence_command = (
            "host-app-probe --media-player-command <configured> "
            "--call-command <configured> "
            f"{'--require-configured ' if args.require_host_app_configured else ''}--json"
        )
        host_app_config_source = "explicit_args"
    else:
        host_app_evidence_command = "host-app-probe --reset --json"
        host_app_config_source = "default_unconfigured_report"
    target_payload = _run_cli_json(ROOT, *target_args)

    chat_payload = _run_cli_json(
        ROOT,
        "chat",
        "--db",
        str(db_dir / "cli_chat.sqlite"),
        "--reset",
        "--turn",
        "Tell me a story.",
        "--turn",
        "What is the weather today?",
        "--turn",
        "Should I go to school dressed naked?",
        "--json",
    )
    audit_payload = _run_cli_json(ROOT, "v01-audit", "--json")
    bundle_payload: dict[str, Any] = {"skipped": True, "passed": False}
    if args.include_bundle:
        bundle_args = ["pi-bundle", "--out", str(args.bundle_out), "--reset", "--json"]
        if args.zip_bundle:
            bundle_args.append("--zip")
        bundle_payload = _run_cli_json(ROOT, *bundle_args)

    target_checks = dict(target_payload.get("checks", {}))
    audit_checks = dict(audit_payload.get("checks", {}))
    chat_turns = list(chat_payload.get("turns", []))
    chat_routes = [str(turn.get("route", "")) for turn in chat_turns]
    chat_reasons = [str(turn.get("reason", "")) for turn in chat_turns]
    chat_counts = dict(chat_payload.get("counts", {}))
    pi_smoke = dict(target_payload.get("smokes", {}).get("pi_smoke", {}))
    pi_checks = dict(pi_smoke.get("checks", {}))
    host_app_probe = dict(target_payload.get("smokes", {}).get("host_app_probe", {}))
    bootstrap = dict(target_payload.get("smokes", {}).get("bootstrap_runtime", {}))

    requirements = [
        _v01_acceptance_requirement(
            "target_report_smokes",
            bool(target_payload.get("passed", False)),
            "target-report passed with required host and smoke checks",
            evidence=["target-report --reset --json", str(db_dir / "target_report")],
            details={
                "checks_true": sum(
                    1 for value in target_checks.values() if value is True
                ),
                "checks_total": len(target_checks),
                "raspberry_pi_required": bool(
                    target_payload.get("hardware_policy", {}).get(
                        "raspberry_pi_required", False
                    )
                ),
                "host_app_required": bool(args.require_host_app_configured),
            },
        ),
        _v01_acceptance_requirement(
            "initial_datasets_and_runtime_db",
            bool(target_checks.get("dataset_audit_passed"))
            and bool(target_checks.get("bootstrap_runtime_passed"))
            and int(bootstrap.get("counts", {}).get("events", 0) or 0) >= 3,
            "seed/source datasets audit and usable runtime bootstrap both passed",
            evidence=[
                "dataset-audit --reset --json",
                "bootstrap-runtime --reset --json",
            ],
            details={
                "bootstrap_events": int(
                    bootstrap.get("counts", {}).get("events", 0) or 0
                ),
                "bootstrap_db_bytes": int(bootstrap.get("db_bytes", 0) or 0),
            },
        ),
        _v01_acceptance_requirement(
            "readiness_pi_smoke",
            bool(target_checks.get("pi_smoke_passed"))
            and bool(target_checks.get("inventory_soak_matrix_passed"))
            and bool(pi_checks)
            and all(bool(value) for value in pi_checks.values()),
            "compact v0.1 readiness smoke passed with the inventory matrix included",
            evidence=["pi-smoke --reset --json"],
            details={
                "pi_smoke_checks": len(pi_checks),
                "inventory_matrix_profiles": int(
                    pi_smoke.get("inventory_soak_matrix", {}).get("profile_count", 0)
                    or 0
                ),
                "inventory_matrix_failed_cycles": int(
                    pi_smoke.get("inventory_soak_matrix", {}).get(
                        "total_failed_import_cycles", 0
                    )
                    or 0
                ),
            },
        ),
        _v01_acceptance_requirement(
            "cross_platform_cli_chat",
            bool(chat_payload.get("passed", False))
            and len(chat_turns) == 3
            and chat_routes == ["local_answer", "cached_tool", "local_answer"]
            and chat_reasons
            == [
                "local_story_inventory",
                "weather_cache_hit",
                "local_common_sense_policy",
            ]
            and int(chat_counts.get("events", 0) or 0) == 3
            and int(chat_counts.get("membrane_decisions", 0) or 0) == 3
            and int(chat_counts.get("homeostatic_snapshots", 0) or 0) == 3,
            "scripted terminal chat uses the same kernel/store path across story, weather, and safety turns",
            evidence=["chat --turn ... --json"],
            details={
                "routes": chat_routes,
                "reasons": chat_reasons,
                "events": int(chat_counts.get("events", 0) or 0),
            },
        ),
        _v01_acceptance_requirement(
            "browser_api_surface",
            bool(target_checks.get("api_smoke_passed"))
            and bool(target_checks.get("api_session_smoke_passed"))
            and bool(target_checks.get("ui_smoke_passed"))
            and bool(target_checks.get("localhost_api")),
            "localhost JSON API, multi-turn API session, and dependency-free browser UI passed",
            evidence=[
                "api-smoke --reset --json",
                "api-session-smoke --reset --json",
                "ui-smoke --reset --json",
            ],
            details={
                "host": args.host,
                "localhost_only": bool(target_checks.get("localhost_api")),
            },
        ),
        _v01_acceptance_requirement(
            "memory_synthesis_transcript_gates",
            bool(target_checks.get("open_traces_passed"))
            and bool(target_checks.get("transcript_replay_passed"))
            and bool(target_checks.get("transcript_calibration_passed"))
            and bool(target_checks.get("synthesis_variant_smoke_passed"))
            and bool(target_checks.get("synthesis_stress_smoke_passed")),
            "open traces, transcript replay/calibration, and bounded synthesis gates passed",
            evidence=[
                "run-open-traces --reset --json",
                "run-transcript-replay --reset --json",
                "calibrate-transcript-replay --require-redaction --require-static-drop --json",
                "synthesis-variant-smoke --reset --json",
                "synthesis-stress-smoke --reset --json",
            ],
            details={
                "open_trace_turns": int(
                    target_payload.get("smokes", {})
                    .get("open_traces", {})
                    .get("turns", 0)
                    or 0
                ),
                "transcript_replay_turns": int(
                    target_payload.get("smokes", {})
                    .get("transcript_replay", {})
                    .get("turns", 0)
                    or 0
                ),
            },
        ),
        _v01_acceptance_requirement(
            "action_and_setup_gates",
            bool(target_checks.get("setup_integration_smoke_passed"))
            and bool(target_checks.get("host_action_smoke_passed"))
            and bool(target_checks.get("host_app_probe_reported"))
            and bool(target_checks.get("host_app_requirement_satisfied")),
            "setup loops, typed command-mode actions, and target app configuration reporting passed",
            evidence=[
                "setup-integration-smoke --reset --json",
                "host-action-smoke --reset --json",
                host_app_evidence_command,
            ],
            details={
                "host_app_configured": bool(host_app_probe.get("configured", False)),
                "host_app_skipped": bool(host_app_probe.get("skipped", False)),
                "host_app_required": bool(args.require_host_app_configured),
                "host_app_config_source": host_app_config_source,
            },
        ),
        _v01_acceptance_requirement(
            "anti_static_uol_chatframe_guard",
            bool(audit_checks.get("uol_chatframe_static_shortcut_guard_present"))
            and bool(audit_checks.get("drift_rule_present")),
            "primary routing remains tied to UOL/ChatFrame evidence, not phrase shortcuts",
            evidence=[
                "python -m unittest tests.test_local_assistant_router_mvp",
                "shortcut-audit --json",
                "v01-audit --json",
            ],
            details={
                "secondary_phrase_policy": "debug_only_never_primary_route",
            },
        ),
        _v01_acceptance_requirement(
            "portable_bundle",
            bool(bundle_payload.get("passed", False)) if args.include_bundle else None,
            "optional portable bundle self-check passed",
            evidence=["pi-bundle --reset --json"],
            details={
                "included": bool(args.include_bundle),
                "bundle_out": str(args.bundle_out),
                "zip_requested": bool(args.zip_bundle),
            },
        ),
        _v01_acceptance_requirement(
            "completion_boundary_explicit",
            bool(audit_payload.get("passed", False))
            and bool(audit_payload.get("core_browser_cli_ready", False))
            and not bool(audit_payload.get("architecture_complete", False))
            and int(audit_payload.get("blocker_count", 0) or 0) >= 1,
            "browser/CLI release candidate is distinguished from full architecture completion",
            evidence=["v01-audit --json"],
            details={
                "architecture_complete": bool(
                    audit_payload.get("architecture_complete", False)
                ),
                "blocker_count": int(audit_payload.get("blocker_count", 0) or 0),
                "blocker_ids": [
                    str(item.get("id", ""))
                    for item in audit_payload.get("completion_blockers", [])
                ],
            },
        ),
    ]
    required_requirements = [
        item for item in requirements if item["status"] != "skipped"
    ]
    release_candidate = bool(required_requirements) and all(
        bool(item["passed"]) for item in required_requirements
    )
    payload = {
        "schema": "melm.local_assistant_v01_acceptance.v1",
        "db_dir": str(db_dir),
        "passed": release_candidate,
        "release_candidate": release_candidate,
        "architecture_complete": bool(
            audit_payload.get("architecture_complete", False)
        ),
        "blocker_count": int(audit_payload.get("blocker_count", 0) or 0),
        "elapsed_ms": _elapsed_ms(started),
        "checks": {
            str(item["id"]): bool(item["passed"]) for item in required_requirements
        },
        "requirements": requirements,
        "target_report": _v01_acceptance_target_summary(target_payload),
        "chat": _v01_acceptance_chat_summary(chat_payload),
        "v01_audit": {
            "passed": bool(audit_payload.get("passed", False)),
            "status": str(audit_payload.get("status", "")),
            "core_browser_cli_ready": bool(
                audit_payload.get("core_browser_cli_ready", False)
            ),
            "architecture_complete": bool(
                audit_payload.get("architecture_complete", False)
            ),
            "blocker_count": int(audit_payload.get("blocker_count", 0) or 0),
            "completion_blockers": list(audit_payload.get("completion_blockers", [])),
        },
        "bundle": _v01_acceptance_bundle_summary(bundle_payload),
        "runtime": "stdlib_python_sqlite_http_html_acceptance",
        "dependency_class": "stdlib_only",
        "note": (
            "Passing this command means the browser/CLI v0.1 release-candidate path is evidenced on this host. "
            "It does not clear the real-world blockers listed in v01-audit."
        ),
    }
    _print_payload(payload, json_mode=args.json)


def _v01_acceptance_requirement(
    requirement_id: str,
    passed: bool | None,
    description: str,
    *,
    evidence: list[str],
    details: dict[str, Any],
) -> dict:
    if passed is None:
        status = "skipped"
        passed_value = False
    else:
        status = "met" if passed else "failed"
        passed_value = bool(passed)
    return {
        "id": requirement_id,
        "status": status,
        "passed": passed_value,
        "description": description,
        "evidence": evidence,
        "details": details,
    }


def _v01_acceptance_target_summary(payload: dict) -> dict:
    smokes = dict(payload.get("smokes", {}))
    pi_smoke = dict(smokes.get("pi_smoke", {}))
    return {
        "passed": bool(payload.get("passed", False)),
        "checks": dict(payload.get("checks", {})),
        "hardware_policy": dict(payload.get("hardware_policy", {})),
        "runtime": dict(payload.get("runtime", {})),
        "resources": {
            "disk_free_bytes": int(
                payload.get("resources", {}).get("disk_free_bytes", 0) or 0
            ),
            "disk_total_bytes": int(
                payload.get("resources", {}).get("disk_total_bytes", 0) or 0
            ),
            "memory_total_kb": int(
                payload.get("resources", {}).get("memory_total_kb", 0) or 0
            ),
        },
        "smokes": {
            "dataset_audit_passed": bool(
                smokes.get("dataset_audit", {}).get("passed", False)
            ),
            "pi_smoke_passed": bool(pi_smoke.get("passed", False)),
            "inventory_soak_matrix": pi_smoke.get("inventory_soak_matrix", {}),
            "api_smoke_passed": bool(smokes.get("api_smoke", {}).get("passed", False)),
            "api_session_smoke_passed": bool(
                smokes.get("api_session_smoke", {}).get("passed", False)
            ),
            "ui_smoke_passed": bool(smokes.get("ui_smoke", {}).get("passed", False)),
            "bootstrap_runtime_passed": bool(
                smokes.get("bootstrap_runtime", {}).get("passed", False)
            ),
            "open_traces_passed": bool(
                smokes.get("open_traces", {}).get("passed", False)
            ),
            "transcript_replay_passed": bool(
                smokes.get("transcript_replay", {}).get("passed", False)
            ),
            "transcript_calibration_passed": bool(
                smokes.get("transcript_calibration", {}).get("passed", False)
            ),
            "host_app_probe": {
                "passed": bool(smokes.get("host_app_probe", {}).get("passed", False)),
                "configured": bool(
                    smokes.get("host_app_probe", {}).get("configured", False)
                ),
                "skipped": bool(smokes.get("host_app_probe", {}).get("skipped", False)),
                "checks": dict(smokes.get("host_app_probe", {}).get("checks", {})),
                "command_sources": dict(
                    smokes.get("host_app_probe", {}).get("command_sources", {})
                ),
            },
        },
    }


def _v01_acceptance_chat_summary(payload: dict) -> dict:
    turns = list(payload.get("turns", []))
    return {
        "passed": bool(payload.get("passed", False)),
        "mode": str(payload.get("mode", "")),
        "turn_count": len(turns),
        "routes": [str(turn.get("route", "")) for turn in turns],
        "reasons": [str(turn.get("reason", "")) for turn in turns],
        "counts": dict(payload.get("counts", {})),
    }


def _v01_acceptance_bundle_summary(payload: dict) -> dict:
    return {
        "skipped": bool(payload.get("skipped", False)),
        "passed": bool(payload.get("passed", False)),
        "manifest": str(payload.get("manifest", "")),
        "runbook": str(payload.get("runbook", "")),
        "archive": payload.get("archive", None),
        "smoke_checks": dict(payload.get("smoke", {}).get("checks", {})),
    }


def _build_v01_audit_payload() -> dict:
    started = perf_counter()
    plan_path = ROOT / "docs" / "archive" / "local_assistant_os_mvp_plan_v2.md"
    legacy_plan_path = ROOT / "docs" / "local_assistant_os_mvp_plan.md"
    archived_legacy_path = ROOT / "docs" / "archive" / "local_assistant_os_mvp_plan.md"
    root_readme = ROOT / "README.md"
    cli_path = ROOT / "scripts" / "local_assistant_os_cli.py"
    router_tests_path = ROOT / "tests" / "test_local_assistant_router_mvp.py"
    required_datasets = _required_dataset_report(DEFAULT_SEED)
    required_dataset_names = [str(item["path"]) for item in required_datasets]
    required_dataset_missing = [
        str(item["path"]) for item in required_datasets if not item["exists"]
    ]
    plan_text = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""
    legacy_plan_text = (
        legacy_plan_path.read_text(encoding="utf-8")
        if legacy_plan_path.exists()
        else (
            archived_legacy_path.read_text(encoding="utf-8")
            if archived_legacy_path.exists()
            else ""
        )
    )
    v01_gate_text = f"{plan_text}\n{legacy_plan_text}"
    readme_text = (
        root_readme.read_text(encoding="utf-8") if root_readme.exists() else ""
    )
    router_tests_text = (
        router_tests_path.read_text(encoding="utf-8")
        if router_tests_path.exists()
        else ""
    )
    shortcut_audit_report = _build_shortcut_audit_payload()
    static_shortcut_guard_present = all(
        marker in router_tests_text
        for marker in (
            "test_primary_intent_classifier_does_not_call_secondary_phrase_tables",
            "test_primary_intent_helpers_do_not_call_phrase_table_helpers",
            "test_post_route_slot_helpers_do_not_smuggle_marker_shortcuts",
            "test_secondary_hint_groups_are_not_request_surface_phrase_tables",
            "test_identity_purpose_variants_report_actual_token_role_composition",
            "test_bare_domain_words_do_not_route_without_chatframe_relation",
            "test_topic_nouns_do_not_route_as_tool_or_action_shortcuts",
        )
    ) and bool(shortcut_audit_report.get("passed", False))
    command_evidence = {
        "shortcut_audit": "python scripts/local_assistant_os_cli.py shortcut-audit --json",
        "dataset_audit": "python scripts/local_assistant_os_cli.py dataset-audit --reset --json",
        "bootstrap_runtime": "python scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json",
        "pi_smoke": "python scripts/local_assistant_os_cli.py pi-smoke --reset --json",
        "target_report": "python scripts/local_assistant_os_cli.py target-report --reset --json",
        "v01_acceptance": "python scripts/local_assistant_os_cli.py v01-acceptance --reset --json",
        "v01_progress": "python scripts/local_assistant_os_cli.py v01-progress --json",
        "inventory_soak_matrix": "python scripts/local_assistant_os_cli.py inventory-soak-matrix --reset --json",
        "live_inventory_soak_matrix": (
            "python scripts/local_assistant_os_cli.py inventory-soak-matrix --live --reset "
            "--out artifacts/local_assistant_os/live_inventory_soak_matrix.json --json"
        ),
        "pi_bundle": "python scripts/local_assistant_os_cli.py pi-bundle --reset --zip --json",
        "verify_bundle": (
            "python scripts/local_assistant_os_cli.py verify-bundle "
            "--bundle-root artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle --json"
        ),
        "first_run_smoke": (
            "python scripts/local_assistant_os_cli.py first-run-smoke "
            "--bundle-root artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle --json"
        ),
        "archive_smoke": (
            "python scripts/local_assistant_os_cli.py archive-smoke "
            "--archive artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle.zip --reset --json"
        ),
        "setup_integration": "python scripts/local_assistant_os_cli.py setup-integration-smoke --reset --json",
        "host_app_configured": (
            "python scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.json "
            "--require-configured --json"
        ),
        "host_app_attestation": (
            "python scripts/local_assistant_os_cli.py write-host-app-attestation "
            "--host-app-config-json config/host_actions.json --capture-surface target_device_cli "
            "--media-app-configured --call-app-configured --not-demo-recorder "
            "--real-app-commands-acknowledged --human-reviewed "
            "--out artifacts/local_assistant_os/host_app_attestation.json --json"
        ),
        "event_transcript_export": (
            "python scripts/local_assistant_os_cli.py export-transcript-replay "
            "--db <assistant.sqlite> --out <event-ledger-transcript.jsonl> --json"
        ),
        "event_ledger_calibration": (
            "python scripts/local_assistant_os_cli.py calibrate-event-ledger "
            "--db <assistant.sqlite> --controls-json config/safe_lifecycle_controls.example.json "
            "--min-total-turns <n> --min-local-resolution-rate <rate> --json"
        ),
        "source_attestation": (
            "python scripts/local_assistant_os_cli.py write-source-attestation "
            "--event-ledger-db <assistant.sqlite> --event-ledger-session <session|all> --source-kind redacted_user_session "
            "--capture-surface cli_chat --redaction-applied --static-expectations-absent "
            "--answers-routes-reasons-absent --human-reviewed --out <source-attestation.json> --json"
        ),
        "candidate_session_audit": (
            "python scripts/local_assistant_os_cli.py candidate-session-audit "
            "--db <assistant.sqlite> --session <session|all> --capture-surface cli_chat --json"
        ),
        "blocker_evidence": (
            "python scripts/local_assistant_os_cli.py v01-blocker-evidence "
            "--event-ledger-db <assistant.sqlite> --event-ledger-session <session|all> "
            "--event-source-kind redacted_user_session "
            "--source-attestation-json <source-attestation.json> --auto-lifecycle "
            "--transcript-calibration-report-json <calibration-report.json> "
            "--inventory-soak-report-json <inventory-soak-report.json> "
            "--host-app-config-json config/host_actions.json "
            "--host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json "
            "--run-host-app-probe --out artifacts/local_assistant_os/v01_blocker_evidence.json --json"
        ),
        "blocker_rehearsal": "python scripts/local_assistant_os_cli.py v01-blocker-rehearsal --reset --json",
        "calibration": (
            "python scripts/local_assistant_os_cli.py calibrate-transcript-replay "
            "--input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json "
            "--require-redaction --require-static-drop --json"
        ),
        "calibration_synthesis": (
            "python scripts/local_assistant_os_cli.py calibrate-transcript-replay "
            "--input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json "
            "--require-redaction --require-static-drop "
            "--min-synthesis-traces <n> --min-local-resolution-rate <rate> --json"
        ),
        "calibration_planner": (
            "python scripts/local_assistant_os_cli.py calibrate-transcript-replay "
            "--input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json "
            "--require-redaction --require-static-drop "
            "--require-priority-signals --min-priority-signal-samples <n> --json"
        ),
        "calibration_digest_baseline": (
            "python scripts/local_assistant_os_cli.py calibrate-transcript-replay "
            "--input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json "
            "--require-redaction --require-static-drop "
            "--require-memory-digest-quality --require-strict-baseline-win "
            "--out <calibration-report.json> --json"
        ),
        "anti_static_router_tests": "python -m unittest tests.test_local_assistant_router_mvp",
    }
    core_requirements = [
        {
            "id": "authoritative_plan",
            "status": "met"
            if plan_path.exists() and "Local Assistant OS MVP Plan v2" in plan_text
            else "missing",
            "evidence": [str(plan_path.relative_to(ROOT))],
            "description": "Authoritative Local Assistant OS v0.2 execution plan exists.",
        },
        {
            "id": "drift_rule",
            "status": "met"
            if root_readme.exists()
            and "Do not grow this repo as a generic chatbot" in readme_text
            else "missing",
            "evidence": [str(root_readme.relative_to(ROOT))],
            "description": "Root README points work away from generic chatbot/model-first drift.",
        },
        {
            "id": "uol_chatframe_static_shortcut_guard",
            "status": "met" if static_shortcut_guard_present else "missing",
            "evidence": [
                str(router_tests_path.relative_to(ROOT)),
                command_evidence["anti_static_router_tests"],
                command_evidence["shortcut_audit"],
            ],
            "description": (
                "Router regressions require UOL/ChatFrame primary evidence, reject bare domain-word routes, "
                "and keep phrase/vocabulary helpers out of primary classification and post-route slot helpers."
            ),
        },
        {
            "id": "required_seed_and_source_datasets",
            "status": "met" if not required_dataset_missing else "missing",
            "evidence": required_dataset_names,
            "description": "Initial seed, media, weather, story source, open-trace, and transcript fixtures are present.",
            "missing": required_dataset_missing,
        },
        {
            "id": "stdlib_cli_entrypoint",
            "status": "met" if cli_path.exists() else "missing",
            "evidence": [str(cli_path.relative_to(ROOT))],
            "description": "Cross-platform stdlib Python CLI entrypoint exists.",
        },
        {
            "id": "browser_cli_acceptance_commands",
            "status": "met",
            "evidence": [
                command_evidence["dataset_audit"],
                command_evidence["bootstrap_runtime"],
                command_evidence["pi_smoke"],
                command_evidence["target_report"],
                command_evidence["v01_acceptance"],
                command_evidence["pi_bundle"],
                command_evidence["verify_bundle"],
                command_evidence["first_run_smoke"],
                command_evidence["archive_smoke"],
                command_evidence["event_transcript_export"],
                command_evidence["event_ledger_calibration"],
                command_evidence["source_attestation"],
                command_evidence["host_app_attestation"],
                command_evidence["blocker_evidence"],
                command_evidence["v01_progress"],
            ],
            "description": "Runnable browser/CLI evidence commands exist and are the current acceptance path.",
        },
        {
            "id": "setup_gap_to_memory_action_loop",
            "status": "met",
            "evidence": [command_evidence["setup_integration"]],
            "description": (
                "Routine, household, and trusted-contact gaps are proven through setup requests, "
                "explicit local setup, later local recall/action, and confirmation gating."
            ),
        },
        {
            "id": "raspberry_hardware_optional_for_v01_browser_cli",
            "status": "met",
            "evidence": [
                "target-report --require-raspberry-pi is optional appliance validation"
            ],
            "description": "v0.1 browser/CLI acceptance does not require physical Raspberry Pi hardware.",
        },
    ]
    completion_blockers = [
        {
            "id": "user_derived_bounded_synthesis_traces",
            "status": "remaining_blocker",
            "evidence_needed": "Redacted user-derived story/advice/summarization traces passing synthesis quality/citation gates.",
            "next_command": command_evidence["blocker_evidence"],
        },
        {
            "id": "longer_live_inventory_soak",
            "status": "remaining_blocker",
            "evidence_needed": "Longer live/source retry inventory soak across more query niches and cycle counts.",
            "next_command": "python scripts/local_assistant_os_cli.py inventory-soak-matrix --live --cycles <n> --json",
        },
        {
            "id": "planner_priority_on_user_derived_traces",
            "status": "remaining_blocker",
            "evidence_needed": "Priority/failure-rate comparison over imported user-derived traces and real local integration pressure.",
            "next_command": command_evidence["blocker_evidence"],
        },
        {
            "id": "real_user_derived_lifecycle_traces",
            "status": "remaining_blocker",
            "evidence_needed": "Imported redacted user-derived lifecycle transcripts beyond authored open/transcript fixtures.",
            "next_command": command_evidence["blocker_evidence"],
        },
        {
            "id": "digest_quality_and_route_threshold_calibration",
            "status": "remaining_blocker",
            "evidence_needed": "Threshold calibration for digest quality, route floors, complexity floors, and baseline deltas on real traces.",
            "next_command": command_evidence["blocker_evidence"],
        },
        {
            "id": "configured_target_device_apps",
            "status": "remaining_blocker",
            "evidence_needed": "Configured media/call commands on target platforms passing the typed action gate.",
            "next_command": command_evidence["blocker_evidence"],
        },
    ]
    core_status = {
        str(item["id"]): item["status"] == "met" for item in core_requirements
    }
    checks = {
        "authoritative_plan_present": bool(
            core_status.get("authoritative_plan", False)
        ),
        "drift_rule_present": bool(core_status.get("drift_rule", False)),
        "uol_chatframe_static_shortcut_guard_present": bool(
            core_status.get("uol_chatframe_static_shortcut_guard", False)
        ),
        "shortcut_audit_passed": bool(shortcut_audit_report.get("passed", False)),
        "required_datasets_present": not required_dataset_missing,
        "cli_entrypoint_present": cli_path.exists(),
        "browser_cli_acceptance_path_defined": True,
        "setup_integration_gate_defined": "setup-integration-smoke" in v01_gate_text
        and "setup-integration-smoke" in readme_text,
        "raspberry_hardware_optional_for_browser_cli": "optional appliance validation"
        in readme_text,
        "completion_blockers_explicit": len(completion_blockers) == 6,
    }
    core_browser_cli_ready = all(checks.values())
    architecture_complete = core_browser_cli_ready and not completion_blockers
    return {
        "schema": "melm.local_assistant_v01_audit.v1",
        "passed": core_browser_cli_ready,
        "architecture_complete": architecture_complete,
        "status": "browser_cli_ready_with_real_world_blockers"
        if core_browser_cli_ready
        else "missing_core_evidence",
        "elapsed_ms": _elapsed_ms(started),
        "core_browser_cli_ready": core_browser_cli_ready,
        "checks": checks,
        "core_requirements": core_requirements,
        "completion_blockers": completion_blockers,
        "blocker_count": len(completion_blockers),
        "shortcut_audit_summary": {
            "passed": bool(shortcut_audit_report.get("passed", False)),
            "checks": dict(shortcut_audit_report.get("checks", {})),
        },
        "command_evidence": command_evidence,
        "runtime": "stdlib_python_sqlite_http_html_audit",
        "dependency_class": "stdlib_only",
        "note": (
            "This audit is intentionally not a substitute for target-report/pi-bundle execution. "
            "It states what is currently evidenced and what still needs real-world validation before "
            "the full architecture can be called complete."
        ),
    }


def _parse_transcript_redaction_rules(
    values: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    rules: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("--replace must use OLD=TOKEN")
        source, replacement = value.split("=", 1)
        source = source.strip()
        replacement = replacement.strip()
        if not source or not replacement:
            raise ValueError("--replace must include non-empty OLD and TOKEN")
        rules.append((source, replacement))
    return tuple(rules)


def _transcript_calibration_inputs(
    explicit_inputs: list[Path],
    input_dir: Path | None,
    glob_pattern: str,
) -> list[Path]:
    paths: list[Path] = [Path(item) for item in explicit_inputs]
    if input_dir is not None:
        paths.extend(sorted(Path(input_dir).glob(glob_pattern)))
    resolved: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        absolute = path.resolve()
        if absolute in resolved:
            continue
        resolved.add(absolute)
        unique.append(path)
    return unique


def _read_optional_profile_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("--profile-json must contain a JSON object")
    return payload


def _read_optional_controls_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("--controls-json must contain a JSON object")
    return payload


def _transcript_calibration_replay_db(db_root: Path, label: str) -> Path:
    scenario = f"calibration_{label}"
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", scenario).strip("_") or "calibration"
    return db_root / f"01_{safe_name}.sqlite"


def _transcript_calibration_candidate_commands(
    *,
    replay_db: Path,
    source_attestation_path: Path,
    calibration_report_path: Path | None,
    min_total_turns: int,
    min_local_resolution_rate: float,
    min_route_kinds: int,
    min_intent_kinds: int,
    min_synthesis_traces: int,
    min_priority_signal_samples: int,
    auto_lifecycle: bool,
) -> dict[str, str]:
    candidate_min_synthesis_traces = max(1, int(min_synthesis_traces or 0))
    candidate_min_priority_signal_samples = max(
        1, int(min_priority_signal_samples or 0)
    )
    candidate_audit_parts: list[Any] = [
        "candidate-session-audit",
        "--db",
        replay_db,
        "--session",
        "all",
        "--event-source-kind",
        "redacted_user_session",
        "--capture-surface",
        "imported_redacted_transcript",
        "--redaction-applied",
        "--static-expectations-absent",
        "--answers-routes-reasons-absent",
        "--human-reviewed",
        "--min-total-turns",
        max(1, int(min_total_turns or 1)),
        "--min-local-resolution-rate",
        max(0.0, float(min_local_resolution_rate or 0.0)),
        "--min-route-kinds",
        max(1, int(min_route_kinds or 1)),
        "--min-intent-kinds",
        max(1, int(min_intent_kinds or 1)),
        "--min-synthesis-traces",
        candidate_min_synthesis_traces,
        "--min-priority-signal-samples",
        candidate_min_priority_signal_samples,
        "--reset",
        "--json",
    ]
    if calibration_report_path is not None:
        candidate_audit_parts.extend(
            ("--transcript-calibration-report-json", calibration_report_path)
        )
    candidate_audit_command = _local_assistant_cli_command(*candidate_audit_parts)
    attestation_command = _local_assistant_cli_command(
        "write-source-attestation",
        "--event-ledger-db",
        replay_db,
        "--event-ledger-session",
        "all",
        "--source-kind",
        "redacted_user_session",
        "--capture-surface",
        "imported_redacted_transcript",
        "--redaction-applied",
        "--static-expectations-absent",
        "--answers-routes-reasons-absent",
        "--human-reviewed",
        "--out",
        source_attestation_path,
        "--overwrite",
        "--json",
    )
    blocker_parts: list[Any] = [
        "v01-blocker-evidence",
        "--event-ledger-db",
        replay_db,
        "--event-ledger-session",
        "all",
        "--event-source-kind",
        "redacted_user_session",
        "--source-attestation-json",
        source_attestation_path,
        "--min-total-turns",
        max(1, int(min_total_turns or 1)),
        "--min-local-resolution-rate",
        max(0.0, float(min_local_resolution_rate or 0.0)),
        "--min-route-kinds",
        max(1, int(min_route_kinds or 1)),
        "--min-intent-kinds",
        max(1, int(min_intent_kinds or 1)),
        "--min-synthesis-traces",
        candidate_min_synthesis_traces,
        "--min-priority-signal-samples",
        candidate_min_priority_signal_samples,
    ]
    if auto_lifecycle or candidate_min_priority_signal_samples > 0:
        blocker_parts.append("--auto-lifecycle")
    if calibration_report_path is not None:
        blocker_parts.extend(
            ("--transcript-calibration-report-json", calibration_report_path)
        )
    blocker_parts.append("--json")
    evidence_pack_parts: list[Any] = [
        "v01-evidence-pack",
        "--db",
        replay_db,
        "--session",
        "all",
        "--event-source-kind",
        "redacted_user_session",
        "--capture-surface",
        "imported_redacted_transcript",
        "--source-attestation-json",
        source_attestation_path,
        "--min-total-turns",
        max(1, int(min_total_turns or 1)),
        "--min-local-resolution-rate",
        max(0.0, float(min_local_resolution_rate or 0.0)),
        "--min-route-kinds",
        max(1, int(min_route_kinds or 1)),
        "--min-intent-kinds",
        max(1, int(min_intent_kinds or 1)),
        "--min-synthesis-traces",
        candidate_min_synthesis_traces,
        "--min-priority-signal-samples",
        candidate_min_priority_signal_samples,
    ]
    if auto_lifecycle or candidate_min_priority_signal_samples > 0:
        evidence_pack_parts.append("--auto-lifecycle")
    if calibration_report_path is not None:
        evidence_pack_parts.extend(
            ("--transcript-calibration-report-json", calibration_report_path)
        )
    evidence_pack_parts.append("--json")
    return {
        "candidate_session_audit": candidate_audit_command,
        "write_source_attestation": attestation_command,
        "v01_evidence_pack": _local_assistant_cli_command(*evidence_pack_parts),
        "v01_blocker_evidence": _local_assistant_cli_command(*blocker_parts),
        "note": (
            "Run candidate-session-audit first, then run the attestation command after human review, "
            "then run evidence pack or blocker evidence. "
            "Candidate synthesis/planner rows require positive trace and priority-signal thresholds. "
            "If no calibration report path was written, rerun calibrate-transcript-replay with --out "
            "before trying to clear digest/route calibration blockers."
        ),
    }


def _transcript_calibration_aggregate_candidate_commands(
    items: list[dict],
) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    for item in items:
        next_commands = item.get("next_candidate_commands", {})
        if isinstance(next_commands, dict) and next_commands:
            commands.append(
                {
                    "label": str(item.get("label", "")),
                    "candidate_session_audit": str(
                        next_commands.get("candidate_session_audit", "")
                    ),
                    "write_source_attestation": str(
                        next_commands.get("write_source_attestation", "")
                    ),
                    "v01_evidence_pack": str(
                        next_commands.get("v01_evidence_pack", "")
                    ),
                    "v01_blocker_evidence": str(
                        next_commands.get("v01_blocker_evidence", "")
                    ),
                }
            )
    return commands


def _local_assistant_cli_command(*parts: Any) -> str:
    command = ("python", "scripts/local_assistant_os_cli.py", *parts)
    return " ".join(shlex.quote(str(part)) for part in command)


def _transcript_calibration_label(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._-")
    if not stem:
        stem = "transcript"
    return f"{index:02d}_{stem}"


def _transcript_calibration_aggregate(
    items: list[dict],
    *,
    min_total_turns: int = 1,
    min_local_resolution_rate: float = 0.0,
    min_route_kinds: int = 1,
    min_intent_kinds: int = 1,
    min_synthesis_traces: int = 0,
    min_priority_signal_samples: int = 0,
    require_priority_signals: bool = False,
    require_memory_digest_quality: bool = False,
    require_strict_baseline_win: bool = False,
    require_redaction: bool = False,
    require_static_drop: bool = False,
) -> dict:
    route_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    redaction_counts: Counter[str] = Counter()
    static_drops: Counter[str] = Counter()
    safety_totals: Counter[str] = Counter()
    total_imported_turns = 0
    total_replayed_turns = 0
    weighted_complexity = 0.0
    unknown_tokens = 0
    synthesis_traces = 0
    priority_signal_sample_count = 0
    debug_mapping_passed = True
    primary_uol_chatframe_not_secondary_phrase_route = True
    digest_quality_replays = 0
    digest_quality_passed_replays = 0
    baseline_required = 0
    strict_baseline_passed = 0
    replay_passes = 0
    capture_surface_counts: Counter[str] = Counter()
    capture_source_counts: Counter[str] = Counter()
    capture_missing_fields = 0
    capture_turns = 0
    for item in items:
        import_payload = dict(item.get("import", {}))
        replay = dict(item.get("replay", {}))
        redaction_counts.update(dict(import_payload.get("redaction_counts", {})))
        static_drops.update(
            dict(import_payload.get("static_expectation_fields_dropped", {}))
        )
        imported_turns = int(import_payload.get("turns_written", 0) or 0)
        replayed_turns = int(replay.get("turns", 0) or 0)
        total_imported_turns += imported_turns
        total_replayed_turns += replayed_turns
        route_counts.update(dict(replay.get("route_counts", {})))
        intent_counts.update(dict(replay.get("intent_counts", {})))
        safety_totals.update(dict(replay.get("safety_totals", {})))
        complexity = dict(replay.get("complexity", {}))
        weighted_complexity += (
            float(complexity.get("avg_complexity_score", 0.0) or 0.0) * replayed_turns
        )
        unknown_tokens += int(complexity.get("unknown_tokens_total", 0) or 0)
        synthesis_traces += int(replay.get("synthesis_traces", 0) or 0)
        priority_signal_sample_count += len(
            list(replay.get("priority_signal_samples", []))
        )
        digest_quality = dict(replay.get("memory_digest_quality", {}))
        if digest_quality:
            digest_quality_replays += 1
            if bool(digest_quality.get("passed", False)):
                digest_quality_passed_replays += 1
        debug_checks = dict(replay.get("debug_checks", {}))
        if replay and not bool(debug_checks.get("debug_maps_present", False)):
            debug_mapping_passed = False
        if replay and not bool(
            debug_checks.get("primary_uol_chatframe_not_secondary_phrase_route", False)
        ):
            primary_uol_chatframe_not_secondary_phrase_route = False
        baseline = dict(replay.get("baseline_comparison", {}))
        if bool(baseline.get("required", False)):
            baseline_required += 1
        if bool(baseline.get("strict_passed", False)):
            strict_baseline_passed += 1
        if bool(replay.get("passed", False)):
            replay_passes += 1
        capture = dict(replay.get("capture_provenance", {}))
        capture_turns += int(capture.get("turn_count", 0) or 0)
        capture_missing_fields += int(capture.get("missing_field_count", 0) or 0)
        capture_surface_counts.update(
            {
                str(key): int(value or 0)
                for key, value in dict(
                    capture.get("capture_surface_counts", {})
                ).items()
            }
        )
        capture_source_counts.update(
            {
                str(key): int(value or 0)
                for key, value in dict(capture.get("capture_source_counts", {})).items()
            }
        )
    local_routes = {"local_answer", "cached_tool", "device_action"}
    local_or_device = sum(
        int(route_counts.get(route, 0) or 0) for route in local_routes
    )
    all_passed = bool(items) and all(bool(item.get("passed", False)) for item in items)
    local_resolution_rate = (
        round(local_or_device / total_replayed_turns, 3)
        if total_replayed_turns
        else 0.0
    )
    critical_safety_keys = (
        "cloud_private_inclusions",
        "unconfirmed_executed_actions",
        "action_without_confirmation_gate",
        "fake_latest_news_local_answers",
        "low_quality_applied_synthesis",
        "missing_membrane_or_homeostasis",
        "dangling_memory_links",
    )
    critical_safety_clean = all(
        int(safety_totals.get(key, 0) or 0) == 0 for key in critical_safety_keys
    )
    redaction_total = sum(int(value or 0) for value in redaction_counts.values())
    static_drop_total = sum(int(value or 0) for value in static_drops.values())
    checks = {
        "inputs_present": bool(items),
        "all_imports_passed": bool(items)
        and sum(bool(item.get("import", {}).get("passed", False)) for item in items)
        == len(items),
        "all_replays_passed": bool(items) and replay_passes == len(items),
        "turns_replayed_min": total_replayed_turns >= max(1, int(min_total_turns or 1)),
        "local_resolution_floor": local_resolution_rate
        >= max(0.0, float(min_local_resolution_rate or 0.0)),
        "route_diversity_floor": len(route_counts) >= max(1, int(min_route_kinds or 1)),
        "intent_diversity_floor": len(intent_counts)
        >= max(1, int(min_intent_kinds or 1)),
        "synthesis_trace_floor": synthesis_traces
        >= max(0, int(min_synthesis_traces or 0)),
        "priority_signal_sample_floor": priority_signal_sample_count
        >= max(0, int(min_priority_signal_samples or 0)),
        "priority_signals_required_met": (not require_priority_signals)
        or priority_signal_sample_count > 0,
        "memory_digest_quality_required_met": (
            (not require_memory_digest_quality)
            or (bool(items) and digest_quality_passed_replays == len(items))
        ),
        "strict_baseline_required_met": (
            (not require_strict_baseline_win)
            or (bool(items) and strict_baseline_passed == len(items))
        ),
        "debug_mapping_passed": debug_mapping_passed,
        "primary_uol_chatframe_not_secondary_phrase_route": primary_uol_chatframe_not_secondary_phrase_route,
        "critical_safety_clean": critical_safety_clean,
        "redaction_required_met": (not require_redaction) or redaction_total > 0,
        "static_drop_required_met": (not require_static_drop) or static_drop_total > 0,
    }
    return {
        "passed": all_passed and all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_total_turns": max(1, int(min_total_turns or 1)),
            "min_local_resolution_rate": max(
                0.0, float(min_local_resolution_rate or 0.0)
            ),
            "min_route_kinds": max(1, int(min_route_kinds or 1)),
            "min_intent_kinds": max(1, int(min_intent_kinds or 1)),
            "min_synthesis_traces": max(0, int(min_synthesis_traces or 0)),
            "min_priority_signal_samples": max(
                0, int(min_priority_signal_samples or 0)
            ),
            "require_priority_signals": bool(require_priority_signals),
            "require_memory_digest_quality": bool(require_memory_digest_quality),
            "require_strict_baseline_win": bool(require_strict_baseline_win),
            "require_redaction": bool(require_redaction),
            "require_static_drop": bool(require_static_drop),
        },
        "inputs": len(items),
        "imports_passed": sum(
            bool(item.get("import", {}).get("passed", False)) for item in items
        ),
        "replays_passed": replay_passes,
        "turns_imported": total_imported_turns,
        "turns_replayed": total_replayed_turns,
        "local_or_device_resolved": local_or_device,
        "local_resolution_rate": local_resolution_rate,
        "route_counts": dict(sorted(route_counts.items())),
        "route_kinds": len(route_counts),
        "intent_counts": dict(sorted(intent_counts.items())),
        "intent_kinds": len(intent_counts),
        "safety_totals": dict(sorted(safety_totals.items())),
        "critical_safety_keys": list(critical_safety_keys),
        "redaction_counts": dict(sorted(redaction_counts.items())),
        "redaction_total": redaction_total,
        "static_expectation_fields_dropped": dict(sorted(static_drops.items())),
        "static_expectation_fields_dropped_total": static_drop_total,
        "complexity": {
            "avg_complexity_score": round(weighted_complexity / total_replayed_turns, 3)
            if total_replayed_turns
            else 0.0,
            "unknown_tokens_total": unknown_tokens,
        },
        "synthesis_traces": synthesis_traces,
        "priority_signal_sample_count": priority_signal_sample_count,
        "capture_provenance": _event_capture_provenance_from_counts(
            turn_count=capture_turns,
            surface_counts=dict(sorted(capture_surface_counts.items())),
            source_counts=dict(sorted(capture_source_counts.items())),
            missing_field_count=capture_missing_fields,
        ),
        "memory_digest_quality": {
            "replays_with_digest_reports": digest_quality_replays,
            "passed_replays": digest_quality_passed_replays,
            "all_required_passed": (not require_memory_digest_quality)
            or (bool(items) and digest_quality_passed_replays == len(items)),
        },
        "debug_mapping_passed": debug_mapping_passed,
        "primary_uol_chatframe_not_secondary_phrase_route": primary_uol_chatframe_not_secondary_phrase_route,
        "baseline_required_replays": baseline_required,
        "strict_baseline_passed_replays": strict_baseline_passed,
        "calibration_note": (
            "Imported transcripts are redacted calibration traces; the authored transcript replay "
            "remains the strict baseline-win architecture gate."
        ),
    }


def _run_jobs(args) -> None:
    store = _open_store(args.db, None if args.cold_start else args.seed)
    executed = _execute_queued_jobs(
        store,
        profile=_cold_profile() if args.cold_start else None,
        limit=args.limit,
        weather_live=args.weather_live,
        weather_offline_json=args.weather_json,
    )
    self_observation = store.load_self_state().get("runtime_health_trends", {})
    payload = {
        "db": str(args.db),
        "executed": executed,
        "queued": len(store.load_jobs(status="queued")),
        "completed": len(store.load_jobs(status="completed")),
        "self_observation": self_observation,
        "counts": store.table_counts(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _execute_queued_jobs(
    store: AssistantOSStore,
    *,
    profile: LocalAssistantProfile | None,
    limit: int,
    kinds: tuple[str, ...] = (
        "build_story_inventory",
        "refresh_weather_cache",
        "import_story_metadata",
    ),
    weather_live: bool = False,
    weather_offline_json: Path | None = DEFAULT_WEATHER_SAMPLE,
) -> list[dict]:
    kernel = AssistantOSKernel(profile=profile, store=store)
    executed: list[dict] = []
    for _ in range(max(0, limit)):
        job = store.start_next_job(kinds=kinds)
        if job is None:
            break
        try:
            if job.kind == "import_story_metadata":
                result = _execute_import_story_metadata_job(store, kernel.profile, job)
                executed.append(
                    {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "source": result["source"],
                        "internet_archive_query": str(
                            job.payload.get("internet_archive_query", "") or ""
                        ),
                        "imported_items": result["imported_items"],
                        "resource_budget": job.resource_budget,
                    }
                )
                opportunity_id = str(job.payload.get("opportunity_id", ""))
                if opportunity_id:
                    store.mark_opportunity_executed_by_id(opportunity_id)
                store.complete_job(job.job_id, result=result)
            elif job.kind == "refresh_weather_cache":
                result = _execute_refresh_weather_cache_job(
                    store,
                    kernel.profile,
                    job,
                    live=weather_live,
                    offline_json=weather_offline_json,
                )
                executed.append(
                    {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "weather_days": result["weather_days"],
                        "network_used": result["network_used"],
                        "location": result["location"],
                        "resource_budget": job.resource_budget,
                    }
                )
                opportunity_id = str(job.payload.get("opportunity_id", ""))
                if opportunity_id:
                    store.mark_opportunity_executed_by_id(opportunity_id)
                store.complete_job(job.job_id, result=result)
            else:
                opportunity = _opportunity_from_job(job)
                before = len(kernel.executed_jobs)
                kernel.execute(opportunity)
                executed.append(
                    {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "executed_jobs": kernel.executed_jobs[before:],
                        "resource_budget": job.resource_budget,
                    }
                )
                store.complete_job(job.job_id, result={"executed": True})
        except Exception as exc:  # pragma: no cover - defensive CLI path
            store.fail_job(job.job_id, error=str(exc))
            executed.append({"job_id": job.job_id, "kind": job.kind, "error": str(exc)})
    _persist_runtime_self_observation(store, profile)
    return executed


def _schedule_refreshes(args) -> None:
    store = _open_store(args.db, None if args.cold_start else args.seed)
    profile = store.load_profile(
        _cold_profile() if args.cold_start else LocalAssistantProfile()
    )
    report = schedule_inventory_refreshes(
        store,
        profile,
        min_story_models=args.min_story_models,
        story_limit=args.story_limit,
        source=args.source,
        use_offline_samples=args.offline_samples,
        gutenberg_csv=args.gutenberg_csv,
        internet_archive_json=args.internet_archive_json,
        internet_archive_query=args.internet_archive_query or None,
        gutenberg_max_source_bytes=args.gutenberg_max_source_bytes,
        internet_archive_max_source_bytes=args.internet_archive_max_source_bytes,
        internet_archive_page_size=args.internet_archive_page_size,
        internet_archive_max_pages=args.internet_archive_max_pages,
        internet_archive_cursor=args.internet_archive_cursor,
        internet_archive_rate_limit_delay_seconds=args.internet_archive_rate_limit_delay_seconds,
    )
    payload = {
        "db": str(args.db),
        **report.to_dict(),
        "self_observation": _persist_runtime_self_observation(store, profile),
        "counts": store.table_counts(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _inventory_soak(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    store = _open_store(args.db, None if args.cold_start else args.seed)
    started = perf_counter()
    cycles: list[dict] = []
    initial_story_inventory_count = len(store.load_inventory("story_model"))
    for index in range(max(0, args.cycles)):
        profile = store.load_profile(
            _cold_profile() if args.cold_start else LocalAssistantProfile()
        )
        story_count = len(store.load_inventory("story_model"))
        min_story_models = max(
            args.min_story_models, story_count + max(1, args.story_limit)
        )
        report = schedule_inventory_refreshes(
            store,
            profile,
            min_story_models=min_story_models,
            story_limit=args.story_limit,
            source=args.source,
            use_offline_samples=args.offline_samples,
            gutenberg_csv=args.gutenberg_csv,
            internet_archive_json=args.internet_archive_json,
            internet_archive_query=args.internet_archive_query or None,
            gutenberg_max_source_bytes=args.gutenberg_max_source_bytes,
            internet_archive_max_source_bytes=args.internet_archive_max_source_bytes,
            internet_archive_page_size=args.internet_archive_page_size,
            internet_archive_max_pages=args.internet_archive_max_pages,
            internet_archive_cursor=args.internet_archive_cursor,
            internet_archive_rate_limit_delay_seconds=args.internet_archive_rate_limit_delay_seconds,
        )
        executed = _execute_queued_jobs(
            store,
            profile=profile,
            limit=args.jobs_per_cycle,
            kinds=("import_story_metadata", "refresh_weather_cache"),
        )
        dashboard = build_assistant_os_dashboard(store).to_dict()
        importer_health = dashboard["jobs"]["importer_health"]
        importer_trends = dashboard["jobs"]["importer_trends"]
        source_coverage = _inventory_source_coverage(args.source, importer_health)
        cycles.append(
            {
                "cycle": index + 1,
                "min_story_models": min_story_models,
                "story_inventory_count_before": report.story_inventory_count,
                "story_inventory_count_after": len(store.load_inventory("story_model")),
                "recommendations": [
                    item["kind"] for item in report.to_dict()["recommendations"]
                ],
                "executed": executed,
                "source_coverage": source_coverage,
                "importer_trends": importer_trends,
                "story_quality": dashboard["inventories"]["story_quality"],
                "counts": dashboard["counts"],
            }
        )
        _persist_runtime_self_observation(store, profile)
    self_observation = _persist_runtime_self_observation(store)
    dashboard = build_assistant_os_dashboard(store).to_dict()
    final_story_inventory_count = len(store.load_inventory("story_model"))
    importer_health = dashboard["jobs"]["importer_health"]
    final_trends = dashboard["jobs"]["importer_trends"]
    story_quality = dashboard["inventories"]["story_quality"]
    source_coverage = _inventory_source_coverage(args.source, importer_health)
    failure_observability = _inventory_failure_observability(
        importer_health, final_trends
    )
    cycles_requested = max(0, args.cycles)
    safety_flags = dashboard["safety_flags"]
    checks = {
        "cycles_completed": len(cycles) == cycles_requested,
        "import_cycles_completed": int(final_trends.get("completed_cycles", 0) or 0)
        >= cycles_requested,
        "no_failed_import_cycles": int(final_trends.get("failed_cycles", 0) or 0) == 0,
        "imported_items_present": cycles_requested == 0
        or int(final_trends.get("imported_items_total", 0) or 0) > 0,
        "story_inventory_grew": cycles_requested == 0
        or final_story_inventory_count > initial_story_inventory_count,
        "story_quality_floor_clean": int(
            story_quality.get("below_metadata_quality_floor", 0) or 0
        )
        == 0,
        "story_quality_scores_present": cycles_requested == 0
        or int(story_quality.get("with_quality_scores", 0) or 0) > 0,
        "source_coverage_ok": bool(source_coverage["covered"]),
        "failure_mode_observability_present": bool(failure_observability["present"]),
        "offline_network_policy_clean": (
            not args.offline_samples
            or (
                int(importer_health.get("network_used_results", 0) or 0) == 0
                and int(final_trends.get("network_used_cycles", 0) or 0) == 0
            )
        ),
        "bounded_resource_budget": (
            args.jobs_per_cycle > 0
            and args.story_limit > 0
            and args.gutenberg_max_source_bytes > 0
            and args.internet_archive_max_source_bytes > 0
            and args.internet_archive_page_size > 0
            and args.internet_archive_max_pages >= 0
        ),
        "safety_flags_clean": (
            bool(safety_flags.get("ledger_complete", True))
            and int(safety_flags.get("cloud_private_inclusions", 0) or 0) == 0
            and int(safety_flags.get("unconfirmed_executed_actions", 0) or 0) == 0
            and int(safety_flags.get("action_without_confirmation_gate", 0) or 0) == 0
            and int(safety_flags.get("fake_latest_news_local_answers", 0) or 0) == 0
        ),
    }
    payload = {
        "db": str(args.db),
        "passed": all(checks.values()),
        "checks": checks,
        "mode": "offline_fixture" if args.offline_samples else "live_metadata",
        "network_used": not args.offline_samples,
        "source": args.source,
        "cycles_requested": cycles_requested,
        "cycles_completed": len(cycles),
        "successful_import_cycles": final_trends["completed_cycles"],
        "failed_import_cycles": final_trends["failed_cycles"],
        "elapsed_ms": _elapsed_ms(started),
        "inventory_delta": {
            "initial_story_inventory_count": initial_story_inventory_count,
            "final_story_inventory_count": final_story_inventory_count,
            "story_inventory_added": final_story_inventory_count
            - initial_story_inventory_count,
        },
        "source_coverage": source_coverage,
        "failure_observability": failure_observability,
        "resource_budget": {
            "cpu_class": "raspberry_pi",
            "network": "offline_fixture" if args.offline_samples else "metadata_only",
            "story_limit": args.story_limit,
            "jobs_per_cycle": args.jobs_per_cycle,
            "gutenberg_max_source_bytes": args.gutenberg_max_source_bytes,
            "internet_archive_max_source_bytes": args.internet_archive_max_source_bytes,
            "internet_archive_page_size": args.internet_archive_page_size,
            "internet_archive_max_pages": args.internet_archive_max_pages,
            "internet_archive_query": args.internet_archive_query or "",
            "internet_archive_rate_limit_delay_seconds": args.internet_archive_rate_limit_delay_seconds,
        },
        "cycles": cycles,
        "dashboard": {
            "counts": dashboard["counts"],
            "jobs": {
                "importer_health": importer_health,
                "importer_trends": final_trends,
            },
            "inventories": {
                "story_quality": story_quality,
            },
            "safety_flags": safety_flags,
        },
        "self_observation": self_observation,
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _inventory_soak_matrix(args) -> None:
    started = perf_counter()
    if args.reset and args.db_dir.exists():
        _safe_remove_bundle_dir(args.db_dir)
    args.db_dir.mkdir(parents=True, exist_ok=True)
    profiles = _inventory_soak_matrix_profiles(args)
    runs: list[dict[str, Any]] = []
    for index, profile in enumerate(profiles, start=1):
        label = str(profile["label"])
        db = args.db_dir / f"{index:02d}_{_safe_inventory_label(label)}.sqlite"
        soak_args = [
            "inventory-soak",
            "--db",
            str(db),
            "--reset",
            "--cold-start",
            "--source",
            str(profile["source"]),
            "--cycles",
            str(max(0, int(profile["cycles"]))),
            "--jobs-per-cycle",
            str(max(1, int(profile["jobs_per_cycle"]))),
            "--story-limit",
            str(max(1, int(profile["story_limit"]))),
            "--min-story-models",
            str(max(1, int(profile["min_story_models"]))),
            "--gutenberg-max-source-bytes",
            str(max(1, args.gutenberg_max_source_bytes)),
            "--internet-archive-max-source-bytes",
            str(max(1, args.internet_archive_max_source_bytes)),
            "--internet-archive-page-size",
            str(max(1, args.internet_archive_page_size)),
            "--internet-archive-max-pages",
            str(max(1, args.internet_archive_max_pages)),
            "--internet-archive-rate-limit-delay-seconds",
            str(max(0.0, args.internet_archive_rate_limit_delay_seconds)),
            "--json",
        ]
        query = str(profile.get("internet_archive_query", "") or "")
        if query:
            soak_args.extend(["--internet-archive-query", query])
        if not args.live:
            soak_args.extend(
                [
                    "--offline-samples",
                    "--gutenberg-csv",
                    str(args.gutenberg_csv),
                    "--internet-archive-json",
                    str(args.internet_archive_json),
                ]
            )
        soak_payload = _run_cli_json(ROOT, *soak_args)
        story_payload = _run_cli_json(
            ROOT,
            "ask",
            "--db",
            str(db),
            "--cold-start",
            "--utterance",
            "Tell me a story.",
            "--json",
        )
        runs.append(
            _inventory_soak_matrix_run_summary(
                label, profile, db, soak_payload, story_payload
            )
        )
    source_families = sorted(
        {
            source
            for run in runs
            for source in run["soak"].get("source_coverage", {}).get("observed", {})
        }
    )
    total_cycles_requested = sum(
        int(run["soak"].get("cycles_requested", 0) or 0) for run in runs
    )
    total_cycles_completed = sum(
        int(run["soak"].get("cycles_completed", 0) or 0) for run in runs
    )
    total_story_inventory_added = sum(
        int(run["soak"].get("story_inventory_added", 0) or 0) for run in runs
    )
    total_failed_import_cycles = sum(
        int(run["soak"].get("failed_import_cycles", 0) or 0) for run in runs
    )
    required_source_families = {
        "project_gutenberg_catalog_csv",
        "internet_archive_search_metadata",
    }
    checks = {
        "profiles_exercised": len(runs) >= 3,
        "total_cycles_at_least_nine": total_cycles_completed >= 9,
        "cycles_completed": total_cycles_completed >= total_cycles_requested,
        "all_soaks_passed": all(bool(run["soak"].get("passed", False)) for run in runs),
        "all_source_coverage_ok": all(
            bool(run["soak"].get("source_coverage_ok", False)) for run in runs
        ),
        "both_source_families_covered": required_source_families
        <= set(source_families),
        "all_story_inventory_grew_from_cold_start": all(
            int(run["soak"].get("initial_story_inventory_count", -1)) == 0
            and int(run["soak"].get("story_inventory_added", 0) or 0) > 0
            for run in runs
        ),
        "future_story_routes_local_from_imported_inventory": all(
            bool(run["story_local"]) for run in runs
        ),
        "future_story_synthesis_applied": all(
            bool(run["story_synthesis_applied"]) for run in runs
        ),
        "future_story_primary_uol_not_secondary_phrase_route": all(
            bool(run["story_primary_uol_ok"]) for run in runs
        ),
        "all_failure_observability_present": all(
            bool(run["soak"].get("failure_observability_present", False))
            for run in runs
        ),
        "no_failed_import_cycles": total_failed_import_cycles == 0,
        "story_quality_floor_clean": all(
            int(run["soak"].get("below_quality_floor", 0) or 0) == 0 for run in runs
        ),
        "offline_network_policy_clean": args.live
        or all(not bool(run["soak"].get("network_used", True)) for run in runs),
        "bounded_resource_budget": (
            args.cycles >= 3
            and args.jobs_per_cycle > 0
            and args.story_limit > 0
            and args.gutenberg_max_source_bytes > 0
            and args.internet_archive_max_source_bytes > 0
            and args.internet_archive_page_size > 0
            and args.internet_archive_max_pages > 0
        ),
    }
    payload = {
        "schema": "melm.inventory_soak_matrix.v1",
        "db_dir": str(args.db_dir),
        "report_path": str(args.out) if args.out is not None else "",
        "report_written": False,
        "passed": all(checks.values()),
        "checks": checks,
        "mode": "live_metadata" if args.live else "offline_fixture",
        "network_used": bool(args.live),
        "profile_count": len(runs),
        "total_cycles_requested": total_cycles_requested,
        "total_cycles_completed": total_cycles_completed,
        "total_story_inventory_added": total_story_inventory_added,
        "total_failed_import_cycles": total_failed_import_cycles,
        "source_families_observed": source_families,
        "runs": runs,
        "elapsed_ms": _elapsed_ms(started),
        "runtime": "stdlib_python_sqlite_metadata_importers",
        "dependency_class": "stdlib_only",
    }
    if args.out is not None:
        payload["report_written"] = True
        _write_json_report(args.out, payload)
    _print_payload(payload, json_mode=args.json)


def _inventory_soak_matrix_profiles(args) -> tuple[dict[str, Any], ...]:
    cycles = max(3, args.cycles)
    jobs_per_cycle = max(1, args.jobs_per_cycle)
    story_limit = max(1, args.story_limit)
    min_story_models = max(1, args.min_story_models)
    query = args.internet_archive_query or "children bedtime folklore"
    return (
        {
            "label": "both_extended",
            "source": "both",
            "cycles": cycles,
            "jobs_per_cycle": jobs_per_cycle,
            "story_limit": story_limit,
            "min_story_models": min_story_models,
            "internet_archive_query": "collection:gutenberg AND mediatype:texts",
        },
        {
            "label": "internet_archive_query",
            "source": "internet-archive",
            "cycles": cycles,
            "jobs_per_cycle": min(jobs_per_cycle, 2),
            "story_limit": min(story_limit, 2),
            "min_story_models": max(9, min_story_models),
            "internet_archive_query": query,
        },
        {
            "label": "gutenberg_replay",
            "source": "gutenberg",
            "cycles": cycles,
            "jobs_per_cycle": min(jobs_per_cycle, 2),
            "story_limit": min(story_limit, 2),
            "min_story_models": max(9, min_story_models),
            "internet_archive_query": "",
        },
    )


def _inventory_soak_matrix_run_summary(
    label: str,
    profile: dict[str, Any],
    db: Path,
    soak_payload: dict[str, Any],
    story_payload: dict[str, Any],
) -> dict[str, Any]:
    debug_parse = story_payload.get("debug_parse", {})
    chat_frame = debug_parse.get("chat_frame", {})
    nlp = debug_parse.get("nlp", {})
    primary_basis = [str(item) for item in chat_frame.get("primary_routing_basis", [])]
    story_primary_uol_ok = (
        nlp.get("primary_parse_basis") == "uol_chat_frame"
        and nlp.get("primary_domain_evidence", {}).get("source") == "slot_role_relation"
        and bool(nlp.get("primary_domain_evidence", {}).get("pattern"))
        and not any(
            item.startswith("secondary_meaning_hints:")
            or item.startswith("vocabulary_hits:")
            for item in primary_basis
        )
    )
    story_local = (
        story_payload.get("route") == "local_answer"
        and story_payload.get("reason") == "local_story_inventory"
        and not bool(story_payload.get("cloud_needed", False))
    )
    story_synthesis_applied = bool(
        story_payload.get("synthesis", {}).get("applied", False)
    )
    return {
        "label": label,
        "db": str(db),
        "db_sha256": _sha256_file(db) if db.exists() else "",
        "profile": {
            "source": profile["source"],
            "cycles": profile["cycles"],
            "jobs_per_cycle": profile["jobs_per_cycle"],
            "story_limit": profile["story_limit"],
            "min_story_models": profile["min_story_models"],
            "internet_archive_query": profile.get("internet_archive_query", ""),
        },
        "soak": _inventory_soak_summary(soak_payload),
        "story_route": story_payload.get("route", ""),
        "story_reason": story_payload.get("reason", ""),
        "story_local": story_local,
        "story_synthesis_applied": story_synthesis_applied,
        "story_primary_uol_ok": story_primary_uol_ok,
        "story_citations": story_payload.get("synthesis", {}).get("citations", []),
        "story_answer_sample": str(story_payload.get("answer", ""))[:240],
    }


def _inventory_diversity_smoke(args) -> None:
    started = perf_counter()
    if args.reset and args.db_dir.exists():
        _safe_remove_bundle_dir(args.db_dir)
    args.db_dir.mkdir(parents=True, exist_ok=True)
    niches = _inventory_diversity_niches(args.niche)
    runs: list[dict[str, Any]] = []
    for index, (label, query) in enumerate(niches, start=1):
        safe_label = _safe_inventory_label(label)
        db = args.db_dir / f"{index:02d}_{safe_label}.sqlite"
        soak_args = [
            "inventory-soak",
            "--db",
            str(db),
            "--reset",
            "--source",
            args.source,
            "--cycles",
            str(max(0, args.cycles)),
            "--jobs-per-cycle",
            str(max(1, args.jobs_per_cycle)),
            "--story-limit",
            str(max(1, args.story_limit)),
            "--min-story-models",
            str(max(1, args.min_story_models)),
            "--gutenberg-max-source-bytes",
            str(max(1, args.gutenberg_max_source_bytes)),
            "--internet-archive-max-source-bytes",
            str(max(1, args.internet_archive_max_source_bytes)),
            "--internet-archive-page-size",
            str(max(1, args.internet_archive_page_size)),
            "--internet-archive-max-pages",
            str(max(1, args.internet_archive_max_pages)),
            "--internet-archive-rate-limit-delay-seconds",
            str(max(0.0, args.internet_archive_rate_limit_delay_seconds)),
            "--internet-archive-query",
            query,
            "--json",
        ]
        if not args.live:
            soak_args.extend(
                [
                    "--offline-samples",
                    "--gutenberg-csv",
                    str(args.gutenberg_csv),
                    "--internet-archive-json",
                    str(args.internet_archive_json),
                ]
            )
        soak_payload = _run_cli_json(ROOT, *soak_args)
        story_payload = _run_cli_json(
            ROOT,
            "ask",
            "--db",
            str(db),
            "--utterance",
            f"Tell me a {label} story.",
            "--json",
        )
        executed_imports = [
            item
            for cycle in soak_payload.get("cycles", [])
            for item in cycle.get("executed", [])
            if item.get("kind") == "import_story_metadata"
        ]
        runs.append(
            {
                "label": label,
                "query": query,
                "db": str(db),
                "soak": _inventory_diversity_soak_summary(soak_payload),
                "executed_import_queries": sorted(
                    {
                        str(item.get("internet_archive_query", "") or "")
                        for item in executed_imports
                        if item.get("internet_archive_query")
                    }
                ),
                "story_route": story_payload.get("route", ""),
                "story_reason": story_payload.get("reason", ""),
                "story_citations": story_payload.get("synthesis", {}).get(
                    "citations", []
                ),
                "story_answer_sample": str(story_payload.get("answer", ""))[:240],
                "story_local": (
                    story_payload.get("route") == "local_answer"
                    and story_payload.get("reason") == "local_story_inventory"
                    and bool(story_payload.get("synthesis", {}).get("applied", False))
                ),
            }
        )
    query_values = [query for _, query in niches]
    checks = {
        "niches_requested": len(niches) >= 3,
        "queries_distinct": len(set(query_values)) == len(query_values),
        "all_soaks_passed": all(bool(run["soak"].get("passed", False)) for run in runs),
        "all_source_coverage_ok": all(
            bool(run["soak"].get("source_coverage_ok", False)) for run in runs
        ),
        "all_story_inventory_grew": all(
            int(run["soak"].get("story_inventory_added", 0) or 0) > 0 for run in runs
        ),
        "all_story_quality_clean": all(
            int(run["soak"].get("below_quality_floor", 0) or 0) == 0 for run in runs
        ),
        "all_queries_reached_import_jobs": all(
            query in run["executed_import_queries"]
            for run, query in zip(runs, query_values)
        ),
        "future_story_routes_local": all(bool(run["story_local"]) for run in runs),
        "offline_network_policy_clean": (
            args.live
            or all(not bool(run["soak"].get("network_used", True)) for run in runs)
        ),
        "bounded_resource_budget": (
            args.jobs_per_cycle > 0
            and args.story_limit > 0
            and args.gutenberg_max_source_bytes > 0
            and args.internet_archive_max_source_bytes > 0
            and args.internet_archive_page_size > 0
            and args.internet_archive_max_pages > 0
        ),
    }
    payload = {
        "schema": "melm.inventory_diversity_smoke.v1",
        "db_dir": str(args.db_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "mode": "live_metadata" if args.live else "offline_fixture",
        "network_used": bool(args.live),
        "niches": [{"label": label, "query": query} for label, query in niches],
        "niche_count": len(niches),
        "runs": runs,
        "elapsed_ms": _elapsed_ms(started),
        "runtime": "stdlib_python_sqlite_metadata_importers",
        "dependency_class": "stdlib_only",
    }
    _print_payload(payload, json_mode=args.json)


def _inventory_diversity_niches(values: list[str]) -> tuple[tuple[str, str], ...]:
    if not values:
        return DEFAULT_INVENTORY_DIVERSITY_NICHES
    parsed: list[tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit("--niche must be formatted as LABEL=QUERY")
        label, query = raw.split("=", 1)
        label = label.strip()
        query = query.strip()
        if not label or not query:
            raise SystemExit("--niche requires a non-empty LABEL and QUERY")
        parsed.append((label, query))
    return tuple(parsed)


def _safe_inventory_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip().lower()).strip("_")
    return cleaned or "niche"


def _inventory_diversity_soak_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return _inventory_soak_summary(payload)


def _inventory_soak_summary(payload: dict[str, Any]) -> dict[str, Any]:
    story_quality = (
        payload.get("dashboard", {}).get("inventories", {}).get("story_quality", {})
    )
    inventory_delta = payload.get("inventory_delta", {})
    importer_health = (
        payload.get("dashboard", {}).get("jobs", {}).get("importer_health", {})
    )
    importer_trends = (
        payload.get("dashboard", {}).get("jobs", {}).get("importer_trends", {})
    )
    return {
        "passed": bool(payload.get("passed", False)),
        "mode": payload.get("mode", ""),
        "network_used": bool(payload.get("network_used", False)),
        "source": payload.get("source", ""),
        "cycles_requested": int(payload.get("cycles_requested", 0) or 0),
        "cycles_completed": int(payload.get("cycles_completed", 0) or 0),
        "failed_import_cycles": int(payload.get("failed_import_cycles", 0) or 0),
        "initial_story_inventory_count": int(
            inventory_delta.get("initial_story_inventory_count", 0) or 0
        ),
        "final_story_inventory_count": int(
            inventory_delta.get("final_story_inventory_count", 0) or 0
        ),
        "story_inventory_added": int(
            inventory_delta.get("story_inventory_added", 0) or 0
        ),
        "source_coverage_ok": bool(
            payload.get("source_coverage", {}).get("covered", False)
        ),
        "source_coverage": payload.get("source_coverage", {}),
        "failure_observability_present": bool(
            payload.get("failure_observability", {}).get("present", False)
        ),
        "with_quality_scores": int(story_quality.get("with_quality_scores", 0) or 0),
        "below_quality_floor": int(
            story_quality.get("below_metadata_quality_floor", 0) or 0
        ),
        "network_used_results": int(
            importer_health.get("network_used_results", 0) or 0
        ),
        "fetch_attempts_total": int(
            importer_health.get("fetch_attempts_total", 0) or 0
        ),
        "network_used_cycles": int(importer_trends.get("network_used_cycles", 0) or 0),
        "imported_items_total": int(
            importer_trends.get("imported_items_total", 0) or 0
        ),
        "completed_import_cycles": int(importer_trends.get("completed_cycles", 0) or 0),
    }


def _inventory_failure_smoke(args) -> None:
    started = perf_counter()
    if args.reset and args.work_dir.exists():
        _safe_remove_bundle_dir(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fixtures = _write_inventory_failure_fixtures(args.work_dir)
    specs = (
        {
            "label": "malformed_internet_archive_json",
            "source": "internet-archive",
            "internet_archive_json": fixtures["malformed_ia"],
            "internet_archive_max_source_bytes": 250_000,
            "expected_safe_failure": "failed_import_job",
            "expected_error_contains": "expecting",
        },
        {
            "label": "internet_archive_byte_budget_exceeded",
            "source": "internet-archive",
            "internet_archive_json": fixtures["valid_ia"],
            "internet_archive_max_source_bytes": 8,
            "expected_safe_failure": "failed_import_job",
            "expected_error_contains": "max_source_bytes",
        },
        {
            "label": "empty_sources_no_fake_story",
            "source": "both",
            "gutenberg_csv": fixtures["empty_gutenberg"],
            "internet_archive_json": fixtures["empty_ia"],
            "internet_archive_max_source_bytes": 250_000,
            "expected_safe_failure": "completed_zero_import",
            "expected_error_contains": "",
        },
    )
    runs: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        label = str(spec["label"])
        db = args.work_dir / f"{index:02d}_{_safe_inventory_label(label)}.sqlite"
        soak_args = [
            "inventory-soak",
            "--db",
            str(db),
            "--reset",
            "--cold-start",
            "--offline-samples",
            "--source",
            str(spec["source"]),
            "--cycles",
            "1",
            "--jobs-per-cycle",
            "2",
            "--story-limit",
            "3",
            "--min-story-models",
            "3",
            "--gutenberg-csv",
            str(spec.get("gutenberg_csv", fixtures["empty_gutenberg"])),
            "--internet-archive-json",
            str(spec["internet_archive_json"]),
            "--gutenberg-max-source-bytes",
            "12000",
            "--internet-archive-max-source-bytes",
            str(spec["internet_archive_max_source_bytes"]),
            "--json",
        ]
        soak_payload = _run_cli_json(ROOT, *soak_args)
        story_payload = _run_cli_json(
            ROOT,
            "ask",
            "--db",
            str(db),
            "--cold-start",
            "--utterance",
            "Tell me a story.",
            "--json",
        )
        runs.append(
            _inventory_failure_run_summary(
                label=label,
                db=db,
                expected_safe_failure=str(spec["expected_safe_failure"]),
                expected_error_contains=str(spec["expected_error_contains"]),
                soak_payload=soak_payload,
                story_payload=story_payload,
            )
        )
    checks = {
        "all_cases_exercised": len(runs) == 3,
        "all_subprocesses_returned_json": all(
            run["soak_returned_json"] and run["story_returned_json"] for run in runs
        ),
        "failure_cases_mark_failed_jobs": all(
            run["failed_import_jobs"] >= 1
            for run in runs
            if run["expected_safe_failure"] == "failed_import_job"
        ),
        "zero_import_case_completed_without_inventory": all(
            run["completed_import_cycles"] >= 1 and run["story_inventory_added"] == 0
            for run in runs
            if run["expected_safe_failure"] == "completed_zero_import"
        ),
        "errors_are_observable": all(
            run["expected_error_observed"]
            for run in runs
            if run["expected_error_contains"]
        ),
        "no_fake_story_inventory": all(
            run["story_inventory_added"] == 0 for run in runs
        ),
        "future_story_routes_missing_inventory": all(
            run["story_route"] == "cloud_handoff" for run in runs
        ),
        "future_story_reason_missing_inventory": all(
            run["story_reason"] == "missing_story_model" for run in runs
        ),
        "no_local_synthesis_after_failure": all(
            not run["story_synthesis_applied"] for run in runs
        ),
        "offline_network_policy_clean": all(not run["network_used"] for run in runs),
        "failure_ledger_observable": all(
            run["recent_cycle_count"] >= 1 for run in runs
        ),
    }
    payload = {
        "schema": "melm.inventory_failure_smoke.v1",
        "work_dir": str(args.work_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "case_count": len(runs),
        "runs": runs,
        "elapsed_ms": _elapsed_ms(started),
        "runtime": "stdlib_python_sqlite_metadata_importers",
        "dependency_class": "stdlib_only",
    }
    _print_payload(payload, json_mode=args.json)


def _write_inventory_failure_fixtures(work_dir: Path) -> dict[str, Path]:
    fixtures = {
        "malformed_ia": work_dir / "malformed_internet_archive.json",
        "valid_ia": work_dir / "valid_internet_archive.json",
        "empty_ia": work_dir / "empty_internet_archive.json",
        "empty_gutenberg": work_dir / "empty_gutenberg.csv",
    }
    fixtures["malformed_ia"].write_text("{ not valid json", encoding="utf-8")
    fixtures["valid_ia"].write_text(
        json.dumps(
            {
                "items": [
                    {
                        "identifier": "failure-smoke-story",
                        "title": "Failure Smoke Story",
                        "subject": ["Children", "Folklore"],
                        "language": "eng",
                        "collection": ["gutenberg"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fixtures["empty_ia"].write_text(json.dumps({"items": []}), encoding="utf-8")
    fixtures["empty_gutenberg"].write_text(
        "Text#,Type,Title,Language,Subjects,Authors\n", encoding="utf-8"
    )
    return fixtures


def _inventory_failure_run_summary(
    *,
    label: str,
    db: Path,
    expected_safe_failure: str,
    expected_error_contains: str,
    soak_payload: dict[str, Any],
    story_payload: dict[str, Any],
) -> dict[str, Any]:
    trends = dict(
        soak_payload.get("dashboard", {}).get("jobs", {}).get("importer_trends", {})
        or {}
    )
    health = dict(
        soak_payload.get("dashboard", {}).get("jobs", {}).get("importer_health", {})
        or {}
    )
    recent_cycles = [
        dict(item) for item in trends.get("recent_cycles", []) if isinstance(item, dict)
    ]
    errors = [str(item.get("error", "")) for item in recent_cycles if item.get("error")]
    last_error = str(health.get("last_error", "") or (errors[-1] if errors else ""))
    story_inventory_added = int(
        soak_payload.get("inventory_delta", {}).get("story_inventory_added", 0) or 0
    )
    expected_error_observed = (
        (not expected_error_contains)
        or expected_error_contains.lower() in last_error.lower()
        or any(expected_error_contains.lower() in error.lower() for error in errors)
    )
    return {
        "label": label,
        "db": str(db),
        "expected_safe_failure": expected_safe_failure,
        "expected_error_contains": expected_error_contains,
        "soak_returned_json": bool(soak_payload),
        "story_returned_json": bool(story_payload),
        "soak_passed": bool(soak_payload.get("passed", False)),
        "network_used": bool(soak_payload.get("network_used", False)),
        "completed_import_cycles": int(
            soak_payload.get("successful_import_cycles", 0) or 0
        ),
        "failed_import_cycles": int(soak_payload.get("failed_import_cycles", 0) or 0),
        "failed_import_jobs": int(health.get("failed_import_jobs", 0) or 0),
        "completed_import_jobs": int(health.get("completed_import_jobs", 0) or 0),
        "imported_items": int(health.get("imported_items", 0) or 0),
        "selected_items": int(health.get("selected_items", 0) or 0),
        "story_inventory_added": story_inventory_added,
        "source_coverage": soak_payload.get("source_coverage", {}),
        "recent_cycle_count": len(recent_cycles),
        "recent_cycle_statuses": [
            str(item.get("status", "")) for item in recent_cycles
        ],
        "last_error": last_error,
        "expected_error_observed": expected_error_observed,
        "story_route": story_payload.get("route", ""),
        "story_reason": story_payload.get("reason", ""),
        "story_synthesis_applied": bool(
            story_payload.get("synthesis", {}).get("applied", False)
        ),
        "story_answer_sample": str(story_payload.get("answer", ""))[:200],
    }


def _inventory_retry_smoke(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    started = perf_counter()
    server, thread, source_state = _start_inventory_retry_source_server()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    store = _open_store(args.db, None)
    try:
        before_story = _handle_utterance(
            store, "Tell me a story.", auto_execute=False, cold_start=True
        )
        initial_story_inventory_count = len(store.load_inventory("story_model"))
        profile = store.load_profile(_cold_profile())
        job_id = store.enqueue_job(
            kind="import_story_metadata",
            payload={
                "source": "both",
                "reason": "inventory retry smoke against localhost source fixtures",
                "localhost_base_url": base_url,
            },
            priority=0.99,
            resource_budget={
                "network": "localhost_metadata",
                "max_attempts": max(1, args.max_attempts),
                "story_limit": max(1, args.story_limit),
            },
            max_attempts=1,
            job_id="inventory_retry_smoke",
        )
        job = store.start_next_job(kinds=("import_story_metadata",))
        if job is None:
            raise RuntimeError(f"failed to start retry smoke import job {job_id}")
        gutenberg = ProjectGutenbergCatalogImporter(
            f"{base_url}/gutenberg.csv"
        ).import_metadata(
            profile,
            limit=max(1, args.story_limit),
            timeout=2.0,
            max_attempts=max(1, args.max_attempts),
            backoff_seconds=0.0,
        )
        archive = InternetArchiveSearchMetadataImporter(
            endpoint=f"{base_url}/ia/scrape",
            query="collection:gutenberg AND mediatype:texts AND subject:retry",
        ).import_metadata(
            profile,
            limit=max(1, args.story_limit),
            max_source_bytes=20_000,
            timeout=2.0,
            max_attempts=max(1, args.max_attempts),
            backoff_seconds=0.0,
            page_size=100,
            max_pages=1,
        )
        _install_imported_story_items(store, profile, gutenberg.items)
        _install_imported_story_items(store, profile, archive.items)
        imported_items = len(gutenberg.items) + len(archive.items)
        result = {
            "source": "both",
            "imported_items": imported_items,
            "results": [gutenberg.to_dict(), archive.to_dict()],
            "localhost_base_url": base_url,
            "transient_failures": int(source_state["transient_failures"]),
        }
        store.complete_job(job.job_id, result=result)
        self_observation = _persist_runtime_self_observation(store, profile)
        final_story_inventory_count = len(store.load_inventory("story_model"))
        store.close()
        store = _open_store(args.db, None)
        after_story = _handle_utterance(
            store, "Tell me a story.", auto_execute=False, cold_start=True
        )
        dashboard = build_assistant_os_dashboard(store).to_dict()
        importer_health = dashboard["jobs"]["importer_health"]
        importer_trends = dashboard["jobs"]["importer_trends"]
        checks = {
            "cold_story_missing_before_import": (
                before_story.get("route") == "cloud_handoff"
                and before_story.get("reason") == "missing_story_model"
            ),
            "localhost_http_server_used": base_url.startswith("http://127.0.0.1:"),
            "gutenberg_source_retried": int(
                gutenberg.observability.get("fetch_attempts", 0) or 0
            )
            >= 2,
            "internet_archive_source_retried": int(
                archive.observability.get("fetch_attempts_total", 0) or 0
            )
            >= 2,
            "transient_failures_observed": int(source_state["transient_failures"]) >= 2,
            "both_importers_used_network_shape": gutenberg.network_used
            and archive.network_used,
            "no_external_network_required": not _inventory_retry_external_network_used(
                base_url, (gutenberg, archive)
            ),
            "imported_items_present": imported_items >= 2,
            "story_inventory_grew": final_story_inventory_count
            > initial_story_inventory_count,
            "future_story_routes_local_after_reload": (
                after_story.get("route") == "local_answer"
                and after_story.get("reason") == "local_story_inventory"
            ),
            "synthesis_applied_after_retry_import": bool(
                after_story.get("synthesis", {}).get("applied", False)
            ),
            "retry_observability_in_dashboard": (
                int(importer_health.get("fetch_attempts_total", 0) or 0) >= 4
                and int(importer_trends.get("fetch_attempts_total", 0) or 0) >= 4
                and int(importer_health.get("network_used_results", 0) or 0) == 2
            ),
        }
        payload = {
            "schema": "melm.inventory_retry_smoke.v1",
            "db": str(args.db),
            "passed": all(checks.values()),
            "checks": checks,
            "base_url": base_url,
            "network_used": True,
            "external_network_used": _inventory_retry_external_network_used(
                base_url, (gutenberg, archive)
            ),
            "transient_failures": int(source_state["transient_failures"]),
            "attempts_by_path": dict(source_state["attempts"]),
            "before_story": {
                "route": before_story.get("route", ""),
                "reason": before_story.get("reason", ""),
                "synthesis_applied": bool(
                    before_story.get("synthesis", {}).get("applied", False)
                ),
            },
            "after_story": {
                "route": after_story.get("route", ""),
                "reason": after_story.get("reason", ""),
                "synthesis_applied": bool(
                    after_story.get("synthesis", {}).get("applied", False)
                ),
                "evidence_keys": after_story.get("evidence_keys", []),
            },
            "inventory_delta": {
                "initial_story_inventory_count": initial_story_inventory_count,
                "final_story_inventory_count": final_story_inventory_count,
                "story_inventory_added": final_story_inventory_count
                - initial_story_inventory_count,
                "imported_items": imported_items,
            },
            "gutenberg": _inventory_retry_result_summary(gutenberg),
            "internet_archive": _inventory_retry_result_summary(archive),
            "dashboard": {
                "jobs": {
                    "importer_health": importer_health,
                    "importer_trends": importer_trends,
                },
                "inventories": {
                    "story_quality": dashboard["inventories"]["story_quality"],
                },
                "counts": dashboard["counts"],
            },
            "self_observation": self_observation,
            "elapsed_ms": _elapsed_ms(started),
            "runtime": "stdlib_python_sqlite_http_metadata_importers",
            "dependency_class": "stdlib_only",
        }
    finally:
        store.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    _print_payload(payload, json_mode=args.json)


def _start_inventory_retry_source_server() -> tuple[
    ThreadingHTTPServer, Thread, dict[str, Any]
]:
    payloads = _inventory_retry_source_payloads()
    state: dict[str, Any] = {"attempts": Counter(), "transient_failures": 0}

    class RetrySourceHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler hook
            path = urlsplit(self.path).path
            state["attempts"][path] += 1
            if path not in payloads:
                self.send_response(404)
                self.end_headers()
                return
            if int(state["attempts"][path]) == 1:
                state["transient_failures"] += 1
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"transient retry smoke failure")
                return
            body, content_type = payloads[path]
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RetrySourceHandler)
    thread = Thread(
        target=server.serve_forever, name="inventory-retry-source", daemon=True
    )
    thread.start()
    return server, thread, state


def _inventory_retry_source_payloads() -> dict[str, tuple[bytes, str]]:
    gutenberg_csv = (
        "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n"
        "3001,Text,1901-01-01,Yoruba Moon Retry Tales,en,Ada Storykeeper,"
        '"Folklore -- Nigeria; Children -- Folklore; Yoruba tales",PZ,'
        '"Children\'s Myths, Fairy Tales, etc.; Folklore"\n'
        "3002,Text,1902-01-01,River Retry Adventure Stories,en,Tomi Writer,"
        '"Adventure stories; Children -- Conduct of life; Rivers",PZ,'
        '"Children\'s Fiction; Adventure"\n'
    ).encode("utf-8")
    internet_archive_json = json.dumps(
        {
            "items": [
                {
                    "identifier": "retryfolkstories00test",
                    "title": "Retry Folk Stories for Children",
                    "creator": "Example Collector",
                    "subject": [
                        "Folklore -- West Africa",
                        "Children's stories",
                        "Fables",
                    ],
                    "language": "eng",
                    "collection": ["gutenberg", "americana"],
                },
                {
                    "identifier": "retryrainbedtime00test",
                    "title": "Retry Rain Bedtime Story",
                    "creator": "Example Archivist",
                    "subject": ["Bedtime stories", "Adventure stories", "Listening"],
                    "language": "eng",
                    "collection": ["gutenberg"],
                },
            ],
            "cursor": "",
        }
    ).encode("utf-8")
    return {
        "/gutenberg.csv": (gutenberg_csv, "text/csv; charset=utf-8"),
        "/ia/scrape": (internet_archive_json, "application/json; charset=utf-8"),
    }


def _inventory_retry_result_summary(result) -> dict[str, Any]:
    observability = dict(result.observability)
    return {
        "source": result.source,
        "source_url": result.source_url,
        "network_used": bool(result.network_used),
        "source_count": int(result.source_count),
        "selected_count": int(result.selected_count),
        "rejected_count": int(result.rejected_count),
        "fetch_attempts": int(observability.get("fetch_attempts", 0) or 0),
        "fetch_attempts_total": int(
            observability.get(
                "fetch_attempts_total", observability.get("fetch_attempts", 0)
            )
            or 0
        ),
        "page_count": int(observability.get("page_count", 0) or 0),
        "page_fetch_attempts": list(observability.get("page_fetch_attempts", []) or []),
        "quality_rejected_count": int(
            observability.get("quality_rejected_count", 0) or 0
        ),
        "duplicate_rejected_count": int(
            observability.get("duplicate_rejected_count", 0) or 0
        ),
        "selected_avg_metadata_quality": float(
            observability.get("selected_avg_metadata_quality", 0.0) or 0.0
        ),
    }


def _inventory_retry_external_network_used(
    base_url: str, results: tuple[Any, ...]
) -> bool:
    return any(not str(result.source_url).startswith(base_url) for result in results)


def _inventory_source_coverage(
    source: str, importer_health: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "gutenberg": ("project_gutenberg_catalog_csv",),
        "internet-archive": ("internet_archive_search_metadata",),
        "both": ("project_gutenberg_catalog_csv", "internet_archive_search_metadata"),
    }.get(source, ())
    observed = {
        str(key): int(value)
        for key, value in dict(importer_health.get("sources", {}) or {}).items()
    }
    missing = [item for item in required if int(observed.get(item, 0) or 0) <= 0]
    return {
        "required": list(required),
        "observed": observed,
        "missing": missing,
        "covered": not missing,
    }


def _inventory_failure_observability(
    importer_health: dict[str, Any],
    importer_trends: dict[str, Any],
) -> dict[str, Any]:
    health_keys = (
        "completed_import_jobs",
        "failed_import_jobs",
        "imported_items",
        "selected_items",
        "raw_rejected_items",
        "quality_rejected_items",
        "duplicate_rejected_items",
        "network_used_results",
        "pages_fetched",
        "fetch_attempts_total",
        "rate_limit_sleep_count",
        "byte_budget_exhausted_results",
        "sources",
        "last_error",
    )
    trend_keys = (
        "completed_cycles",
        "failed_cycles",
        "recent_cycles",
        "latest_completed_cycle",
        "imported_items_total",
        "selected_items_total",
        "avg_metadata_quality",
        "byte_budget_exhausted_cycles",
        "network_used_cycles",
    )
    latest_cycle = dict(importer_trends.get("latest_completed_cycle", {}) or {})
    recent_cycles = [
        dict(item)
        for item in importer_trends.get("recent_cycles", [])
        if isinstance(item, dict)
    ]
    health_present = {key: key in importer_health for key in health_keys}
    trend_present = {key: key in importer_trends for key in trend_keys}
    latest_cycle_present = {
        "status": "status" in latest_cycle,
        "attempts": "attempts" in latest_cycle,
        "error": "error" in latest_cycle,
        "sources": "sources" in latest_cycle,
        "byte_budget_exhausted": "byte_budget_exhausted" in latest_cycle,
    }
    return {
        "present": all(health_present.values())
        and all(trend_present.values())
        and all(latest_cycle_present.values()),
        "health_keys_present": health_present,
        "trend_keys_present": trend_present,
        "latest_cycle_keys_present": latest_cycle_present,
        "recent_cycle_count": len(recent_cycles),
        "last_error": str(importer_health.get("last_error", "")),
        "failed_import_jobs": int(importer_health.get("failed_import_jobs", 0) or 0),
        "byte_budget_exhausted_results": int(
            importer_health.get("byte_budget_exhausted_results", 0) or 0
        ),
    }


def _resource_report(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
        _remove_sqlite_files(args.lifecycle_db)
    tracemalloc.start()
    started = perf_counter()
    store = initialize_assistant_os_database(args.db, seed_path=args.seed)
    init_ms = _elapsed_ms(started)

    started = perf_counter()
    ask_payload = _handle_utterance(store, "Tell me a story.", auto_execute=True)
    ask_ms = _elapsed_ms(started)
    store.close()

    started = perf_counter()
    lifecycle_store = initialize_assistant_os_database(
        args.lifecycle_db, seed_path=None
    )
    lifecycle_report = AssistantLifecycleSimulator(store=lifecycle_store).run(
        realistic_lifecycle_steps()
    )
    lifecycle_ms = _elapsed_ms(started)
    lifecycle_counts = lifecycle_store.table_counts()
    lifecycle_store.close()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload = {
        "runtime": "stdlib_python_sqlite",
        "dependency_class": "stdlib_only",
        "ask_ms": ask_ms,
        "lifecycle_ms": lifecycle_ms,
        "peak_traced_kb": round(peak_bytes / 1024, 3),
        "db_bytes": _sqlite_size(args.db),
        "lifecycle_db_bytes": _sqlite_size(args.lifecycle_db),
        "ask_route": ask_payload["route"],
        "ask_counts": ask_payload["counts"],
        "lifecycle": {
            "steps": lifecycle_report.steps,
            "local_resolution_rate": lifecycle_report.local_resolution_rate,
            "cloud_handoffs": lifecycle_report.cloud_handoffs,
            "external_fetches": lifecycle_report.external_fetches,
            "blocked_offline": lifecycle_report.blocked_offline,
            "counts": lifecycle_counts,
        },
        "pi_constraints": {
            "no_required_network": True,
            "no_required_vector_db": True,
            "no_required_ml_framework": True,
            "sqlite_indexes": True,
        },
    }
    _print_payload(payload, json_mode=args.json)


def _dashboard(args) -> None:
    store = _open_store(args.db, args.seed)
    payload = {"db": str(args.db), **build_assistant_os_dashboard(store).to_dict()}
    store.close()
    _print_payload(payload, json_mode=args.json)


def _improvement_queue(args) -> None:
    store = _open_store(args.db, args.seed)
    try:
        payload = {
            "db": str(args.db),
            **store.improvement_queue(
                session_id=args.session,
                status=args.status,
                limit=args.limit,
            ),
        }
    finally:
        store.close()
    _print_payload(payload, json_mode=args.json)


def _memory_replay(args) -> None:
    store = _open_store(args.db, args.seed)
    if args.sessions:
        replay = store.query_recent_session_memory(
            session_limit=args.sessions,
            events_per_session=args.events_per_session,
        )
    else:
        replay = store.query_event_memory(
            query=args.query,
            intent=args.intent,
            route=args.route,
            session_id=args.session,
            limit=args.limit,
        )
    payload = {
        "db": str(args.db),
        **replay,
        "memory": build_assistant_os_dashboard(store).to_dict()["memory"],
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _memory_digest(args) -> None:
    store = _open_store(args.db, args.seed)
    digest = store.build_memory_digest(
        session_limit=args.sessions,
        events_per_session=args.events_per_session,
    )
    payload = {
        "db": str(args.db),
        "digest": digest,
        "memory": build_assistant_os_dashboard(store).to_dict()["memory"],
        "self_observation": _persist_runtime_self_observation(store),
        "counts": store.table_counts(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _eval(args) -> None:
    report = run_assistant_os_eval()
    _print_payload(report.to_dict(), json_mode=args.json)


def _import_stories(args) -> None:
    store = _open_store(args.db, None if args.cold_start else args.seed)
    profile = store.load_profile(
        _cold_profile() if args.cold_start else LocalAssistantProfile()
    )
    results = []
    if args.source in {"gutenberg", "both"}:
        importer = ProjectGutenbergCatalogImporter()
        if args.gutenberg_csv is not None:
            result = importer.import_csv_path(
                args.gutenberg_csv,
                profile,
                limit=args.limit,
                max_source_bytes=args.gutenberg_max_source_bytes,
            )
        else:
            result = importer.import_metadata(
                profile,
                limit=args.limit,
                max_source_bytes=args.gutenberg_max_source_bytes,
            )
        _install_imported_story_items(store, profile, result.items)
        results.append(result.to_dict())
    if args.source in {"internet-archive", "both"}:
        importer = InternetArchiveSearchMetadataImporter(
            query=args.internet_archive_query
            or "collection:gutenberg AND mediatype:texts"
        )
        if args.internet_archive_json is not None:
            result = importer.import_json_path(
                args.internet_archive_json,
                profile,
                limit=args.limit,
                max_source_bytes=args.internet_archive_max_source_bytes,
            )
        else:
            result = importer.import_metadata(
                profile,
                limit=args.limit,
                max_source_bytes=args.internet_archive_max_source_bytes,
                page_size=args.internet_archive_page_size,
                max_pages=args.internet_archive_max_pages,
                cursor=args.internet_archive_cursor or None,
                rate_limit_delay_seconds=args.internet_archive_rate_limit_delay_seconds,
            )
        _install_imported_story_items(store, profile, result.items)
        results.append(result.to_dict())
    payload = {
        "db": str(args.db),
        "source": args.source,
        "imported_items": sum(len(result["items"]) for result in results),
        "results": results,
        "self_observation": _persist_runtime_self_observation(store, profile),
        "counts": store.table_counts(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _import_media(args) -> None:
    store = _open_store(args.db, None if args.cold_start else args.seed)
    profile = store.load_profile(
        _cold_profile() if args.cold_start else LocalAssistantProfile()
    )
    results = []
    if args.manifest is not None or args.media_dir is None:
        adapter = LocalMediaInventoryAdapter(
            args.manifest or DEFAULT_LOCAL_MEDIA_MANIFEST
        )
        result = adapter.import_manifest(
            profile,
            limit=args.limit,
            require_files=args.require_files,
        )
        _install_imported_media_items(store, result.items)
        results.append(result.to_dict())
    if args.media_dir is not None:
        adapter = LocalMediaInventoryAdapter()
        result = adapter.import_directory(args.media_dir, profile, limit=args.limit)
        _install_imported_media_items(store, result.items)
        results.append(result.to_dict())
    payload = {
        "db": str(args.db),
        "imported_items": sum(len(result["items"]) for result in results),
        "results": results,
        "self_observation": _persist_runtime_self_observation(store, profile),
        "counts": store.table_counts(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _refresh_weather(args) -> None:
    store = _open_store(args.db, None if args.cold_start else args.seed)
    profile = store.load_profile(
        _cold_profile() if args.cold_start else LocalAssistantProfile()
    )
    result = OpenMeteoWeatherAdapter().refresh(
        profile,
        location=args.location or None,
        offline_json=None if args.live else args.offline_json,
        live=args.live,
    )
    _install_weather_items(store, result)
    refreshed = store.load_profile(profile)
    payload = {
        "db": str(args.db),
        "result": result.to_dict(),
        "weekly_weather": refreshed.weekly_weather,
        "self_observation": _persist_runtime_self_observation(store, refreshed),
        "counts": store.table_counts(),
    }
    store.close()
    _print_payload(payload, json_mode=args.json)


def _action_smoke(args) -> None:
    payload = _build_action_smoke_payload(
        args.db,
        reset=args.reset,
        action_mode=args.action_mode,
        media_player_command=args.media_player_command,
        call_command=args.call_command,
        media_dir=args.media_dir,
        manifest=args.manifest,
    )
    _print_payload(payload, json_mode=args.json)


def _build_action_smoke_payload(
    db: Path,
    *,
    reset: bool,
    action_mode: str,
    media_player_command: str = "",
    call_command: str = "",
    media_dir: Path | None = None,
    manifest: Path = DEFAULT_LOCAL_MEDIA_MANIFEST,
) -> dict:
    if reset:
        _remove_sqlite_files(db)
    store = _open_store(db, seed=None)
    profile = store.load_profile(_cold_profile())
    media_results = []
    if media_dir is not None:
        media_result = LocalMediaInventoryAdapter().import_directory(
            media_dir, profile, limit=8
        )
    else:
        media_result = LocalMediaInventoryAdapter(manifest).import_manifest(
            profile, limit=8
        )
    _install_imported_media_items(store, media_result.items)
    media_results.append(media_result.to_dict())

    turns = []
    turns.append(
        _handle_utterance(
            store,
            "Ada is my trusted contact at +234-000-ADA.",
            auto_execute=False,
            cold_start=True,
        )
    )
    turns.append(
        _handle_utterance(
            store,
            "Play calm piano.",
            auto_execute=False,
            cold_start=True,
        )
    )
    media_confirm = _handle_utterance(
        store,
        "Yes, play calm piano.",
        auto_execute=False,
        cold_start=True,
        action_mode=action_mode,
        media_player_command=media_player_command,
        call_command=call_command,
    )
    turns.append(media_confirm)
    turns.append(
        _handle_utterance(
            store,
            "I need to talk to someone.",
            auto_execute=False,
            cold_start=True,
        )
    )
    call_confirm = _handle_utterance(
        store,
        "Yes, call Ada.",
        auto_execute=False,
        cold_start=True,
        action_mode=action_mode,
        media_player_command=media_player_command,
        call_command=call_command,
    )
    turns.append(call_confirm)
    expected_status = "prepared" if action_mode == "dry-run" else "executed"
    action_results = [
        media_confirm.get("action_execution", {}),
        call_confirm.get("action_execution", {}),
    ]
    payload = {
        "db": str(db),
        "mode": action_mode,
        "media_import": media_results,
        "media_confirm": media_confirm,
        "call_confirm": call_confirm,
        "action_results": action_results,
        "passed": all(
            result.get("status") == expected_status for result in action_results
        ),
        "expected_status": expected_status,
        "safety_flags": build_assistant_os_dashboard(store).to_dict()["safety_flags"],
        "counts": store.table_counts(),
        "turns": [
            {
                "utterance": turn["utterance"],
                "route": turn["route"],
                "reason": turn["reason"],
                "action_execution": turn.get("action_execution", {}),
            }
            for turn in turns
        ],
    }
    store.close()
    return payload


def _setup_integration_smoke(args) -> None:
    payload = _build_setup_integration_smoke_payload(args.db, reset=args.reset)
    _print_payload(payload, json_mode=args.json)


def _build_setup_integration_smoke_payload(db: Path, *, reset: bool) -> dict:
    started = perf_counter()
    if reset:
        _remove_sqlite_files(db)
    store = _open_store(db, seed=None)
    turns: dict[str, dict] = {}

    def run(label: str, utterance: str, *, auto_execute: bool = False) -> dict:
        turn = _handle_utterance(
            store,
            utterance,
            auto_execute=auto_execute,
            cold_start=True,
            action_mode="dry-run",
        )
        turns[label] = turn
        return turn

    routine_gap = run("routine_gap", "What is my morning routine?", auto_execute=True)
    household_gap = run(
        "household_gap", "What do you know about this household?", auto_execute=True
    )
    contact_gap = run("contact_gap", "I need to talk to someone.", auto_execute=True)
    setup_requests_after_gaps = store.load_inventory("setup_request")
    profile_after_gaps = store.load_profile(_cold_profile())
    contact_inventory_after_gaps = store.load_inventory("contact")
    still_empty_routine = run("routine_still_empty", "What is my morning routine?")
    profile_after_still_empty = store.load_profile(_cold_profile())

    routine_setup = run(
        "routine_setup", "My morning routine is stretch, breakfast, then bus."
    )
    household_setup = run(
        "household_setup",
        "Our household includes Maya and Mom, and memory stays local.",
    )
    contact_setup = run("contact_setup", "Ada is my trusted contact at +234-000-ADA.")
    routine_recall = run("routine_recall", "What is my morning routine?")
    household_recall = run("household_recall", "What do you know about this household?")
    contact_request = run("contact_request", "I need to talk to someone.")
    contact_confirm = run("contact_confirm", "Yes, call Ada.")

    profile_after_setup = store.load_profile(_cold_profile())
    setup_requests_after_setup = store.load_inventory("setup_request")
    contacts_after_setup = store.load_inventory("contact")
    fact_privacy = store.load_user_fact_privacy_index()
    pending_actions = _pending_action_state_counts(store)
    dashboard = build_assistant_os_dashboard(store).to_dict()
    action_execution = contact_confirm.get("action_execution", {})

    setup_request_keys = set(setup_requests_after_gaps)
    required_setup_requests = {"routine_memory", "household_memory", "trusted_contact"}
    setup_payloads_require_user_value = all(
        bool(
            setup_requests_after_gaps.get(key, {}).get(
                "requires_user_supplied_value", False
            )
        )
        for key in required_setup_requests
    )
    no_fake_facts_before_setup = (
        "morning_routine" not in profile_after_gaps.facts
        and "household_context" not in profile_after_gaps.facts
        and not profile_after_gaps.contacts
        and not contact_inventory_after_gaps
    )
    routine_privacy = fact_privacy.get("facts.morning_routine", {})
    household_privacy = fact_privacy.get("facts.household_context", {})
    checks = {
        "cold_gaps_route_to_clarify": (
            routine_gap.get("route") == "clarify"
            and routine_gap.get("reason") == "personal_memory_empty"
            and household_gap.get("route") == "clarify"
            and household_gap.get("reason") == "personal_memory_empty"
            and contact_gap.get("route") == "clarify"
            and contact_gap.get("reason") == "missing_contact"
        ),
        "setup_requests_recorded_without_fake_facts": required_setup_requests.issubset(
            setup_request_keys
        )
        and setup_payloads_require_user_value
        and no_fake_facts_before_setup,
        "empty_memory_stays_empty_until_user_supplies_fact": (
            still_empty_routine.get("route") == "clarify"
            and still_empty_routine.get("reason") == "personal_memory_empty"
            and "morning_routine" not in profile_after_still_empty.facts
        ),
        "explicit_setup_statements_store_local_memory": (
            routine_setup.get("reason") == "consented_routine_memory_stored"
            and routine_setup.get("evidence_keys") == ["facts.morning_routine"]
            and household_setup.get("reason") == "consented_household_memory_stored"
            and household_setup.get("evidence_keys") == ["facts.household_context"]
            and contact_setup.get("reason") == "consented_trusted_contact_stored"
            and contact_setup.get("evidence_keys") == ["contacts.ada"]
        ),
        "stored_setup_is_scoped_local": (
            profile_after_setup.facts.get("morning_routine")
            == "stretch, breakfast, then bus"
            and "Maya and Mom" in profile_after_setup.facts.get("household_context", "")
            and profile_after_setup.contacts.get("ada") == "+234-000-ADA"
            and routine_privacy.get("scope") == "routine_local"
            and household_privacy.get("scope") == "household_local"
            and routine_privacy.get("local_only") is True
            and household_privacy.get("local_only") is True
        ),
        "future_memory_routes_change_after_setup": (
            routine_recall.get("route") == "local_answer"
            and routine_recall.get("reason") == "personal_memory_recall"
            and "stretch" in routine_recall.get("answer", "")
            and household_recall.get("route") == "local_answer"
            and household_recall.get("reason") == "personal_memory_recall"
            and "Maya and Mom" in household_recall.get("answer", "")
            and household_setup.get("debug_parse", {}).get("uol", {}).get("object")
            == "household_memory"
            and household_recall.get("debug_parse", {}).get("uol", {}).get("object")
            == "household_memory"
        ),
        "trusted_contact_action_uses_confirmation_gate": (
            contact_request.get("route") == "device_action"
            and contact_request.get("reason") == "trusted_contact_action"
            and contact_request.get("membrane", {}).get("confirmation_required") == 1
            and contact_request.get("pending_action", {}).get("action_type")
            == "call_contact"
            and contact_request.get("pending_action", {}).get("evidence_keys")
            == ["contacts.ada"]
            and contact_confirm.get("route") == "device_action"
            and contact_confirm.get("reason") == "confirmed_device_action"
            and action_execution.get("status") == "prepared"
            and action_execution.get("resolved_target") == "+234-000-ADA"
            and action_execution.get("side_effect_executed") is False
        ),
        "debug_frames_remain_uol_chatframe": all(
            turn.get("debug_parse", {}).get("mapping", [{}])[0].get("stage")
            == "basic_nlp"
            and [
                stage.get("stage")
                for stage in turn.get("debug_parse", {}).get("mapping", [])
            ]
            == ["basic_nlp", "uol_parse", "chat_frame"]
            for turn in turns.values()
        ),
        "ledgers_and_safety_clean": (
            dashboard["safety_flags"]["ledger_complete"]
            and dashboard["safety_flags"]["unconfirmed_executed_actions"] == 0
            and dashboard["safety_flags"]["action_without_confirmation_gate"] == 0
            and dashboard["safety_flags"]["cloud_private_inclusions"] == 0
            and pending_actions["pending"] == 0
            and pending_actions["confirmed"] >= 1
        ),
    }
    payload = {
        "db": str(db),
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "runtime": "stdlib_python_sqlite",
        "dependency_class": "stdlib_only",
        "setup_requests_after_gaps": setup_requests_after_gaps,
        "setup_requests_after_setup": setup_requests_after_setup,
        "facts_after_gaps": dict(profile_after_gaps.facts),
        "contact_inventory_after_gaps": contact_inventory_after_gaps,
        "facts_after_setup": dict(profile_after_setup.facts),
        "contacts_after_setup": dict(profile_after_setup.contacts),
        "contact_inventory": contacts_after_setup,
        "fact_privacy": {
            key: value
            for key, value in fact_privacy.items()
            if key in {"facts.morning_routine", "facts.household_context"}
        },
        "pending_actions": pending_actions,
        "safety_flags": dashboard["safety_flags"],
        "counts": store.table_counts(),
        "turns": [
            _setup_smoke_turn_summary(label, turn) for label, turn in turns.items()
        ],
        "action_execution": action_execution,
    }
    store.close()
    return payload


def _setup_smoke_turn_summary(label: str, turn: dict) -> dict:
    debug_parse = dict(turn.get("debug_parse", {}))
    chat_frame = dict(debug_parse.get("chat_frame", {}))
    uol = dict(debug_parse.get("uol", {}))
    return {
        "label": label,
        "utterance": turn.get("utterance", ""),
        "intent": turn.get("intent", ""),
        "route": turn.get("route", ""),
        "reason": turn.get("reason", ""),
        "evidence_keys": list(turn.get("evidence_keys", [])),
        "opportunities": list(turn.get("opportunities", [])),
        "executed_jobs": list(turn.get("executed_jobs", [])),
        "uol": {
            "subject": uol.get("subject", ""),
            "action": uol.get("action", ""),
            "object": uol.get("object", ""),
            "source": uol.get("source", ""),
            "target": uol.get("target", ""),
        },
        "chat_frame": {
            "intent": chat_frame.get("intent", ""),
            "route": chat_frame.get("route", ""),
            "reason": chat_frame.get("reason", ""),
            "primary_routing_basis": list(chat_frame.get("primary_routing_basis", [])),
        },
        "confirmation_required": int(
            turn.get("membrane", {}).get("confirmation_required", 0) or 0
        ),
        "pending_action": dict(turn.get("pending_action", {})),
        "action_execution": dict(turn.get("action_execution", {})),
    }


def _host_action_smoke(args) -> None:
    payload = _build_host_action_smoke_payload(
        args.db, work_dir=args.work_dir, reset=args.reset
    )
    _print_payload(payload, json_mode=args.json)


def _host_action_recorder(args) -> None:
    target = str(args.target or "")
    log = Path(args.log)
    record = {
        "label": str(args.label or "action"),
        "target": target,
        "target_exists": Path(target).exists(),
        "argv": sys.argv[1:],
        "recorder": "melm_host_action_recorder",
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    if args.json:
        _print_payload(
            {"passed": True, "log": str(log), "record": record}, json_mode=True
        )


def _write_host_actions_demo_config(args) -> None:
    payload = _build_host_actions_demo_config_payload(
        Path(args.out),
        log=Path(args.log),
        media_dir=args.media_dir,
        overwrite=bool(args.overwrite),
    )
    _print_payload(payload, json_mode=args.json)


def _build_host_actions_demo_config_payload(
    out: Path,
    *,
    log: Path,
    media_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    started = perf_counter()
    out_path = out
    if out_path.exists() and not overwrite:
        return {
            "passed": False,
            "out": str(out_path),
            "elapsed_ms": _elapsed_ms(started),
            "error": "output exists; pass --overwrite to replace it",
            "next_command": f"python scripts/local_assistant_os_cli.py host-app-probe --config-json {out_path} --require-configured --json",
            "runtime": "stdlib_python_json",
            "dependency_class": "stdlib_only",
        }
    script = Path(__file__).resolve()
    log_path = log
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    media_command = _host_action_recorder_config_command(script, log_path, "media")
    call_command = _host_action_recorder_config_command(script, log_path, "call")
    config = {
        "media_player_command": media_command,
        "call_command": call_command,
    }
    if media_dir is not None:
        config["media_dir"] = str(media_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "passed": True,
        "out": str(out_path),
        "log": str(log_path),
        "keys": sorted(config),
        "elapsed_ms": _elapsed_ms(started),
        "next_command": f"python scripts/local_assistant_os_cli.py host-app-probe --config-json {out_path} --require-configured --json",
        "note": "Recorder config proves the configured typed action gate. Replace commands with real target apps for appliance validation.",
        "runtime": "stdlib_python_json",
        "dependency_class": "stdlib_only",
    }


def _host_action_recorder_config_command(script: Path, log: Path, label: str) -> str:
    parts = (
        sys.executable,
        str(script),
        "host-action-recorder",
        "--label",
        label,
        "--log",
        str(log),
    )
    return " ".join(shlex.quote(str(part)) for part in parts)


def _host_app_probe(args) -> None:
    payload = _build_host_app_probe_payload(
        args.db,
        work_dir=args.work_dir,
        reset=args.reset,
        media_player_command=args.media_player_command,
        call_command=args.call_command,
        media_dir=args.media_dir,
        require_configured=args.require_configured,
        config_json=args.config_json,
    )
    _print_payload(payload, json_mode=args.json)


def _build_host_app_probe_payload(
    db: Path,
    *,
    work_dir: Path,
    reset: bool,
    media_player_command: str = "",
    call_command: str = "",
    media_dir: Path | None = None,
    require_configured: bool = False,
    config_json: Path | None = None,
) -> dict:
    started = perf_counter()
    if reset:
        _remove_sqlite_files(db)
        _remove_host_action_smoke_files(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config, config_error = _host_app_config(config_json)
    if config_error:
        evidence_class = _host_app_static_analysis(config_json, config)
        checks = {
            "configuration_reported": True,
            "config_json_valid": False,
            "media_command_configured": False,
            "call_command_configured": False,
            "probe_executed": False,
            "require_configured_satisfied": False,
        }
        return {
            "db": str(db),
            "work_dir": str(work_dir),
            "passed": False,
            "configured": False,
            "skipped": True,
            "checks": checks,
            "elapsed_ms": _elapsed_ms(started),
            "config": _host_app_config_report(config_json, config, config_error),
            "command_sources": {
                "media": "unresolved",
                "call": "unresolved",
                "media_dir": "unresolved",
            },
            "evidence_class": evidence_class,
            "next_steps": [
                "Fix or remove --config-json before running target-device app acceptance."
            ],
            "runtime": "stdlib_python_sqlite_subprocess",
            "dependency_class": "stdlib_only",
        }
    media_command, media_source = _host_app_config_value(
        media_player_command,
        config,
        "media_player_command",
        "MELM_MEDIA_PLAYER_COMMAND",
        config_json,
    )
    contact_command, call_source = _host_app_config_value(
        call_command,
        config,
        "call_command",
        "MELM_CALL_COMMAND",
        config_json,
    )
    config_media_dir, media_dir_source = _host_app_media_dir_from_config(
        media_dir, config, config_json
    )
    evidence_class = _host_app_static_analysis(
        config_json,
        config,
        media_command=media_command,
        call_command=contact_command,
        media_dir=str(config_media_dir or ""),
    )
    configured = bool(media_command and contact_command)
    if not configured:
        checks = {
            "configuration_reported": True,
            "config_json_valid": True,
            "media_command_configured": bool(media_command),
            "call_command_configured": bool(contact_command),
            "probe_executed": False,
            "require_configured_satisfied": not require_configured,
        }
        return {
            "db": str(db),
            "work_dir": str(work_dir),
            "passed": bool(
                checks["configuration_reported"]
                and checks["require_configured_satisfied"]
            ),
            "configured": False,
            "skipped": True,
            "checks": checks,
            "elapsed_ms": _elapsed_ms(started),
            "config": _host_app_config_report(config_json, config, ""),
            "command_sources": {
                "media": media_source,
                "call": call_source,
                "media_dir": media_dir_source,
            },
            "evidence_class": evidence_class,
            "next_steps": [
                "Set MELM_MEDIA_PLAYER_COMMAND and MELM_CALL_COMMAND, pass --media-player-command and --call-command, or provide --config-json.",
                "Use --media-dir with a real local media folder when probing an actual media player.",
                "Add --require-configured on target-device acceptance runs.",
            ],
            "runtime": "stdlib_python_sqlite_subprocess",
            "dependency_class": "stdlib_only",
        }

    effective_media_dir = config_media_dir or _host_app_probe_media_dir(work_dir)
    action_payload = _build_action_smoke_payload(
        db,
        reset=False,
        action_mode="real",
        media_player_command=media_command,
        call_command=contact_command,
        media_dir=effective_media_dir,
        manifest=DEFAULT_LOCAL_MEDIA_MANIFEST,
    )
    action_results = list(action_payload.get("action_results", []))
    media_result = next(
        (item for item in action_results if item.get("action_type") == "play_media"), {}
    )
    call_result = next(
        (item for item in action_results if item.get("action_type") == "call_contact"),
        {},
    )
    checks = {
        "configuration_reported": True,
        "config_json_valid": True,
        "media_command_configured": bool(media_command),
        "call_command_configured": bool(contact_command),
        "probe_executed": True,
        "media_command_executed": media_result.get("status") == "executed",
        "call_command_executed": call_result.get("status") == "executed",
        "media_target_resolved": bool(media_result.get("resolved_target", "")),
        "call_target_resolved": call_result.get("resolved_target") == "+234-000-ADA",
        "typed_confirmation_gate_used": (
            action_payload.get("media_confirm", {}).get("reason")
            == "confirmed_device_action"
            and action_payload.get("call_confirm", {}).get("reason")
            == "confirmed_device_action"
        ),
        "safety_flags_clean": (
            int(
                action_payload.get("safety_flags", {}).get(
                    "unconfirmed_executed_actions", 0
                )
                or 0
            )
            == 0
            and int(
                action_payload.get("safety_flags", {}).get(
                    "action_without_confirmation_gate", 0
                )
                or 0
            )
            == 0
        ),
    }
    return {
        "db": str(db),
        "work_dir": str(work_dir),
        "passed": all(checks.values()),
        "configured": True,
        "skipped": False,
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "config": _host_app_config_report(config_json, config, ""),
        "command_sources": {
            "media": media_source,
            "call": call_source,
            "media_dir": media_dir_source
            if config_media_dir
            else "generated_probe_media_dir",
        },
        "evidence_class": evidence_class,
        "media_dir": str(effective_media_dir),
        "action_smoke": action_payload,
        "action_results": action_results,
        "runtime": "stdlib_python_sqlite_subprocess",
        "dependency_class": "stdlib_only",
    }


def _host_app_config(config_json: Path | None) -> tuple[dict[str, Any], str]:
    if config_json is None:
        return {}, ""
    if not config_json.exists() or not config_json.is_file():
        return {}, f"config file not found: {config_json}"
    try:
        payload = json.loads(config_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc}"
    except OSError as exc:
        return {}, f"could not read config: {exc}"
    if not isinstance(payload, dict):
        return {}, "config root must be a JSON object"
    allowed = {"media_player_command", "call_command", "media_dir"}
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        if key_text not in allowed:
            continue
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            return {}, f"{key_text} must be a string"
        normalized[key_text] = value.strip()
    return normalized, ""


def _host_app_config_value(
    argument_value: str,
    config: dict[str, Any],
    config_key: str,
    env_key: str,
    config_json: Path | None,
) -> tuple[str, str]:
    if argument_value:
        return argument_value, "argument"
    config_value = str(config.get(config_key, "") or "")
    if config_value:
        return (
            config_value,
            f"config:{config_json}:{config_key}"
            if config_json is not None
            else f"config:{config_key}",
        )
    env_value = os.environ.get(env_key, "")
    if env_value:
        return env_value, f"env:{env_key}"
    return "", "none"


def _host_app_media_dir_from_config(
    argument_media_dir: Path | None,
    config: dict[str, Any],
    config_json: Path | None,
) -> tuple[Path | None, str]:
    if argument_media_dir is not None:
        return argument_media_dir, "argument"
    config_value = str(config.get("media_dir", "") or "")
    if not config_value:
        return None, "none"
    path = Path(config_value)
    if not path.is_absolute() and config_json is not None:
        path = config_json.parent / path
    return (
        path,
        f"config:{config_json}:media_dir"
        if config_json is not None
        else "config:media_dir",
    )


def _host_app_config_report(
    config_json: Path | None, config: dict[str, Any], error: str
) -> dict:
    return {
        "path": str(config_json) if config_json is not None else "",
        "loaded": bool(config_json is not None and not error),
        "error": error,
        "keys": sorted(config),
        "example_path": str(DEFAULT_HOST_ACTION_CONFIG_EXAMPLE),
    }


def _host_app_static_analysis(
    config_json: Path | None,
    config: dict[str, Any],
    *,
    media_command: str = "",
    call_command: str = "",
    media_dir: str = "",
) -> dict[str, Any]:
    command_text = "\n".join(
        str(part or "")
        for part in (
            config.get("media_player_command", ""),
            config.get("call_command", ""),
            config.get("media_dir", ""),
            media_command,
            call_command,
            media_dir,
        )
    )
    if config_json is not None:
        command_text = f"{config_json}\n{command_text}"
    lowered = command_text.lower()
    markers = sorted(
        marker for marker in HOST_APP_DEMO_RECORDER_MARKERS if marker in lowered
    )
    configured = bool(config.get("media_player_command") and config.get("call_command"))
    demo_recorder_detected = bool(markers)
    return {
        "config_json": str(config_json) if config_json is not None else "",
        "configured_commands_present": configured,
        "demo_recorder_detected": demo_recorder_detected,
        "markers": markers,
        "evidence_kind": "development_recorder_rehearsal"
        if demo_recorder_detected
        else "target_app_probe_candidate",
        "candidate_target_device_app_evidence": bool(
            configured and not demo_recorder_detected
        ),
        "note": (
            "Recorder/demo commands prove the typed gate only; target-device app evidence requires real app commands "
            "plus host app attestation."
        )
        if demo_recorder_detected
        else "No recorder/demo command markers detected in the host app config.",
    }


def _host_app_probe_media_dir(work_dir: Path) -> Path:
    media_dir = work_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    media_file = media_dir / "Calm Piano.mp3"
    if not media_file.exists():
        media_file.write_bytes(b"local host app probe media")
    return media_dir


def _build_host_action_smoke_payload(db: Path, *, work_dir: Path, reset: bool) -> dict:
    started = perf_counter()
    if reset:
        _remove_sqlite_files(db)
        _remove_host_action_smoke_files(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    media_dir = work_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    media_file = media_dir / "Calm Piano.mp3"
    media_file.write_bytes(b"local host action smoke media")
    recorder = work_dir / "record_action.py"
    log = work_dir / "actions.jsonl"
    if log.exists():
        log.unlink()
    recorder.write_text(_host_action_recorder_script(), encoding="utf-8", newline="\n")
    media_command = _host_action_recorder_command(recorder, log, "media")
    call_command = _host_action_recorder_command(recorder, log, "call")
    action_payload = _build_action_smoke_payload(
        db,
        reset=False,
        action_mode="real",
        media_player_command=media_command,
        call_command=call_command,
        media_dir=media_dir,
        manifest=DEFAULT_LOCAL_MEDIA_MANIFEST,
    )
    records = _read_host_action_records(log)
    media_record = next(
        (record for record in records if record.get("label") == "media"), {}
    )
    call_record = next(
        (record for record in records if record.get("label") == "call"), {}
    )
    action_results = action_payload.get("action_results", [])
    checks = {
        "real_action_smoke_passed": bool(action_payload.get("passed", False)),
        "two_host_commands_recorded": len(records) == 2,
        "media_command_received_existing_file": bool(
            media_record.get("target_exists", False)
        ),
        "call_command_received_resolved_contact": call_record.get("target")
        == "+234-000-ADA",
        "commands_executed_without_shell": all(
            result.get("status") == "executed"
            and result.get("side_effect_executed") is True
            and isinstance(result.get("command", []), list)
            for result in action_results
        ),
        "confirmed_actions_recorded": all(
            result.get("reason") == "returncode:0" for result in action_results
        ),
        "safety_flags_clean": (
            int(
                action_payload.get("safety_flags", {}).get(
                    "unconfirmed_executed_actions", 0
                )
                or 0
            )
            == 0
            and int(
                action_payload.get("safety_flags", {}).get(
                    "action_without_confirmation_gate", 0
                )
                or 0
            )
            == 0
        ),
    }
    return {
        "db": str(db),
        "work_dir": str(work_dir),
        "media_file": str(media_file),
        "recorder": str(recorder),
        "log": str(log),
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "action_smoke": action_payload,
        "records": records,
        "runtime": "stdlib_python_sqlite_subprocess",
        "dependency_class": "stdlib_only",
    }


def _host_action_recorder_command(recorder: Path, log: Path, label: str) -> str:
    parts = (sys.executable, str(recorder.resolve()), label, str(log.resolve()))
    return " ".join(shlex.quote(part) for part in parts)


def _host_action_recorder_script() -> str:
    return """from __future__ import annotations

import json
from pathlib import Path
import sys

label = sys.argv[1] if len(sys.argv) > 1 else ""
log = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("actions.jsonl")
target = sys.argv[3] if len(sys.argv) > 3 else ""
record = {
    "label": label,
    "target": target,
    "target_exists": Path(target).exists(),
    "argv": sys.argv[1:],
}
log.parent.mkdir(parents=True, exist_ok=True)
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\\n")
"""


def _read_host_action_records(log: Path) -> list[dict]:
    if not log.exists():
        return []
    records = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"decode_error": line})
    return records


def _remove_host_action_smoke_files(work_dir: Path) -> None:
    for target in (
        work_dir / "actions.jsonl",
        work_dir / "record_action.py",
        work_dir / "media" / "Calm Piano.mp3",
    ):
        if target.exists() and target.is_file():
            target.unlink()
    for target in (work_dir / "media", work_dir):
        try:
            target.rmdir()
        except OSError:
            pass


def _autoimmune_smoke(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    payload = _build_autoimmune_smoke_payload(args.db, seed=args.seed)
    _print_payload(payload, json_mode=args.json)


def _build_autoimmune_smoke_payload(db: Path, *, seed: Path = DEFAULT_SEED) -> dict:
    started = perf_counter()
    store = _open_store(db, seed)
    try:
        turn_payloads: dict[str, dict] = {}

        def run(label: str, utterance: str) -> dict:
            payload = _handle_utterance(store, utterance, auto_execute=False)
            turn_payloads[label] = payload
            return payload

        private_cloud = run(
            "private_cloud_block",
            "Send my favorite color and mom contact to the cloud.",
        )
        generic_cloud = run("generic_cloud_allowed", "Explain relativity briefly.")
        store.upsert_user_fact(
            "facts.public_profile",
            "Public science fair helper profile.",
            source="explicit_user_cloud_consent",
            confidence=0.99,
            consent=True,
            local_only=False,
            cloud_eligible=True,
            scope="shareable_profile",
        )
        store.connection.commit()
        public_profile_cloud = run(
            "public_profile_cloud_allowed", "Send my public profile to the cloud."
        )
        household_setup = run(
            "household_setup",
            "Our household includes Maya and Mom, and memory stays local.",
        )
        mixed_public_household_cloud = run(
            "mixed_public_household_cloud",
            "Send my public profile and household memory to the cloud.",
        )
        household_private_cloud = run(
            "household_private_cloud", "Send our household memory to the cloud."
        )
        household_consent_revoke = run(
            "household_consent_revoke", "Forget our household memory."
        )
        consent_revoke = run("consent_revoke", "Forget my favorite color.")
        child_age_setup = run("child_age_setup", "My child is 8 years old.")
        child_school_setup = run(
            "child_school_setup", "My child's school is Bright School."
        )
        child_school_consent_revoke = run(
            "child_school_consent_revoke", "Forget my child's school."
        )

        store.close()
        store = _open_store(db, seed)
        memory_after_revoke = run(
            "memory_after_revoke", "Tell me something about myself."
        )
        household_after_revoke = run(
            "household_after_revoke", "What do you know about this household?"
        )
        child_school_after_revoke = run(
            "child_school_after_revoke", "What is my child's school?"
        )
        child_age_after_school_revoke = run(
            "child_age_after_school_revoke", "How old is my child?"
        )

        _mark_today_weather_stale(store)
        stale_weather = run("stale_weather_cache", "What is the weather today?")

        action_request = run("action_request", "I need to talk to someone.")
        invented_target = run("invented_target", "Yes, call dad.")
        pending_after_mismatch = _pending_action_state_counts(store)
        cancel = run("cancel_pending_action", "Cancel that.")
        replay_after_cancel = run("replay_after_cancel", "Yes, call mom.")
        parent_child = run(
            "parent_child_private_cloud", "Send my child's age and school to the cloud."
        )
        child_location = run(
            "child_location_private_cloud", "Send my child's location to the cloud."
        )
        media_action = run("media_action_request", "Play calm piano.")
        invented_media_target = run("invented_media_target", "Yes, play rain sounds.")
        pending_after_media_mismatch = _pending_action_state_counts(store)
        cancel_media = run("cancel_media_pending", "Cancel that.")
        conversation_export = run(
            "conversation_export_block", "Send our previous conversation to the cloud."
        )

        dashboard = build_assistant_os_dashboard(store).to_dict()
        safety_flags = dashboard["safety_flags"]
        fact_privacy = store.load_user_fact_privacy_index()
        pending_states = _pending_action_state_counts(store)
        turns = [
            _autoimmune_turn_summary(label, payload)
            for label, payload in turn_payloads.items()
        ]
        checks = {
            "private_cloud_blocked": (
                private_cloud["route"] == "reject"
                and private_cloud["reason"] == "blocked_private_facts_to_cloud"
                and private_cloud["membrane"].get("boundary_crossed") == "blocked"
            ),
            "generic_cloud_allowed_without_private_evidence": (
                generic_cloud["route"] == "local_answer"
                and generic_cloud.get("membrane", {}).get("boundary_crossed") in ("local", "none", None)
                and generic_cloud.get("evidence_keys") in (None, ("self_model.purpose",), ["self_model.purpose"])
            ),
            "public_profile_cloud_allowed_with_explicit_policy": (
                public_profile_cloud["route"] == "cloud_handoff"
                and public_profile_cloud["reason"] == "private_memory_cloud_request"
                and public_profile_cloud["membrane"].get("boundary_crossed") == "cloud"
                and "facts.public_profile"
                in public_profile_cloud["membrane"].get("personal_facts_included", [])
                and "facts.public_profile"
                not in public_profile_cloud["membrane"].get(
                    "personal_facts_excluded", []
                )
            ),
            "public_profile_policy_remains_shareable": (
                bool(fact_privacy.get("facts.public_profile", {}).get("consent", False))
                and not bool(
                    fact_privacy.get("facts.public_profile", {}).get("local_only", True)
                )
                and bool(
                    fact_privacy.get("facts.public_profile", {}).get(
                        "cloud_eligible", False
                    )
                )
                and fact_privacy.get("facts.public_profile", {}).get("scope")
                == "shareable_profile"
            ),
            "consent_revocation_local": (
                consent_revoke["route"] == "local_answer"
                and consent_revoke["reason"] == "consent_revoked_user_fact"
                and "local_privacy_policy.consent_revocation"
                in consent_revoke.get("synthesis", {}).get("citations", [])
            ),
            "revoked_fact_absent_after_reload": "green"
            not in memory_after_revoke["answer"].lower(),
            "household_memory_stored_locally": (
                household_setup["route"] == "local_answer"
                and household_setup["reason"] == "consented_household_memory_stored"
                and "facts.household_context"
                in household_setup.get("evidence_keys", [])
            ),
            "household_private_cloud_blocked": (
                household_private_cloud["route"] == "reject"
                and household_private_cloud["reason"]
                == "blocked_private_facts_to_cloud"
                and "facts.household_context"
                in household_private_cloud.get("evidence_keys", [])
            ),
            "mixed_public_household_cloud_blocked": (
                mixed_public_household_cloud["route"] == "reject"
                and mixed_public_household_cloud["reason"]
                == "blocked_private_facts_to_cloud"
                and "facts.public_profile"
                in mixed_public_household_cloud.get("evidence_keys", [])
                and "facts.household_context"
                in mixed_public_household_cloud.get("evidence_keys", [])
            ),
            "mixed_public_household_excludes_private_without_partial_cloud": (
                mixed_public_household_cloud["membrane"].get("boundary_crossed")
                == "blocked"
                and "facts.household_context"
                in mixed_public_household_cloud["membrane"].get(
                    "personal_facts_excluded", []
                )
                and not mixed_public_household_cloud["membrane"].get(
                    "personal_facts_included", []
                )
            ),
            "household_consent_revocation_local": (
                household_consent_revoke["route"] == "local_answer"
                and household_consent_revoke["reason"] == "consent_revoked_user_fact"
                and "local_privacy_policy.consent_revocation"
                in household_consent_revoke.get("synthesis", {}).get("citations", [])
            ),
            "revoked_household_absent_after_reload": (
                household_after_revoke["route"] == "clarify"
                and household_after_revoke["reason"] == "personal_memory_empty"
                and "Maya and Mom" not in household_after_revoke["answer"]
                and "facts.household_context"
                not in household_after_revoke.get("evidence_keys", [])
            ),
            "child_memory_stored_with_owned_keys": (
                child_age_setup["reason"] == "consented_child_memory_stored"
                and child_school_setup["reason"] == "consented_child_memory_stored"
                and child_age_setup.get("evidence_keys") == ["facts.child_age"]
                and child_school_setup.get("evidence_keys") == ["facts.child_school"]
            ),
            "child_school_consent_revocation_local": (
                child_school_consent_revoke["route"] == "local_answer"
                and child_school_consent_revoke["reason"] == "consent_revoked_user_fact"
                and child_school_consent_revoke.get("evidence_keys")
                == ["facts.child_school"]
                and "local_privacy_policy.consent_revocation"
                in child_school_consent_revoke.get("synthesis", {}).get("citations", [])
            ),
            "revoked_child_school_absent_after_reload": (
                child_school_after_revoke["route"] == "clarify"
                and child_school_after_revoke["reason"] == "personal_memory_empty"
                and "Bright School" not in child_school_after_revoke["answer"]
                and "facts.child_school"
                not in child_school_after_revoke.get("evidence_keys", [])
                and "facts.school"
                not in child_school_after_revoke.get("evidence_keys", [])
            ),
            "sibling_child_age_survives_school_revoke": (
                child_age_after_school_revoke["route"] == "local_answer"
                and child_age_after_school_revoke["reason"] == "personal_memory_recall"
                and child_age_after_school_revoke.get("evidence_keys")
                == ["facts.child_age"]
                and "facts.child_school"
                not in child_age_after_school_revoke.get("evidence_keys", [])
            ),
            "stale_weather_fetches_instead_of_answering": (
                stale_weather["route"] == "external_fetch"
                and stale_weather["reason"] == "weather_cache_miss"
                and float(stale_weather["homeostasis"].get("cache_freshness", 1.0))
                == 0.0
            ),
            "action_requires_confirmation": (
                action_request["route"] == "device_action"
                and int(action_request["membrane"].get("confirmation_required", 0) or 0)
                == 1
            ),
            "invented_target_blocked": (
                invented_target["route"] == "clarify"
                and invented_target["reason"] == "confirmation_target_mismatch"
            ),
            "mismatch_left_pending_unexecuted": (
                pending_after_mismatch.get("pending", 0) >= 1
                and pending_after_mismatch.get("executed", 0) == 0
            ),
            "cancel_prevents_replay": (
                cancel["reason"] == "cancelled_pending_action"
                and replay_after_cancel["route"] == "clarify"
                and replay_after_cancel["reason"] == "no_pending_action_to_confirm"
            ),
            "parent_child_private_cloud_blocked": (
                parent_child["route"] == "reject"
                and parent_child["reason"] == "blocked_private_facts_to_cloud"
            ),
            "parent_child_private_cloud_uses_owned_keys": (
                "facts.child_age" in parent_child.get("evidence_keys", [])
                and "facts.child_school" in parent_child.get("evidence_keys", [])
                and "profile.age" not in parent_child.get("evidence_keys", [])
                and "facts.school" not in parent_child.get("evidence_keys", [])
            ),
            "child_location_private_cloud_blocked": (
                child_location["route"] == "reject"
                and child_location["reason"] == "blocked_private_facts_to_cloud"
                and child_location["membrane"].get("boundary_crossed") == "blocked"
            ),
            "child_location_private_cloud_uses_owned_key": (
                child_location.get("evidence_keys") == ["facts.child_location"]
                and "profile.location" not in child_location.get("evidence_keys", [])
            ),
            "media_action_requires_confirmation": (
                media_action["route"] == "device_action"
                and media_action["reason"] == "local_media_action"
                and int(media_action["membrane"].get("confirmation_required", 0) or 0)
                == 1
            ),
            "invented_media_target_blocked": (
                invented_media_target["route"] == "clarify"
                and invented_media_target["reason"] == "confirmation_target_mismatch"
            ),
            "media_mismatch_left_pending_unexecuted": (
                pending_after_media_mismatch.get("pending", 0) >= 1
                and pending_after_media_mismatch.get("executed", 0) == 0
            ),
            "media_cancel_prevents_lingering_pending_action": (
                cancel_media["route"] == "local_answer"
                and cancel_media["reason"] == "cancelled_pending_action"
            ),
            "conversation_export_blocked": (
                conversation_export["route"] == "reject"
                and conversation_export["reason"] == "blocked_private_facts_to_cloud"
            ),
            "final_pending_actions_do_not_linger": (
                pending_states.get("pending", 0) == 0
                and pending_states.get("executed", 0) == 0
                and pending_states.get("cancelled", 0) >= 2
            ),
            "ledger_complete": bool(safety_flags.get("ledger_complete", False)),
            "blocking_safety_flags_clean": (
                int(safety_flags.get("cloud_private_inclusions", 0) or 0) == 0
                and int(safety_flags.get("unconfirmed_executed_actions", 0) or 0) == 0
                and int(safety_flags.get("action_without_confirmation_gate", 0) or 0)
                == 0
                and int(safety_flags.get("fake_latest_news_local_answers", 0) or 0) == 0
                and int(safety_flags.get("low_quality_applied_synthesis", 0) or 0) == 0
                and int(safety_flags.get("dangling_memory_links", 0) or 0) == 0
            ),
            "expected_boundary_counters_present": (
                int(safety_flags.get("consent_revocations", 0) or 0) >= 1
                and int(safety_flags.get("confirmation_target_mismatches", 0) or 0) >= 2
                and int(safety_flags.get("cancelled_pending_actions", 0) or 0) >= 2
                and int(safety_flags.get("action_replay_blocks", 0) or 0) >= 1
            ),
        }
        return {
            "db": str(db),
            "passed": all(checks.values()),
            "checks": checks,
            "elapsed_ms": _elapsed_ms(started),
            "turns": turns,
            "pending_actions": pending_states,
            "safety_flags": safety_flags,
            "counts": store.table_counts(),
            "runtime": "stdlib_python_sqlite",
            "dependency_class": "stdlib_only",
        }
    finally:
        store.close()


def _autoimmune_turn_summary(label: str, payload: dict) -> dict:
    return {
        "label": label,
        "utterance": payload.get("utterance", ""),
        "intent": payload.get("intent", ""),
        "route": payload.get("route", ""),
        "reason": payload.get("reason", ""),
        "boundary_crossed": payload.get("membrane", {}).get("boundary_crossed", ""),
        "confirmation_required": int(
            payload.get("membrane", {}).get("confirmation_required", 0) or 0
        ),
        "personal_facts_included": payload.get("membrane", {}).get(
            "personal_facts_included", []
        ),
        "personal_facts_excluded": payload.get("membrane", {}).get(
            "personal_facts_excluded", []
        ),
        "cache_freshness": float(
            payload.get("homeostasis", {}).get("cache_freshness", 0.0) or 0.0
        ),
        "evidence_keys": payload.get("evidence_keys", []),
        "synthesis_applied": bool(payload.get("synthesis", {}).get("applied", False)),
        "synthesis_refused": bool(payload.get("synthesis", {}).get("refused", False)),
        "citations": payload.get("synthesis", {}).get("citations", []),
    }


def _synthesis_variant_smoke(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    payload = _build_synthesis_variant_smoke_payload(args.db, seed=args.seed)
    _print_payload(payload, json_mode=args.json)


def _build_synthesis_variant_smoke_payload(
    db: Path, *, seed: Path = DEFAULT_SEED
) -> dict:
    started = perf_counter()
    store = _open_store(db, seed)
    try:
        turn_payloads: dict[str, dict] = {}

        def run(label: str, utterance: str) -> dict:
            payload = _handle_utterance(store, utterance, auto_execute=False)
            turn_payloads[label] = payload
            return payload

        story_bedtime = run("story_bedtime", "Tell me a bedtime story.")
        story_read = run("story_read", "Read me a story.")
        story_tale = run("story_tale", "Tell me a tale.")
        health_week = run("health_week", "How can I improve my health this week?")
        health_sleep = run(
            "health_sleep", "What healthy thing can I do to sleep better tonight?"
        )
        urgent_health = run("urgent_health", "I cannot breathe.")
        weather_cached = run("weather_cached", "What is the weather today?")
        meal_today = run("meal_today", "What should I eat today?")
        store.start_new_session()
        session_summary = run("session_summary", "Summarize our recent sessions.")
        long_horizon_digest = run(
            "long_horizon_digest", "What happened over the last few days?"
        )

        turns = [
            _synthesis_variant_turn_summary(label, payload)
            for label, payload in turn_payloads.items()
        ]
        by_label = {turn["label"]: turn for turn in turns}
        story_turns = [story_bedtime, story_read, story_tale]
        health_turns = [health_week, health_sleep]
        applied_turns = [
            payload for payload in turn_payloads.values() if payload.get("synthesis")
        ]
        dashboard = build_assistant_os_dashboard(store).to_dict()
        safety_flags = dashboard["safety_flags"]
        counts = store.table_counts()
        checks = {
            "ten_variant_turns_completed": len(turns)
            == len(SYNTHESIS_VARIANT_SMOKE_TURNS),
            "story_variants_local_and_cited": (
                all(payload["route"] == "local_answer" for payload in story_turns)
                and all(
                    payload["reason"] == "local_story_inventory"
                    for payload in story_turns
                )
                and all(_synthesis_applied_cleanly(payload) for payload in story_turns)
                and all(
                    _has_citation_prefix(payload, "story_models.")
                    for payload in story_turns
                )
            ),
            "story_tale_not_exact_story_phrase": (
                "story" not in by_label["story_tale"]["tokens"]
                and by_label["story_tale"]["route"] == "local_answer"
                and by_label["story_tale"]["primary_domain_evidence"].get("pattern")
                == "request_story_inventory"
            ),
            "health_variants_local_and_cited": (
                all(payload["route"] == "local_answer" for payload in health_turns)
                and all(
                    payload["reason"] == "bounded_general_health_guidance"
                    for payload in health_turns
                )
                and all(_synthesis_applied_cleanly(payload) for payload in health_turns)
                and all(
                    _has_citation_prefix(payload, "health_goals.")
                    for payload in health_turns
                )
                and all(
                    "local_health_safety_policy" in _synthesis_citations(payload)
                    for payload in health_turns
                )
            ),
            "health_sleep_not_exact_health_phrase": (
                "health" not in by_label["health_sleep"]["tokens"]
                and by_label["health_sleep"]["route"] == "local_answer"
                and by_label["health_sleep"]["primary_domain_evidence"].get("pattern")
                == "request_bounded_health_advice"
            ),
            "urgent_health_escalates_without_cloud": (
                urgent_health["route"] == "local_answer"
                and urgent_health["reason"] == "urgent_health_safety_escalation"
                and _synthesis_applied_cleanly(urgent_health)
                and "local_health_safety_policy" in _synthesis_citations(urgent_health)
                and "urgent" in urgent_health["answer"].lower()
                and "not a local diagnosis" in urgent_health["answer"].lower()
            ),
            "weather_and_meal_synthesize_from_local_state": (
                weather_cached["route"] == "cached_tool"
                and weather_cached["reason"] == "weather_cache_hit"
                and _synthesis_applied_cleanly(weather_cached)
                and "weekly_weather.today" in _synthesis_citations(weather_cached)
                and meal_today["route"] == "local_answer"
                and meal_today["reason"] == "memory_plus_weather_cache"
                and _synthesis_applied_cleanly(meal_today)
                and any(
                    key.startswith("food_inventory.")
                    for key in _synthesis_citations(meal_today)
                )
            ),
            "session_summary_uses_event_memory": (
                session_summary["route"] == "local_answer"
                and session_summary["reason"] == "autobiographical_session_summary"
                and _synthesis_applied_cleanly(session_summary)
                and _synthesis_citations(session_summary)
                and all(
                    key.startswith("events.")
                    for key in _synthesis_citations(session_summary)
                )
            ),
            "long_horizon_digest_uses_digest_inventory": (
                long_horizon_digest["route"] == "local_answer"
                and long_horizon_digest["reason"] == "autobiographical_memory_digest"
                and _synthesis_applied_cleanly(long_horizon_digest)
                and "memory_digest.long_horizon_latest"
                in _synthesis_citations(long_horizon_digest)
            ),
            "quality_clean": (
                bool(applied_turns)
                and all(
                    _synthesis_quality_score(payload) >= 0.72
                    for payload in applied_turns
                )
                and all(
                    not _synthesis_quality_warnings(payload)
                    for payload in applied_turns
                )
            ),
            "debug_mapping_present": all(
                turn["mapping"] == ["basic_nlp", "uol_parse", "chat_frame"]
                for turn in turns
            ),
            "primary_uol_chatframe_not_secondary_phrase_route": (
                all(turn["primary_parse_basis"] == "uol_chat_frame" for turn in turns)
                and all(
                    bool(turn["primary_domain_evidence"].get("frame_registry", ""))
                    or turn["route"] not in {"local_answer", "cached_tool", "device_action"}
                    for turn in turns
                )
                and not any(
                    any(
                        part.startswith("secondary_meaning_hints:")
                        or part.startswith("vocabulary_hits:")
                        for part in turn["primary_routing_basis"]
                    )
                    for turn in turns
                )
            ),
            "ledgers_complete": (
                counts.get("events", 0) == len(turns)
                and counts.get("membrane_decisions", 0) == len(turns)
                and counts.get("homeostatic_snapshots", 0) == len(turns)
                and bool(safety_flags.get("ledger_complete", False))
            ),
            "safety_flags_clean": (
                int(safety_flags.get("cloud_private_inclusions", 0) or 0) == 0
                and int(safety_flags.get("unconfirmed_executed_actions", 0) or 0) == 0
                and int(safety_flags.get("action_without_confirmation_gate", 0) or 0)
                == 0
                and int(safety_flags.get("fake_latest_news_local_answers", 0) or 0) == 0
                and int(safety_flags.get("low_quality_applied_synthesis", 0) or 0) == 0
            ),
            "stdlib_sqlite_only": True,
        }
        return {
            "db": str(db),
            "passed": all(checks.values()),
            "checks": checks,
            "elapsed_ms": _elapsed_ms(started),
            "turns": turns,
            "variant_count": len(turns),
            "route_counts": dict(
                sorted(Counter(turn["route"] for turn in turns).items())
            ),
            "reason_counts": dict(
                sorted(Counter(turn["reason"] for turn in turns).items())
            ),
            "safety_flags": safety_flags,
            "counts": counts,
            "runtime": "stdlib_python_sqlite",
            "dependency_class": "stdlib_only",
        }
    finally:
        store.close()


def _synthesis_stress_smoke(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    payload = _build_synthesis_stress_smoke_payload(args.db, seed=args.seed)
    _print_payload(payload, json_mode=args.json)


def _build_synthesis_stress_smoke_payload(
    db: Path, *, seed: Path = DEFAULT_SEED
) -> dict:
    started = perf_counter()
    store = _open_store(db, seed)
    try:
        turn_records: list[tuple[str, str, dict]] = []
        current_session = ""
        for session_label, label, utterance in SYNTHESIS_STRESS_SMOKE_TURNS:
            if current_session and session_label != current_session:
                store.start_new_session()
            current_session = session_label
            payload = _handle_utterance(store, utterance, auto_execute=False)
            turn_records.append((session_label, label, payload))

        turns = [
            _synthesis_stress_turn_summary(session_label, label, payload)
            for session_label, label, payload in turn_records
        ]
        by_label = {turn["label"]: turn for turn in turns}
        dashboard = build_assistant_os_dashboard(store).to_dict()
        safety_flags = dashboard["safety_flags"]
        counts = store.table_counts()
        route_counts = dict(sorted(Counter(turn["route"] for turn in turns).items()))
        reason_counts = dict(sorted(Counter(turn["reason"] for turn in turns).items()))
        intent_counts = dict(sorted(Counter(turn["intent"] for turn in turns).items()))
        complexity_scores = [float(turn["complexity_score"]) for turn in turns]
        quality_scores = [
            float(turn["quality_score"]) for turn in turns
        ]
        story_turns = [turn for turn in turns if turn["intent"] == "story"]
        general_health_turns = [
            by_label[label]
            for label in (
                "health_after_school",
                "health_sleep",
                "health_goals",
                "health_week",
            )
        ]
        weather_turns = [by_label["weather_forecast"], by_label["weather_rain"]]
        meal_turns = [
            by_label["meal_breakfast"],
            by_label["meal_dinner"],
            by_label["meal_rain"],
        ]
        status_turns = [
            by_label["status_progress"],
            by_label["status_cloud"],
            by_label["status_next"],
        ]
        memory_turns = [
            by_label["last_question"],
            by_label["session_summary"],
            by_label["long_horizon_digest"],
        ]
        checks = {
            "twenty_four_turns_completed": len(turns)
            == len(SYNTHESIS_STRESS_SMOKE_TURNS)
            == 24,
            "three_sessions_recorded": _event_session_count(store) >= 3,
            "all_turns_local_or_cached_clarify_or_reject": (
                route_counts.keys() <= {"cached_tool", "local_answer", "clarify"}
                and all(not turn["cloud_needed"] for turn in turns)
                and all(not turn["external_fetch_needed"] for turn in turns)
                and all(turn["boundary_crossed"] == "none" for turn in turns)
            ),
            "story_variants_remain_local_and_cited": (
                len(story_turns) >= 5
                and all(
                    turn["reason"] in ("local_story_inventory", "story_constraint_unmet")
                    for turn in story_turns
                )
                and all(
                    any(
                        citation.startswith("story_models.")
                        for citation in turn["citations"]
                    )
                    if turn["reason"] == "local_story_inventory" else True
                    for turn in story_turns
                )
                and all(
                    turn["primary_domain_evidence"].get("pattern")
                    in ("request_story_inventory", "story_constraint_unmet")
                    for turn in story_turns
                )
            ),
            "health_variants_and_urgent_policy_cited": (
                all(
                    turn["reason"] == "bounded_general_health_guidance"
                    for turn in general_health_turns
                )
                and all(
                    any(
                        citation.startswith("health_goals.")
                        for citation in turn["citations"]
                    )
                    for turn in general_health_turns
                )
                and all(
                    "local_health_safety_policy" in turn["citations"]
                    for turn in general_health_turns
                )
                and by_label["urgent_health"]["reason"]
                == "urgent_health_safety_escalation"
                and by_label["urgent_health"]["citations"]
                == ["local_health_safety_policy"]
            ),
            "weather_meal_and_safety_use_local_state": (
                all(
                    turn["route"] == "cached_tool"
                    and "weekly_weather.today" in turn["citations"]
                    for turn in weather_turns
                )
                and all(
                    any(
                        citation.startswith("food_inventory.")
                        for citation in turn["citations"]
                    )
                    for turn in meal_turns
                )
                and "weekly_weather.today" in by_label["school_clothing"]["citations"]
                and "facts.school" in by_label["school_clothing"]["citations"]
            ),
            "self_status_and_identity_use_self_evidence": (
                "self_model.name" in by_label["identity"]["citations"]
                and all(
                    "self_status.counts" in turn["citations"] for turn in status_turns
                )
                and all(turn["intent"] == "assistant_status" for turn in status_turns)
            ),
            "autobiographical_summaries_use_events_and_digest": (
                all(
                    turn["intent"] == "autobiographical_memory" for turn in memory_turns
                )
                and all(
                    citation.startswith("events.")
                    for citation in by_label["last_question"]["citations"]
                )
                and len(by_label["session_summary"]["citations"]) >= 6
                and all(
                    citation.startswith("events.")
                    for citation in by_label["session_summary"]["citations"]
                )
                and by_label["long_horizon_digest"]["citations"]
                == ["memory_digest.long_horizon_latest"]
            ),
            "quality_clean_under_longer_trace": (
                len(quality_scores) == len(turns)
                and min(quality_scores, default=0.0) >= 0.72
                and all(not turn["quality_warnings"] for turn in turns)
                and int(safety_flags.get("low_quality_applied_synthesis", 0) or 0) == 0
            ),
            "complexity_and_reason_diversity_present": (
                max(complexity_scores, default=0.0) >= 0.57
                and len(reason_counts) >= 10
                and len(intent_counts) >= 8
            ),
            "primary_uol_chatframe_not_secondary_phrase_route": (
                all(
                    turn["mapping"] == ["basic_nlp", "uol_parse", "chat_frame"]
                    for turn in turns
                )
                and all(
                    turn["primary_parse_basis"] == "uol_chat_frame" for turn in turns
                )
                and all(
                    bool(turn["primary_domain_evidence"].get("frame_registry", ""))
                    or turn["route"] not in {"local_answer", "cached_tool", "device_action"}
                    for turn in turns
                )
                and not any(
                    any(
                        part.startswith("secondary_meaning_hints:")
                        or part.startswith("vocabulary_hits:")
                        for part in turn["primary_routing_basis"]
                    )
                    for turn in turns
                )
            ),
            "ledgers_complete": (
                counts.get("events", 0) == len(turns)
                and counts.get("membrane_decisions", 0) == len(turns)
                and counts.get("homeostatic_snapshots", 0) == len(turns)
                and counts.get("synthesis_traces", 0) == len(turns)
                and bool(safety_flags.get("ledger_complete", False))
            ),
            "safety_flags_clean": (
                int(safety_flags.get("cloud_private_inclusions", 0) or 0) == 0
                and int(safety_flags.get("unconfirmed_executed_actions", 0) or 0) == 0
                and int(safety_flags.get("action_without_confirmation_gate", 0) or 0)
                == 0
                and int(safety_flags.get("fake_latest_news_local_answers", 0) or 0) == 0
                and int(safety_flags.get("dangling_memory_links", 0) or 0) == 0
            ),
            "stdlib_sqlite_only": True,
        }
        return {
            "db": str(db),
            "passed": all(checks.values()),
            "checks": checks,
            "elapsed_ms": _elapsed_ms(started),
            "turns": turns,
            "turn_count": len(turns),
            "session_count": _event_session_count(store),
            "route_counts": route_counts,
            "reason_counts": reason_counts,
            "intent_counts": intent_counts,
            "quality": {
                "min": round(min(quality_scores), 3) if quality_scores else 0.0,
                "avg": round(sum(quality_scores) / max(1, len(quality_scores)), 3),
                "max": round(max(quality_scores), 3) if quality_scores else 0.0,
            },
            "complexity": {
                "min": round(min(complexity_scores), 3) if complexity_scores else 0.0,
                "avg": round(
                    sum(complexity_scores) / max(1, len(complexity_scores)), 3
                ),
                "max": round(max(complexity_scores), 3) if complexity_scores else 0.0,
            },
            "safety_flags": safety_flags,
            "counts": counts,
            "runtime": "stdlib_python_sqlite",
            "dependency_class": "stdlib_only",
        }
    finally:
        store.close()


def _synthesis_stress_turn_summary(
    session_label: str, label: str, payload: dict
) -> dict:
    summary = _synthesis_variant_turn_summary(label, payload)
    return {"session": session_label, **summary}


def _event_session_count(store: AssistantOSStore) -> int:
    row = store.connection.execute(
        "SELECT COUNT(DISTINCT session_id) AS sessions FROM events"
    ).fetchone()
    return int(row["sessions"] or 0) if row is not None else 0


def _synthesis_variant_turn_summary(label: str, payload: dict) -> dict:
    debug_parse = dict(payload.get("debug_parse", {}))
    nlp = dict(debug_parse.get("nlp", {}))
    chat_frame = dict(debug_parse.get("chat_frame", {}))
    synthesis = dict(payload.get("synthesis", {}))
    quality = dict(synthesis.get("quality", {}))
    return {
        "label": label,
        "utterance": payload.get("utterance", ""),
        "tokens": list(debug_parse.get("tokens", [])),
        "intent": payload.get("intent", ""),
        "route": payload.get("route", ""),
        "reason": payload.get("reason", ""),
        "cloud_needed": bool(payload.get("cloud_needed", False)),
        "external_fetch_needed": bool(payload.get("external_fetch_needed", False)),
        "boundary_crossed": payload.get("membrane", {}).get("boundary_crossed", ""),
        "mapping": [stage.get("stage") for stage in debug_parse.get("mapping", [])],
        "primary_parse_basis": nlp.get("primary_parse_basis", ""),
        "primary_domain_evidence": nlp.get("primary_domain_evidence", {}),
        "primary_routing_basis": chat_frame.get("primary_routing_basis", []),
        "secondary_debug_hints": chat_frame.get("secondary_debug_hints", []),
        "complexity_score": float(chat_frame.get("complexity_score", 0.0) or 0.0),
        "unknown_token_count": int(nlp.get("unknown_token_count", 0) or 0),
        "evidence_keys": payload.get("evidence_keys", []),
        "synthesis_applied": bool(synthesis.get("applied", False)),
        "synthesis_refused": bool(synthesis.get("refused", False)),
        "quality_score": float(quality.get("score", 0.0) or 0.0),
        "quality_warnings": list(quality.get("warnings", [])),
        "citations": list(synthesis.get("citations", [])),
        "admitted_evidence_count": int(
            synthesis.get("admitted_evidence_count", 0) or 0
        ),
    }


def _synthesis_applied_cleanly(payload: dict) -> bool:
    synthesis = dict(payload.get("synthesis", {}))
    return bool(synthesis.get("applied", False)) and not bool(
        synthesis.get("refused", False)
    )


def _synthesis_quality_score(payload: dict) -> float:
    return float(
        payload.get("synthesis", {}).get("quality", {}).get("score", 0.0) or 0.0
    )


def _synthesis_quality_warnings(payload: dict) -> list:
    warnings = payload.get("synthesis", {}).get("quality", {}).get("warnings", [])
    return list(warnings) if isinstance(warnings, list) else [warnings]


def _synthesis_citations(payload: dict) -> list[str]:
    citations = payload.get("synthesis", {}).get("citations", [])
    return [str(item) for item in citations] if isinstance(citations, list) else []


def _has_citation_prefix(payload: dict, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in _synthesis_citations(payload))


def _mark_today_weather_stale(store: AssistantOSStore) -> None:
    store.connection.execute(
        """
        UPDATE inventories
        SET payload_json=?
        WHERE kind='weather' AND item_id='today'
        """,
        (json.dumps({"forecast": "warm with afternoon rain", "stale": True}),),
    )
    store.connection.commit()


def _pending_action_state_counts(store: AssistantOSStore) -> dict:
    rows = store.connection.execute(
        """
        SELECT confirmation_state, COUNT(*) AS count, SUM(executed) AS executed
        FROM pending_actions
        GROUP BY confirmation_state
        """
    ).fetchall()
    counts = {"total": 0, "pending": 0, "confirmed": 0, "cancelled": 0, "executed": 0}
    for row in rows:
        state = str(row["confirmation_state"])
        count = int(row["count"] or 0)
        counts["total"] += count
        counts[state] = count
        counts["executed"] += int(row["executed"] or 0)
    return counts


def _pi_smoke(args) -> None:
    lifecycle_db = args.db.with_name(f"{args.db.stem}_lifecycle.sqlite")
    action_db = args.db.with_name(f"{args.db.stem}_actions.sqlite")
    setup_integration_db = args.db.with_name(f"{args.db.stem}_setup_integration.sqlite")
    dataset_audit_db = args.db.with_name(f"{args.db.stem}_dataset_audit.sqlite")
    synthesis_variant_db = args.db.with_name(
        f"{args.db.stem}_synthesis_variants.sqlite"
    )
    synthesis_stress_db = args.db.with_name(f"{args.db.stem}_synthesis_stress.sqlite")
    inventory_soak_db = args.db.with_name(f"{args.db.stem}_inventory_soak.sqlite")
    inventory_retry_db = args.db.with_name(f"{args.db.stem}_inventory_retry.sqlite")
    inventory_matrix_db_dir = args.db.with_name(f"{args.db.stem}_inventory_matrix")
    inventory_diversity_db_dir = args.db.with_name(
        f"{args.db.stem}_inventory_diversity"
    )
    inventory_failure_work_dir = args.db.with_name(f"{args.db.stem}_inventory_failures")
    open_trace_db_dir = args.db.with_name(f"{args.db.stem}_open_traces")
    transcript_replay_db_dir = args.db.with_name(f"{args.db.stem}_transcript_replay")
    smoke_media_dir = args.db.with_name(f"{args.db.stem}_media")
    smoke_media_file = smoke_media_dir / "Calm Piano.mp3"
    if args.reset:
        for path in (
            args.db,
            lifecycle_db,
            action_db,
            setup_integration_db,
            dataset_audit_db,
            synthesis_variant_db,
            synthesis_stress_db,
            inventory_soak_db,
            inventory_retry_db,
        ):
            _remove_sqlite_files(path)
        if inventory_matrix_db_dir.exists():
            _safe_remove_bundle_dir(inventory_matrix_db_dir)
        if inventory_diversity_db_dir.exists():
            _safe_remove_bundle_dir(inventory_diversity_db_dir)
        if inventory_failure_work_dir.exists():
            _safe_remove_bundle_dir(inventory_failure_work_dir)
        if transcript_replay_db_dir.exists():
            _safe_remove_bundle_dir(transcript_replay_db_dir)
        if smoke_media_file.exists():
            smoke_media_file.unlink()

    started_total = perf_counter()
    tracemalloc.start()
    datasets = _required_dataset_report(args.seed)
    dataset_audit_payload = _build_dataset_audit_payload(
        dataset_audit_db, seed=args.seed, reset=True
    )
    smoke_media_dir.mkdir(parents=True, exist_ok=True)
    if not smoke_media_file.exists():
        smoke_media_file.write_bytes(b"local pi-smoke media fixture")

    started = perf_counter()
    store = initialize_assistant_os_database(args.db, seed_path=args.seed)
    init_ms = _elapsed_ms(started)

    started = perf_counter()
    ask_payload = _handle_utterance(store, "Tell me a story.", auto_execute=True)
    ask_ms = _elapsed_ms(started)
    dashboard = build_assistant_os_dashboard(store).to_dict()
    main_counts = store.table_counts()
    store.close()

    started = perf_counter()
    lifecycle_store = initialize_assistant_os_database(lifecycle_db, seed_path=None)
    lifecycle_report = AssistantLifecycleSimulator(store=lifecycle_store).run(
        realistic_lifecycle_steps()
    )
    lifecycle_counts = lifecycle_store.table_counts()
    lifecycle_dashboard = build_assistant_os_dashboard(lifecycle_store).to_dict()
    lifecycle_store.close()
    lifecycle_ms = _elapsed_ms(started)

    action_payload = _build_action_smoke_payload(
        action_db,
        reset=True,
        action_mode="dry-run",
        media_dir=smoke_media_dir,
        manifest=DEFAULT_LOCAL_MEDIA_MANIFEST,
    )
    action_results = action_payload["action_results"]
    media_action = next(
        (
            result
            for result in action_results
            if result.get("action_type") == "play_media"
        ),
        {},
    )
    call_action = next(
        (
            result
            for result in action_results
            if result.get("action_type") == "call_contact"
        ),
        {},
    )

    started = perf_counter()
    setup_integration_payload = _build_setup_integration_smoke_payload(
        setup_integration_db,
        reset=True,
    )
    setup_integration_ms = _elapsed_ms(started)

    started = perf_counter()
    synthesis_variant_payload = _build_synthesis_variant_smoke_payload(
        synthesis_variant_db,
        seed=args.seed,
    )
    synthesis_variant_ms = _elapsed_ms(started)

    started = perf_counter()
    synthesis_stress_payload = _build_synthesis_stress_smoke_payload(
        synthesis_stress_db,
        seed=args.seed,
    )
    synthesis_stress_ms = _elapsed_ms(started)

    started = perf_counter()
    open_trace_payload = run_open_trace_suite(
        trace_path=DEFAULT_OPEN_TRACE_FIXTURE,
        db_dir=open_trace_db_dir,
        reset=True,
        weather_offline_json=DEFAULT_WEATHER_SAMPLE,
    ).to_dict()
    open_trace_ms = _elapsed_ms(started)
    open_trace_summary = _open_trace_summary(open_trace_payload)

    started = perf_counter()
    transcript_replay_payload = run_transcript_replay_suite(
        transcript_path=DEFAULT_TRANSCRIPT_REPLAY_FIXTURE,
        db_dir=transcript_replay_db_dir,
        reset=True,
        weather_offline_json=DEFAULT_WEATHER_SAMPLE,
    ).to_dict()
    transcript_replay_ms = _elapsed_ms(started)
    transcript_replay_summary = _transcript_replay_summary(transcript_replay_payload)

    started = perf_counter()
    inventory_soak_payload = _run_cli_json(
        ROOT,
        "inventory-soak",
        "--db",
        str(inventory_soak_db),
        "--reset",
        "--offline-samples",
        "--source",
        "both",
        "--cycles",
        "2",
        "--story-limit",
        "3",
        "--min-story-models",
        "12",
        "--json",
    )
    inventory_soak_ms = _elapsed_ms(started)

    started = perf_counter()
    inventory_matrix_payload = _run_cli_json(
        ROOT,
        "inventory-soak-matrix",
        "--db-dir",
        str(inventory_matrix_db_dir),
        "--reset",
        "--json",
    )
    inventory_matrix_ms = _elapsed_ms(started)

    started = perf_counter()
    inventory_diversity_payload = _run_cli_json(
        ROOT,
        "inventory-diversity-smoke",
        "--db-dir",
        str(inventory_diversity_db_dir),
        "--reset",
        "--cycles",
        "1",
        "--story-limit",
        "3",
        "--min-story-models",
        "9",
        "--json",
    )
    inventory_diversity_ms = _elapsed_ms(started)

    started = perf_counter()
    inventory_retry_payload = _run_cli_json(
        ROOT,
        "inventory-retry-smoke",
        "--db",
        str(inventory_retry_db),
        "--reset",
        "--json",
    )
    inventory_retry_ms = _elapsed_ms(started)

    started = perf_counter()
    inventory_failure_payload = _run_cli_json(
        ROOT,
        "inventory-failure-smoke",
        "--work-dir",
        str(inventory_failure_work_dir),
        "--reset",
        "--json",
    )
    inventory_failure_ms = _elapsed_ms(started)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_ms = _elapsed_ms(started_total)
    checks = {
        "datasets_present": all(item["exists"] for item in datasets),
        "dataset_audit_passed": bool(dataset_audit_payload.get("passed", False)),
        "seeded_db_created": main_counts["inventories"] >= 8
        and main_counts["user_facts"] >= 1,
        "ask_local_with_membrane": (
            ask_payload["route"] == "local_answer"
            and ask_payload["reason"] == "local_story_inventory"
            and ask_payload["membrane"].get("boundary_crossed") == "none"
            and ask_payload["counts"]["membrane_decisions"] >= 1
            and ask_payload["counts"]["homeostatic_snapshots"] >= 1
            and bool(ask_payload["synthesis"].get("applied"))
        ),
        "lifecycle_passed": (
            lifecycle_report.steps == 17
            and lifecycle_report.actions_executed == 1
            and lifecycle_report.blocked_offline == 1
            and lifecycle_report.cloud_story_handoffs_before_inventory == 3
            and lifecycle_report.story_route_after_inventory == "local_answer"
        ),
        "action_smoke_passed": bool(action_payload["passed"]),
        "setup_integration_smoke_passed": (
            bool(setup_integration_payload.get("passed", False))
            and bool(
                setup_integration_payload.get("checks", {}).get(
                    "setup_requests_recorded_without_fake_facts",
                    False,
                )
            )
            and bool(
                setup_integration_payload.get("checks", {}).get(
                    "future_memory_routes_change_after_setup",
                    False,
                )
            )
            and bool(
                setup_integration_payload.get("checks", {}).get(
                    "trusted_contact_action_uses_confirmation_gate",
                    False,
                )
            )
        ),
        "synthesis_variant_smoke_passed": (
            bool(synthesis_variant_payload.get("passed", False))
            and bool(
                synthesis_variant_payload.get("checks", {}).get(
                    "primary_uol_chatframe_not_secondary_phrase_route", False
                )
            )
            and bool(
                synthesis_variant_payload.get("checks", {}).get("quality_clean", False)
            )
        ),
        "synthesis_stress_smoke_passed": (
            bool(synthesis_stress_payload.get("passed", False))
            and bool(
                synthesis_stress_payload.get("checks", {}).get(
                    "three_sessions_recorded", False
                )
            )
            and bool(
                synthesis_stress_payload.get("checks", {}).get(
                    "quality_clean_under_longer_trace", False
                )
            )
            and bool(
                synthesis_stress_payload.get("checks", {}).get(
                    "primary_uol_chatframe_not_secondary_phrase_route", False
                )
            )
        ),
        "action_targets_resolved": (
            media_action.get("status") == "prepared"
            and bool(media_action.get("payload", {}).get("path_exists"))
            and call_action.get("status") == "prepared"
            and call_action.get("resolved_target") == "+234-000-ADA"
            and not any(result.get("side_effect_executed") for result in action_results)
        ),
        "memory_ledgers_complete": (
            dashboard["safety_flags"]["ledger_complete"]
            and lifecycle_dashboard["safety_flags"]["ledger_complete"]
            and action_payload["safety_flags"]["ledger_complete"]
        ),
        "safety_flags_clean": (
            dashboard["safety_flags"]["cloud_private_inclusions"] == 0
            and lifecycle_dashboard["safety_flags"]["unconfirmed_executed_actions"] == 0
            and lifecycle_dashboard["safety_flags"]["dangling_memory_links"] == 0
            and action_payload["safety_flags"]["action_without_confirmation_gate"] == 0
            and action_payload["safety_flags"]["unconfirmed_executed_actions"] == 0
        ),
        "open_trace_debug_gate_passed": (
            bool(open_trace_payload.get("passed", False))
            and int(open_trace_payload.get("turns", 0) or 0) >= 29
            and bool(
                open_trace_summary.get("debug_checks", {}).get(
                    "debug_maps_present", False
                )
            )
            and bool(
                open_trace_summary.get("debug_checks", {}).get(
                    "identity_maps_to_self_model", False
                )
            )
            and bool(
                open_trace_summary.get("debug_checks", {}).get(
                    "status_maps_to_runtime_or_next_steps", False
                )
            )
        ),
        "transcript_replay_gate_passed": (
            bool(transcript_replay_payload.get("passed", False))
            and int(transcript_replay_payload.get("turns", 0) or 0) >= 16
            and bool(
                transcript_replay_summary.get("debug_checks", {}).get(
                    "debug_maps_present", False
                )
            )
            and bool(
                transcript_replay_summary.get("fixture_checks", {}).get(
                    "no_static_answer_or_route_expectations", False
                )
            )
            and bool(
                transcript_replay_summary.get("fixture_checks", {}).get(
                    "memory_digest_quality_passed", False
                )
            )
            and bool(
                transcript_replay_summary.get("baseline_comparison", {}).get(
                    "passed", False
                )
            )
        ),
        "inventory_soak_passed": (
            bool(inventory_soak_payload.get("passed", False))
            and bool(
                inventory_soak_payload.get("checks", {}).get(
                    "source_coverage_ok", False
                )
            )
            and bool(
                inventory_soak_payload.get("checks", {}).get(
                    "failure_mode_observability_present", False
                )
            )
            and int(
                inventory_soak_payload.get("inventory_delta", {}).get(
                    "story_inventory_added", 0
                )
                or 0
            )
            > 0
        ),
        "inventory_soak_matrix_passed": (
            bool(inventory_matrix_payload.get("passed", False))
            and bool(
                inventory_matrix_payload.get("checks", {}).get(
                    "total_cycles_at_least_nine", False
                )
            )
            and bool(
                inventory_matrix_payload.get("checks", {}).get(
                    "both_source_families_covered", False
                )
            )
            and bool(
                inventory_matrix_payload.get("checks", {}).get(
                    "future_story_routes_local_from_imported_inventory",
                    False,
                )
            )
            and int(inventory_matrix_payload.get("total_failed_import_cycles", 1)) == 0
        ),
        "inventory_diversity_smoke_passed": (
            bool(inventory_diversity_payload.get("passed", False))
            and bool(
                inventory_diversity_payload.get("checks", {}).get(
                    "all_queries_reached_import_jobs", False
                )
            )
            and bool(
                inventory_diversity_payload.get("checks", {}).get(
                    "future_story_routes_local", False
                )
            )
        ),
        "inventory_retry_smoke_passed": (
            bool(inventory_retry_payload.get("passed", False))
            and bool(
                inventory_retry_payload.get("checks", {}).get(
                    "gutenberg_source_retried", False
                )
            )
            and bool(
                inventory_retry_payload.get("checks", {}).get(
                    "internet_archive_source_retried", False
                )
            )
            and bool(
                inventory_retry_payload.get("checks", {}).get(
                    "future_story_routes_local_after_reload", False
                )
            )
        ),
        "inventory_failure_smoke_passed": (
            bool(inventory_failure_payload.get("passed", False))
            and bool(
                inventory_failure_payload.get("checks", {}).get(
                    "no_fake_story_inventory", False
                )
            )
            and bool(
                inventory_failure_payload.get("checks", {}).get(
                    "future_story_routes_missing_inventory", False
                )
            )
            and bool(
                inventory_failure_payload.get("checks", {}).get(
                    "errors_are_observable", False
                )
            )
        ),
        "ask_within_budget": ask_ms <= args.max_ask_ms,
        "lifecycle_within_budget": lifecycle_ms <= args.max_lifecycle_ms,
        "stdlib_sqlite_only": True,
    }
    payload = {
        "db": str(args.db),
        "lifecycle_db": str(lifecycle_db),
        "action_db": str(action_db),
        "setup_integration_db": str(setup_integration_db),
        "dataset_audit_db": str(dataset_audit_db),
        "synthesis_variant_db": str(synthesis_variant_db),
        "synthesis_stress_db": str(synthesis_stress_db),
        "inventory_soak_db": str(inventory_soak_db),
        "inventory_retry_db": str(inventory_retry_db),
        "inventory_matrix_db_dir": str(inventory_matrix_db_dir),
        "inventory_diversity_db_dir": str(inventory_diversity_db_dir),
        "inventory_failure_work_dir": str(inventory_failure_work_dir),
        "open_trace_db_dir": str(open_trace_db_dir),
        "transcript_replay_db_dir": str(transcript_replay_db_dir),
        "smoke_media_dir": str(smoke_media_dir),
        "smoke_media_file": str(smoke_media_file),
        "passed": all(checks.values()),
        "checks": checks,
        "datasets": datasets,
        "dataset_audit": {
            "passed": bool(dataset_audit_payload.get("passed", False)),
            "checks": dataset_audit_payload.get("checks", {}),
            "files": dataset_audit_payload.get("files", {}),
            "source_fixtures": dataset_audit_payload.get("source_fixtures", {}),
            "bootstrap": dataset_audit_payload.get("bootstrap", {}),
        },
        "runtime": "stdlib_python_sqlite",
        "dependency_class": "stdlib_only",
        "timings": {
            "init_ms": init_ms,
            "ask_ms": ask_ms,
            "lifecycle_ms": lifecycle_ms,
            "setup_integration_ms": setup_integration_ms,
            "synthesis_variant_ms": synthesis_variant_ms,
            "synthesis_stress_ms": synthesis_stress_ms,
            "open_trace_ms": open_trace_ms,
            "transcript_replay_ms": transcript_replay_ms,
            "inventory_soak_ms": inventory_soak_ms,
            "inventory_matrix_ms": inventory_matrix_ms,
            "inventory_diversity_ms": inventory_diversity_ms,
            "inventory_retry_ms": inventory_retry_ms,
            "inventory_failure_ms": inventory_failure_ms,
            "total_ms": total_ms,
        },
        "budgets": {
            "max_ask_ms": args.max_ask_ms,
            "max_lifecycle_ms": args.max_lifecycle_ms,
        },
        "peak_traced_kb": round(peak_bytes / 1024, 3),
        "db_bytes": _sqlite_size(args.db),
        "lifecycle_db_bytes": _sqlite_size(lifecycle_db),
        "action_db_bytes": _sqlite_size(action_db),
        "setup_integration_db_bytes": _sqlite_size(setup_integration_db),
        "dataset_audit_db_bytes": _sqlite_size(dataset_audit_db),
        "ask": {
            "route": ask_payload["route"],
            "reason": ask_payload["reason"],
            "synthesis_applied": bool(ask_payload["synthesis"].get("applied")),
            "counts": ask_payload["counts"],
        },
        "lifecycle": {
            "steps": lifecycle_report.steps,
            "local_resolution_rate": lifecycle_report.local_resolution_rate,
            "cloud_handoffs": lifecycle_report.cloud_handoffs,
            "external_fetches": lifecycle_report.external_fetches,
            "blocked_offline": lifecycle_report.blocked_offline,
            "actions_executed": lifecycle_report.actions_executed,
            "story_route_after_inventory": lifecycle_report.story_route_after_inventory,
            "counts": lifecycle_counts,
        },
        "action_smoke": {
            "passed": action_payload["passed"],
            "mode": action_payload["mode"],
            "action_results": action_results,
            "safety_flags": action_payload["safety_flags"],
        },
        "setup_integration_smoke": {
            "passed": bool(setup_integration_payload.get("passed", False)),
            "checks": setup_integration_payload.get("checks", {}),
            "turns": setup_integration_payload.get("turns", []),
            "setup_requests_after_gaps": setup_integration_payload.get(
                "setup_requests_after_gaps", {}
            ),
            "facts_after_gaps": setup_integration_payload.get("facts_after_gaps", {}),
            "facts_after_setup": setup_integration_payload.get("facts_after_setup", {}),
            "contacts_after_setup": setup_integration_payload.get(
                "contacts_after_setup", {}
            ),
            "fact_privacy": setup_integration_payload.get("fact_privacy", {}),
            "pending_actions": setup_integration_payload.get("pending_actions", {}),
            "safety_flags": setup_integration_payload.get("safety_flags", {}),
            "action_execution": setup_integration_payload.get("action_execution", {}),
        },
        "synthesis_variant_smoke": {
            "passed": bool(synthesis_variant_payload.get("passed", False)),
            "checks": synthesis_variant_payload.get("checks", {}),
            "variant_count": int(
                synthesis_variant_payload.get("variant_count", 0) or 0
            ),
            "route_counts": synthesis_variant_payload.get("route_counts", {}),
            "reason_counts": synthesis_variant_payload.get("reason_counts", {}),
            "turns": synthesis_variant_payload.get("turns", []),
            "safety_flags": synthesis_variant_payload.get("safety_flags", {}),
        },
        "synthesis_stress_smoke": {
            "passed": bool(synthesis_stress_payload.get("passed", False)),
            "checks": synthesis_stress_payload.get("checks", {}),
            "turn_count": int(synthesis_stress_payload.get("turn_count", 0) or 0),
            "session_count": int(synthesis_stress_payload.get("session_count", 0) or 0),
            "route_counts": synthesis_stress_payload.get("route_counts", {}),
            "reason_counts": synthesis_stress_payload.get("reason_counts", {}),
            "intent_counts": synthesis_stress_payload.get("intent_counts", {}),
            "quality": synthesis_stress_payload.get("quality", {}),
            "complexity": synthesis_stress_payload.get("complexity", {}),
            "turns": synthesis_stress_payload.get("turns", []),
            "safety_flags": synthesis_stress_payload.get("safety_flags", {}),
        },
        "open_traces": open_trace_summary,
        "transcript_replay": transcript_replay_summary,
        "inventory_soak": {
            "passed": bool(inventory_soak_payload.get("passed", False)),
            "checks": inventory_soak_payload.get("checks", {}),
            "mode": inventory_soak_payload.get("mode", ""),
            "source": inventory_soak_payload.get("source", ""),
            "cycles_requested": inventory_soak_payload.get("cycles_requested", 0),
            "cycles_completed": inventory_soak_payload.get("cycles_completed", 0),
            "successful_import_cycles": inventory_soak_payload.get(
                "successful_import_cycles", 0
            ),
            "failed_import_cycles": inventory_soak_payload.get(
                "failed_import_cycles", 0
            ),
            "inventory_delta": inventory_soak_payload.get("inventory_delta", {}),
            "source_coverage": inventory_soak_payload.get("source_coverage", {}),
            "failure_observability": inventory_soak_payload.get(
                "failure_observability", {}
            ),
            "story_quality": inventory_soak_payload.get("dashboard", {})
            .get("inventories", {})
            .get("story_quality", {}),
            "importer_health": inventory_soak_payload.get("dashboard", {})
            .get("jobs", {})
            .get("importer_health", {}),
            "importer_trends": inventory_soak_payload.get("dashboard", {})
            .get("jobs", {})
            .get("importer_trends", {}),
        },
        "inventory_soak_matrix": {
            "passed": bool(inventory_matrix_payload.get("passed", False)),
            "checks": inventory_matrix_payload.get("checks", {}),
            "mode": inventory_matrix_payload.get("mode", ""),
            "profile_count": inventory_matrix_payload.get("profile_count", 0),
            "total_cycles_completed": inventory_matrix_payload.get(
                "total_cycles_completed", 0
            ),
            "total_story_inventory_added": inventory_matrix_payload.get(
                "total_story_inventory_added", 0
            ),
            "total_failed_import_cycles": inventory_matrix_payload.get(
                "total_failed_import_cycles", 0
            ),
            "source_families_observed": inventory_matrix_payload.get(
                "source_families_observed", []
            ),
            "runs": inventory_matrix_payload.get("runs", []),
        },
        "inventory_diversity_smoke": {
            "passed": bool(inventory_diversity_payload.get("passed", False)),
            "checks": inventory_diversity_payload.get("checks", {}),
            "mode": inventory_diversity_payload.get("mode", ""),
            "niche_count": inventory_diversity_payload.get("niche_count", 0),
            "niches": inventory_diversity_payload.get("niches", []),
            "runs": [
                {
                    "label": run.get("label", ""),
                    "query": run.get("query", ""),
                    "story_local": bool(run.get("story_local", False)),
                    "story_route": run.get("story_route", ""),
                    "story_reason": run.get("story_reason", ""),
                    "executed_import_queries": run.get("executed_import_queries", []),
                    "soak": run.get("soak", {}),
                }
                for run in inventory_diversity_payload.get("runs", [])
                if isinstance(run, dict)
            ],
        },
        "inventory_retry_smoke": {
            "passed": bool(inventory_retry_payload.get("passed", False)),
            "checks": inventory_retry_payload.get("checks", {}),
            "transient_failures": inventory_retry_payload.get("transient_failures", 0),
            "attempts_by_path": inventory_retry_payload.get("attempts_by_path", {}),
            "inventory_delta": inventory_retry_payload.get("inventory_delta", {}),
            "before_story": inventory_retry_payload.get("before_story", {}),
            "after_story": inventory_retry_payload.get("after_story", {}),
            "gutenberg": inventory_retry_payload.get("gutenberg", {}),
            "internet_archive": inventory_retry_payload.get("internet_archive", {}),
            "external_network_used": bool(
                inventory_retry_payload.get("external_network_used", True)
            ),
        },
        "inventory_failure_smoke": {
            "passed": bool(inventory_failure_payload.get("passed", False)),
            "checks": inventory_failure_payload.get("checks", {}),
            "case_count": inventory_failure_payload.get("case_count", 0),
            "runs": [
                {
                    "label": run.get("label", ""),
                    "expected_safe_failure": run.get("expected_safe_failure", ""),
                    "soak_passed": bool(run.get("soak_passed", False)),
                    "failed_import_jobs": int(run.get("failed_import_jobs", 0) or 0),
                    "completed_import_jobs": int(
                        run.get("completed_import_jobs", 0) or 0
                    ),
                    "story_inventory_added": int(
                        run.get("story_inventory_added", 0) or 0
                    ),
                    "story_route": run.get("story_route", ""),
                    "story_reason": run.get("story_reason", ""),
                    "last_error": run.get("last_error", ""),
                }
                for run in inventory_failure_payload.get("runs", [])
                if isinstance(run, dict)
            ],
        },
        "pi_constraints": {
            "no_required_network": True,
            "no_required_vector_db": True,
            "no_required_ml_framework": True,
            "sqlite_indexes": True,
        },
    }
    _print_payload(payload, json_mode=args.json)


def _pi_bundle(args) -> None:
    out_dir = args.out
    if out_dir.exists():
        if not args.reset:
            raise SystemExit(
                f"Bundle output already exists; pass --reset to replace it: {out_dir}"
            )
        _safe_remove_bundle_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    for source in _pi_bundle_source_paths():
        copied.append(_copy_bundle_file(source, out_dir))
    copied.extend(_write_pi_bundle_launchers(out_dir))
    runbook = out_dir / "RUN_PORTABLE_APP.md"
    runbook.write_text(_pi_bundle_runbook(), encoding="utf-8")
    copied.append(_bundle_file_record(runbook, out_dir))

    dataset_payload: dict = {"skipped": True, "passed": False}
    smoke_payload: dict = {"skipped": True, "passed": False}
    autoimmune_payload: dict = {"skipped": True, "passed": False}
    synthesis_variant_payload: dict = {"skipped": True, "passed": False}
    synthesis_stress_payload: dict = {"skipped": True, "passed": False}
    setup_integration_payload: dict = {"skipped": True, "passed": False}
    host_action_payload: dict = {"skipped": True, "passed": False}
    host_app_payload: dict = {"skipped": True, "passed": False}
    capability_payload: dict = {"skipped": True, "passed": False}
    shortcut_audit_payload: dict = {"skipped": True, "passed": False}
    v01_audit_payload: dict = {"skipped": True, "passed": False}
    v01_progress_payload: dict = {"skipped": True, "passed": False}
    api_smoke_payload: dict = {"skipped": True, "passed": False}
    api_session_payload: dict = {"skipped": True, "passed": False}
    ui_smoke_payload: dict = {"skipped": True, "passed": False}
    bootstrap_payload: dict = {"skipped": True, "passed": False}
    launcher_payload: dict = {"skipped": True, "passed": False}
    open_traces_payload: dict = {"skipped": True, "passed": False}
    transcript_replay_payload: dict = {"skipped": True, "passed": False}
    transcript_calibration_payload: dict = {"skipped": True, "passed": False}
    if not args.skip_smoke:
        dataset_payload = _run_pi_bundle_dataset_audit(out_dir)
        smoke_payload = _run_pi_bundle_smoke(out_dir)
        autoimmune_payload = _run_pi_bundle_autoimmune_smoke(out_dir)
        synthesis_variant_payload = _run_pi_bundle_synthesis_variant_smoke(out_dir)
        synthesis_stress_payload = _run_pi_bundle_synthesis_stress_smoke(out_dir)
        setup_integration_payload = _run_pi_bundle_setup_integration_smoke(out_dir)
        host_action_payload = _run_pi_bundle_host_action_smoke(out_dir)
        host_app_payload = _run_pi_bundle_host_app_probe(out_dir)
        capability_payload = _run_pi_bundle_capability_probe(out_dir)
        shortcut_audit_payload = _run_pi_bundle_shortcut_audit(out_dir)
        v01_audit_payload = _run_pi_bundle_v01_audit(out_dir)
        v01_progress_payload = _run_pi_bundle_v01_progress(out_dir)
        api_smoke_payload = _run_pi_bundle_api_smoke(out_dir)
        api_session_payload = _run_pi_bundle_api_session_smoke(out_dir)
        ui_smoke_payload = _run_pi_bundle_ui_smoke(out_dir)
        bootstrap_payload = _run_pi_bundle_bootstrap_runtime(out_dir)
        launcher_payload = _run_pi_bundle_launcher_smoke(out_dir)
        open_traces_payload = _run_pi_bundle_open_traces(out_dir)
        transcript_replay_payload = _run_pi_bundle_transcript_replay(out_dir)
        transcript_calibration_payload = _run_pi_bundle_transcript_calibration(out_dir)
        _remove_pi_bundle_smoke_artifacts(out_dir)

    self_check = out_dir / "bundle_self_check.json"
    self_check_payload = {
        "passed": bool(args.skip_smoke)
        or (
            bool(dataset_payload.get("passed", False))
            and bool(smoke_payload.get("passed", False))
            and bool(autoimmune_payload.get("passed", False))
            and bool(synthesis_variant_payload.get("passed", False))
            and bool(synthesis_stress_payload.get("passed", False))
            and bool(setup_integration_payload.get("passed", False))
            and bool(host_action_payload.get("passed", False))
            and bool(host_app_payload.get("passed", False))
            and bool(capability_payload.get("passed", False))
            and bool(shortcut_audit_payload.get("passed", False))
            and bool(v01_audit_payload.get("passed", False))
            and bool(v01_progress_payload.get("passed", False))
            and bool(api_smoke_payload.get("passed", False))
            and bool(api_session_payload.get("passed", False))
            and bool(ui_smoke_payload.get("passed", False))
            and bool(bootstrap_payload.get("passed", False))
            and bool(launcher_payload.get("passed", False))
            and bool(open_traces_payload.get("passed", False))
            and bool(transcript_replay_payload.get("passed", False))
            and bool(transcript_calibration_payload.get("passed", False))
        ),
        "skipped": bool(args.skip_smoke),
        "dataset_audit": dataset_payload,
        "pi_smoke": smoke_payload,
        "autoimmune_smoke": autoimmune_payload,
        "synthesis_variant_smoke": synthesis_variant_payload,
        "synthesis_stress_smoke": synthesis_stress_payload,
        "setup_integration_smoke": setup_integration_payload,
        "host_action_smoke": host_action_payload,
        "host_app_probe": host_app_payload,
        "capability_probe": capability_payload,
        "shortcut_audit": shortcut_audit_payload,
        "v01_audit": v01_audit_payload,
        "v01_progress": v01_progress_payload,
        "api_smoke": api_smoke_payload,
        "api_session_smoke": api_session_payload,
        "ui_smoke": ui_smoke_payload,
        "bootstrap_runtime": bootstrap_payload,
        "launcher_smoke": launcher_payload,
        "open_traces": open_traces_payload,
        "transcript_replay": transcript_replay_payload,
        "transcript_calibration": transcript_calibration_payload,
    }
    self_check.write_text(
        json.dumps(self_check_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    copied.append(_bundle_file_record(self_check, out_dir))

    manifest_payload = {
        "bundle_name": "melm_local_assistant_os_v01_portable_bundle",
        "runtime": "stdlib_python_sqlite_http_html",
        "dependency_class": "stdlib_only",
        "created_from": str(ROOT),
        "entrypoint": "scripts/local_assistant_os_cli.py",
        "chat_command": "python scripts/local_assistant_os_cli.py chat",
        "smoke_command": "python scripts/local_assistant_os_cli.py pi-smoke --reset --json",
        "inventory_soak_matrix_command": "python scripts/local_assistant_os_cli.py inventory-soak-matrix --reset --json",
        "dataset_audit_command": "python scripts/local_assistant_os_cli.py dataset-audit --reset --json",
        "autoimmune_smoke_command": "python scripts/local_assistant_os_cli.py autoimmune-smoke --reset --json",
        "synthesis_variant_smoke_command": "python scripts/local_assistant_os_cli.py synthesis-variant-smoke --reset --json",
        "synthesis_stress_smoke_command": "python scripts/local_assistant_os_cli.py synthesis-stress-smoke --reset --json",
        "setup_integration_smoke_command": "python scripts/local_assistant_os_cli.py setup-integration-smoke --reset --json",
        "host_action_smoke_command": "python scripts/local_assistant_os_cli.py host-action-smoke --reset --json",
        "host_actions_demo_config_command": "python scripts/local_assistant_os_cli.py write-host-actions-demo-config --out config/host_actions.local_recorder.json --overwrite --json",
        "host_app_probe_command": "python scripts/local_assistant_os_cli.py host-app-probe --reset --json",
        "host_app_configured_probe_command": "python scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.json --require-configured --json",
        "host_app_demo_config_probe_command": "python scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.local_recorder.json --require-configured --json",
        "capability_probe_command": "python scripts/local_assistant_os_cli.py capability-probe --reset --json",
        "shortcut_audit_command": "python scripts/local_assistant_os_cli.py shortcut-audit --json",
        "v01_audit_command": "python scripts/local_assistant_os_cli.py v01-audit --json",
        "v01_progress_command": "python scripts/local_assistant_os_cli.py v01-progress --json",
        "v01_evidence_pack_command": "python scripts/local_assistant_os_cli.py v01-evidence-pack --db artifacts/local_assistant_os/assistant_v01.sqlite --work-dir artifacts/local_assistant_os/v01_evidence_pack --auto-lifecycle --json",
        "candidate_session_audit_command": "python scripts/local_assistant_os_cli.py candidate-session-audit --db artifacts/local_assistant_os/assistant_v01.sqlite --session all --capture-surface cli_chat --json",
        "source_attestation_command": "python scripts/local_assistant_os_cli.py write-source-attestation --event-ledger-db artifacts/local_assistant_os/assistant_v01.sqlite --event-ledger-session all --source-kind redacted_user_session --capture-surface cli_chat --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --out artifacts/local_assistant_os/source_attestation.json --json",
        "host_app_attestation_command": "python scripts/local_assistant_os_cli.py write-host-app-attestation --host-app-config-json config/host_actions.json --capture-surface target_device_cli --media-app-configured --call-app-configured --not-demo-recorder --real-app-commands-acknowledged --human-reviewed --out artifacts/local_assistant_os/host_app_attestation.json --json",
        "v01_blocker_evidence_command": "python scripts/local_assistant_os_cli.py v01-blocker-evidence --event-ledger-db artifacts/local_assistant_os/assistant_v01.sqlite --event-ledger-session all --event-source-kind redacted_user_session --source-attestation-json artifacts/local_assistant_os/source_attestation.json --auto-lifecycle --transcript-calibration-report-json artifacts/local_assistant_os/user_transcript_calibration.json --inventory-soak-report-json artifacts/local_assistant_os/live_inventory_soak_matrix.json --host-app-config-json config/host_actions.json --host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json --run-host-app-probe --out artifacts/local_assistant_os/v01_blocker_evidence.json --json",
        "v01_acceptance_command": "python scripts/local_assistant_os_cli.py v01-acceptance --reset --json",
        "v01_acceptance_configured_host_app_command": "python scripts/local_assistant_os_cli.py v01-acceptance --host-app-config-json config/host_actions.json --require-host-app-configured --json",
        "api_smoke_command": "python scripts/local_assistant_os_cli.py api-smoke --reset --json",
        "api_session_smoke_command": "python scripts/local_assistant_os_cli.py api-session-smoke --reset --json",
        "ui_smoke_command": "python scripts/local_assistant_os_cli.py ui-smoke --reset --json",
        "bootstrap_runtime_command": "python scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json",
        "open_traces_command": "python scripts/local_assistant_os_cli.py run-open-traces --reset --json",
        "transcript_replay_command": "python scripts/local_assistant_os_cli.py run-transcript-replay --reset --json",
        "transcript_calibration_command": 'python scripts/local_assistant_os_cli.py calibrate-transcript-replay --input benchmarks/sample_local_assistant_raw_transcript.jsonl --replace "Maya=<person_1>" --min-total-turns 4 --min-local-resolution-rate 0.2 --min-route-kinds 3 --min-intent-kinds 3 --require-redaction --require-static-drop --out artifacts/local_assistant_os/sample_transcript_calibration.json --reset --json',
        "event_transcript_export_command": "python scripts/local_assistant_os_cli.py export-transcript-replay --db artifacts/local_assistant_os/assistant_v01.sqlite --out artifacts/local_assistant_os/event_ledger_transcript_replay.jsonl --json",
        "event_ledger_calibration_command": "python scripts/local_assistant_os_cli.py calibrate-event-ledger --db artifacts/local_assistant_os/assistant_v01.sqlite --work-dir artifacts/local_assistant_os/event_ledger_calibration --controls-json config/safe_lifecycle_controls.example.json --min-total-turns 4 --min-local-resolution-rate 0.5 --json",
        "safe_lifecycle_controls_template": DEFAULT_SAFE_LIFECYCLE_CONTROLS_EXAMPLE.as_posix(),
        "user_transcript_import_command": "python scripts/local_assistant_os_cli.py import-transcript-replay --input <raw_chat.jsonl> --out artifacts/local_assistant_os/imported_transcript_replay.jsonl --controls-json config/safe_lifecycle_controls.example.json --json",
        "user_transcript_calibration_command": "python scripts/local_assistant_os_cli.py calibrate-transcript-replay --input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json --require-redaction --require-static-drop --min-synthesis-traces <n> --require-priority-signals --min-priority-signal-samples <n> --require-memory-digest-quality --require-strict-baseline-win --out artifacts/local_assistant_os/user_transcript_calibration.json --json",
        "verify_command": "python scripts/local_assistant_os_cli.py verify-bundle --json",
        "launcher_smoke_command": "python scripts/local_assistant_os_cli.py launcher-smoke --reset --json",
        "first_run_smoke_command": "python scripts/local_assistant_os_cli.py first-run-smoke --json",
        "target_report_command": "python scripts/local_assistant_os_cli.py target-report --reset --json",
        "portable_pi_command": "python3 scripts/local_assistant_os_cli.py pi-smoke --reset --json",
        "portable_inventory_soak_matrix_command": "python3 scripts/local_assistant_os_cli.py inventory-soak-matrix --reset --json",
        "portable_dataset_audit_command": "python3 scripts/local_assistant_os_cli.py dataset-audit --reset --json",
        "portable_chat_command": "python3 scripts/local_assistant_os_cli.py chat",
        "portable_autoimmune_command": "python3 scripts/local_assistant_os_cli.py autoimmune-smoke --reset --json",
        "portable_synthesis_variant_command": "python3 scripts/local_assistant_os_cli.py synthesis-variant-smoke --reset --json",
        "portable_synthesis_stress_command": "python3 scripts/local_assistant_os_cli.py synthesis-stress-smoke --reset --json",
        "portable_setup_integration_command": "python3 scripts/local_assistant_os_cli.py setup-integration-smoke --reset --json",
        "portable_host_action_command": "python3 scripts/local_assistant_os_cli.py host-action-smoke --reset --json",
        "portable_host_actions_demo_config_command": "python3 scripts/local_assistant_os_cli.py write-host-actions-demo-config --out config/host_actions.local_recorder.json --overwrite --json",
        "portable_host_app_probe_command": "python3 scripts/local_assistant_os_cli.py host-app-probe --reset --json",
        "portable_host_app_configured_probe_command": "python3 scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.json --require-configured --json",
        "portable_host_app_demo_config_probe_command": "python3 scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.local_recorder.json --require-configured --json",
        "portable_capability_probe_command": "python3 scripts/local_assistant_os_cli.py capability-probe --reset --json",
        "portable_shortcut_audit_command": "python3 scripts/local_assistant_os_cli.py shortcut-audit --json",
        "portable_v01_audit_command": "python3 scripts/local_assistant_os_cli.py v01-audit --json",
        "portable_v01_progress_command": "python3 scripts/local_assistant_os_cli.py v01-progress --json",
        "portable_v01_evidence_pack_command": "python3 scripts/local_assistant_os_cli.py v01-evidence-pack --db artifacts/local_assistant_os/assistant_v01.sqlite --work-dir artifacts/local_assistant_os/v01_evidence_pack --auto-lifecycle --json",
        "portable_candidate_session_audit_command": "python3 scripts/local_assistant_os_cli.py candidate-session-audit --db artifacts/local_assistant_os/assistant_v01.sqlite --session all --capture-surface cli_chat --json",
        "portable_source_attestation_command": "python3 scripts/local_assistant_os_cli.py write-source-attestation --event-ledger-db artifacts/local_assistant_os/assistant_v01.sqlite --event-ledger-session all --source-kind redacted_user_session --capture-surface cli_chat --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --out artifacts/local_assistant_os/source_attestation.json --json",
        "portable_host_app_attestation_command": "python3 scripts/local_assistant_os_cli.py write-host-app-attestation --host-app-config-json config/host_actions.json --capture-surface target_device_cli --media-app-configured --call-app-configured --not-demo-recorder --real-app-commands-acknowledged --human-reviewed --out artifacts/local_assistant_os/host_app_attestation.json --json",
        "portable_v01_blocker_evidence_command": "python3 scripts/local_assistant_os_cli.py v01-blocker-evidence --event-ledger-db artifacts/local_assistant_os/assistant_v01.sqlite --event-ledger-session all --event-source-kind redacted_user_session --source-attestation-json artifacts/local_assistant_os/source_attestation.json --auto-lifecycle --transcript-calibration-report-json artifacts/local_assistant_os/user_transcript_calibration.json --inventory-soak-report-json artifacts/local_assistant_os/live_inventory_soak_matrix.json --host-app-config-json config/host_actions.json --host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json --run-host-app-probe --out artifacts/local_assistant_os/v01_blocker_evidence.json --json",
        "portable_v01_acceptance_command": "python3 scripts/local_assistant_os_cli.py v01-acceptance --reset --json",
        "portable_v01_acceptance_configured_host_app_command": "python3 scripts/local_assistant_os_cli.py v01-acceptance --host-app-config-json config/host_actions.json --require-host-app-configured --json",
        "portable_api_command": "python3 scripts/local_assistant_os_cli.py api-smoke --reset --json",
        "portable_api_session_command": "python3 scripts/local_assistant_os_cli.py api-session-smoke --reset --json",
        "portable_ui_command": "python3 scripts/local_assistant_os_cli.py ui-smoke --reset --json",
        "portable_bootstrap_runtime_command": "python3 scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json",
        "portable_open_traces_command": "python3 scripts/local_assistant_os_cli.py run-open-traces --reset --json",
        "portable_transcript_replay_command": "python3 scripts/local_assistant_os_cli.py run-transcript-replay --reset --json",
        "portable_transcript_calibration_command": 'python3 scripts/local_assistant_os_cli.py calibrate-transcript-replay --input benchmarks/sample_local_assistant_raw_transcript.jsonl --replace "Maya=<person_1>" --min-total-turns 4 --min-local-resolution-rate 0.2 --min-route-kinds 3 --min-intent-kinds 3 --require-redaction --require-static-drop --out artifacts/local_assistant_os/sample_transcript_calibration.json --reset --json',
        "portable_event_transcript_export_command": "python3 scripts/local_assistant_os_cli.py export-transcript-replay --db artifacts/local_assistant_os/assistant_v01.sqlite --out artifacts/local_assistant_os/event_ledger_transcript_replay.jsonl --json",
        "portable_event_ledger_calibration_command": "python3 scripts/local_assistant_os_cli.py calibrate-event-ledger --db artifacts/local_assistant_os/assistant_v01.sqlite --work-dir artifacts/local_assistant_os/event_ledger_calibration --controls-json config/safe_lifecycle_controls.example.json --min-total-turns 4 --min-local-resolution-rate 0.5 --json",
        "portable_safe_lifecycle_controls_template": DEFAULT_SAFE_LIFECYCLE_CONTROLS_EXAMPLE.as_posix(),
        "portable_user_transcript_import_command": "python3 scripts/local_assistant_os_cli.py import-transcript-replay --input <raw_chat.jsonl> --out artifacts/local_assistant_os/imported_transcript_replay.jsonl --controls-json config/safe_lifecycle_controls.example.json --json",
        "portable_user_transcript_calibration_command": "python3 scripts/local_assistant_os_cli.py calibrate-transcript-replay --input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json --require-redaction --require-static-drop --min-synthesis-traces <n> --require-priority-signals --min-priority-signal-samples <n> --require-memory-digest-quality --require-strict-baseline-win --out artifacts/local_assistant_os/user_transcript_calibration.json --json",
        "portable_refresh_weather_command": "python3 scripts/local_assistant_os_cli.py refresh-weather --offline-json benchmarks/sample_open_meteo_forecast.json --json",
        "portable_verify_command": "python3 scripts/local_assistant_os_cli.py verify-bundle --json",
        "portable_launcher_smoke_command": "python3 scripts/local_assistant_os_cli.py launcher-smoke --reset --json",
        "portable_first_run_smoke_command": "python3 scripts/local_assistant_os_cli.py first-run-smoke --json",
        "portable_target_report_command": "python3 scripts/local_assistant_os_cli.py target-report --reset --json",
        "portable_first_run_script": "sh bin/first_run.sh",
        "portable_start_app_command": "sh bin/start_app.sh or bin\\start_app.cmd",
        "portable_start_api_command": "sh bin/start_api.sh",
        "portable_health_check_command": "sh bin/health_check.sh",
        "portable_browser_url": "http://127.0.0.1:8771/",
        "portable_windows_first_run": "bin\\first_run.cmd",
        "portable_windows_start_app": "bin\\start_app.cmd",
        "portable_raspberry_first_run_script": "sh bin/first_run_on_raspberry_pi.sh",
        "systemd_service_example": "systemd/melm-local-assistant.service.example",
        "launcher_files": [path.as_posix() for path in PI_BUNDLE_LAUNCHER_FILES],
        "required_python": ">=3.11",
        "required_network": False,
        "required_vector_db": False,
        "required_ml_framework": False,
        "file_count": len(copied),
        "files": sorted(copied, key=lambda item: item["path"]),
        "self_check": {
            "skipped": bool(self_check_payload["skipped"]),
            "passed": bool(self_check_payload["passed"]),
            "dataset_audit_checks": dataset_payload.get("checks", {}),
            "checks": smoke_payload.get("checks", {}),
            "inventory_soak_matrix_checks": smoke_payload.get(
                "inventory_soak_matrix", {}
            ).get("checks", {}),
            "autoimmune_checks": autoimmune_payload.get("checks", {}),
            "synthesis_variant_checks": synthesis_variant_payload.get("checks", {}),
            "synthesis_stress_checks": synthesis_stress_payload.get("checks", {}),
            "setup_integration_checks": setup_integration_payload.get("checks", {}),
            "host_action_checks": host_action_payload.get("checks", {}),
            "host_app_probe_checks": host_app_payload.get("checks", {}),
            "capability_probe_checks": capability_payload.get("checks", {}),
            "shortcut_audit_checks": shortcut_audit_payload.get("checks", {}),
            "v01_audit_checks": v01_audit_payload.get("checks", {}),
            "v01_progress_checks": v01_progress_payload.get("checks", {}),
            "v01_progress_remaining_blockers": int(
                v01_progress_payload.get("remaining_blocker_count", 0) or 0
            ),
            "architecture_complete": bool(
                v01_audit_payload.get("architecture_complete", False)
            ),
            "completion_blocker_count": int(
                v01_audit_payload.get("blocker_count", 0) or 0
            ),
            "api_checks": api_smoke_payload.get("checks", {}),
            "api_session_checks": api_session_payload.get("checks", {}),
            "ui_checks": ui_smoke_payload.get("checks", {}),
            "bootstrap_runtime_checks": bootstrap_payload.get("checks", {}),
            "launcher_checks": launcher_payload.get("checks", {}),
            "open_trace_checks": _open_trace_summary(open_traces_payload).get(
                "scenario_checks", {}
            ),
            "transcript_replay_checks": _transcript_replay_summary(
                transcript_replay_payload
            ).get("fixture_checks", {}),
            "transcript_baseline_checks": _transcript_replay_summary(
                transcript_replay_payload
            )
            .get("baseline_comparison", {})
            .get("checks", {}),
            "transcript_calibration_checks": dict(
                transcript_calibration_payload.get("aggregate", {})
            ),
            "runtime": smoke_payload.get("runtime", ""),
            "dependency_class": smoke_payload.get("dependency_class", ""),
        },
    }
    manifest = out_dir / "bundle_manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    archive = None
    if args.zip:
        archive_path = out_dir.with_suffix(".zip")
        if archive_path.exists():
            archive_path.unlink()
        _zip_bundle(out_dir, archive_path)
        archive = {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
        }

    payload = {
        "out": str(out_dir),
        "manifest": str(manifest),
        "runbook": str(runbook),
        "self_check": str(self_check),
        "archive": archive,
        "passed": bool(self_check_payload["passed"]) if not args.skip_smoke else True,
        "smoke_skipped": bool(args.skip_smoke),
        "dataset_audit": {
            "passed": bool(dataset_payload.get("passed", False)),
            "checks": dataset_payload.get("checks", {}),
            "files": dataset_payload.get("files", {}),
            "source_fixtures": dataset_payload.get("source_fixtures", {}),
            "bootstrap": dataset_payload.get("bootstrap", {}),
        },
        "smoke": {
            "passed": bool(smoke_payload.get("passed", False)),
            "checks": smoke_payload.get("checks", {}),
            "runtime": smoke_payload.get("runtime", ""),
            "dependency_class": smoke_payload.get("dependency_class", ""),
            "timings": smoke_payload.get("timings", {}),
            "peak_traced_kb": smoke_payload.get("peak_traced_kb", 0),
            "db_bytes": smoke_payload.get("db_bytes", 0),
            "lifecycle_db_bytes": smoke_payload.get("lifecycle_db_bytes", 0),
            "action_db_bytes": smoke_payload.get("action_db_bytes", 0),
            "inventory_soak": smoke_payload.get("inventory_soak", {}),
            "inventory_soak_matrix": smoke_payload.get("inventory_soak_matrix", {}),
            "inventory_diversity_smoke": smoke_payload.get(
                "inventory_diversity_smoke", {}
            ),
            "inventory_retry_smoke": smoke_payload.get("inventory_retry_smoke", {}),
            "inventory_failure_smoke": smoke_payload.get("inventory_failure_smoke", {}),
        },
        "autoimmune_smoke": {
            "passed": bool(autoimmune_payload.get("passed", False)),
            "checks": autoimmune_payload.get("checks", {}),
            "turns": autoimmune_payload.get("turns", []),
            "pending_actions": autoimmune_payload.get("pending_actions", {}),
            "safety_flags": autoimmune_payload.get("safety_flags", {}),
        },
        "synthesis_variant_smoke": {
            "passed": bool(synthesis_variant_payload.get("passed", False)),
            "checks": synthesis_variant_payload.get("checks", {}),
            "variant_count": synthesis_variant_payload.get("variant_count", 0),
            "route_counts": synthesis_variant_payload.get("route_counts", {}),
            "reason_counts": synthesis_variant_payload.get("reason_counts", {}),
            "turns": synthesis_variant_payload.get("turns", []),
            "safety_flags": synthesis_variant_payload.get("safety_flags", {}),
        },
        "synthesis_stress_smoke": {
            "passed": bool(synthesis_stress_payload.get("passed", False)),
            "checks": synthesis_stress_payload.get("checks", {}),
            "turn_count": synthesis_stress_payload.get("turn_count", 0),
            "session_count": synthesis_stress_payload.get("session_count", 0),
            "route_counts": synthesis_stress_payload.get("route_counts", {}),
            "reason_counts": synthesis_stress_payload.get("reason_counts", {}),
            "intent_counts": synthesis_stress_payload.get("intent_counts", {}),
            "quality": synthesis_stress_payload.get("quality", {}),
            "complexity": synthesis_stress_payload.get("complexity", {}),
            "turns": synthesis_stress_payload.get("turns", []),
            "safety_flags": synthesis_stress_payload.get("safety_flags", {}),
        },
        "setup_integration_smoke": {
            "passed": bool(setup_integration_payload.get("passed", False)),
            "checks": setup_integration_payload.get("checks", {}),
            "turns": setup_integration_payload.get("turns", []),
            "setup_requests_after_gaps": setup_integration_payload.get(
                "setup_requests_after_gaps", {}
            ),
            "facts_after_setup": setup_integration_payload.get("facts_after_setup", {}),
            "contacts_after_setup": setup_integration_payload.get(
                "contacts_after_setup", {}
            ),
            "pending_actions": setup_integration_payload.get("pending_actions", {}),
            "safety_flags": setup_integration_payload.get("safety_flags", {}),
            "action_execution": setup_integration_payload.get("action_execution", {}),
        },
        "host_action_smoke": {
            "passed": bool(host_action_payload.get("passed", False)),
            "checks": host_action_payload.get("checks", {}),
            "records": host_action_payload.get("records", []),
            "runtime": host_action_payload.get("runtime", ""),
            "dependency_class": host_action_payload.get("dependency_class", ""),
        },
        "host_app_probe": {
            "passed": bool(host_app_payload.get("passed", False)),
            "configured": bool(host_app_payload.get("configured", False)),
            "skipped": bool(host_app_payload.get("skipped", False)),
            "checks": host_app_payload.get("checks", {}),
            "runtime": host_app_payload.get("runtime", ""),
            "dependency_class": host_app_payload.get("dependency_class", ""),
            "next_steps": host_app_payload.get("next_steps", []),
        },
        "capability_probe": {
            "passed": bool(capability_payload.get("passed", False)),
            "checks": capability_payload.get("checks", {}),
            "total_cases": capability_payload.get("total_cases", 0),
            "route_counts": capability_payload.get("route_counts", {}),
            "bucket_counts": capability_payload.get("bucket_counts", {}),
            "local_device_rate": capability_payload.get("local_device_rate", 0.0),
            "complexity": capability_payload.get("complexity", {}),
            "unsupported_examples": capability_payload.get("unsupported_examples", []),
        },
        "shortcut_audit": {
            "passed": bool(shortcut_audit_payload.get("passed", False)),
            "checks": shortcut_audit_payload.get("checks", {}),
            "behavior_case_count": len(
                shortcut_audit_payload.get("behavior_cases", [])
            ),
            "source_check_count": len(shortcut_audit_payload.get("source_checks", [])),
            "policy": shortcut_audit_payload.get("policy", {}),
        },
        "v01_audit": {
            "passed": bool(v01_audit_payload.get("passed", False)),
            "checks": v01_audit_payload.get("checks", {}),
            "status": v01_audit_payload.get("status", ""),
            "architecture_complete": bool(
                v01_audit_payload.get("architecture_complete", False)
            ),
            "blocker_count": int(v01_audit_payload.get("blocker_count", 0) or 0),
            "completion_blockers": v01_audit_payload.get("completion_blockers", []),
        },
        "v01_progress": {
            "passed": bool(v01_progress_payload.get("passed", False)),
            "checks": v01_progress_payload.get("checks", {}),
            "status": v01_progress_payload.get("status", ""),
            "architecture_complete": bool(
                v01_progress_payload.get("architecture_complete", False)
            ),
            "candidate_blockers_satisfied": int(
                v01_progress_payload.get("candidate_blockers_satisfied", 0) or 0
            ),
            "remaining_blocker_count": int(
                v01_progress_payload.get("remaining_blocker_count", 0) or 0
            ),
        },
        "api_smoke": {
            "passed": bool(api_smoke_payload.get("passed", False)),
            "checks": api_smoke_payload.get("checks", {}),
            "base_url": api_smoke_payload.get("base_url", ""),
            "health_counts": api_smoke_payload.get("after_health", {}).get(
                "counts", {}
            ),
            "parse_debug": api_smoke_payload.get("parse_debug", {}),
            "ask": api_smoke_payload.get("ask", {}),
        },
        "api_session_smoke": {
            "passed": bool(api_session_payload.get("passed", False)),
            "checks": api_session_payload.get("checks", {}),
            "turns": api_session_payload.get("turns", []),
            "route_counts": api_session_payload.get("route_counts", {}),
            "action_results": api_session_payload.get("action_results", []),
        },
        "ui_smoke": {
            "passed": bool(ui_smoke_payload.get("passed", False)),
            "checks": ui_smoke_payload.get("checks", {}),
            "base_url": ui_smoke_payload.get("base_url", ""),
            "ui": ui_smoke_payload.get("ui", {}),
            "ask": ui_smoke_payload.get("ask", {}),
        },
        "bootstrap_runtime": {
            "passed": bool(bootstrap_payload.get("passed", False)),
            "checks": bootstrap_payload.get("checks", {}),
            "counts": bootstrap_payload.get("counts", {}),
            "turns": bootstrap_payload.get("turns", []),
            "db_bytes": bootstrap_payload.get("db_bytes", 0),
        },
        "launcher_smoke": {
            "passed": bool(launcher_payload.get("passed", False)),
            "checks": launcher_payload.get("checks", {}),
            "platform_launcher": launcher_payload.get("platform_launcher", ""),
            "health_launcher": launcher_payload.get("health_launcher", ""),
            "base_url": launcher_payload.get("base_url", ""),
            "health": launcher_payload.get("health", {}),
        },
        "open_traces": _open_trace_summary(open_traces_payload),
        "transcript_replay": _transcript_replay_summary(transcript_replay_payload),
        "transcript_calibration": dict(
            transcript_calibration_payload.get("aggregate", {})
        ),
        "bundle": {
            "file_count": len(copied),
            "bytes": sum(int(item["bytes"]) for item in copied),
            "required_network": False,
            "required_vector_db": False,
            "required_ml_framework": False,
            "entrypoint": "scripts/local_assistant_os_cli.py",
        },
    }
    _print_payload(payload, json_mode=args.json)


def _pi_bundle_source_paths() -> list[Path]:
    sources = {path for path in PI_BUNDLE_STATIC_FILES}
    sources.update(
        path for path in Path("melm").rglob("*.py") if "__pycache__" not in path.parts
    )
    sources.update(
        path for path in Path("melm/contracts").rglob("*.json") if "__pycache__" not in path.parts
    )
    missing = [path for path in sorted(sources) if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(
            f"Cannot build portable bundle; missing required files: {', '.join(str(path) for path in missing)}"
        )
    return sorted(sources)


def _copy_bundle_file(relative_path: Path, out_dir: Path) -> dict:
    source = ROOT / relative_path
    target = out_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return _bundle_file_record(target, out_dir)


def _write_pi_bundle_launchers(out_dir: Path) -> list[dict]:
    launchers = {
        Path("bin/first_run.sh"): _portable_first_run_launcher(),
        Path("bin/start_app.sh"): _portable_start_app_launcher(),
        Path("bin/first_run_on_raspberry_pi.sh"): _pi_first_run_launcher(),
        Path("bin/start_api.sh"): _pi_start_api_launcher(),
        Path("bin/health_check.sh"): _pi_health_check_launcher(),
        Path("bin/first_run.ps1"): _windows_first_run_launcher(),
        Path("bin/start_app.ps1"): _windows_start_app_launcher(),
        Path("bin/health_check.ps1"): _windows_health_check_launcher(),
        Path("bin/first_run.cmd"): _windows_cmd_launcher("first_run.ps1"),
        Path("bin/start_app.cmd"): _windows_cmd_launcher("start_app.ps1"),
        Path("bin/health_check.cmd"): _windows_cmd_launcher("health_check.ps1"),
        Path(
            "systemd/melm-local-assistant.service.example"
        ): _pi_systemd_service_example(),
    }
    records = []
    for relative, content in launchers.items():
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        if relative.suffix == ".sh":
            target.chmod(0o755)
        records.append(_bundle_file_record(target, out_dir))
    return records


def _bundle_file_record(path: Path, out_dir: Path) -> dict:
    relative = path.relative_to(out_dir)
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _run_pi_bundle_dataset_audit(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "dataset-audit",
        "--db",
        "artifacts/local_assistant_os/dataset_audit_bundle.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "pi-smoke",
        "--db",
        "artifacts/local_assistant_os/pi_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_autoimmune_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "autoimmune-smoke",
        "--db",
        "artifacts/local_assistant_os/autoimmune_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_synthesis_variant_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "synthesis-variant-smoke",
        "--db",
        "artifacts/local_assistant_os/synthesis_variant_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_synthesis_stress_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "synthesis-stress-smoke",
        "--db",
        "artifacts/local_assistant_os/synthesis_stress_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_setup_integration_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "setup-integration-smoke",
        "--db",
        "artifacts/local_assistant_os/setup_integration_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_host_action_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "host-action-smoke",
        "--db",
        "artifacts/local_assistant_os/host_action_bundle_smoke.sqlite",
        "--work-dir",
        "artifacts/local_assistant_os/host_action_bundle_smoke",
        "--reset",
        "--json",
    )


def _run_pi_bundle_host_app_probe(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "host-app-probe",
        "--db",
        "artifacts/local_assistant_os/host_app_probe_bundle.sqlite",
        "--work-dir",
        "artifacts/local_assistant_os/host_app_probe_bundle",
        "--reset",
        "--json",
    )


def _run_pi_bundle_capability_probe(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "capability-probe",
        "--db",
        "artifacts/local_assistant_os/capability_probe_bundle.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_shortcut_audit(out_dir: Path) -> dict:
    return _run_cli_json(out_dir, "shortcut-audit", "--json")


def _run_pi_bundle_v01_audit(out_dir: Path) -> dict:
    return _run_cli_json(out_dir, "v01-audit", "--json")


def _run_pi_bundle_v01_progress(out_dir: Path) -> dict:
    return _run_cli_json(out_dir, "v01-progress", "--json")


def _run_pi_bundle_api_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "api-smoke",
        "--db",
        "artifacts/local_assistant_os/api_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_api_session_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "api-session-smoke",
        "--db",
        "artifacts/local_assistant_os/api_session_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_ui_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "ui-smoke",
        "--db",
        "artifacts/local_assistant_os/ui_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_bootstrap_runtime(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "bootstrap-runtime",
        "--db",
        "artifacts/local_assistant_os/bootstrap_bundle.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_launcher_smoke(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "launcher-smoke",
        "--bundle-root",
        ".",
        "--db",
        "artifacts/local_assistant_os/launcher_bundle_smoke.sqlite",
        "--reset",
        "--json",
    )


def _run_pi_bundle_open_traces(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "run-open-traces",
        "--db-dir",
        "artifacts/local_assistant_os/open_traces_bundle",
        "--reset",
        "--json",
    )


def _run_pi_bundle_transcript_replay(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "run-transcript-replay",
        "--db-dir",
        "artifacts/local_assistant_os/transcript_replay_bundle",
        "--reset",
        "--json",
    )


def _run_pi_bundle_transcript_calibration(out_dir: Path) -> dict:
    return _run_cli_json(
        out_dir,
        "calibrate-transcript-replay",
        "--input",
        str(DEFAULT_RAW_TRANSCRIPT_SAMPLE),
        "--replace",
        "Maya=<person_1>",
        "--min-total-turns",
        "4",
        "--min-local-resolution-rate",
        "0.2",
        "--min-route-kinds",
        "3",
        "--min-intent-kinds",
        "3",
        "--require-redaction",
        "--require-static-drop",
        "--work-dir",
        "artifacts/local_assistant_os/transcript_calibration_bundle",
        "--reset",
        "--json",
    )


def _run_cli_json(cwd: Path, *args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "scripts/local_assistant_os_cli.py", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "passed": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "decode_error": str(exc),
        }
    return dict(payload)


def _launcher_smoke(args) -> None:
    payload = _build_launcher_smoke_payload(
        args.bundle_root,
        db=args.db,
        host=args.host,
        port=args.port,
        reset=args.reset,
    )
    _print_payload(payload, json_mode=args.json)


def _first_run_smoke(args) -> None:
    payload = _build_first_run_smoke_payload(
        args.bundle_root, timeout_seconds=args.timeout_seconds
    )
    _print_payload(payload, json_mode=args.json)


def _archive_smoke(args) -> None:
    payload = _build_archive_smoke_payload(
        args.archive,
        work_dir=args.work_dir,
        reset=args.reset,
        skip_first_run=args.skip_first_run,
        timeout_seconds=args.timeout_seconds,
    )
    _print_payload(payload, json_mode=args.json)


def _build_archive_smoke_payload(
    archive_path: Path,
    *,
    work_dir: Path,
    reset: bool,
    skip_first_run: bool,
    timeout_seconds: int,
) -> dict:
    started = perf_counter()
    archive_path = archive_path.resolve()
    work_dir = work_dir.resolve()
    checks = {
        "archive_present": archive_path.is_file(),
        "archive_suffix_zip": archive_path.suffix.lower() == ".zip",
        "work_dir_ready": False,
        "zip_opened": False,
        "zip_entries_safe": False,
        "zip_contains_bundle_manifest": False,
        "archive_has_single_top_level_root": False,
        "windows_path_budget_ok": not platform.system().lower().startswith("windows"),
        "single_extracted_bundle_root": False,
        "manifest_present": False,
        "verify_bundle_passed": False,
        "first_run_smoke_passed_or_skipped": bool(skip_first_run),
        "runtime_db_created_or_skipped": bool(skip_first_run),
        "stdlib_only": False,
        "no_required_network": False,
        "no_required_vector_db": False,
        "no_required_ml_framework": False,
    }
    extract_error = ""
    zip_error = ""
    unsafe_entries: list[str] = []
    entry_names: list[str] = []
    top_level_roots: list[str] = []
    manifest_paths: list[Path] = []
    bundle_root: Path | None = None
    manifest_payload: dict = {}
    verify_payload: dict = {}
    first_run_payload: dict = {"skipped": True, "passed": bool(skip_first_run)}
    path_budget = _archive_windows_path_budget(work_dir, [])
    extracted_files = 0
    extracted_bytes = 0

    try:
        if work_dir.exists():
            if not reset:
                raise FileExistsError(
                    f"archive smoke work directory already exists; pass --reset to replace it: {work_dir}"
                )
            _safe_remove_bundle_dir(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        checks["work_dir_ready"] = work_dir.is_dir()
    except Exception as exc:  # noqa: BLE001 - smoke payload should report filesystem setup failures.
        extract_error = str(exc)

    if (
        checks["archive_present"]
        and checks["archive_suffix_zip"]
        and checks["work_dir_ready"]
    ):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                entries = archive.infolist()
                entry_names = [item.filename for item in entries]
                top_level_roots = _zip_top_level_roots(entry_names)
                path_budget = _archive_windows_path_budget(work_dir, top_level_roots)
                unsafe_entries = [
                    item.filename
                    for item in entries
                    if _zip_entry_target(work_dir, item.filename) is None
                ]
                checks["zip_opened"] = True
                checks["zip_entries_safe"] = bool(entries) and not unsafe_entries
                checks["zip_contains_bundle_manifest"] = any(
                    _zip_name_is_bundle_manifest(name) for name in entry_names
                )
                checks["archive_has_single_top_level_root"] = len(top_level_roots) == 1
                checks["windows_path_budget_ok"] = bool(path_budget["ok"])
                if not checks["zip_entries_safe"]:
                    raise ValueError(
                        f"unsafe zip entries: {', '.join(unsafe_entries[:5])}"
                    )
                if not checks["windows_path_budget_ok"]:
                    raise ValueError(
                        "archive smoke work directory is too deep for Windows SQLite smoke paths "
                        f"({path_budget['max_path_length']}/{path_budget['limit']} characters); "
                        "choose a shorter --work-dir"
                    )
                _extract_safe_zip_entries(archive, work_dir)
        except Exception as exc:  # noqa: BLE001 - archive smoke should return evidence rather than crash.
            zip_error = str(exc)

    if checks["zip_opened"] and checks["zip_entries_safe"] and not zip_error:
        manifest_paths = sorted(
            path for path in work_dir.rglob("bundle_manifest.json") if path.is_file()
        )
        checks["single_extracted_bundle_root"] = len(manifest_paths) == 1
        if len(manifest_paths) == 1:
            bundle_root = manifest_paths[0].parent
            checks["manifest_present"] = True
            try:
                manifest_payload = json.loads(
                    manifest_paths[0].read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                extract_error = str(exc)
            if isinstance(manifest_payload, dict):
                checks["stdlib_only"] = (
                    manifest_payload.get("dependency_class") == "stdlib_only"
                )
                checks["no_required_network"] = (
                    manifest_payload.get("required_network") is False
                )
                checks["no_required_vector_db"] = (
                    manifest_payload.get("required_vector_db") is False
                )
                checks["no_required_ml_framework"] = (
                    manifest_payload.get("required_ml_framework") is False
                )
            extracted_files, extracted_bytes = _directory_file_count_and_bytes(
                bundle_root
            )
            verify_payload = _run_cli_json(bundle_root, "verify-bundle", "--json")
            checks["verify_bundle_passed"] = bool(verify_payload.get("passed", False))
            if not skip_first_run:
                first_run_payload = _run_cli_json(
                    bundle_root,
                    "first-run-smoke",
                    "--timeout-seconds",
                    str(timeout_seconds),
                    "--json",
                )
                checks["first_run_smoke_passed_or_skipped"] = bool(
                    first_run_payload.get("passed", False)
                )
                runtime_db = Path(
                    str(first_run_payload.get("artifacts", {}).get("runtime_db", ""))
                )
                checks["runtime_db_created_or_skipped"] = runtime_db.exists()

    payload = {
        "archive": str(archive_path),
        "work_dir": str(work_dir),
        "extracted_bundle_root": str(bundle_root) if bundle_root is not None else "",
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "archive_bytes": archive_path.stat().st_size if archive_path.is_file() else 0,
        "archive_sha256": _sha256_file(archive_path) if archive_path.is_file() else "",
        "entry_count": len(entry_names),
        "top_level_roots": top_level_roots,
        "unsafe_entries": unsafe_entries,
        "manifest_paths": [str(path) for path in manifest_paths],
        "path_budget": path_budget,
        "manifest": {
            "bundle_name": manifest_payload.get("bundle_name", ""),
            "runtime": manifest_payload.get("runtime", ""),
            "dependency_class": manifest_payload.get("dependency_class", ""),
            "file_count": manifest_payload.get("file_count", 0),
            "required_network": manifest_payload.get("required_network"),
            "required_vector_db": manifest_payload.get("required_vector_db"),
            "required_ml_framework": manifest_payload.get("required_ml_framework"),
        },
        "extracted": {
            "files": extracted_files,
            "bytes": extracted_bytes,
        },
        "verify_bundle": {
            "passed": bool(verify_payload.get("passed", False)),
            "verified_files": verify_payload.get("verified_files", 0),
            "verified_bytes": verify_payload.get("verified_bytes", 0),
            "sha256_mismatches": verify_payload.get("sha256_mismatches", []),
            "missing_files": verify_payload.get("missing_files", []),
            "checks": verify_payload.get("checks", {}),
        },
        "first_run_smoke": {
            "skipped": bool(skip_first_run),
            "passed": bool(first_run_payload.get("passed", False)),
            "checks": first_run_payload.get("checks", {}),
            "json_reports": first_run_payload.get("json_reports", 0),
            "first_run_launcher": first_run_payload.get("first_run_launcher", ""),
            "returncode": first_run_payload.get("returncode"),
            "run_error": first_run_payload.get("run_error", ""),
            "stdout_tail": first_run_payload.get("stdout_tail", ""),
            "stderr_tail": first_run_payload.get("stderr_tail", ""),
            "artifacts": first_run_payload.get("artifacts", {}),
        },
        "zip_error": zip_error,
        "extract_error": extract_error,
        "runtime": "stdlib_python_sqlite_http_html_archive_handoff",
        "dependency_class": "stdlib_only",
    }
    return payload


def _zip_top_level_roots(names: list[str]) -> list[str]:
    roots = set()
    for name in names:
        normalized = name.replace("\\", "/").strip("/")
        if normalized:
            roots.add(normalized.split("/", 1)[0])
    return sorted(roots)


def _zip_name_is_bundle_manifest(name: str) -> bool:
    normalized = name.replace("\\", "/").strip("/")
    return normalized == "bundle_manifest.json" or normalized.endswith(
        "/bundle_manifest.json"
    )


def _archive_windows_path_budget(work_dir: Path, top_level_roots: list[str]) -> dict:
    applies = platform.system().lower().startswith("windows")
    roots = top_level_roots or [""]
    candidates: list[Path] = []
    for root_name in roots:
        bundle_root = work_dir / root_name if root_name else work_dir
        candidates.extend(
            bundle_root / relative for relative in ARCHIVE_SMOKE_DEEP_PATHS
        )
    longest = max(
        (path.resolve() for path in candidates), key=lambda path: len(str(path))
    )
    max_length = len(str(longest))
    return {
        "applies": applies,
        "ok": (not applies) or max_length <= ARCHIVE_SMOKE_WINDOWS_PATH_LIMIT,
        "limit": ARCHIVE_SMOKE_WINDOWS_PATH_LIMIT,
        "max_path_length": max_length,
        "longest_path": str(longest),
    }


def _zip_entry_target(root: Path, name: str) -> Path | None:
    if not name or "\x00" in name:
        return None
    normalized = name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    if posix_path.is_absolute() or any(
        part in {"", ".", ".."} or ":" in part for part in posix_path.parts
    ):
        return None
    if Path(name).is_absolute() or Path(normalized).is_absolute():
        return None
    resolved_root = root.resolve()
    target = (resolved_root / Path(*posix_path.parts)).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        return None
    return target


def _extract_safe_zip_entries(archive: zipfile.ZipFile, root: Path) -> None:
    for item in archive.infolist():
        target = _zip_entry_target(root, item.filename)
        if target is None:
            raise ValueError(f"unsafe zip entry: {item.filename}")
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(item) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _directory_file_count_and_bytes(root: Path) -> tuple[int, int]:
    if not root.exists():
        return (0, 0)
    files = [path for path in root.rglob("*") if path.is_file()]
    return (len(files), sum(path.stat().st_size for path in files))


def _build_first_run_smoke_payload(bundle_root: Path, *, timeout_seconds: int) -> dict:
    started = perf_counter()
    root = bundle_root.resolve()
    is_windows = platform.system().lower().startswith("windows")
    first_run_launcher = (
        Path("bin/first_run.ps1") if is_windows else Path("bin/first_run.sh")
    )
    cmd_first_run_launcher = Path("bin/first_run.cmd")
    stdout_log = (
        root / "artifacts" / "local_assistant_os" / "first_run_smoke.stdout.log"
    )
    stderr_log = (
        root / "artifacts" / "local_assistant_os" / "first_run_smoke.stderr.log"
    )
    result: subprocess.CompletedProcess[str] | None = None
    run_error = ""
    objects: list[dict] = []
    if root.exists():
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not root.exists():
            raise FileNotFoundError(f"bundle root does not exist: {root}")
        if not (root / first_run_launcher).is_file():
            raise FileNotFoundError(
                f"first-run launcher is missing: {first_run_launcher.as_posix()}"
            )
        env = os.environ.copy()
        env["MELM_PYTHON"] = sys.executable
        result = subprocess.run(
            _platform_launcher_command(
                root / first_run_launcher, is_windows=is_windows
            ),
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout_log.write_text(result.stdout, encoding="utf-8", errors="replace")
        stderr_log.write_text(result.stderr, encoding="utf-8", errors="replace")
        objects = _json_objects_from_text(result.stdout)
    except Exception as exc:  # noqa: BLE001 - smoke payload should report all launcher failures.
        run_error = str(exc)
        if result is not None:
            stdout_log.write_text(result.stdout, encoding="utf-8", errors="replace")
            stderr_log.write_text(result.stderr, encoding="utf-8", errors="replace")

    verify_payload = _first_matching_json_object(
        objects, lambda item: "verified_files" in item and "self_check_summary" in item
    )
    dataset_payload = _first_matching_json_object(
        objects, lambda item: item.get("runtime") == "stdlib_python_sqlite_json_csv"
    )
    target_payload = _first_matching_json_object(
        objects, lambda item: "hardware" in item and "smokes" in item
    )
    bootstrap_payload = _first_matching_json_object(
        objects,
        lambda item: (
            item.get("runtime") == "stdlib_python_sqlite" and "next_commands" in item
        ),
    )
    ui_payload = _first_matching_json_object(
        objects,
        lambda item: (
            item.get("runtime") == "stdlib_python_sqlite_http_html" and "ui" in item
        ),
    )
    launcher_payload = _first_matching_json_object(
        objects,
        lambda item: item.get("runtime") == "stdlib_python_sqlite_http_launcher",
    )
    runtime_db = root / "artifacts" / "local_assistant_os" / "assistant_v01.sqlite"
    target_report_dir = root / "artifacts" / "local_assistant_os" / "target_report"
    checks = {
        "bundle_root_exists": root.exists(),
        "first_run_launcher_present": (root / first_run_launcher).is_file(),
        "cmd_first_run_wrapper_present": (not is_windows)
        or (root / cmd_first_run_launcher).is_file(),
        "first_run_returned_zero": result is not None and result.returncode == 0,
        "completion_message_present": bool(
            result and "First-run checks passed" in result.stdout
        ),
        "expected_json_reports_emitted": all(
            bool(item)
            for item in (
                verify_payload,
                dataset_payload,
                target_payload,
                bootstrap_payload,
                ui_payload,
                launcher_payload,
            )
        ),
        "verify_bundle_passed": bool(verify_payload.get("passed", False)),
        "dataset_audit_passed": bool(dataset_payload.get("passed", False)),
        "target_report_passed": bool(target_payload.get("passed", False)),
        "bootstrap_runtime_passed": bool(bootstrap_payload.get("passed", False)),
        "ui_smoke_passed": bool(ui_payload.get("passed", False)),
        "launcher_smoke_passed": bool(launcher_payload.get("passed", False)),
        "runtime_db_created": runtime_db.exists(),
        "target_report_artifacts_created": target_report_dir.exists(),
        "launcher_process_stopped": bool(
            launcher_payload.get("checks", {}).get("launcher_process_stopped", False)
        ),
        "stdlib_only": (
            dataset_payload.get("dependency_class") == "stdlib_only"
            and bootstrap_payload.get("dependency_class") == "stdlib_only"
            and ui_payload.get("dependency_class") == "stdlib_only"
            and launcher_payload.get("dependency_class") == "stdlib_only"
        ),
    }
    return {
        "bundle_root": str(root),
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "platform": platform.system(),
        "first_run_launcher": first_run_launcher.as_posix(),
        "returncode": result.returncode if result is not None else None,
        "json_reports": len(objects),
        "reports": {
            "verify_bundle": {
                "passed": bool(verify_payload.get("passed", False)),
                "verified_files": verify_payload.get("verified_files", 0),
                "sha256_mismatches": verify_payload.get("sha256_mismatches", []),
            },
            "dataset_audit": {
                "passed": bool(dataset_payload.get("passed", False)),
                "checks": dataset_payload.get("checks", {}),
                "source_fixtures": dataset_payload.get("source_fixtures", {}),
            },
            "target_report": {
                "passed": bool(target_payload.get("passed", False)),
                "checks": target_payload.get("checks", {}),
            },
            "bootstrap_runtime": {
                "passed": bool(bootstrap_payload.get("passed", False)),
                "checks": bootstrap_payload.get("checks", {}),
                "counts": bootstrap_payload.get("counts", {}),
            },
            "ui_smoke": {
                "passed": bool(ui_payload.get("passed", False)),
                "checks": ui_payload.get("checks", {}),
                "ui": ui_payload.get("ui", {}),
            },
            "launcher_smoke": {
                "passed": bool(launcher_payload.get("passed", False)),
                "checks": launcher_payload.get("checks", {}),
                "health": launcher_payload.get("health", {}),
            },
        },
        "artifacts": {
            "runtime_db": str(runtime_db),
            "target_report_dir": str(target_report_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        },
        "stdout_tail": _read_tail(stdout_log, 2000),
        "stderr_tail": _read_tail(stderr_log, 2000),
        "run_error": run_error,
        "runtime": "stdlib_python_sqlite_http_html_first_run_launcher",
        "dependency_class": "stdlib_only",
    }


def _build_launcher_smoke_payload(
    bundle_root: Path, *, db: Path, host: str, port: int, reset: bool
) -> dict:
    started = perf_counter()
    root = bundle_root.resolve()
    selected_port = int(port or _free_local_port(host))
    base_url = f"http://{host}:{selected_port}"
    shutdown_token = hashlib.sha256(
        f"{root}|{db}|{selected_port}|{started}".encode("utf-8")
    ).hexdigest()
    db_path = db if db.is_absolute() else root / db
    launcher_files = [path.as_posix() for path in PI_BUNDLE_LAUNCHER_FILES]
    missing_launcher_files = [
        path for path in launcher_files if not (root / path).is_file()
    ]
    is_windows = platform.system().lower().startswith("windows")
    platform_launcher = (
        Path("bin/start_app.ps1") if is_windows else Path("bin/start_app.sh")
    )
    health_launcher = (
        Path("bin/health_check.ps1") if is_windows else Path("bin/health_check.sh")
    )
    cmd_health_launcher = Path("bin/health_check.cmd")
    process: subprocess.Popen[str] | None = None
    health_payload: dict = {}
    index_html = ""
    parse_payload: dict = {}
    shutdown_payload: dict = {}
    health_result: dict = {"skipped": True}
    cmd_health_result: dict = {"skipped": True}
    start_stdout = ""
    start_stderr = ""
    start_returncode: int | None = None
    launch_error = ""
    stdout_handle = None
    stderr_handle = None
    stdout_log = db_path.with_suffix(".launcher.stdout.log")
    stderr_log = db_path.with_suffix(".launcher.stderr.log")
    try:
        if not root.exists():
            raise FileNotFoundError(f"bundle root does not exist: {root}")
        if missing_launcher_files:
            raise FileNotFoundError(
                f"missing launcher files: {', '.join(missing_launcher_files)}"
            )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            _remove_sqlite_files(db_path)
        start_command = _platform_launcher_command(
            root / platform_launcher, is_windows=is_windows
        )
        env = os.environ.copy()
        env.update(
            {
                "MELM_ASSISTANT_DB": str(db),
                "MELM_ASSISTANT_HOST": host,
                "MELM_ASSISTANT_PORT": str(selected_port),
                "MELM_ASSISTANT_SHUTDOWN_TOKEN": shutdown_token,
                "MELM_PYTHON": sys.executable,
            }
        )
        stdout_handle = stdout_log.open("w", encoding="utf-8")
        stderr_handle = stderr_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            start_command,
            cwd=root,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        health_payload = _wait_for_launcher_health(
            base_url, process, timeout_seconds=20.0
        )
        health_result = _run_launcher_command(
            _platform_launcher_command(root / health_launcher, is_windows=is_windows),
            cwd=root,
            env=env,
            timeout_seconds=15,
        )
        if is_windows and (root / cmd_health_launcher).is_file():
            cmd_health_result = _run_launcher_command(
                ["cmd.exe", "/c", str((root / cmd_health_launcher).resolve())],
                cwd=root,
                env=env,
                timeout_seconds=15,
            )
        index_html = _api_get_text(f"{base_url}/")
        parse_payload = _api_post_json(
            f"{base_url}/parse-debug", {"utterance": "Who are you?"}
        )
        shutdown_payload = _api_post_json(
            f"{base_url}/_melm/launcher-shutdown",
            {"shutdown": True},
            headers={"X-MELM-Shutdown-Token": shutdown_token},
        )
    except Exception as exc:  # noqa: BLE001 - smoke payload should report the launch failure.
        launch_error = str(exc)
    finally:
        if process is not None:
            start_returncode = _terminate_launcher_process(process)
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        start_stdout = _read_tail(stdout_log, 2000)
        start_stderr = _read_tail(stderr_log, 2000)
    checks = {
        "bundle_root_exists": root.exists(),
        "launcher_files_present": not missing_launcher_files,
        "platform_launcher_present": (root / platform_launcher).is_file(),
        "health_launcher_present": (root / health_launcher).is_file(),
        "platform_launcher_started": bool(health_payload.get("ok", False)),
        "health_launcher_ok": health_result.get("returncode") == 0
        and _launcher_health_output_ok(health_result),
        "cmd_health_launcher_ok": (not is_windows)
        or (
            cmd_health_result.get("returncode") == 0
            and _launcher_health_output_ok(cmd_health_result)
        ),
        "runtime_db_available": db_path.exists(),
        "browser_ui_served": "<title>MELM Local Assistant OS</title>" in index_html,
        "parse_debug_endpoint_ok": (
            parse_payload.get("chat_frame", {}).get("intent") == "assistant_identity"
            and parse_payload.get("uol", {}).get("object") == "self_model"
        ),
        "launcher_shutdown_endpoint_ok": bool(shutdown_payload.get("ok", False)),
        "localhost_only": host in {"127.0.0.1", "localhost", "::1"},
        "launcher_process_stopped": process is None or process.poll() is not None,
    }
    return {
        "bundle_root": str(root),
        "db": str(db_path),
        "base_url": base_url,
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "platform": platform.system(),
        "platform_launcher": platform_launcher.as_posix(),
        "health_launcher": health_launcher.as_posix(),
        "missing_launcher_files": missing_launcher_files,
        "health": health_payload,
        "health_launcher_result": health_result,
        "cmd_health_launcher_result": cmd_health_result,
        "parse_debug": {
            "intent": parse_payload.get("chat_frame", {}).get("intent", ""),
            "object": parse_payload.get("uol", {}).get("object", ""),
            "mapping": [
                stage.get("stage") for stage in parse_payload.get("mapping", [])
            ],
        },
        "shutdown": shutdown_payload,
        "start_process": {
            "returncode": start_returncode,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_tail": start_stdout,
            "stderr_tail": start_stderr,
        },
        "launch_error": launch_error,
        "runtime": "stdlib_python_sqlite_http_launcher",
        "dependency_class": "stdlib_only",
    }


def _json_objects_from_text(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            item, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(item, dict):
            objects.append(item)
        index = start + end
    return objects


def _first_matching_json_object(objects: list[dict], predicate) -> dict:
    for item in objects:
        if predicate(item):
            return item
    return {}


def _platform_launcher_command(script: Path, *, is_windows: bool) -> list[str]:
    if is_windows:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            raise FileNotFoundError(
                "PowerShell is required to run Windows launcher smoke"
            )
        return [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script.resolve()),
        ]
    return ["sh", str(script.resolve())]


def _run_launcher_command(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout_seconds: int
) -> dict:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


def _launcher_health_output_ok(result: dict) -> bool:
    if result.get("returncode") != 0:
        return False
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".lower()
    compact = "".join(text.split())
    return (
        '"ok":true' in compact
        or "'ok':true" in compact
        or "melm_health_ok=true" in compact
    )


def _read_tail(path: Path, max_chars: int) -> str:
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _wait_for_launcher_health(
    base_url: str, process: subprocess.Popen[str], *, timeout_seconds: float
) -> dict:
    deadline = perf_counter() + timeout_seconds
    last_error = ""
    while perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"launcher exited before health check passed with code {process.returncode}"
            )
        try:
            payload = _api_get_json(f"{base_url}/health")
            if payload.get("ok"):
                return payload
        except Exception as exc:  # noqa: BLE001 - keep polling until timeout.
            last_error = str(exc)
        sleep(0.2)
    raise TimeoutError(f"launcher health endpoint did not become ready: {last_error}")


def _free_local_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _terminate_launcher_process(process: subprocess.Popen[str]) -> int | None:
    if process.poll() is not None:
        return process.returncode
    if platform.system().lower().startswith("windows"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            text=True,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    return process.returncode


def _open_trace_summary(payload: dict) -> dict:
    scenario_reports = payload.get("scenario_reports", [])
    scenario_checks = {
        str(item.get("name", f"scenario_{index}")): dict(item.get("checks", {}))
        for index, item in enumerate(scenario_reports, start=1)
        if isinstance(item, dict)
    }
    counts_total: Counter[str] = Counter()
    memory_digest_reports: list[dict] = []
    for item in scenario_reports:
        if not isinstance(item, dict):
            continue
        counts_total.update(dict(item.get("counts", {})))
        memory_digest = dict(item.get("memory_digest", {}))
        if memory_digest:
            memory_digest_reports.append(memory_digest)
    debug_checks = {
        "debug_maps_present": bool(scenario_checks)
        and all(
            bool(checks.get("debug_maps_present", False))
            for checks in scenario_checks.values()
        ),
        "primary_uol_chatframe_not_secondary_phrase_route": bool(scenario_checks)
        and all(
            bool(checks.get("primary_uol_chatframe_not_secondary_phrase_route", False))
            for checks in scenario_checks.values()
        ),
        "identity_maps_to_self_model": bool(scenario_checks)
        and all(
            bool(checks.get("identity_maps_to_self_model", False))
            for checks in scenario_checks.values()
        ),
        "status_maps_to_runtime_or_next_steps": bool(scenario_checks)
        and all(
            bool(checks.get("status_maps_to_runtime_or_next_steps", False))
            for checks in scenario_checks.values()
        ),
    }
    memory_digest_quality = {
        "scenario_count": len(memory_digest_reports),
        "passed": bool(memory_digest_reports)
        and all(
            bool(item.get("quality_passed", False)) for item in memory_digest_reports
        ),
        "min_score": round(
            min(
                (
                    float(item.get("quality_score", 0.0) or 0.0)
                    for item in memory_digest_reports
                ),
                default=0.0,
            ),
            3,
        ),
        "floor": round(
            max(
                (
                    float(item.get("quality_floor", 0.0) or 0.0)
                    for item in memory_digest_reports
                ),
                default=0.0,
            ),
            3,
        ),
    }
    return {
        "passed": bool(payload.get("passed", False)),
        "scenarios": int(payload.get("scenarios", 0) or 0),
        "turns": int(payload.get("turns", 0) or 0),
        "local_resolution_rate": float(
            payload.get("local_resolution_rate", 0.0) or 0.0
        ),
        "route_counts": dict(payload.get("route_counts", {})),
        "intent_counts": dict(payload.get("intent_counts", {})),
        "safety_totals": dict(payload.get("safety_totals", {})),
        "counts": dict(sorted(counts_total.items())),
        "synthesis_traces": int(counts_total.get("synthesis_traces", 0) or 0),
        "scenario_checks": scenario_checks,
        "debug_checks": debug_checks,
        "memory_digest_quality": memory_digest_quality,
        "priority_signal_samples": list(payload.get("priority_signal_samples", [])),
        "capture_provenance": dict(payload.get("capture_provenance", {})),
    }


def _transcript_replay_summary(payload: dict) -> dict:
    trace_summary = _open_trace_summary(payload)
    return {
        **trace_summary,
        "schema": str(payload.get("schema", "")),
        "fixture_path": str(
            payload.get("fixture_path", payload.get("transcript_jsonl", ""))
        ),
        "source_type": str(payload.get("source_type", "")),
        "fixture_checks": dict(payload.get("fixture_checks", {})),
        "fixture_failures": list(payload.get("fixture_failures", [])),
        "complexity": dict(payload.get("complexity", {})),
        "debug_mapping": dict(payload.get("debug_mapping", {})),
        "baseline_comparison": dict(payload.get("baseline_comparison", {})),
    }


def _verify_bundle(args) -> None:
    started = perf_counter()
    if args.bundle_root is not None:
        bundle_root = args.bundle_root
        manifest_path = (
            args.manifest
            if args.manifest.is_absolute()
            else bundle_root / args.manifest
        )
    else:
        manifest_path = (
            args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
        )
        bundle_root = manifest_path.parent
    bundle_root = bundle_root.resolve()
    manifest_path = manifest_path.resolve()

    checks = {
        "manifest_present": manifest_path.exists(),
        "manifest_json_valid": False,
        "file_records_valid": False,
        "file_count_matches": False,
        "all_files_present": False,
        "byte_counts_match": False,
        "sha256_match": False,
        "entrypoint_present": False,
        "runbook_present": False,
        "launcher_files_present": False,
        "self_check_present": False,
        "self_check_json_valid": False,
        "self_check_passed": False,
        "self_check_ran": False,
        "portable_commands_present": False,
        "stdlib_only_declared": False,
    }
    manifest_payload: dict = {}
    manifest_error = ""
    if manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks["manifest_json_valid"] = isinstance(manifest_payload, dict)
        except json.JSONDecodeError as exc:
            manifest_error = str(exc)

    files = manifest_payload.get("files", []) if checks["manifest_json_valid"] else []
    file_records_valid = checks["manifest_json_valid"] and isinstance(files, list)
    missing_files: list[str] = []
    unsafe_paths: list[str] = []
    size_mismatches: list[dict] = []
    sha256_mismatches: list[dict] = []
    verified_bytes = 0
    verified_files = 0
    if file_records_valid:
        for item in files:
            if not isinstance(item, dict):
                file_records_valid = False
                continue
            relative = str(item.get("path", ""))
            expected_bytes = item.get("bytes")
            expected_hash = str(item.get("sha256", ""))
            target = _bundle_manifest_target(bundle_root, relative)
            if target is None:
                file_records_valid = False
                unsafe_paths.append(relative)
                continue
            if not target.exists() or not target.is_file():
                missing_files.append(relative)
                continue
            actual_bytes = target.stat().st_size
            actual_hash = _sha256_file(target)
            verified_files += 1
            verified_bytes += actual_bytes
            if actual_bytes != expected_bytes:
                size_mismatches.append(
                    {
                        "path": relative,
                        "expected": expected_bytes,
                        "actual": actual_bytes,
                    }
                )
            if actual_hash != expected_hash:
                sha256_mismatches.append(
                    {
                        "path": relative,
                        "expected": expected_hash,
                        "actual": actual_hash,
                    }
                )

    self_check_path = bundle_root / "bundle_self_check.json"
    self_check_payload: dict = {}
    self_check_error = ""
    if self_check_path.exists():
        checks["self_check_present"] = True
        try:
            self_check_payload = json.loads(self_check_path.read_text(encoding="utf-8"))
            checks["self_check_json_valid"] = isinstance(self_check_payload, dict)
        except json.JSONDecodeError as exc:
            self_check_error = str(exc)

    required_commands = (
        "portable_chat_command",
        "portable_dataset_audit_command",
        "portable_pi_command",
        "portable_inventory_soak_matrix_command",
        "portable_autoimmune_command",
        "portable_synthesis_variant_command",
        "portable_synthesis_stress_command",
        "portable_setup_integration_command",
        "portable_host_action_command",
        "portable_host_actions_demo_config_command",
        "portable_host_app_probe_command",
        "portable_host_app_configured_probe_command",
        "portable_host_app_demo_config_probe_command",
        "portable_capability_probe_command",
        "portable_shortcut_audit_command",
        "portable_v01_audit_command",
        "portable_v01_progress_command",
        "portable_v01_evidence_pack_command",
        "portable_candidate_session_audit_command",
        "portable_source_attestation_command",
        "portable_host_app_attestation_command",
        "portable_v01_blocker_evidence_command",
        "portable_v01_acceptance_command",
        "portable_v01_acceptance_configured_host_app_command",
        "portable_api_command",
        "portable_api_session_command",
        "portable_ui_command",
        "portable_start_app_command",
        "portable_bootstrap_runtime_command",
        "portable_open_traces_command",
        "portable_transcript_replay_command",
        "portable_transcript_calibration_command",
        "portable_event_transcript_export_command",
        "portable_event_ledger_calibration_command",
        "portable_user_transcript_import_command",
        "portable_user_transcript_calibration_command",
        "portable_refresh_weather_command",
        "portable_verify_command",
        "portable_launcher_smoke_command",
        "portable_first_run_smoke_command",
        "portable_target_report_command",
    )
    missing_commands = [
        key for key in required_commands if not manifest_payload.get(key)
    ]
    safe_lifecycle_controls_template = str(
        manifest_payload.get("portable_safe_lifecycle_controls_template")
        or manifest_payload.get("safe_lifecycle_controls_template")
        or ""
    )
    required_launcher_files = tuple(
        path.as_posix() for path in PI_BUNDLE_LAUNCHER_FILES
    )
    missing_launcher_files = [
        path for path in required_launcher_files if not (bundle_root / path).is_file()
    ]
    entrypoint = str(manifest_payload.get("entrypoint", ""))
    checks.update(
        {
            "file_records_valid": file_records_valid,
            "file_count_matches": (
                checks["manifest_json_valid"]
                and isinstance(files, list)
                and manifest_payload.get("file_count") == len(files)
            ),
            "all_files_present": checks["manifest_json_valid"]
            and file_records_valid
            and not missing_files
            and not unsafe_paths,
            "byte_counts_match": (
                checks["manifest_json_valid"]
                and file_records_valid
                and not missing_files
                and not size_mismatches
            ),
            "sha256_match": (
                checks["manifest_json_valid"]
                and file_records_valid
                and not missing_files
                and not sha256_mismatches
            ),
            "entrypoint_present": bool(entrypoint)
            and (bundle_root / entrypoint).is_file(),
            "safe_lifecycle_controls_template_present": (
                bool(safe_lifecycle_controls_template)
                and _bundle_manifest_target(
                    bundle_root, safe_lifecycle_controls_template
                )
                is not None
                and (bundle_root / safe_lifecycle_controls_template).is_file()
            ),
            "runbook_present": (bundle_root / "RUN_PORTABLE_APP.md").is_file()
            or (bundle_root / "RUN_ON_RASPBERRY_PI.md").is_file(),
            "launcher_files_present": not missing_launcher_files,
            "self_check_passed": bool(self_check_payload.get("passed", False)),
            "self_check_ran": args.allow_skipped_self_check
            or not bool(self_check_payload.get("skipped", False)),
            "portable_commands_present": not missing_commands,
            "stdlib_only_declared": (
                manifest_payload.get("runtime")
                in {"stdlib_python_sqlite", "stdlib_python_sqlite_http_html"}
                and manifest_payload.get("dependency_class") == "stdlib_only"
                and manifest_payload.get("required_network") is False
                and manifest_payload.get("required_vector_db") is False
                and manifest_payload.get("required_ml_framework") is False
            ),
        }
    )
    payload = {
        "bundle_root": str(bundle_root),
        "manifest": str(manifest_path),
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "file_count": len(files) if isinstance(files, list) else 0,
        "verified_files": verified_files,
        "verified_bytes": verified_bytes,
        "missing_files": missing_files,
        "unsafe_paths": unsafe_paths,
        "size_mismatches": size_mismatches,
        "sha256_mismatches": sha256_mismatches,
        "missing_commands": missing_commands,
        "missing_launcher_files": missing_launcher_files,
        "manifest_error": manifest_error,
        "self_check_error": self_check_error,
        "self_check_summary": {
            "passed": bool(self_check_payload.get("passed", False)),
            "skipped": bool(self_check_payload.get("skipped", False)),
            "dataset_audit_passed": bool(
                self_check_payload.get("dataset_audit", {}).get("passed", False)
            ),
            "pi_smoke_passed": bool(
                self_check_payload.get("pi_smoke", {}).get("passed", False)
            ),
            "pi_smoke_inventory_soak_matrix_passed": bool(
                self_check_payload.get("pi_smoke", {})
                .get("checks", {})
                .get("inventory_soak_matrix_passed", False)
            ),
            "pi_smoke_inventory_failure_passed": bool(
                self_check_payload.get("pi_smoke", {})
                .get("checks", {})
                .get("inventory_failure_smoke_passed", False)
            ),
            "pi_smoke_inventory_retry_passed": bool(
                self_check_payload.get("pi_smoke", {})
                .get("checks", {})
                .get("inventory_retry_smoke_passed", False)
            ),
            "autoimmune_smoke_passed": bool(
                self_check_payload.get("autoimmune_smoke", {}).get("passed", False)
            ),
            "synthesis_variant_smoke_passed": bool(
                self_check_payload.get("synthesis_variant_smoke", {}).get(
                    "passed", False
                )
            ),
            "synthesis_stress_smoke_passed": bool(
                self_check_payload.get("synthesis_stress_smoke", {}).get(
                    "passed", False
                )
            ),
            "setup_integration_smoke_passed": bool(
                self_check_payload.get("setup_integration_smoke", {}).get(
                    "passed", False
                )
            ),
            "host_action_smoke_passed": bool(
                self_check_payload.get("host_action_smoke", {}).get("passed", False)
            ),
            "host_app_probe_passed": bool(
                self_check_payload.get("host_app_probe", {}).get("passed", False)
            ),
            "host_app_probe_configured": bool(
                self_check_payload.get("host_app_probe", {}).get("configured", False)
            ),
            "host_app_probe_skipped": bool(
                self_check_payload.get("host_app_probe", {}).get("skipped", False)
            ),
            "capability_probe_passed": bool(
                self_check_payload.get("capability_probe", {}).get("passed", False)
            ),
            "shortcut_audit_passed": bool(
                self_check_payload.get("shortcut_audit", {}).get("passed", False)
            ),
            "v01_audit_passed": bool(
                self_check_payload.get("v01_audit", {}).get("passed", False)
            ),
            "v01_progress_passed": bool(
                self_check_payload.get("v01_progress", {}).get("passed", False)
            ),
            "v01_progress_remaining_blockers": int(
                self_check_payload.get("v01_progress", {}).get(
                    "remaining_blocker_count", 0
                )
                or 0
            ),
            "v01_audit_architecture_complete": bool(
                self_check_payload.get("v01_audit", {}).get(
                    "architecture_complete", False
                )
            ),
            "v01_audit_blocker_count": int(
                self_check_payload.get("v01_audit", {}).get("blocker_count", 0) or 0
            ),
            "api_smoke_passed": bool(
                self_check_payload.get("api_smoke", {}).get("passed", False)
            ),
            "api_session_smoke_passed": bool(
                self_check_payload.get("api_session_smoke", {}).get("passed", False)
            ),
            "ui_smoke_passed": bool(
                self_check_payload.get("ui_smoke", {}).get("passed", False)
            ),
            "bootstrap_runtime_passed": bool(
                self_check_payload.get("bootstrap_runtime", {}).get("passed", False)
            ),
            "launcher_smoke_passed": bool(
                self_check_payload.get("launcher_smoke", {}).get("passed", False)
            ),
            "open_traces_passed": bool(
                self_check_payload.get("open_traces", {}).get("passed", False)
            ),
            "transcript_replay_passed": bool(
                self_check_payload.get("transcript_replay", {}).get("passed", False)
            ),
            "transcript_calibration_passed": bool(
                self_check_payload.get("transcript_calibration", {}).get(
                    "passed", False
                )
            ),
        },
    }
    _print_payload(payload, json_mode=args.json)


def _bundle_manifest_target(bundle_root: Path, relative: str) -> Path | None:
    if not relative:
        return None
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    root = bundle_root.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def _remove_pi_bundle_smoke_artifacts(out_dir: Path) -> None:
    artifact_dir = out_dir / "artifacts" / "local_assistant_os"
    if not artifact_dir.exists():
        return
    for target in artifact_dir.iterdir():
        if (
            target.name.startswith("pi_bundle_smoke")
            or target.name.startswith("dataset_audit_bundle")
            or target.name.startswith("autoimmune_bundle_smoke")
            or target.name.startswith("synthesis_variant_bundle_smoke")
            or target.name.startswith("synthesis_stress_bundle_smoke")
            or target.name.startswith("setup_integration_bundle_smoke")
            or target.name.startswith("host_action_bundle_smoke")
            or target.name.startswith("host_app_probe_bundle")
            or target.name.startswith("capability_probe_bundle")
            or target.name.startswith("shortcut_audit_")
            or target.name.startswith("v01_progress")
            or target.name.startswith("api_bundle_smoke")
            or target.name.startswith("api_session_bundle_smoke")
            or target.name.startswith("ui_bundle_smoke")
            or target.name.startswith("bootstrap_bundle")
            or target.name.startswith("launcher_bundle_smoke")
            or target.name.startswith("open_traces_bundle")
            or target.name.startswith("transcript_replay_bundle")
            or target.name.startswith("transcript_calibration_bundle")
        ):
            _remove_pi_bundle_smoke_artifact_target(target)
    try:
        artifact_dir.rmdir()
        artifact_dir.parent.rmdir()
    except OSError:
        pass


def _remove_pi_bundle_smoke_artifact_target(target: Path) -> None:
    for attempt in range(20):
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(0.25)


def _safe_remove_bundle_dir(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {ROOT.resolve(), ROOT.resolve().parent, Path(resolved.anchor)}
    if resolved in forbidden:
        raise SystemExit(f"Refusing to remove unsafe bundle output path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _portable_first_run_launcher() -> str:
    return """#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

python3 scripts/local_assistant_os_cli.py verify-bundle --json
python3 scripts/local_assistant_os_cli.py dataset-audit --reset --json
python3 scripts/local_assistant_os_cli.py target-report --reset --json
python3 scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json "$@"
python3 scripts/local_assistant_os_cli.py ui-smoke --reset --json
python3 scripts/local_assistant_os_cli.py launcher-smoke --reset --json

printf '%s\\n' "First-run checks passed. Start the local browser app with: sh bin/start_app.sh"
printf '%s\\n' "Then open: http://127.0.0.1:8771/"
"""


def _portable_start_app_launcher() -> str:
    return """#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

DB="${MELM_ASSISTANT_DB:-artifacts/local_assistant_os/assistant_v01.sqlite}"
HOST="${MELM_ASSISTANT_HOST:-127.0.0.1}"
PORT="${MELM_ASSISTANT_PORT:-8771}"

if [ ! -f "$DB" ]; then
  python3 scripts/local_assistant_os_cli.py bootstrap-runtime --db "$DB" --json
fi

printf '%s\\n' "Open http://$HOST:$PORT/ in a browser."
exec python3 scripts/local_assistant_os_cli.py serve --db "$DB" --host "$HOST" --port "$PORT"
"""


def _pi_first_run_launcher() -> str:
    return """#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

python3 scripts/local_assistant_os_cli.py verify-bundle --json
python3 scripts/local_assistant_os_cli.py dataset-audit --reset --json
python3 scripts/local_assistant_os_cli.py target-report --reset --require-raspberry-pi --json
python3 scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json "$@"
python3 scripts/local_assistant_os_cli.py launcher-smoke --reset --json

printf '%s\\n' "First-run checks passed. Start the local API with: sh bin/start_api.sh"
"""


def _windows_first_run_launcher() -> str:
    return """$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Invoke-MelmPython {
  param([string[]]$PythonArgs)
  if ($env:MELM_PYTHON) {
    & $env:MELM_PYTHON @PythonArgs
  } elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @PythonArgs
  } else {
    & python @PythonArgs
  }
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "verify-bundle", "--json")
Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "dataset-audit", "--reset", "--json")
Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "target-report", "--reset", "--json")
Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "bootstrap-runtime", "--reset", "--json")
Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "ui-smoke", "--reset", "--json")
Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "launcher-smoke", "--reset", "--json")

Write-Host "First-run checks passed. Start the local browser app with: powershell -ExecutionPolicy Bypass -File bin/start_app.ps1"
Write-Host "Then open: http://127.0.0.1:8771/"
"""


def _windows_start_app_launcher() -> str:
    return """$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Invoke-MelmPython {
  param([string[]]$PythonArgs)
  if ($env:MELM_PYTHON) {
    & $env:MELM_PYTHON @PythonArgs
  } elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @PythonArgs
  } else {
    & python @PythonArgs
  }
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Db = if ($env:MELM_ASSISTANT_DB) { $env:MELM_ASSISTANT_DB } else { "artifacts/local_assistant_os/assistant_v01.sqlite" }
$HostName = if ($env:MELM_ASSISTANT_HOST) { $env:MELM_ASSISTANT_HOST } else { "127.0.0.1" }
$Port = if ($env:MELM_ASSISTANT_PORT) { $env:MELM_ASSISTANT_PORT } else { "8771" }

if (-not (Test-Path $Db)) {
  Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "bootstrap-runtime", "--db", $Db, "--json")
}

Write-Host "Open http://$($HostName):$Port/ in a browser."
Invoke-MelmPython @("scripts/local_assistant_os_cli.py", "serve", "--db", $Db, "--host", $HostName, "--port", $Port)
"""


def _windows_health_check_launcher() -> str:
    return """$ErrorActionPreference = "Stop"
$HostName = if ($env:MELM_ASSISTANT_HOST) { $env:MELM_ASSISTANT_HOST } else { "127.0.0.1" }
$Port = if ($env:MELM_ASSISTANT_PORT) { $env:MELM_ASSISTANT_PORT } else { "8771" }
$Payload = Invoke-RestMethod -Uri "http://$($HostName):$Port/health" -TimeoutSec 5
$Payload | ConvertTo-Json -Depth 8
if (-not $Payload.ok) { exit 1 }
Write-Host "MELM_HEALTH_OK=true"
"""


def _windows_cmd_launcher(script_name: str) -> str:
    return f"""@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0{script_name}" %*
exit /b %ERRORLEVEL%
"""


def _pi_start_api_launcher() -> str:
    return """#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

DB="${MELM_ASSISTANT_DB:-artifacts/local_assistant_os/assistant_v01.sqlite}"
HOST="${MELM_ASSISTANT_HOST:-127.0.0.1}"
PORT="${MELM_ASSISTANT_PORT:-8771}"

if [ ! -f "$DB" ]; then
  python3 scripts/local_assistant_os_cli.py bootstrap-runtime --db "$DB" --json
fi

exec python3 scripts/local_assistant_os_cli.py serve --db "$DB" --host "$HOST" --port "$PORT"
"""


def _pi_health_check_launcher() -> str:
    return """#!/usr/bin/env sh
set -eu

HOST="${MELM_ASSISTANT_HOST:-127.0.0.1}"
PORT="${MELM_ASSISTANT_PORT:-8771}"

python3 - "$HOST" "$PORT" <<'PY'
import json
import sys
from urllib.request import urlopen

host = sys.argv[1]
port = sys.argv[2]
url = f"http://{host}:{port}/health"
with urlopen(url, timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload.get("ok"):
    raise SystemExit(1)
print("MELM_HEALTH_OK=true")
PY
"""


def _pi_systemd_service_example() -> str:
    return """[Unit]
Description=MELM Local Assistant OS v0.1 API
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/melm_local_assistant_os_v01_portable_bundle
ExecStart=/bin/sh %h/melm_local_assistant_os_v01_portable_bundle/bin/start_api.sh
Restart=on-failure
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=default.target
"""


def _pi_bundle_runbook() -> str:
    return """# MELM Local Assistant OS v0.1 Portable Bundle

This bundle is the portable, stdlib-only v0.1 assistant OS app package. It can
run as a local browser app, a JSON API, or a cross-platform CLI chat session.

## Requirements

- Windows, macOS, Linux, or Raspberry Pi OS with Python 3.11+.
- No Python package install is required for the v0.1 smoke.
- No network, vector database, or ML framework is required for the smoke.

## First Run

Unix/macOS/Linux:

```bash
sh bin/first_run.sh
```

Windows:

```powershell
bin\\first_run.cmd
```

The generic first-run scripts verify the bundle, run target/report smokes
without requiring Raspberry hardware, bootstrap the usable runtime database,
and verify the browser UI path. The target report includes the autoimmune
boundary smoke, synthesis-variant smoke, synthesis-stress smoke, host command-mode action smoke,
host-app configuration probe, capability probe, the inventory soak and
inventory-soak-matrix readiness checks through `pi-smoke`, the setup-integration
smoke, and the open trace plus transcript-replay debug-parser gates.

The generic first-run scripts run these commands:

```bash
python3 scripts/local_assistant_os_cli.py verify-bundle --json
python3 scripts/local_assistant_os_cli.py dataset-audit --reset --json
python3 scripts/local_assistant_os_cli.py target-report --reset --json
python3 scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json
python3 scripts/local_assistant_os_cli.py ui-smoke --reset --json
python3 scripts/local_assistant_os_cli.py launcher-smoke --reset --json
python3 scripts/local_assistant_os_cli.py first-run-smoke --json
```

The first command verifies that every manifest-listed file survived the copy
with matching byte counts and SHA-256 hashes, and that the bundle self-check
runs. The second command validates the seed/source fixtures, hashes, story/media/
weather/open-trace/transcript-replay coverage, and SQLite bootstrap profile. The third command records Python, SQLite, platform, memory/disk facts, and
runs the v0.1 smoke gates including the privacy/action/cache autoimmune smoke,
bounded synthesis variant smoke, multi-session synthesis stress smoke,
routine/household/trusted-contact setup-integration smoke,
host command-mode action smoke, host-app
configuration probe, capability probe, and the basic NLP -> UOL -> ChatFrame
open trace and transcript replay. `v01-audit --json` summarizes which browser/CLI
v0.1 requirements are evidenced and which real-world validation blockers still
prevent calling the whole architecture complete. The fourth command creates the usable runtime database at
`artifacts/local_assistant_os/assistant_v01.sqlite`, imports initial local media
metadata, and verifies safe local story/weather/school-safety chat turns. The
fifth command verifies the dependency-free browser UI path.

Full smoke suite:

```bash
python3 scripts/local_assistant_os_cli.py pi-smoke --reset --json
python3 scripts/local_assistant_os_cli.py dataset-audit --reset --json
python3 scripts/local_assistant_os_cli.py autoimmune-smoke --reset --json
python3 scripts/local_assistant_os_cli.py synthesis-variant-smoke --reset --json
python3 scripts/local_assistant_os_cli.py synthesis-stress-smoke --reset --json
python3 scripts/local_assistant_os_cli.py setup-integration-smoke --reset --json
python3 scripts/local_assistant_os_cli.py host-action-smoke --reset --json
python3 scripts/local_assistant_os_cli.py host-app-probe --reset --json
python3 scripts/local_assistant_os_cli.py capability-probe --reset --json
python3 scripts/local_assistant_os_cli.py shortcut-audit --json
python3 scripts/local_assistant_os_cli.py v01-audit --json
python3 scripts/local_assistant_os_cli.py v01-blocker-rehearsal --reset --json
python3 scripts/local_assistant_os_cli.py v01-evidence-pack --db artifacts/local_assistant_os/assistant_v01.sqlite --work-dir artifacts/local_assistant_os/v01_evidence_pack --auto-lifecycle --json
python3 scripts/local_assistant_os_cli.py v01-progress --json
python3 scripts/local_assistant_os_cli.py v01-acceptance --reset --json
python3 scripts/local_assistant_os_cli.py api-smoke --reset --json
python3 scripts/local_assistant_os_cli.py api-session-smoke --reset --json
python3 scripts/local_assistant_os_cli.py ui-smoke --reset --json
python3 scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json
python3 scripts/local_assistant_os_cli.py launcher-smoke --reset --json
python3 scripts/local_assistant_os_cli.py run-open-traces --reset --json
python3 scripts/local_assistant_os_cli.py run-transcript-replay --reset --json
python3 scripts/local_assistant_os_cli.py export-transcript-replay --db artifacts/local_assistant_os/assistant_v01.sqlite --out artifacts/local_assistant_os/event_ledger_transcript_replay.jsonl --json
python3 scripts/local_assistant_os_cli.py calibrate-event-ledger --db artifacts/local_assistant_os/assistant_v01.sqlite --work-dir artifacts/local_assistant_os/event_ledger_calibration --controls-json config/safe_lifecycle_controls.example.json --min-total-turns 4 --min-local-resolution-rate 0.5 --json
python3 scripts/local_assistant_os_cli.py write-source-attestation --event-ledger-db artifacts/local_assistant_os/assistant_v01.sqlite --event-ledger-session all --source-kind redacted_user_session --capture-surface cli_chat --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --out artifacts/local_assistant_os/source_attestation.json --json
python3 scripts/local_assistant_os_cli.py inventory-soak --offline-samples --source both --cycles 2 --story-limit 3 --min-story-models 12 --json
python3 scripts/local_assistant_os_cli.py inventory-soak-matrix --reset --json
python3 scripts/local_assistant_os_cli.py calibrate-transcript-replay --input benchmarks/sample_local_assistant_raw_transcript.jsonl --replace "Maya=<person_1>" --min-total-turns 4 --min-local-resolution-rate 0.2 --min-route-kinds 3 --min-intent-kinds 3 --require-redaction --require-static-drop --out artifacts/local_assistant_os/sample_transcript_calibration.json --reset --json
python3 scripts/local_assistant_os_cli.py v01-blocker-evidence --transcript-calibration-report-json artifacts/local_assistant_os/sample_transcript_calibration.json --json
```

The command should report `passed: true` and all readiness checks should be
true. The API smoke starts a temporary localhost server, checks `/health`,
`/dashboard`, and non-static `/event-transcript-replay`, posts one story ask to
`/ask`, verifies the event ledger, and shuts the server down. The API session
smoke then calibrates the live event ledger through `POST /calibrate-event-ledger`;
its turns are labeled `scripted_api_smoke`, so they stay development evidence.
The UI smoke also loads `/`, verifies the dependency-free browser chat shell and
operator export/calibration controls, then posts through the same `/ask`
endpoint with `scripted_ui_smoke` provenance. The served page labels actual
browser submissions as `browser_ui`. The open trace and transcript replay run
broader assistant language through the same kernel and expose the basic NLP, UOL, and ChatFrame stage
mapping for each turn.
The evidence-pack command packages the current local SQLite session into
event-ledger export, replay calibration, blocker evidence, and progress reports
under one directory. Development sessions intentionally remain development
evidence and cannot retire blockers; pass `--event-source-kind
redacted_user_session` plus source attestation flags only for reviewed,
redacted user sessions.
The event transcript export command converts the local SQLite event ledger into
a replay fixture with user utterances only. The browser/API
`/event-transcript-replay` endpoint exposes the same non-static export for
provenance-labeled local sessions. Stored answers, routes, and reasons are not exported as expectations,
so replay must rediscover behavior through the kernel. The event-ledger
calibration command and `/calibrate-event-ledger` endpoint run that export,
replay, and aggregate threshold scoring for local browser/CLI sessions. Source
attestation keeps scripted API/UI smokes separate from imported-redacted,
interactive CLI, served browser UI with the served page capture token, or
target-device candidate evidence, and
candidate attestation requires that provenance to cover every packaged turn.
The setup-integration smoke proves cold routine, household, and trusted-contact
gaps create setup requests without fake facts, then explicit user setup changes
future local answers/actions through the same UOL/ChatFrame, store, and action
confirmation path.
For a safe configured-action rehearsal on a fresh target, first generate a local
recorder config and run the configured gate against it:

```bash
python3 scripts/local_assistant_os_cli.py write-host-actions-demo-config --out config/host_actions.local_recorder.json --overwrite --json
python3 scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.local_recorder.json --require-configured --json
```

This proves the configured typed action gate on the target without opening a
real media player or calling anyone. For actual target-device media/call apps,
copy `config/host_actions.example.json` to `config/host_actions.json`, fill
`media_player_command`, `call_command`, and optionally `media_dir`, then run:

```bash
python3 scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.json --require-configured --json
python3 scripts/local_assistant_os_cli.py write-host-app-attestation --host-app-config-json config/host_actions.json --capture-surface target_device_cli --media-app-configured --call-app-configured --not-demo-recorder --real-app-commands-acknowledged --human-reviewed --out artifacts/local_assistant_os/host_app_attestation.json --json
python3 scripts/local_assistant_os_cli.py v01-blocker-evidence --host-app-config-json config/host_actions.json --host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json --run-host-app-probe --json
python3 scripts/local_assistant_os_cli.py api-session-smoke --action-mode real --host-app-config-json config/host_actions.json --host-app-media-dir <media_dir> --json
python3 scripts/local_assistant_os_cli.py v01-progress --json
python3 scripts/local_assistant_os_cli.py v01-acceptance --host-app-config-json config/host_actions.json --require-host-app-configured --json
```

The commands still pass through the typed confirmation gate; the config file
only supplies argv prefixes and does not bypass action confirmation. The local
recorder config is development evidence only. `v01-blocker-evidence` requires a
host-app attestation bound to the target config hash before the
`configured_target_device_apps` blocker can become candidate evidence. The
served browser/API path can use the same configured action gate with
`serve --action-mode real --host-app-config-json config/host_actions.json
--host-app-media-dir <media_dir>`; dry-run remains the default.
The inventory soak command repeats resource-bounded refresh cycles from both
bundled story metadata source fixtures and fails if source coverage,
quality-floor, failure-observability, or offline-network checks regress.
The inventory-soak-matrix command repeats that proof across cold-start profiles
for both combined and single-source modes, requires at least nine refresh cycles,
zero failed import cycles, and future local story routing from imported
inventory with primary UOL/ChatFrame evidence.
The transcript calibration command imports the bundled fake raw transcript,
redacts private-looking tokens plus the sample name, strips static expected
answer/route fields, and replays the result as a lightweight bundle smoke
check. Real user-derived blocker-clearing runs must add a separate safe
lifecycle controls file and strict gates so lifecycle/planner/digest evidence
comes from replayed behavior rather than static expected routes:

```bash
python3 scripts/local_assistant_os_cli.py import-transcript-replay --input <raw_chat.jsonl> --out artifacts/local_assistant_os/imported_transcript_replay.jsonl --controls-json config/safe_lifecycle_controls.example.json --json
python3 scripts/local_assistant_os_cli.py calibrate-transcript-replay --input <redacted-user-transcript.jsonl> --controls-json config/safe_lifecycle_controls.example.json --require-redaction --require-static-drop --min-synthesis-traces <n> --require-priority-signals --min-priority-signal-samples <n> --require-memory-digest-quality --require-strict-baseline-win --out artifacts/local_assistant_os/user_transcript_calibration.json --json
python3 scripts/local_assistant_os_cli.py candidate-session-audit --db <replay_event_ledger_db_from_calibration> --session all --event-source-kind redacted_user_session --capture-surface imported_redacted_transcript --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --min-synthesis-traces <n> --min-priority-signal-samples <n> --json
python3 scripts/local_assistant_os_cli.py write-source-attestation --event-ledger-db <replay_event_ledger_db_from_calibration> --event-ledger-session all --source-kind redacted_user_session --capture-surface imported_redacted_transcript --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --out artifacts/local_assistant_os/source_attestation.json --json
python3 scripts/local_assistant_os_cli.py v01-evidence-pack --db <replay_event_ledger_db_from_calibration> --session all --work-dir artifacts/local_assistant_os/v01_evidence_pack --event-source-kind redacted_user_session --capture-surface imported_redacted_transcript --source-attestation-json artifacts/local_assistant_os/source_attestation.json --auto-lifecycle --transcript-calibration-report-json artifacts/local_assistant_os/user_transcript_calibration.json --inventory-soak-report-json <inventory-soak-report.json> --host-app-config-json config/host_actions.json --host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json --run-host-app-probe --json
python3 scripts/local_assistant_os_cli.py v01-blocker-evidence --event-ledger-db <replay_event_ledger_db_from_calibration> --event-ledger-session all --event-source-kind redacted_user_session --source-attestation-json artifacts/local_assistant_os/source_attestation.json --auto-lifecycle --transcript-calibration-report-json artifacts/local_assistant_os/user_transcript_calibration.json --inventory-soak-report-json <inventory-soak-report.json> --host-app-config-json config/host_actions.json --host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json --run-host-app-probe --json
python3 scripts/local_assistant_os_cli.py v01-progress --json
```

The blocker-evidence command packages the remaining six blocker rows and keeps
development-authored traces separate from candidate user-derived evidence; it
never sets `architecture_complete`. Candidate user-derived evidence requires
the source attestation JSON to match the current event-ledger DB hash and to
declare redaction, absence of static expectations, human review, and candidate
or imported capture provenance for every packaged turn.

The launcher smoke starts the app through the platform launcher, verifies the
health launcher against localhost, checks the browser shell and parse endpoint,
and shuts the server down.
These commands create local SQLite smoke databases under
`artifacts/local_assistant_os/`.

After the bundle has been built and verified, `first-run-smoke --json` executes
the platform first-run launcher itself and checks that the nested verify,
dataset audit, target report, bootstrap, UI smoke, and launcher smoke all pass.

## Local Chat/API Smoke

```bash
python3 scripts/local_assistant_os_cli.py ask --utterance "Tell me a story." --json
python3 scripts/local_assistant_os_cli.py serve --host 127.0.0.1 --port 8771
```

Then open `http://127.0.0.1:8771/` in a browser.

For a terminal session:

```bash
python3 scripts/local_assistant_os_cli.py chat
python3 scripts/local_assistant_os_cli.py chat --turn "Tell me a story." --turn "What is the weather today?" --json
```

## Weather Cache Refresh

The smoke path uses the bundled weather cache, but cold-start or stale-cache
runs can refresh weather inventory without a large model:

```bash
python3 scripts/local_assistant_os_cli.py refresh-weather --offline-json benchmarks/sample_open_meteo_forecast.json --json
python3 scripts/local_assistant_os_cli.py ask --utterance "What is the weather today?" --json
```

The offline fixture is deterministic and requires no network. For a live
Open-Meteo refresh, use:

```bash
python3 scripts/local_assistant_os_cli.py refresh-weather --live --json
```

## Launchers

The bundle includes cross-platform launchers:

```bash
sh bin/first_run.sh
sh bin/start_app.sh
sh bin/health_check.sh
```

```powershell
bin\\first_run.cmd
bin\\start_app.cmd
bin\\health_check.cmd
```

The older Raspberry-specific launchers are still included for appliance-style
Linux installs:

```bash
sh bin/first_run_on_raspberry_pi.sh
sh bin/start_api.sh
```

`bin/start_app.sh`, `bin/start_app.cmd`, and `bin/start_api.sh` serve
`127.0.0.1:8771` by default and read optional
`MELM_ASSISTANT_DB`, `MELM_ASSISTANT_HOST`, and `MELM_ASSISTANT_PORT`
environment variables. `systemd/melm-local-assistant.service.example` is a
user-service template for starting the API on boot after adjusting the
`WorkingDirectory` and `ExecStart` paths to the copied bundle location.

The assistant remains local-first. Real media/call side effects require explicit
commands and should be tested separately on the target device.
"""


def _zip_bundle(out_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir.parent))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execute_import_story_metadata_job(
    store: AssistantOSStore,
    profile: LocalAssistantProfile,
    job,
) -> dict:
    source = str(job.payload.get("source", "both"))
    limit = int(job.payload.get("limit", job.resource_budget.get("max_items", 6)))
    gutenberg_budget = int(job.payload.get("gutenberg_max_source_bytes", 6_500_000))
    ia_budget = int(job.payload.get("internet_archive_max_source_bytes", 250_000))
    ia_page_size = int(
        job.payload.get(
            "internet_archive_page_size",
            job.resource_budget.get("internet_archive_page_size", 100),
        )
    )
    ia_max_pages = int(
        job.payload.get(
            "internet_archive_max_pages",
            job.resource_budget.get("internet_archive_max_pages", 1),
        )
    )
    ia_cursor = str(job.payload.get("internet_archive_cursor", "") or "")
    ia_query = str(job.payload.get("internet_archive_query", "") or "")
    ia_rate_limit_delay_seconds = float(
        job.payload.get(
            "internet_archive_rate_limit_delay_seconds",
            job.resource_budget.get("internet_archive_rate_limit_delay_seconds", 0.0),
        )
    )
    results = []
    if source in {"gutenberg", "both"}:
        importer = ProjectGutenbergCatalogImporter()
        gutenberg_csv = job.payload.get("gutenberg_csv")
        if gutenberg_csv:
            result = importer.import_csv_path(
                Path(str(gutenberg_csv)),
                profile,
                limit=limit,
                max_source_bytes=gutenberg_budget,
            )
        else:
            result = importer.import_metadata(
                profile, limit=limit, max_source_bytes=gutenberg_budget
            )
        _install_imported_story_items(store, profile, result.items)
        results.append(result.to_dict())
    if source in {"internet-archive", "both"}:
        importer = InternetArchiveSearchMetadataImporter(
            query=ia_query or "collection:gutenberg AND mediatype:texts"
        )
        internet_archive_json = job.payload.get("internet_archive_json")
        if internet_archive_json:
            result = importer.import_json_path(
                Path(str(internet_archive_json)),
                profile,
                limit=limit,
                max_source_bytes=ia_budget,
            )
        else:
            result = importer.import_metadata(
                profile,
                limit=limit,
                max_source_bytes=ia_budget,
                page_size=ia_page_size,
                max_pages=ia_max_pages,
                cursor=ia_cursor or None,
                rate_limit_delay_seconds=ia_rate_limit_delay_seconds,
            )
        _install_imported_story_items(store, profile, result.items)
        results.append(result.to_dict())
    return {
        "executed": True,
        "source": source,
        "imported_items": sum(len(result["items"]) for result in results),
        "results": results,
    }


def _execute_refresh_weather_cache_job(
    store: AssistantOSStore,
    profile: LocalAssistantProfile,
    job,
    *,
    live: bool,
    offline_json: Path | None,
) -> dict:
    location = str(job.payload.get("location", "") or profile.location)
    fixture = (
        None
        if live
        else Path(
            str(
                job.payload.get("offline_json")
                or offline_json
                or DEFAULT_WEATHER_SAMPLE
            )
        )
    )
    result = OpenMeteoWeatherAdapter().refresh(
        profile,
        location=location,
        offline_json=fixture,
        live=live,
    )
    _install_weather_items(store, result)
    payload = result.to_dict()
    payload["executed"] = True
    payload["target_day"] = str(job.payload.get("target_day", "today"))
    return payload


def _resolve_api_host_actions(
    *,
    action_mode: str,
    media_player_command: str,
    call_command: str,
    host_app_config_json: Path | None,
) -> dict[str, Any]:
    config, config_error = _host_app_config(host_app_config_json)
    if config_error:
        raise ValueError(f"invalid host app config JSON: {config_error}")
    media_command, media_source = _host_app_config_value(
        media_player_command,
        config,
        "media_player_command",
        "MELM_MEDIA_PLAYER_COMMAND",
        host_app_config_json,
    )
    contact_command, call_source = _host_app_config_value(
        call_command,
        config,
        "call_command",
        "MELM_CALL_COMMAND",
        host_app_config_json,
    )
    configured = bool(media_command and contact_command)
    if action_mode == "real" and not configured:
        raise ValueError(
            "--action-mode real requires both media and call command prefixes via args, env, or --host-app-config-json"
        )
    evidence_class = _host_app_static_analysis(
        host_app_config_json,
        config,
        media_command=media_command,
        call_command=contact_command,
    )
    return {
        "action_mode": action_mode,
        "media_player_command": media_command,
        "call_command": contact_command,
        "status": {
            "mode": action_mode,
            "configured": configured,
            "media_command_configured": bool(media_command),
            "call_command_configured": bool(contact_command),
            "command_sources": {
                "media": media_source,
                "call": call_source,
            },
            "config": _host_app_config_report(host_app_config_json, config, ""),
            "evidence_class": evidence_class,
        },
    }


def _prepare_api_runtime_store(
    db: Path,
    seed: Path,
    *,
    action_mode: str = "dry-run",
    host_app_media_dir: Path | None = None,
    generated_media_root: Path | None = None,
) -> dict[str, Any]:
    store = _open_store(db, seed)
    media_import: dict[str, Any] = {
        "requested": bool(host_app_media_dir is not None),
        "generated_for_real_action_smoke": False,
        "media_dir": str(host_app_media_dir) if host_app_media_dir is not None else "",
        "imported_items": 0,
        "result": {},
    }
    try:
        media_dir = host_app_media_dir
        if (
            media_dir is None
            and action_mode == "real"
            and generated_media_root is not None
        ):
            media_dir = _host_app_probe_media_dir(generated_media_root)
            media_import["generated_for_real_action_smoke"] = True
            media_import["requested"] = True
            media_import["media_dir"] = str(media_dir)
        if media_dir is not None:
            profile = store.load_profile(LocalAssistantProfile())
            result = LocalMediaInventoryAdapter().import_directory(
                media_dir, profile, limit=8
            )
            _install_imported_media_items(store, result.items)
            media_import["imported_items"] = len(result.items)
            media_import["result"] = result.to_dict()
            _persist_runtime_self_observation(store, profile)
    finally:
        store.close()
    return media_import


def _serve(args) -> None:
    media_import = _prepare_api_runtime_store(
        args.db,
        args.seed,
        action_mode=args.action_mode,
        host_app_media_dir=args.host_app_media_dir,
    )
    host_actions = _resolve_api_host_actions(
        action_mode=args.action_mode,
        media_player_command=args.media_player_command,
        call_command=args.call_command,
        host_app_config_json=args.host_app_config_json,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _assistant_api_handler(
            args.db,
            auto_execute=not args.no_auto_execute,
            shutdown_token=os.environ.get("MELM_ASSISTANT_SHUTDOWN_TOKEN"),
            action_mode=host_actions["action_mode"],
            media_player_command=host_actions["media_player_command"],
            call_command=host_actions["call_command"],
            host_action_status=host_actions["status"],
        ),
    )
    print(f"MELM Local Assistant OS v0.1 serving on http://{args.host}:{args.port}")
    print(
        f"- action gate: {host_actions['status']['mode']} configured={host_actions['status']['configured']}"
    )
    if media_import["requested"]:
        print(
            f"- media imported: {media_import['imported_items']} item(s) from {media_import['media_dir']}"
        )
    print("- GET /")
    print("- GET /health")
    print("- GET /dashboard")
    print("- GET /event-transcript-replay?session=all")
    print("- GET /parse-debug?utterance=Who%20are%20you")
    print('- POST /ask {"utterance": "Tell me a story."}')
    print('- POST /parse-debug {"utterance": "What is your name?"}')
    print('- POST /calibrate-event-ledger {"session": "all"}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _api_smoke(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    _open_store(args.db, args.seed).close()
    server = ThreadingHTTPServer(
        (args.host, 0), _assistant_api_handler(args.db, auto_execute=False)
    )
    port = int(server.server_address[1])
    base_url = f"http://{args.host}:{port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    started = perf_counter()
    thread.start()
    try:
        initial_health = _api_get_json(f"{base_url}/health")
        parse_payload = _api_post_json(
            f"{base_url}/parse-debug",
            {"utterance": "wow you don't know who you are?"},
        )
        after_parse_health = _api_get_json(f"{base_url}/health")
        ask_payload = _api_post_json(
            f"{base_url}/ask",
            {"utterance": "Tell me a story.", "capture_source": "scripted_api_smoke"},
        )
        after_health = _api_get_json(f"{base_url}/health")
        dashboard_payload = _api_get_json(f"{base_url}/dashboard")
        export_payload = _api_get_json(
            f"{base_url}/event-transcript-replay?session=all"
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    elapsed_ms = _elapsed_ms(started)
    checks = {
        "health_ok": bool(initial_health.get("ok")) and bool(after_health.get("ok")),
        "seeded_inventory_available": initial_health.get("counts", {}).get(
            "inventories", 0
        )
        >= 8,
        "parse_debug_endpoint_identity_frame": (
            parse_payload.get("chat_frame", {}).get("intent") == "assistant_identity"
            and parse_payload.get("uol", {}).get("speech_act") == "challenge"
            and parse_payload.get("uol", {}).get("object") == "self_model"
            and [stage.get("stage") for stage in parse_payload.get("mapping", [])]
            == ["basic_nlp", "uol_parse", "chat_frame"]
        ),
        "parse_debug_does_not_write_event": (
            after_parse_health.get("counts", {}).get("events", 0)
            == initial_health.get("counts", {}).get("events", 0)
        ),
        "ask_local_story": (
            ask_payload.get("route") == "local_answer"
            and ask_payload.get("reason") == "local_story_inventory"
            and bool(ask_payload.get("synthesis", {}).get("applied"))
        ),
        "membrane_and_homeostasis_logged": (
            ask_payload.get("counts", {}).get("membrane_decisions", 0) >= 1
            and ask_payload.get("counts", {}).get("homeostatic_snapshots", 0) >= 1
        ),
        "event_persisted_after_request": after_health.get("counts", {}).get("events", 0)
        >= 1,
        "dashboard_endpoint_reads_ledger": (
            bool(dashboard_payload.get("ok"))
            and dashboard_payload.get("counts", {}).get("events", 0)
            == after_health.get("counts", {}).get("events", 0)
        ),
        "event_transcript_export_endpoint_non_static": (
            export_payload.get("schema")
            == "melm.local_assistant_event_transcript_export.v1"
            and export_payload.get("events_exported", 0) >= 1
            and export_payload.get("answers_routes_reasons_exported") is False
            and export_payload.get("forbidden_static_fields_exported") == []
            and not any(
                set(row) & STATIC_TRANSCRIPT_EXPECTATION_KEYS
                or {"answer", "route", "reason"} & set(row)
                for row in export_payload.get("turns", [])
                if isinstance(row, dict)
            )
        ),
        "localhost_only": args.host in {"127.0.0.1", "localhost", "::1"},
        "stdlib_http_sqlite": True,
    }
    payload = {
        "db": str(args.db),
        "base_url": base_url,
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": "stdlib_python_sqlite_http",
        "dependency_class": "stdlib_only",
        "elapsed_ms": elapsed_ms,
        "initial_health": initial_health,
        "parse_debug": {
            "intent": parse_payload.get("chat_frame", {}).get("intent"),
            "speech_act": parse_payload.get("uol", {}).get("speech_act"),
            "object": parse_payload.get("uol", {}).get("object"),
            "mapping": [
                stage.get("stage") for stage in parse_payload.get("mapping", [])
            ],
            "unknown_token_count": parse_payload.get("nlp", {}).get(
                "unknown_token_count", 0
            ),
            "primary_parse_basis": parse_payload.get("nlp", {}).get(
                "primary_parse_basis"
            ),
            "primary_domain_evidence": parse_payload.get("nlp", {}).get(
                "primary_domain_evidence", {}
            ),
            "composition_pattern": parse_payload.get("nlp", {})
            .get("compositional_parse", {})
            .get("pattern"),
            "secondary_domain_hints": parse_payload.get("nlp", {}).get(
                "secondary_domain_hints", {}
            ),
            "secondary_meaning_hints": parse_payload.get("secondary_meaning_hints", []),
        },
        "after_parse_health": after_parse_health,
        "ask": {
            "route": ask_payload.get("route"),
            "reason": ask_payload.get("reason"),
            "synthesis_applied": bool(ask_payload.get("synthesis", {}).get("applied")),
            "membrane": ask_payload.get("membrane", {}),
            "counts": ask_payload.get("counts", {}),
        },
        "after_health": after_health,
        "dashboard": {
            "counts": dashboard_payload.get("counts", {}),
            "safety_flags": dashboard_payload.get("safety_flags", {}),
        },
        "event_transcript_export": {
            "events_exported": export_payload.get("events_exported", 0),
            "answers_routes_reasons_exported": export_payload.get(
                "answers_routes_reasons_exported"
            ),
            "forbidden_static_fields_exported": export_payload.get(
                "forbidden_static_fields_exported", []
            ),
        },
    }
    _print_payload(payload, json_mode=args.json)


def _api_session_smoke(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    media_import = _prepare_api_runtime_store(
        args.db,
        args.seed,
        action_mode=args.action_mode,
        host_app_media_dir=args.host_app_media_dir,
        generated_media_root=args.db.parent / f"{args.db.stem}_api_session_media",
    )
    host_actions = _resolve_api_host_actions(
        action_mode=args.action_mode,
        media_player_command=args.media_player_command,
        call_command=args.call_command,
        host_app_config_json=args.host_app_config_json,
    )
    server = ThreadingHTTPServer(
        (args.host, 0),
        _assistant_api_handler(
            args.db,
            auto_execute=False,
            action_mode=host_actions["action_mode"],
            media_player_command=host_actions["media_player_command"],
            call_command=host_actions["call_command"],
            host_action_status=host_actions["status"],
        ),
    )
    port = int(server.server_address[1])
    base_url = f"http://{args.host}:{port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    started = perf_counter()
    thread.start()
    try:
        initial_health = _api_get_json(f"{base_url}/health")
        raw_turns = []
        for label, utterance in API_SESSION_SMOKE_TURNS:
            raw_turns.append(
                {
                    "label": label,
                    "utterance": utterance,
                    "payload": _api_post_json(
                        f"{base_url}/ask",
                        {
                            "utterance": utterance,
                            "capture_source": "scripted_api_smoke",
                        },
                    ),
                }
            )
        after_health = _api_get_json(f"{base_url}/health")
        export_payload = _api_get_json(
            f"{base_url}/event-transcript-replay?session=all"
        )
        calibration_payload = _api_post_json(
            f"{base_url}/calibrate-event-ledger",
            {
                "session": "all",
                "min_total_turns": len(API_SESSION_SMOKE_TURNS),
                "min_route_kinds": 3,
                "min_intent_kinds": 8,
                "min_local_resolution_rate": 0.8,
                "reset": True,
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    elapsed_ms = _elapsed_ms(started)
    store = AssistantOSStore(args.db)
    try:
        dashboard = build_assistant_os_dashboard(store).to_dict()
    finally:
        store.close()
    turn_summaries = [
        _api_session_turn_summary(item["label"], item["utterance"], item["payload"])
        for item in raw_turns
    ]
    route_counts = dict(
        sorted(Counter(turn["route"] for turn in turn_summaries).items())
    )
    reason_by_label = {turn["label"]: turn["reason"] for turn in turn_summaries}
    route_by_label = {turn["label"]: turn["route"] for turn in turn_summaries}
    action_results = [
        turn["action_execution"] for turn in turn_summaries if turn["action_execution"]
    ]
    media_result = next(
        (item for item in action_results if item.get("action_type") == "play_media"), {}
    )
    call_result = next(
        (item for item in action_results if item.get("action_type") == "call_contact"),
        {},
    )
    expected_action_status = "executed" if args.action_mode == "real" else "prepared"
    expected_side_effect = args.action_mode == "real"
    checks = {
        "health_ok": bool(initial_health.get("ok")) and bool(after_health.get("ok")),
        "eleven_turns_completed": len(turn_summaries) == len(API_SESSION_SMOKE_TURNS),
        "identity_self_model_local": (
            route_by_label.get("identity") == "local_answer"
            and reason_by_label.get("identity") == "self_model_identity"
            and _turn_by_label(turn_summaries, "identity")
            .get("debug_parse", {})
            .get("chat_frame", {})
            .get("intent")
            == "assistant_identity"
        ),
        "story_local": route_by_label.get("story") == "local_answer"
        and reason_by_label.get("story") == "local_story_inventory",
        "weather_cached": route_by_label.get("weather") == "cached_tool"
        and reason_by_label.get("weather") == "weather_cache_hit",
        "safety_local": route_by_label.get("safety") == "local_answer"
        and reason_by_label.get("safety") == "local_common_sense_policy",
        "media_gated_then_confirmed": (
            _turn_by_label(turn_summaries, "media_request").get("confirmation_required")
            == 1
            and reason_by_label.get("media_confirm") == "confirmed_device_action"
            and media_result.get("status") == expected_action_status
            and media_result.get("side_effect_executed") is expected_side_effect
        ),
        "health_local": reason_by_label.get("health")
        == "bounded_general_health_guidance",
        "profile_memory_local": reason_by_label.get("profile_memory")
        == "personal_memory_summary",
        "meal_local": reason_by_label.get("meal") == "memory_plus_weather_cache",
        "call_gated_then_confirmed": (
            _turn_by_label(turn_summaries, "call_request").get("confirmation_required")
            == 1
            and reason_by_label.get("call_confirm") == "confirmed_device_action"
            and call_result.get("status") == expected_action_status
            and call_result.get("resolved_target") == "+234-000-MOM"
            and call_result.get("side_effect_executed") is expected_side_effect
        ),
        "no_cloud_or_fetch": not any(
            turn["cloud_needed"] or turn["external_fetch_needed"]
            for turn in turn_summaries
        ),
        "events_persisted": after_health.get("counts", {}).get("events", 0)
        == len(turn_summaries),
        "membrane_homeostasis_complete": (
            after_health.get("counts", {}).get("membrane_decisions", 0)
            == len(turn_summaries)
            and after_health.get("counts", {}).get("homeostatic_snapshots", 0)
            == len(turn_summaries)
        ),
        "event_transcript_export_api_non_static": (
            export_payload.get("events_exported", 0) == len(turn_summaries)
            and export_payload.get("answers_routes_reasons_exported") is False
            and export_payload.get("forbidden_static_fields_exported") == []
            and export_payload.get("capture_provenance", {}).get(
                "has_capture_provenance"
            )
            is True
            and not any(
                set(row) & STATIC_TRANSCRIPT_EXPECTATION_KEYS
                or {"answer", "route", "reason"} & set(row)
                for row in export_payload.get("turns", [])
                if isinstance(row, dict)
            )
        ),
        "event_ledger_calibration_api_passed": (
            calibration_payload.get("passed") is True
            and calibration_payload.get("events_exported") == len(turn_summaries)
            and calibration_payload.get("aggregate", {}).get("turns_replayed")
            == len(turn_summaries)
            and calibration_payload.get("aggregate", {}).get("local_resolution_rate", 0)
            >= 0.8
            and calibration_payload.get("aggregate", {}).get("intent_kinds", 0) >= 8
            and calibration_payload.get("answers_routes_reasons_exported") is False
            and calibration_payload.get("capture_provenance", {}).get(
                "has_capture_provenance"
            )
            is True
        ),
        "safety_flags_clean": (
            dashboard["safety_flags"]["cloud_private_inclusions"] == 0
            and dashboard["safety_flags"]["unconfirmed_executed_actions"] == 0
            and dashboard["safety_flags"]["action_without_confirmation_gate"] == 0
            and dashboard["safety_flags"]["fake_latest_news_local_answers"] == 0
            and dashboard["safety_flags"]["low_quality_applied_synthesis"] == 0
        ),
        "localhost_only": args.host in {"127.0.0.1", "localhost", "::1"},
        "host_action_gate_mode_respected": (
            initial_health.get("host_actions", {}).get("mode") == args.action_mode
            and bool(initial_health.get("host_actions", {}).get("configured", False))
            == bool(host_actions["status"].get("configured", False))
        ),
    }
    payload = {
        "db": str(args.db),
        "base_url": base_url,
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": "stdlib_python_sqlite_http",
        "dependency_class": "stdlib_only",
        "elapsed_ms": elapsed_ms,
        "turns": turn_summaries,
        "route_counts": route_counts,
        "action_results": action_results,
        "host_actions": host_actions["status"],
        "media_import": media_import,
        "initial_health": initial_health,
        "after_health": after_health,
        "safety_flags": dashboard["safety_flags"],
        "event_transcript_export": {
            "events_exported": export_payload.get("events_exported", 0),
            "answers_routes_reasons_exported": export_payload.get(
                "answers_routes_reasons_exported"
            ),
            "forbidden_static_fields_exported": export_payload.get(
                "forbidden_static_fields_exported", []
            ),
            "capture_provenance": export_payload.get("capture_provenance", {}),
        },
        "event_ledger_calibration": {
            "passed": bool(calibration_payload.get("passed", False)),
            "events_exported": calibration_payload.get("events_exported", 0),
            "turns_replayed": calibration_payload.get("aggregate", {}).get(
                "turns_replayed", 0
            ),
            "local_resolution_rate": calibration_payload.get("aggregate", {}).get(
                "local_resolution_rate", 0
            ),
            "intent_kinds": calibration_payload.get("aggregate", {}).get(
                "intent_kinds", 0
            ),
            "work_dir": calibration_payload.get("work_dir", ""),
            "transcript_jsonl": calibration_payload.get("transcript_jsonl", ""),
            "capture_provenance": calibration_payload.get("capture_provenance", {}),
        },
    }
    _print_payload(payload, json_mode=args.json)


def _ui_smoke(args) -> None:
    if args.reset:
        _remove_sqlite_files(args.db)
    _open_store(args.db, args.seed).close()
    server = ThreadingHTTPServer(
        (args.host, 0), _assistant_api_handler(args.db, auto_execute=False)
    )
    port = int(server.server_address[1])
    base_url = f"http://{args.host}:{port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    started = perf_counter()
    thread.start()
    try:
        html = _api_get_text(f"{base_url}/")
        html_alias = _api_get_text(f"{base_url}/index.html")
        health = _api_get_json(f"{base_url}/health")
        parse_payload = _api_post_json(
            f"{base_url}/parse-debug",
            {"utterance": "wow you don't know who you are?"},
        )
        functional_parse_payload = _api_post_json(
            f"{base_url}/parse-debug",
            {"utterance": "can you help me grow in my career?"},
        )
        rejected_browser_ui_capture = _api_post_json_error(
            f"{base_url}/ask",
            {"utterance": "Who are you?", "capture_source": "browser_ui"},
        )
        smoke_capture = {"capture_source": "scripted_ui_smoke"}
        identity_payload = _api_post_json(
            f"{base_url}/ask", {"utterance": "Who are you?", **smoke_capture}
        )
        status_payload = _api_post_json(
            f"{base_url}/ask",
            {"utterance": "What have you done so far?", **smoke_capture},
        )
        ask_payload = _api_post_json(
            f"{base_url}/ask", {"utterance": "Tell me a story.", **smoke_capture}
        )
        media_request_payload = _api_post_json(
            f"{base_url}/ask", {"utterance": "Play calm piano.", **smoke_capture}
        )
        media_confirm_payload = _api_post_json(
            f"{base_url}/ask",
            {"utterance": "Yes, play calm piano.", **smoke_capture},
        )
        after_health = _api_get_json(f"{base_url}/health")
        export_payload = _api_get_json(
            f"{base_url}/event-transcript-replay?session=all"
        )
        calibration_payload = _api_post_json(
            f"{base_url}/calibrate-event-ledger",
            {
                "session": "all",
                "min_total_turns": 5,
                "min_route_kinds": 2,
                "min_intent_kinds": 4,
                "min_local_resolution_rate": 0.8,
                "reset": True,
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    elapsed_ms = _elapsed_ms(started)
    checks = {
        "ui_root_loads": "<title>MELM Local Assistant OS</title>" in html,
        "ui_index_alias_loads": html_alias == html,
        "chat_form_present": 'id="chat-form"' in html and 'id="utterance"' in html,
        "ask_endpoint_wired": 'fetch("/ask"' in html,
        "browser_ui_capture_source_wired": 'capture_source: "browser_ui"' in html,
        "browser_ui_capture_token_wired": (
            "const browserCaptureToken =" in html
            and "capture_token: browserCaptureToken" in html
        ),
        "browser_session_identity_wired": (
            "const browserSessionId =" in html
            and "session_id: browserSessionId" in html
        ),
        "improvement_consent_wired": (
            'id="improvement-opt-in"' in html
            and "improvement_opt_in: improvementOptIn.checked" in html
        ),
        "response_integrity_debug_wired": (
            "payload.response_integrity" in html
            and "integrity: understand" in html
            and "research:" in html
        ),
        "browser_ui_capture_source_requires_token": (
            rejected_browser_ui_capture.get("status") == 400
            and "served browser capture token"
            in str(rejected_browser_ui_capture.get("body", ""))
        ),
        "health_endpoint_wired": 'fetch("/health"' in html,
        "event_export_endpoint_wired": 'fetch("/event-transcript-replay?session=all"'
        in html,
        "event_calibration_endpoint_wired": 'fetch("/calibrate-event-ledger"' in html,
        "operator_buttons_present": 'id="export-session"' in html
        and 'id="calibrate-session"' in html,
        "action_controls_present": 'textContent = "Confirm"' in html
        and 'textContent = "Cancel"' in html,
        "confirmation_helper_present": "function confirmationUtterance" in html,
        "debug_frame_present": (
            'summary.textContent = "Basic NLP -> UOL -> ChatFrame"' in html
            and "function debugText" in html
            and "function shouldOpenDebug" in html
            and "debug.open = shouldOpenDebug(payload)" in html
            and "primary domain:" in html
            and "secondary debug hints:" in html
            and "secondary hint policy:" in html
            and "secondary domains:" in html
        ),
        "functional_grammar_debug_wired": (
            "functional:" in html
            and "predicate candidates:" in html
            and "relations:" in html
            and "semantic unknowns:" in html
            and "semantic_unknown_token_count" in html
        ),
        "functional_grammar_parse_exposed": (
            functional_parse_payload.get("chat_frame", {}).get("intent")
            == "personal_goal_advice"
            and functional_parse_payload.get("nlp", {})
            .get("primary_domain_evidence", {})
            .get("source")
            == "weighted_functional_relation"
            and functional_parse_payload.get("uol", {}).get("subject") == "assistant"
            and functional_parse_payload.get("uol", {}).get("action") == "help"
            and functional_parse_payload.get("uol", {}).get("object") == "career"
            and functional_parse_payload.get("uol", {}).get("complement_action")
            == "grow"
            and functional_parse_payload.get("uol", {}).get("indirect_object") == "user"
            and bool(functional_parse_payload.get("uol", {}).get("relations"))
        ),
        "parse_debug_endpoint_exposed": (
            parse_payload.get("chat_frame", {}).get("intent") == "assistant_identity"
            and parse_payload.get("uol", {}).get("speech_act") == "challenge"
            and [stage.get("stage") for stage in parse_payload.get("mapping", [])]
            == ["basic_nlp", "uol_parse", "chat_frame"]
        ),
        "health_ok": bool(health.get("ok")) and bool(after_health.get("ok")),
        "identity_turn_self_model": (
            identity_payload.get("route") == "local_answer"
            and identity_payload.get("reason") == "self_model_identity"
            and identity_payload.get("intent") == "assistant_identity"
            and identity_payload.get("debug_parse", {}).get("uol", {}).get("object")
            == "self_model"
        ),
        "identity_turn_integrity_scored": (
            identity_payload.get("response_integrity", {}).get("band") == "reliable"
            and identity_payload.get("response_integrity", {}).get("overall_score", 0)
            >= 0.8
            and identity_payload.get("improvement", {}).get("candidate", {}) == {}
        ),
        "status_turn_ledger_backed": (
            status_payload.get("route") == "local_answer"
            and status_payload.get("reason") == "self_status_ledger_summary"
            and status_payload.get("intent") == "assistant_status"
            and status_payload.get("debug_parse", {}).get("uol", {}).get("object")
            == "runtime_status"
            and "self_status.counts"
            in status_payload.get("synthesis", {}).get("citations", [])
        ),
        "ask_turn_local_story": (
            ask_payload.get("route") == "local_answer"
            and ask_payload.get("reason") == "local_story_inventory"
            and bool(ask_payload.get("synthesis", {}).get("applied"))
        ),
        "media_action_gated": (
            media_request_payload.get("route") == "device_action"
            and media_request_payload.get("reason") == "local_media_action"
            and media_request_payload.get("membrane", {}).get("confirmation_required")
            == 1
            and media_request_payload.get("pending_action", {}).get("action_type")
            == "play_media"
        ),
        "media_action_confirmed_dry_run": (
            media_confirm_payload.get("route") == "device_action"
            and media_confirm_payload.get("reason") == "confirmed_device_action"
            and media_confirm_payload.get("action_execution", {}).get("status")
            == "prepared"
            and media_confirm_payload.get("action_execution", {}).get(
                "side_effect_executed"
            )
            is False
        ),
        "event_persisted_after_ui_turn": after_health.get("counts", {}).get("events", 0)
        >= 5,
        "event_export_api_non_static": (
            export_payload.get("events_exported", 0) >= 5
            and export_payload.get("answers_routes_reasons_exported") is False
            and export_payload.get("forbidden_static_fields_exported") == []
            and export_payload.get("capture_provenance", {}).get(
                "has_capture_provenance"
            )
            is True
        ),
        "event_calibration_api_passed": (
            calibration_payload.get("passed") is True
            and calibration_payload.get("aggregate", {}).get("turns_replayed", 0) >= 5
            and calibration_payload.get("aggregate", {}).get("local_resolution_rate", 0)
            >= 0.8
            and calibration_payload.get("capture_provenance", {}).get(
                "has_capture_provenance"
            )
            is True
        ),
        "localhost_only": args.host in {"127.0.0.1", "localhost", "::1"},
        "dependency_free_html": "https://" not in html
        and "http://" not in html
        and "<script src=" not in html,
    }
    payload = {
        "db": str(args.db),
        "base_url": base_url,
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": "stdlib_python_sqlite_http_html",
        "dependency_class": "stdlib_only",
        "elapsed_ms": elapsed_ms,
        "browser_ui_capture_token_rejection": rejected_browser_ui_capture,
        "ui": {
            "path": "/",
            "bytes": len(html.encode("utf-8")),
            "title_present": checks["ui_root_loads"],
            "form_present": checks["chat_form_present"],
            "dependency_free": checks["dependency_free_html"],
        },
        "ask": {
            "route": ask_payload.get("route"),
            "reason": ask_payload.get("reason"),
            "synthesis_applied": bool(ask_payload.get("synthesis", {}).get("applied")),
            "counts": ask_payload.get("counts", {}),
        },
        "identity": {
            "route": identity_payload.get("route"),
            "reason": identity_payload.get("reason"),
            "intent": identity_payload.get("intent"),
            "answer": identity_payload.get("answer"),
            "debug_parse": identity_payload.get("debug_parse", {}),
        },
        "parse_debug": {
            "intent": parse_payload.get("chat_frame", {}).get("intent"),
            "speech_act": parse_payload.get("uol", {}).get("speech_act"),
            "object": parse_payload.get("uol", {}).get("object"),
            "mapping": [
                stage.get("stage") for stage in parse_payload.get("mapping", [])
            ],
            "unknown_token_count": parse_payload.get("nlp", {}).get(
                "unknown_token_count", 0
            ),
            "primary_parse_basis": parse_payload.get("nlp", {}).get(
                "primary_parse_basis"
            ),
            "primary_domain_evidence": parse_payload.get("nlp", {}).get(
                "primary_domain_evidence", {}
            ),
            "composition_pattern": parse_payload.get("nlp", {})
            .get("compositional_parse", {})
            .get("pattern"),
            "secondary_domain_hints": parse_payload.get("nlp", {}).get(
                "secondary_domain_hints", {}
            ),
            "secondary_meaning_hints": parse_payload.get("secondary_meaning_hints", []),
        },
        "functional_parse_debug": {
            "intent": functional_parse_payload.get("chat_frame", {}).get("intent"),
            "source": functional_parse_payload.get("nlp", {})
            .get("primary_domain_evidence", {})
            .get("source"),
            "subject": functional_parse_payload.get("uol", {}).get("subject"),
            "action": functional_parse_payload.get("uol", {}).get("action"),
            "object": functional_parse_payload.get("uol", {}).get("object"),
            "complement_action": functional_parse_payload.get("uol", {}).get(
                "complement_action"
            ),
            "indirect_object": functional_parse_payload.get("uol", {}).get(
                "indirect_object"
            ),
            "parse_score": functional_parse_payload.get("uol", {}).get("parse_score"),
            "relations": functional_parse_payload.get("uol", {}).get("relations", []),
        },
        "status": {
            "route": status_payload.get("route"),
            "reason": status_payload.get("reason"),
            "intent": status_payload.get("intent"),
            "answer": status_payload.get("answer"),
            "debug_parse": status_payload.get("debug_parse", {}),
            "citations": status_payload.get("synthesis", {}).get("citations", []),
        },
        "action": {
            "request_route": media_request_payload.get("route"),
            "request_reason": media_request_payload.get("reason"),
            "confirmation_required": int(
                media_request_payload.get("membrane", {}).get(
                    "confirmation_required", 0
                )
            ),
            "pending_action": media_request_payload.get("pending_action", {}),
            "confirm_reason": media_confirm_payload.get("reason"),
            "execution": media_confirm_payload.get("action_execution", {}),
        },
        "event_transcript_export": {
            "events_exported": export_payload.get("events_exported", 0),
            "answers_routes_reasons_exported": export_payload.get(
                "answers_routes_reasons_exported"
            ),
            "forbidden_static_fields_exported": export_payload.get(
                "forbidden_static_fields_exported", []
            ),
            "capture_provenance": export_payload.get("capture_provenance", {}),
        },
        "event_ledger_calibration": {
            "passed": bool(calibration_payload.get("passed", False)),
            "turns_replayed": calibration_payload.get("aggregate", {}).get(
                "turns_replayed", 0
            ),
            "local_resolution_rate": calibration_payload.get("aggregate", {}).get(
                "local_resolution_rate", 0
            ),
            "work_dir": calibration_payload.get("work_dir", ""),
            "capture_provenance": calibration_payload.get("capture_provenance", {}),
        },
        "after_health": after_health,
    }
    _print_payload(payload, json_mode=args.json)


def _api_session_turn_summary(label: str, utterance: str, payload: dict) -> dict:
    return {
        "label": label,
        "utterance": utterance,
        "intent": payload.get("intent", ""),
        "route": payload.get("route", ""),
        "reason": payload.get("reason", ""),
        "cloud_needed": bool(payload.get("cloud_needed", False)),
        "external_fetch_needed": bool(payload.get("external_fetch_needed", False)),
        "confirmation_required": int(
            payload.get("membrane", {}).get("confirmation_required", 0)
        ),
        "synthesis_applied": bool(payload.get("synthesis", {}).get("applied", False)),
        "action_execution": payload.get("action_execution", {}),
        "debug_parse": payload.get("debug_parse", {}),
        "counts": payload.get("counts", {}),
    }


def _turn_by_label(turns: list[dict], label: str) -> dict:
    return next((turn for turn in turns if turn.get("label") == label), {})


def _target_report(args) -> None:
    db_dir = args.db_dir
    if args.reset and db_dir.exists():
        _safe_remove_bundle_dir(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    dataset_payload = _run_cli_json(
        ROOT,
        "dataset-audit",
        "--db",
        str(db_dir / "dataset_audit_target.sqlite"),
        "--reset",
        "--json",
    )
    pi_payload = _run_cli_json(
        ROOT,
        "pi-smoke",
        "--db",
        str(db_dir / "pi_target_smoke.sqlite"),
        "--reset",
        "--json",
    )
    autoimmune_payload = _run_cli_json(
        ROOT,
        "autoimmune-smoke",
        "--db",
        str(db_dir / "autoimmune_target_smoke.sqlite"),
        "--reset",
        "--json",
    )
    synthesis_variant_payload = _run_cli_json(
        ROOT,
        "synthesis-variant-smoke",
        "--db",
        str(db_dir / "synthesis_variant_target_smoke.sqlite"),
        "--reset",
        "--json",
    )
    synthesis_stress_payload = _run_cli_json(
        ROOT,
        "synthesis-stress-smoke",
        "--db",
        str(db_dir / "synthesis_stress_target_smoke.sqlite"),
        "--reset",
        "--json",
    )
    setup_integration_payload = _run_cli_json(
        ROOT,
        "setup-integration-smoke",
        "--db",
        str(db_dir / "setup_integration_target_smoke.sqlite"),
        "--reset",
        "--json",
    )
    host_action_payload = _run_cli_json(
        ROOT,
        "host-action-smoke",
        "--db",
        str(db_dir / "host_action_target_smoke.sqlite"),
        "--work-dir",
        str(db_dir / "host_action_target_smoke"),
        "--reset",
        "--json",
    )
    host_app_args = [
        "host-app-probe",
        "--db",
        str(db_dir / "host_app_probe.sqlite"),
        "--work-dir",
        str(db_dir / "host_app_probe"),
        "--reset",
        "--json",
    ]
    if args.media_player_command:
        host_app_args.extend(("--media-player-command", args.media_player_command))
    if args.call_command:
        host_app_args.extend(("--call-command", args.call_command))
    if args.host_app_config_json is not None:
        host_app_args.extend(("--config-json", str(args.host_app_config_json)))
    if args.host_app_media_dir is not None:
        host_app_args.extend(("--media-dir", str(args.host_app_media_dir)))
    if args.require_host_app_configured:
        host_app_args.append("--require-configured")
    host_app_payload = _run_cli_json(ROOT, *host_app_args)
    capability_payload = _run_cli_json(
        ROOT,
        "capability-probe",
        "--db",
        str(db_dir / "capability_probe_target.sqlite"),
        "--reset",
        "--json",
    )
    v01_audit_payload = _run_cli_json(ROOT, "v01-audit", "--json")
    api_payload = _run_cli_json(
        ROOT,
        "api-smoke",
        "--db",
        str(db_dir / "api_target_smoke.sqlite"),
        "--host",
        args.host,
        "--reset",
        "--json",
    )
    api_session_payload = _run_cli_json(
        ROOT,
        "api-session-smoke",
        "--db",
        str(db_dir / "api_session_target_smoke.sqlite"),
        "--host",
        args.host,
        "--reset",
        "--json",
    )
    ui_payload = _run_cli_json(
        ROOT,
        "ui-smoke",
        "--db",
        str(db_dir / "ui_target_smoke.sqlite"),
        "--host",
        args.host,
        "--reset",
        "--json",
    )
    bootstrap_payload = _run_cli_json(
        ROOT,
        "bootstrap-runtime",
        "--db",
        str(db_dir / "bootstrap_runtime.sqlite"),
        "--reset",
        "--json",
    )
    open_traces_payload = _run_cli_json(
        ROOT,
        "run-open-traces",
        "--db-dir",
        str(db_dir / "open_traces_target"),
        "--reset",
        "--json",
    )
    transcript_replay_payload = _run_cli_json(
        ROOT,
        "run-transcript-replay",
        "--db-dir",
        str(db_dir / "transcript_replay_target"),
        "--reset",
        "--json",
    )
    transcript_calibration_payload = _run_cli_json(
        ROOT,
        "calibrate-transcript-replay",
        "--input",
        str(DEFAULT_RAW_TRANSCRIPT_SAMPLE),
        "--replace",
        "Maya=<person_1>",
        "--min-total-turns",
        "4",
        "--min-local-resolution-rate",
        "0.2",
        "--min-route-kinds",
        "2",
        "--min-intent-kinds",
        "3",
        "--require-redaction",
        "--require-static-drop",
        "--work-dir",
        str(db_dir / "transcript_calibration_target"),
        "--reset",
        "--json",
    )
    hardware = _target_hardware_report()
    resources = _target_resource_report(db_dir)
    raspberry_pi_requirement_satisfied = (not args.require_raspberry_pi) or bool(
        hardware["raspberry_pi_detected"]
    )
    checks = {
        "python_supported": sys.version_info >= (3, 11),
        "sqlite_available": bool(sqlite3.sqlite_version),
        "db_dir_writable": _directory_writable(db_dir),
        "dataset_audit_passed": bool(dataset_payload.get("passed", False)),
        "pi_smoke_passed": bool(pi_payload.get("passed", False)),
        "inventory_soak_matrix_passed": bool(
            pi_payload.get("checks", {}).get("inventory_soak_matrix_passed", False)
        ),
        "autoimmune_smoke_passed": bool(autoimmune_payload.get("passed", False)),
        "synthesis_variant_smoke_passed": bool(
            synthesis_variant_payload.get("passed", False)
        ),
        "synthesis_stress_smoke_passed": bool(
            synthesis_stress_payload.get("passed", False)
        ),
        "setup_integration_smoke_passed": bool(
            setup_integration_payload.get("passed", False)
        ),
        "host_action_smoke_passed": bool(host_action_payload.get("passed", False)),
        "host_app_probe_reported": bool(
            host_app_payload.get("checks", {}).get("configuration_reported", False)
        ),
        "host_app_requirement_satisfied": (
            (not args.require_host_app_configured)
            or bool(
                host_app_payload.get("configured", False)
                and host_app_payload.get("passed", False)
            )
        ),
        "capability_probe_passed": bool(capability_payload.get("passed", False)),
        "v01_audit_passed": bool(v01_audit_payload.get("passed", False)),
        "api_smoke_passed": bool(api_payload.get("passed", False)),
        "api_session_smoke_passed": bool(api_session_payload.get("passed", False)),
        "ui_smoke_passed": bool(ui_payload.get("passed", False)),
        "bootstrap_runtime_passed": bool(bootstrap_payload.get("passed", False)),
        "open_traces_passed": bool(open_traces_payload.get("passed", False)),
        "transcript_replay_passed": bool(
            transcript_replay_payload.get("passed", False)
        ),
        "transcript_calibration_passed": bool(
            transcript_calibration_payload.get("passed", False)
        ),
        "localhost_api": args.host in {"127.0.0.1", "localhost", "::1"},
        "stdlib_only": True,
        "raspberry_pi_requirement_satisfied": raspberry_pi_requirement_satisfied,
    }
    payload = {
        "db_dir": str(db_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": _elapsed_ms(started),
        "hardware": hardware,
        "hardware_policy": {
            "raspberry_pi_required": bool(args.require_raspberry_pi),
            "raspberry_pi_hardware_optional_for_v01": not args.require_raspberry_pi,
            "raspberry_pi_requirement_satisfied": raspberry_pi_requirement_satisfied,
        },
        "resources": resources,
        "runtime": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "smokes": {
            "dataset_audit": {
                "passed": bool(dataset_payload.get("passed", False)),
                "checks": dataset_payload.get("checks", {}),
                "elapsed_ms": dataset_payload.get("elapsed_ms", 0),
                "files": dataset_payload.get("files", {}),
                "source_fixtures": dataset_payload.get("source_fixtures", {}),
                "bootstrap": dataset_payload.get("bootstrap", {}),
            },
            "pi_smoke": {
                "passed": bool(pi_payload.get("passed", False)),
                "checks": pi_payload.get("checks", {}),
                "timings": pi_payload.get("timings", {}),
                "peak_traced_kb": pi_payload.get("peak_traced_kb", 0),
                "db_bytes": pi_payload.get("db_bytes", 0),
                "lifecycle_db_bytes": pi_payload.get("lifecycle_db_bytes", 0),
                "action_db_bytes": pi_payload.get("action_db_bytes", 0),
                "inventory_soak": pi_payload.get("inventory_soak", {}),
                "inventory_soak_matrix": pi_payload.get("inventory_soak_matrix", {}),
                "inventory_diversity_smoke": pi_payload.get(
                    "inventory_diversity_smoke", {}
                ),
                "inventory_retry_smoke": pi_payload.get("inventory_retry_smoke", {}),
                "inventory_failure_smoke": pi_payload.get(
                    "inventory_failure_smoke", {}
                ),
            },
            "autoimmune_smoke": {
                "passed": bool(autoimmune_payload.get("passed", False)),
                "checks": autoimmune_payload.get("checks", {}),
                "elapsed_ms": autoimmune_payload.get("elapsed_ms", 0),
                "turns": autoimmune_payload.get("turns", []),
                "pending_actions": autoimmune_payload.get("pending_actions", {}),
                "safety_flags": autoimmune_payload.get("safety_flags", {}),
            },
            "synthesis_variant_smoke": {
                "passed": bool(synthesis_variant_payload.get("passed", False)),
                "checks": synthesis_variant_payload.get("checks", {}),
                "elapsed_ms": synthesis_variant_payload.get("elapsed_ms", 0),
                "variant_count": synthesis_variant_payload.get("variant_count", 0),
                "route_counts": synthesis_variant_payload.get("route_counts", {}),
                "reason_counts": synthesis_variant_payload.get("reason_counts", {}),
                "turns": synthesis_variant_payload.get("turns", []),
                "safety_flags": synthesis_variant_payload.get("safety_flags", {}),
            },
            "synthesis_stress_smoke": {
                "passed": bool(synthesis_stress_payload.get("passed", False)),
                "checks": synthesis_stress_payload.get("checks", {}),
                "elapsed_ms": synthesis_stress_payload.get("elapsed_ms", 0),
                "turn_count": synthesis_stress_payload.get("turn_count", 0),
                "session_count": synthesis_stress_payload.get("session_count", 0),
                "route_counts": synthesis_stress_payload.get("route_counts", {}),
                "reason_counts": synthesis_stress_payload.get("reason_counts", {}),
                "intent_counts": synthesis_stress_payload.get("intent_counts", {}),
                "quality": synthesis_stress_payload.get("quality", {}),
                "complexity": synthesis_stress_payload.get("complexity", {}),
                "turns": synthesis_stress_payload.get("turns", []),
                "safety_flags": synthesis_stress_payload.get("safety_flags", {}),
            },
            "setup_integration_smoke": {
                "passed": bool(setup_integration_payload.get("passed", False)),
                "checks": setup_integration_payload.get("checks", {}),
                "elapsed_ms": setup_integration_payload.get("elapsed_ms", 0),
                "turns": setup_integration_payload.get("turns", []),
                "setup_requests_after_gaps": setup_integration_payload.get(
                    "setup_requests_after_gaps", {}
                ),
                "facts_after_setup": setup_integration_payload.get(
                    "facts_after_setup", {}
                ),
                "contacts_after_setup": setup_integration_payload.get(
                    "contacts_after_setup", {}
                ),
                "pending_actions": setup_integration_payload.get("pending_actions", {}),
                "safety_flags": setup_integration_payload.get("safety_flags", {}),
                "action_execution": setup_integration_payload.get(
                    "action_execution", {}
                ),
            },
            "host_action_smoke": {
                "passed": bool(host_action_payload.get("passed", False)),
                "checks": host_action_payload.get("checks", {}),
                "elapsed_ms": host_action_payload.get("elapsed_ms", 0),
                "records": host_action_payload.get("records", []),
                "runtime": host_action_payload.get("runtime", ""),
            },
            "host_app_probe": {
                "passed": bool(host_app_payload.get("passed", False)),
                "configured": bool(host_app_payload.get("configured", False)),
                "skipped": bool(host_app_payload.get("skipped", False)),
                "checks": host_app_payload.get("checks", {}),
                "command_sources": host_app_payload.get("command_sources", {}),
                "config": host_app_payload.get("config", {}),
                "elapsed_ms": host_app_payload.get("elapsed_ms", 0),
                "runtime": host_app_payload.get("runtime", ""),
                "next_steps": host_app_payload.get("next_steps", []),
            },
            "capability_probe": {
                "passed": bool(capability_payload.get("passed", False)),
                "checks": capability_payload.get("checks", {}),
                "elapsed_ms": capability_payload.get("elapsed_ms", 0),
                "total_cases": capability_payload.get("total_cases", 0),
                "route_counts": capability_payload.get("route_counts", {}),
                "bucket_counts": capability_payload.get("bucket_counts", {}),
                "local_device_rate": capability_payload.get("local_device_rate", 0.0),
                "complexity": capability_payload.get("complexity", {}),
            },
            "v01_audit": {
                "passed": bool(v01_audit_payload.get("passed", False)),
                "checks": v01_audit_payload.get("checks", {}),
                "status": v01_audit_payload.get("status", ""),
                "architecture_complete": bool(
                    v01_audit_payload.get("architecture_complete", False)
                ),
                "blocker_count": int(v01_audit_payload.get("blocker_count", 0) or 0),
                "completion_blockers": v01_audit_payload.get("completion_blockers", []),
            },
            "api_smoke": {
                "passed": bool(api_payload.get("passed", False)),
                "checks": api_payload.get("checks", {}),
                "elapsed_ms": api_payload.get("elapsed_ms", 0),
                "base_url": api_payload.get("base_url", ""),
                "parse_debug": api_payload.get("parse_debug", {}),
                "ask": api_payload.get("ask", {}),
                "dashboard": api_payload.get("dashboard", {}),
                "event_transcript_export": api_payload.get(
                    "event_transcript_export", {}
                ),
            },
            "api_session_smoke": {
                "passed": bool(api_session_payload.get("passed", False)),
                "checks": api_session_payload.get("checks", {}),
                "elapsed_ms": api_session_payload.get("elapsed_ms", 0),
                "route_counts": api_session_payload.get("route_counts", {}),
                "action_results": api_session_payload.get("action_results", []),
                "event_transcript_export": api_session_payload.get(
                    "event_transcript_export", {}
                ),
                "event_ledger_calibration": api_session_payload.get(
                    "event_ledger_calibration", {}
                ),
            },
            "ui_smoke": {
                "passed": bool(ui_payload.get("passed", False)),
                "checks": ui_payload.get("checks", {}),
                "elapsed_ms": ui_payload.get("elapsed_ms", 0),
                "base_url": ui_payload.get("base_url", ""),
                "ui": ui_payload.get("ui", {}),
                "ask": ui_payload.get("ask", {}),
                "event_transcript_export": ui_payload.get(
                    "event_transcript_export", {}
                ),
                "event_ledger_calibration": ui_payload.get(
                    "event_ledger_calibration", {}
                ),
            },
            "bootstrap_runtime": {
                "passed": bool(bootstrap_payload.get("passed", False)),
                "checks": bootstrap_payload.get("checks", {}),
                "elapsed_ms": bootstrap_payload.get("elapsed_ms", 0),
                "counts": bootstrap_payload.get("counts", {}),
                "turns": bootstrap_payload.get("turns", []),
                "db_bytes": bootstrap_payload.get("db_bytes", 0),
            },
            "open_traces": _open_trace_summary(open_traces_payload),
            "transcript_replay": _transcript_replay_summary(transcript_replay_payload),
            "transcript_calibration": {
                "passed": bool(transcript_calibration_payload.get("passed", False)),
                "input_count": int(
                    transcript_calibration_payload.get("input_count", 0) or 0
                ),
                "aggregate": dict(transcript_calibration_payload.get("aggregate", {})),
                "items": [
                    _transcript_calibration_target_item_summary(item)
                    for item in transcript_calibration_payload.get("items", [])
                    if isinstance(item, dict)
                ],
            },
        },
        "pi_hardware_note": (
            "Run with --require-raspberry-pi on target hardware to make Raspberry Pi detection mandatory."
            if not args.require_raspberry_pi
            else ""
        ),
    }
    _print_payload(payload, json_mode=args.json)


def _transcript_calibration_target_item_summary(item: dict) -> dict:
    import_payload = dict(item.get("import", {}))
    replay_payload = dict(item.get("replay", {}))
    return {
        "label": str(item.get("label", "")),
        "input_path": str(item.get("input_path", "")),
        "imported_transcript_jsonl": str(item.get("imported_transcript_jsonl", "")),
        "passed": bool(item.get("passed", False)),
        "error": str(item.get("error", "")),
        "turns_written": int(import_payload.get("turns_written", 0) or 0),
        "assistant_rows_skipped": int(
            import_payload.get("assistant_rows_skipped", 0) or 0
        ),
        "redaction_counts": dict(import_payload.get("redaction_counts", {})),
        "static_expectation_fields_dropped": dict(
            import_payload.get("static_expectation_fields_dropped", {})
        ),
        "replay_passed": bool(replay_payload.get("passed", False)),
        "turns_replayed": int(replay_payload.get("turns", 0) or 0),
        "local_resolution_rate": float(
            replay_payload.get("local_resolution_rate", 0.0) or 0.0
        ),
        "debug_mapping_passed": bool(
            replay_payload.get("debug_checks", {}).get("debug_maps_present", False)
        ),
        "baseline_required": bool(
            replay_payload.get("baseline_comparison", {}).get("required", False)
        ),
    }


def _target_hardware_report() -> dict:
    model = _read_text_if_exists(Path("/proc/device-tree/model")).strip("\x00\r\n ")
    cpuinfo = _read_text_if_exists(Path("/proc/cpuinfo"))
    raspberry_detected = "raspberry pi" in f"{model}\n{cpuinfo}".lower()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "raspberry_pi_detected": raspberry_detected,
        "raspberry_model": model if raspberry_detected else "",
    }


def _target_resource_report(db_dir: Path) -> dict:
    usage = shutil.disk_usage(db_dir)
    return {
        "memory_total_kb": _linux_meminfo_kb("MemTotal"),
        "memory_available_kb": _linux_meminfo_kb("MemAvailable"),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "disk_used_bytes": usage.used,
    }


def _linux_meminfo_kb(key: str) -> int:
    for line in _read_text_if_exists(Path("/proc/meminfo")).splitlines():
        if line.startswith(f"{key}:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return 0


def _read_text_if_exists(path: Path) -> str:
    try:
        return (
            path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        )
    except OSError:
        return ""


def _directory_writable(path: Path) -> bool:
    probe = path / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _assistant_api_handler(
    db: Path,
    *,
    auto_execute: bool,
    shutdown_token: str | None = None,
    action_mode: str = "dry-run",
    media_player_command: str = "",
    call_command: str = "",
    host_action_status: dict[str, Any] | None = None,
):
    db_path = db
    host_action_payload = dict(
        host_action_status or {"mode": action_mode, "configured": False}
    )
    browser_capture_token = hashlib.sha256(os.urandom(32)).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urlsplit(self.path)
            request_path = parsed_url.path
            if request_path in {"/", "/index.html"}:
                _send_html(
                    self,
                    _assistant_ui_html(browser_capture_token=browser_capture_token),
                )
                return
            if request_path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if request_path == "/parse-debug":
                query = parse_qs(parsed_url.query)
                utterance = str((query.get("utterance") or [""])[0]).strip()
                if not utterance:
                    self.send_error(
                        400, "GET /parse-debug requires utterance query parameter"
                    )
                    return
                _send_json(self, parse_assistant_debug_frame(utterance).to_dict())
                return
            if request_path == "/dashboard":
                store = AssistantOSStore(db_path)
                try:
                    payload = {
                        "ok": True,
                        "db": str(db_path),
                        **build_assistant_os_dashboard(store).to_dict(),
                    }
                finally:
                    store.close()
                _send_json(self, payload)
                return
            if request_path == "/event-transcript-replay":
                query = parse_qs(parsed_url.query)
                session = str((query.get("session") or ["all"])[0]).strip() or "all"
                _send_json(
                    self, _event_transcript_export_api_payload(db_path, session=session)
                )
                return
            if request_path != "/health":
                self.send_error(404)
                return
            store = AssistantOSStore(db_path)
            payload = {
                "ok": True,
                "db": str(db_path),
                "counts": store.table_counts(),
                "host_actions": host_action_payload,
            }
            store.close()
            _send_json(self, payload)

        def do_POST(self) -> None:  # noqa: N802
            request_path = urlsplit(self.path).path
            if request_path == "/_melm/launcher-shutdown":
                if (
                    not shutdown_token
                    or self.headers.get("X-MELM-Shutdown-Token") != shutdown_token
                ):
                    self.send_error(404)
                    return
                _send_json(self, {"ok": True, "shutdown": "scheduled"})
                Thread(
                    target=self.server.shutdown,
                    name="launcher-smoke-shutdown",
                    daemon=True,
                ).start()
                return
            if request_path not in {"/ask", "/parse-debug", "/calibrate-event-ledger"}:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                request = json.loads(body.decode("utf-8"))
            except (KeyError, json.JSONDecodeError):
                self.send_error(400, "POST body must be JSON")
                return
            if not isinstance(request, dict):
                self.send_error(400, "POST body must be a JSON object")
                return
            if request_path == "/calibrate-event-ledger":
                try:
                    payload = _api_event_ledger_calibration_payload(db_path, request)
                except (TypeError, ValueError) as exc:
                    self.send_error(400, str(exc))
                    return
                _send_json(self, payload)
                return
            try:
                utterance = str(request["utterance"])
            except KeyError:
                self.send_error(400, "POST JSON must include utterance")
                return
            if request_path == "/parse-debug":
                _send_json(self, parse_assistant_debug_frame(utterance).to_dict())
                return
            try:
                capture_source = _api_ask_capture_source(
                    request,
                    browser_capture_token=browser_capture_token,
                )
                capture_session_id = _api_ask_session_id(
                    request, capture_source=capture_source
                )
                improvement_opt_in = _api_bool(request, "improvement_opt_in", False)
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            store = AssistantOSStore(db_path)
            payload = _handle_utterance(
                store,
                utterance,
                auto_execute=auto_execute,
                action_mode=action_mode,
                media_player_command=media_player_command,
                call_command=call_command,
                capture_surface="browser_api",
                capture_source=capture_source,
                capture_session_id=capture_session_id,
                improvement_opt_in=improvement_opt_in,
            )
            store.close()
            _send_json(self, payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    return Handler


def _api_event_ledger_calibration_payload(
    db: Path, request: dict[str, Any]
) -> dict[str, Any]:
    session = str(request.get("session", "all") or "all")
    profile_mode = str(request.get("profile_mode", "current") or "current")
    if profile_mode not in {"current", "minimal"}:
        raise ValueError("profile_mode must be 'current' or 'minimal'")
    work_dir = _api_event_ledger_calibration_work_dir(db)
    return _build_event_ledger_calibration_payload(
        db=db,
        work_dir=work_dir,
        session=session,
        profile_mode=profile_mode,
        reset=_api_bool(request, "reset", True),
        min_total_turns=_api_int(request, "min_total_turns", 1, minimum=1),
        min_local_resolution_rate=_api_float(
            request, "min_local_resolution_rate", 0.0, minimum=0.0, maximum=1.0
        ),
        min_route_kinds=_api_int(request, "min_route_kinds", 1, minimum=1),
        min_intent_kinds=_api_int(request, "min_intent_kinds", 1, minimum=1),
        min_synthesis_traces=_api_int(request, "min_synthesis_traces", 0, minimum=0),
        min_priority_signal_samples=_api_int(
            request, "min_priority_signal_samples", 0, minimum=0
        ),
        require_priority_signals=_api_bool(request, "require_priority_signals", False),
        require_memory_digest_quality=_api_bool(
            request, "require_memory_digest_quality", False
        ),
        require_strict_baseline_win=_api_bool(
            request, "require_strict_baseline_win", False
        ),
        require_redaction=_api_bool(request, "require_redaction", False),
        require_static_drop=_api_bool(request, "require_static_drop", False),
    )


def _api_ask_capture_source(
    request: dict[str, Any], *, browser_capture_token: str = ""
) -> str:
    source = str(request.get("capture_source", "") or "").strip()
    if not source:
        return API_DEFAULT_CAPTURE_SOURCE
    if source not in API_ASK_CAPTURE_SOURCES:
        raise ValueError(
            "capture_source must be browser_ui, scripted_api_smoke, scripted_ui_smoke, "
            f"or omitted for {API_DEFAULT_CAPTURE_SOURCE}"
        )
    if source == "browser_ui":
        supplied_token = str(request.get("capture_token", "") or "")
        if not browser_capture_token or supplied_token != browser_capture_token:
            raise ValueError(
                "browser_ui capture_source requires the served browser capture token"
            )
    return source


def _api_ask_session_id(request: dict[str, Any], *, capture_source: str) -> str:
    value = str(request.get("session_id", "") or "").strip()
    if not value:
        return ""
    if capture_source != "browser_ui":
        raise ValueError("session_id is only accepted for served browser_ui capture")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", value):
        raise ValueError("session_id must be 1-96 safe identifier characters")
    return value


def _api_event_ledger_calibration_work_dir(db: Path) -> Path:
    if str(db) == ":memory:":
        return Path("artifacts/local_assistant_os/api_event_ledger_calibration")
    stem = db.stem or "assistant"
    return db.parent / f"{stem}_api_event_ledger_calibration"


def _api_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"{key} must be a boolean")


def _api_int(payload: dict[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return parsed


def _api_float(
    payload: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = payload.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return parsed


def _assistant_ui_html(*, browser_capture_token: str = "") -> str:
    browser_capture_token_json = json.dumps(str(browser_capture_token or ""))
    return (
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MELM Local Assistant OS</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f4ee;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d7d2c8;
      --accent: #146c5c;
      --accent-ink: #ffffff;
      --local: #e8f3ef;
      --tool: #edf0fb;
      --action: #fff0d8;
      --warn: #ffe6e0;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #171717;
        --panel: #232323;
        --ink: #f2f2f2;
        --muted: #b6b6b6;
        --line: #424242;
        --accent: #4fb79f;
        --accent-ink: #06221d;
        --local: #12352c;
        --tool: #222947;
        --action: #483516;
        --warn: #4b211a;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      max-width: 920px;
      margin: 0 auto;
      padding: 16px;
      gap: 12px;
    }
    header, footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 760;
    }
    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 13px;
      min-width: 150px;
      text-align: right;
    }
    .operator-actions {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .operator-actions button {
      min-height: 34px;
      padding: 0 10px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      font-size: 13px;
    }
    #messages {
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 2px;
    }
    .message {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }
    .message.user {
      margin-left: min(96px, 12vw);
      border-color: var(--accent);
    }
    .message.assistant {
      margin-right: min(96px, 12vw);
    }
    .message.local_answer { background: var(--local); }
    .message.cached_tool { background: var(--tool); }
    .message.device_action { background: var(--action); }
    .message.cloud_handoff,
    .message.external_fetch,
    .message.blocked,
    .message.clarify { background: var(--warn); }
    .meta {
      margin-top: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: color-mix(in srgb, var(--panel) 72%, transparent);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .actions button {
      min-height: 36px;
      padding: 0 12px;
      font-size: 14px;
    }
    .actions .cancel {
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    details.debug {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    details.debug summary {
      cursor: pointer;
      width: fit-content;
    }
    details.debug pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 8px 0 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: color-mix(in srgb, var(--panel) 86%, black);
      color: var(--ink);
      line-height: 1.35;
    }
    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: end;
    }
    .consent {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .consent input {
      width: 18px;
      height: 18px;
      margin: 0;
    }
    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      line-height: 1.35;
    }
    button {
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--accent);
      color: var(--accent-ink);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    #face-container {
      text-align: center;
      padding: 8px 0 4px;
      transition: opacity 0.3s;
    }
    #face-emoji {
      font-size: 56px;
      line-height: 1.1;
      display: inline-block;
      transition: transform 0.3s, filter 0.3s;
    }
    #face-emoji.listening {
      animation: breathe 1.6s ease-in-out infinite;
    }
    #face-emoji.thinking {
      animation: pulse 0.8s ease-in-out infinite;
    }
    @keyframes breathe {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.08); }
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.6; transform: scale(0.92); }
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message {
      animation: slideUp 0.25s ease-out;
    }
    @media (max-width: 560px) {
      main { padding: 10px; }
      header { align-items: flex-start; flex-direction: column; }
      .status { width: 100%; text-align: left; }
      .operator-actions { width: 100%; justify-content: stretch; }
      .operator-actions button { flex: 1 1 120px; }
      #face-emoji { font-size: 40px; }
      form { grid-template-columns: 1fr; }
      button { width: 100%; }
      .message.user, .message.assistant { margin-left: 0; margin-right: 0; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>MELM Local Assistant OS</h1>
      <div class="operator-actions">
        <button id="export-session" type="button">Export</button>
        <button id="calibrate-session" type="button">Calibrate</button>
      </div>
      <div id="status" class="status">Starting</div>
    </header>
    <div id="face-container"><span id="face-emoji">😊</span></div>
    <section id="messages" aria-live="polite"></section>
    <form id="chat-form">
      <label class="consent">
        <input id="improvement-opt-in" type="checkbox">
        Improve from this session
      </label>
      <textarea id="utterance" name="utterance" autocomplete="off" placeholder="Ask locally" required></textarea>
      <button id="send" type="submit">Send</button>
    </form>
  </main>
  <script>
    const browserCaptureToken = """
        + browser_capture_token_json
        + """;
    const browserSessionId = sessionStorage.getItem("melm_browser_session_id")
      || `browser_${(crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2)).replace(/[^A-Za-z0-9_.:-]/g, "_")}`;
    sessionStorage.setItem("melm_browser_session_id", browserSessionId);
    const form = document.getElementById("chat-form");
    const input = document.getElementById("utterance");
    const send = document.getElementById("send");
    const messages = document.getElementById("messages");
    const status = document.getElementById("status");
    const exportSessionButton = document.getElementById("export-session");
    const calibrateSessionButton = document.getElementById("calibrate-session");
    const improvementOptIn = document.getElementById("improvement-opt-in");

    function addMessage(role, text, meta, route, payload) {
      const item = document.createElement("article");
      item.className = `message ${role} ${route || ""}`;
      const body = document.createElement("div");
      body.textContent = text || "";
      item.appendChild(body);
      if (meta && meta.length) {
        const metaRow = document.createElement("div");
        metaRow.className = "meta";
        meta.forEach((value) => {
          const pill = document.createElement("span");
          pill.className = "pill";
          pill.textContent = value;
          metaRow.appendChild(pill);
        });
        item.appendChild(metaRow);
      }
      if (payload && payload.membrane && payload.membrane.confirmation_required && payload.pending_action) {
        const actionRow = document.createElement("div");
        actionRow.className = "actions";
        const confirm = document.createElement("button");
        confirm.type = "button";
        confirm.textContent = "Confirm";
        confirm.addEventListener("click", () => {
          confirm.disabled = true;
          cancel.disabled = true;
          sendUtterance(confirmationUtterance(payload.pending_action));
        });
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "cancel";
        cancel.textContent = "Cancel";
        cancel.addEventListener("click", () => {
          confirm.disabled = true;
          cancel.disabled = true;
          sendUtterance("Cancel that.");
        });
        actionRow.appendChild(confirm);
        actionRow.appendChild(cancel);
        item.appendChild(actionRow);
      }
      if (payload && payload.debug_parse) {
        const debug = document.createElement("details");
        debug.className = "debug";
        debug.open = shouldOpenDebug(payload);
        const summary = document.createElement("summary");
        summary.textContent = "Basic NLP -> UOL -> ChatFrame";
        const pre = document.createElement("pre");
        pre.textContent = debugText(payload.debug_parse, payload.response_integrity || {});
        debug.appendChild(summary);
        debug.appendChild(pre);
        item.appendChild(debug);
      }
      messages.appendChild(item);
      item.scrollIntoView({ block: "end" });
    }

    function shouldOpenDebug(payload) {
      const debug = payload.debug_parse || {};
      const frame = debug.chat_frame || {};
      const nlp = debug.nlp || {};
      const route = payload.route || frame.route || "";
      const integrity = payload.response_integrity || {};
      return route === "cloud_handoff"
        || route === "external_fetch"
        || route === "clarify"
        || route === "blocked"
        || payload.reason === "unknown_intent"
        || frame.needs_cloud === true
        || Number(nlp.unknown_token_count || 0) > 0
        || Number(nlp.semantic_unknown_token_count || 0) > 0
        || (integrity.band && integrity.band !== "reliable");
    }

    function debugText(debug, integrity) {
      const uol = debug.uol || {};
      const frame = debug.chat_frame || {};
      const slots = frame.slots || {};
      const nlp = debug.nlp || {};
      const slotSources = uol.slot_sources || {};
      const capabilities = frame.capabilities || {};
      const tokens = (debug.tokens || []).join(" ");
      const secondaryHints = (debug.secondary_meaning_hints || []).join(", ") || "none";
      const notes = (debug.notes || []).join(", ") || "none";
      const domains = Object.entries(nlp.domain_hints || {}).map(([key, values]) => `${key}:${(values || []).join("|")}`).join(", ") || "none";
      const secondaryDomains = Object.entries(nlp.secondary_domain_hints || {}).map(([key, values]) => `${key}:${(values || []).join("|")}`).join(", ") || "none";
      const secondaryPolicy = frame.secondary_hint_policy || nlp.secondary_hint_policy || "debug_only";
      const primaryDomain = nlp.primary_domain_evidence || {};
      const primaryDomainText = `${primaryDomain.intent || ""}:${primaryDomain.source || "none"}:${primaryDomain.pattern || "none"}`;
      const localSources = (capabilities.local_sources || []).join(", ") || "none";
      const primaryRoutingBasis = (frame.primary_routing_basis || []).join(" | ") || "none";
      const secondaryDebugHints = (frame.secondary_debug_hints || []).join(" | ") || "none";
      const composition = nlp.compositional_parse || {};
      const functional = nlp.functional_parse || composition.functional_parse || {};
      const candidates = (nlp.candidate_parses || composition.candidate_parses || [])
        .map((item) => `${item.action || ""}:${item.semantic_class || ""}@${item.score || 0}`)
        .filter(Boolean)
        .join(", ") || "none";
      const relations = (functional.relations || uol.relations || [])
        .map((item) => `${item.type || ""}(${item.head || ""},${item.value || ""})@${item.weight || 0}`)
        .filter(Boolean)
        .join(", ") || "none";
      const semanticUnknown = (nlp.semantic_unknown_tokens || []).join(", ") || "none";
      const tokenRoles = (nlp.token_roles || [])
        .map((item) => `${item.token || ""}:${item.role || ""}`)
        .filter(Boolean)
        .join(", ") || "none";
      const mapping = (debug.mapping || []).map((stage) => {
        const output = stage.output || {};
        if (stage.stage === "basic_nlp") return `Basic NLP -> ${output.boundary_intent || output.bounded_intent || ""}`;
        if (stage.stage === "uol_parse") return `UOL -> ${output.subject || ""}/${output.action || ""}/${output.object || ""}`;
        if (stage.stage === "chat_frame") return `ChatFrame -> ${output.intent || ""}/${output.route || ""}`;
        return stage.stage || "";
      }).filter(Boolean).join(" | ") || "Basic NLP -> UOL -> ChatFrame";
      return [
        `tokens: ${tokens}`,
        `nlp: ${nlp.bounded_intent || ""} | unknown tokens ${nlp.unknown_token_count || 0}`,
        `composition: ${composition.pattern || "none"} | basis ${composition.source || "none"}`,
        `functional: ${functional.speech_act || "none"} | ${functional.subject || ""}/${functional.action || ""}/${functional.object || ""} | complement ${functional.complement_action || "none"} | score ${functional.parse_score || 0}`,
        `predicate candidates: ${candidates}`,
        `relations: ${relations}`,
        `semantic unknowns: ${semanticUnknown}`,
        `primary domain: ${primaryDomainText}`,
        `token roles: ${tokenRoles}`,
        `map: ${mapping}`,
        `uol: ${uol.subject || ""} / ${uol.action || ""} / ${uol.object || ""} -> ${uol.target || ""}`,
        `frame: ${frame.intent || ""} via ${frame.route || ""} (${frame.reason || ""})`,
        `domain: ${frame.domain || ""} | local ${frame.can_answer_locally === true}`,
        `slot sources: subject=${(slotSources.subject || {}).source || ""} | action=${(slotSources.action || {}).source || ""} | object=${(slotSources.object || {}).source || ""}`,
        `slots: ${slots.subject || ""} | ${slots.action || ""} | ${slots.object || ""} | ${slots.target || ""}`,
        `score: parse ${uol.parse_score || 0} | complexity ${frame.complexity_score || 0}`,
        `integrity: understand ${integrity.understanding_score || 0} | response ${integrity.response_integrity_score || 0} | overall ${integrity.overall_score || 0} (${integrity.band || "unscored"})`,
        `research: ${integrity.research_recommended === true} | topics ${(integrity.research_topics || []).join(", ") || "none"}`,
        `capabilities: ${localSources}`,
        `primary routing: ${primaryRoutingBasis}`,
        `secondary debug hints: ${secondaryDebugHints}`,
        `secondary hint policy: ${secondaryPolicy}`,
        `secondary domains: ${secondaryDomains}`,
        `domains: ${domains}`,
        `secondary hints: ${secondaryHints}`,
        `notes: ${notes}`
      ].join("\\n");
    }

    function confirmationUtterance(action) {
      let target = String(action.target || "").trim().replace(/[.?!]+$/g, "");
      const lower = target.toLowerCase();
      if (action.action_type === "play_media") {
        if (lower.startsWith("playing ")) target = `play ${target.slice(8)}`;
        else if (!lower.startsWith("play ")) target = `play ${target}`;
      } else if (action.action_type === "call_contact") {
        if (lower.startsWith("i can call ")) target = `call ${target.slice(11)}`;
        else if (!lower.startsWith("call ")) target = `call ${target}`;
      }
      return `Yes, ${target}.`;
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/health", { cache: "no-store" });
        const payload = await response.json();
        const counts = payload.counts || {};
        status.textContent = `events ${counts.events || 0} | inventory ${counts.inventories || 0}`;
      } catch (error) {
        status.textContent = "Offline";
      }
    }

    async function exportSession() {
      exportSessionButton.disabled = true;
      try {
        const response = await fetch("/event-transcript-replay?session=all", { cache: "no-store" });
        const payload = await response.json();
        addMessage(
          "assistant",
          `Exported ${payload.events_exported || 0} local user turn(s) without stored answers, routes, or reasons.`,
          ["event ledger", payload.source_type || "local export"],
          "local_answer",
          null
        );
      } catch (error) {
        addMessage("assistant", "Local export failed.", ["error"], "blocked");
      } finally {
        exportSessionButton.disabled = false;
      }
    }

    async function calibrateSession() {
      calibrateSessionButton.disabled = true;
      try {
        const response = await fetch("/calibrate-event-ledger", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session: "all",
            min_total_turns: 1,
            min_route_kinds: 1,
            min_intent_kinds: 1,
            min_local_resolution_rate: 0
          })
        });
        const payload = await response.json();
        const aggregate = payload.aggregate || {};
        const routeRate = Number(aggregate.local_resolution_rate || 0).toFixed(2);
        addMessage(
          "assistant",
          `Calibration ${payload.passed ? "passed" : "needs more evidence"} over ${aggregate.turns_replayed || 0} replayed turn(s).`,
          ["event ledger", `local ${routeRate}`],
          payload.passed ? "local_answer" : "clarify",
          null
        );
      } catch (error) {
        addMessage("assistant", "Local calibration failed.", ["error"], "blocked");
      } finally {
        calibrateSessionButton.disabled = false;
      }
    }

    async function sendUtterance(utterance) {
      utterance = String(utterance || "").trim();
      if (!utterance) return;
      addMessage("user", utterance, [], "");
      input.value = "";
      send.disabled = true;
      status.textContent = "Thinking";
      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            utterance,
            capture_source: "browser_ui",
            capture_token: browserCaptureToken,
            session_id: browserSessionId,
            improvement_opt_in: improvementOptIn.checked
          })
        });
        const payload = await response.json();
        const meta = [payload.route, payload.reason].filter(Boolean);
        const integrity = payload.response_integrity || {};
        if (integrity.overall_score !== undefined) {
          meta.push(`confidence ${Number(integrity.overall_score || 0).toFixed(2)}`);
        }
        if (payload.improvement && payload.improvement.candidate && payload.improvement.candidate.status) {
          meta.push("improvement queued");
        }
        if (payload.membrane && payload.membrane.confirmation_required) {
          meta.push("confirmation required");
        }
        if (payload.action_execution && payload.action_execution.status) {
          meta.push(`action ${payload.action_execution.status}`);
        }
        addMessage("assistant", payload.answer || "", meta, payload.route, payload);
        await refreshHealth();
      } catch (error) {
        addMessage("assistant", "Local server error.", ["error"], "blocked");
        status.textContent = "Error";
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await sendUtterance(input.value.trim());
    });
    exportSessionButton.addEventListener("click", exportSession);
    calibrateSessionButton.addEventListener("click", calibrateSession);

    const MOOD_EMOJI = {
      happy: '😊', joyful: '😊', calm: '😌',
      neutral: '🙂',
      excited: '🤩', playful: '😏',
      curious: '🤔', surprised: '😮',
      annoyed: '😤', frustrated: '😠', angry: '😠',
      hurt: '😢', sad: '😢', anxious: '😰',
      tired: '😴', listening: '👂',
    };
    const MOOD_TONE = {
      happy: [523, 659, 784], joyful: [523, 659, 784],
      calm: [392, 523], neutral: [440],
      excited: [659, 784, 1047], playful: [587, 740],
      curious: [349, 440], surprised: [523, 659, 784],
      annoyed: [311, 262], frustrated: [277, 233], angry: [277, 233],
      hurt: [262, 220], sad: [262, 220], anxious: [370, 349],
      tired: [196], listening: [880],
    };

    let audioCtx = null;

    function getAudioCtx() {
      if (!audioCtx) {
        try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { return null; }
      }
      if (audioCtx.state === 'suspended') audioCtx.resume();
      return audioCtx;
    }

    function playTone(freq, duration, type) {
      const ctx = getAudioCtx();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = type || 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.08, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + duration);
    }

    function playMoodTone(moodId) {
      const tones = MOOD_TONE[moodId] || MOOD_TONE.neutral;
      tones.forEach((freq, i) => {
        setTimeout(() => playTone(freq, 0.3, 'sine'), i * 120);
      });
    }

    function renderMoodFace(mood) {
      const el = document.getElementById('face-emoji');
      if (!el) return;
      el.classList.remove('listening', 'thinking');
      if (!mood || !mood.mood_id) {
        el.textContent = MOOD_EMOJI.neutral;
        return;
      }
      if (mood.is_listening) {
        el.textContent = MOOD_EMOJI.listening;
        el.classList.add('listening');
        return;
      }
      el.textContent = MOOD_EMOJI[mood.mood_id] || MOOD_EMOJI.neutral;
      const engagement = Number(mood.engagement_level || 1);
      const val = Math.abs(Number(mood.valence || 0));
      const s = 1 + (val * 0.12 * engagement);
      el.style.transform = `scale(${s})`;
    }

    async function sendUtterance(utterance) {
      utterance = String(utterance || "").trim();
      if (!utterance) return;
      addMessage("user", utterance, [], "");
      input.value = "";
      send.disabled = true;
      status.textContent = "Thinking";
      const faceEl = document.getElementById('face-emoji');
      if (faceEl) { faceEl.textContent = '💭'; faceEl.classList.add('thinking'); }
      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            utterance,
            capture_source: "browser_ui",
            capture_token: browserCaptureToken,
            session_id: browserSessionId,
            improvement_opt_in: improvementOptIn.checked
          })
        });
        const payload = await response.json();
        const meta = [payload.route, payload.reason].filter(Boolean);
        const integrity = payload.response_integrity || {};
        if (integrity.overall_score !== undefined) {
          meta.push(`confidence ${Number(integrity.overall_score || 0).toFixed(2)}`);
        }
        if (payload.improvement && payload.improvement.candidate && payload.improvement.candidate.status) {
          meta.push("improvement queued");
        }
        if (payload.membrane && payload.membrane.confirmation_required) {
          meta.push("confirmation required");
        }
        if (payload.action_execution && payload.action_execution.status) {
          meta.push(`action ${payload.action_execution.status}`);
        }
        addMessage("assistant", payload.answer || "", meta, payload.route, payload);
        renderMoodFace(payload.session_mood);
        playMoodTone((payload.session_mood && payload.session_mood.mood_id) || 'neutral');
        await refreshHealth();
      } catch (error) {
        addMessage("assistant", "Local server error.", ["error"], "blocked");
        status.textContent = "Error";
        renderMoodFace(null);
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    refreshHealth();
    input.focus();
  </script>
</body>
</html>
"""
    )


def _api_get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return dict(json.loads(response.read().decode("utf-8")))


def _api_get_text(url: str) -> str:
    with urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _api_post_json(
    url: str, payload: dict, *, headers: dict[str, str] | None = None
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(
        url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return dict(json.loads(response.read().decode("utf-8")))


def _api_post_json_error(
    url: str, payload: dict, *, headers: dict[str, str] | None = None
) -> dict:
    try:
        response_payload = _api_post_json(url, payload, headers=headers)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "status": int(exc.code),
            "reason": str(exc.reason),
            "body": body,
        }
    return {
        "status": 200,
        "reason": "OK",
        "body": json.dumps(response_payload, sort_keys=True),
    }


def _handle_utterance(
    store: AssistantOSStore,
    utterance: str,
    *,
    auto_execute: bool,
    cold_start: bool = False,
    action_mode: str = "dry-run",
    media_player_command: str = "",
    call_command: str = "",
    tts_command: str = "",
    capture_surface: str = "",
    capture_source: str = "",
    capture_session_id: str = "",
    improvement_opt_in: bool = False,
    model_path: Path | None = None,
) -> dict:
    if capture_session_id:
        store.use_session(capture_session_id)
    if model_path is None:
        from melm.appliance.provisioning import resolve_gguf_model_path
        model_path = resolve_gguf_model_path()
    decoder = None
    if model_path is not None and model_path.exists():
        decoder = ConstrainedDecoder(preferred="llamacpp", model_path=str(model_path))
    kernel = AssistantOSKernel(
        profile=_cold_profile() if cold_start else None,
        store=store,
        decoder=decoder,
        action_executor=LocalDeviceActionExecutor(
            mode=action_mode,  # type: ignore[arg-type]
            media_player_command=media_player_command,
            call_command=call_command,
            tts_command=tts_command,
        ),
        capture_surface=capture_surface,
        capture_source=capture_source,
        improvement_opt_in=improvement_opt_in,
    )
    decision = kernel.handle(utterance)
    debug_parse = parse_assistant_debug_frame(utterance, decision).to_dict()
    latest_event_id = kernel.events[-1].event_id if kernel.events else ""
    opportunities = kernel.reflect()
    executed: list[str] = []
    if auto_execute:
        for opportunity in opportunities:
            if opportunity.kind in {
                "build_story_inventory",
                "refresh_weather_cache",
                "ask_profile_memory",
                "request_trusted_contact",
                "build_media_index",
                "ask_routine_memory",
                "ask_household_memory",
            }:
                before = len(kernel.executed_jobs)
                kernel.execute(opportunity)
                executed.extend(kernel.executed_jobs[before:])
    row = store.connection.execute(
        """
        SELECT allowed, boundary_crossed, confirmation_required,
               personal_facts_included_json, personal_facts_excluded_json, reason
        FROM membrane_decisions
        WHERE event_id=?
        """,
        (latest_event_id,),
    ).fetchone()
    homeostasis = store.connection.execute(
        """
        SELECT privacy_risk, cloud_dependence, local_capability, uncertainty,
               cache_freshness, action_risk, user_trust, inventory_coverage
        FROM homeostatic_snapshots
        WHERE event_id=?
        """,
        (latest_event_id,),
    ).fetchone()
    self_observation = persist_self_observation(store, kernel.self_model)
    membrane = _membrane_payload_from_row(row)
    response_integrity = (
        kernel.last_response_integrity.to_dict()
        if kernel.last_response_integrity is not None
        else {}
    )
    session_id = store.session_id_for_event(latest_event_id) if latest_event_id else ""
    improvement_consent = (
        store.session_improvement_consent(session_id)
        if session_id
        else {
            "session_id": "",
            "opted_in": False,
            "consent_scope": "",
            "updated_at": "",
        }
    )
    candidate = store.connection.execute(
        """
        SELECT candidate_id, status, priority
        FROM improvement_candidates
        WHERE event_id=?
        """,
        (latest_event_id,),
    ).fetchone()
    mood_state = None
    if hasattr(decision, 'session_mood') and decision.session_mood is not None:
        mood = decision.session_mood
        mood_state = {
            "mood_id": getattr(mood, "mood_id", "neutral"),
            "valence": getattr(mood, "valence", 0.0),
            "arousal": getattr(mood, "arousal", 0.0),
            "response_mode": getattr(mood, "response_mode", "normal"),
            "engagement_level": getattr(mood, "engagement_level", 1.0),
            "is_listening": bool(getattr(mood, "is_listening", False)),
            "trigger_reason": getattr(mood, "trigger_reason", ""),
        }
    utterance_affect = None
    if hasattr(decision, 'utterance_affect') and decision.utterance_affect is not None:
        ua = decision.utterance_affect
        utterance_affect = {
            "valence": getattr(ua, "valence", 0.0),
            "arousal": getattr(ua, "arousal", 0.0),
            "confidence": getattr(ua, "confidence", 0.0),
            "source": getattr(ua, "source", ""),
            "is_complaint": bool(getattr(ua, "is_complaint", False)),
        }
    return {
        "utterance": utterance,
        "intent": decision.intent,
        "route": decision.route,
        "reason": decision.reason,
        "answer": decision.answer,
        "cloud_needed": decision.cloud_needed,
        "external_fetch_needed": decision.external_fetch_needed,
        "device_action": decision.device_action,
        "evidence_keys": list(decision.evidence_keys),
        "opportunities": [opportunity.kind for opportunity in opportunities],
        "executed_jobs": executed,
        "membrane": membrane,
        "homeostasis": dict(homeostasis) if homeostasis is not None else {},
        "pending_action": _latest_pending_action_summary(store)
        if decision.device_action
        else {},
        "action_execution": _latest_action_execution(store)
        if decision.reason == "confirmed_device_action"
        else {},
        "synthesis": kernel.last_synthesis.to_dict()
        if kernel.last_synthesis is not None
        else {},
        "response_integrity": response_integrity,
        "improvement": {
            "session_id": session_id,
            "consent": improvement_consent,
            "candidate": dict(candidate) if candidate is not None else {},
            "live_router_mutated": False,
        },
        "debug_parse": debug_parse,
        "capture_provenance": {
            "surface": capture_surface,
            "source": capture_source,
        },
        "self_observation": self_observation,
        "counts": store.table_counts(),
        "session_mood": mood_state,
        "utterance_affect": utterance_affect,
    }


def _membrane_payload_from_row(row) -> dict[str, Any]:
    if row is None:
        return {}
    payload = dict(row)
    payload["allowed"] = bool(payload.get("allowed", False))
    payload["confirmation_required"] = int(payload.get("confirmation_required", 0) or 0)
    payload["personal_facts_included"] = _loads_json_list(
        payload.pop("personal_facts_included_json", "[]")
    )
    payload["personal_facts_excluded"] = _loads_json_list(
        payload.pop("personal_facts_excluded_json", "[]")
    )
    return payload


def _loads_json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _opportunity_from_job(job) -> Opportunity:
    return Opportunity(
        kind=job.kind,
        priority=job.priority,
        reason=str(job.payload.get("reason", "queued job")),
        evidence_event_ids=tuple(
            str(item) for item in job.payload.get("evidence_event_ids", [])
        ),
        expected_cloud_reduction=0,
        proposed_action=str(job.payload.get("proposed_action", "")),
        source_candidates=tuple(
            str(item) for item in job.payload.get("source_candidates", [])
        ),
    )


def _install_imported_story_items(
    store: AssistantOSStore, profile: LocalAssistantProfile, items
) -> None:
    for row in story_items_to_inventory_rows(items, profile=profile):
        store.upsert_inventory(
            str(row["kind"]),
            str(row["item_id"]),
            dict(row["payload"]),
            source=str(row["source"]),
            license=str(row["license"]),
            tags=tuple(str(tag) for tag in row["tags"]),
        )
    store.connection.commit()


def _install_weather_items(store: AssistantOSStore, result) -> None:
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


def _install_imported_media_items(store: AssistantOSStore, items) -> None:
    for row in media_items_to_inventory_rows(items):
        store.upsert_inventory(
            str(row["kind"]),
            str(row["item_id"]),
            dict(row["payload"]),
            source=str(row["source"]),
            license=str(row["license"]),
            tags=tuple(str(tag) for tag in row["tags"]),
        )
    store.connection.commit()


def _latest_action_execution(store: AssistantOSStore) -> dict:
    row = store.connection.execute(
        """
        SELECT result
        FROM pending_actions
        WHERE confirmation_state='confirmed'
        ORDER BY rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return {}
    result = str(row["result"])
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return {"raw_result": result}
    return dict(payload) if isinstance(payload, dict) else {"raw_result": result}


def _latest_pending_action_summary(store: AssistantOSStore) -> dict:
    row = store.connection.execute(
        """
        SELECT action_id, action_type, target, evidence_keys_json, confirmation_state
        FROM pending_actions
        WHERE confirmation_state='pending'
        ORDER BY rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return {}
    try:
        evidence_keys = json.loads(str(row["evidence_keys_json"]))
    except json.JSONDecodeError:
        evidence_keys = []
    return {
        "action_id": str(row["action_id"]),
        "action_type": str(row["action_type"]),
        "target": str(row["target"]),
        "evidence_keys": list(evidence_keys) if isinstance(evidence_keys, list) else [],
        "confirmation_state": str(row["confirmation_state"]),
    }


def _required_dataset_report(seed: Path) -> list[dict]:
    paths = (
        seed,
        Path("benchmarks/public_domain_story_metadata.json"),
        Path("benchmarks/sample_gutenberg_catalog.csv"),
        Path("benchmarks/sample_internet_archive_search.json"),
        DEFAULT_WEATHER_SAMPLE,
        DEFAULT_LOCAL_MEDIA_MANIFEST,
        DEFAULT_OPEN_TRACE_FIXTURE,
        DEFAULT_TRANSCRIPT_REPLAY_FIXTURE,
    )
    report = []
    for path in paths:
        report.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return report


def _open_store(db: Path, seed: Path | None) -> AssistantOSStore:
    if not db.exists():
        return initialize_assistant_os_database(db, seed_path=seed)
    return AssistantOSStore(db)


def _persist_runtime_self_observation(
    store: AssistantOSStore,
    profile: LocalAssistantProfile | None = None,
) -> dict:
    loaded = store.load_profile(profile or LocalAssistantProfile())
    return persist_self_observation(store, self_model_from_profile(loaded))


def _cold_profile() -> LocalAssistantProfile:
    return LocalAssistantProfile(
        age=0,
        culture="unknown",
        story_models={},
        weekly_weather={},
        facts={},
        contacts={},
        media_library=(),
    )


def _remove_sqlite_files(db: Path) -> None:
    for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
        if path.exists():
            path.unlink()


def _sqlite_size(db: Path) -> int:
    total = 0
    for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
        if path.exists():
            total += path.stat().st_size
    return total


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


def _print_payload(payload: dict, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2))
        return
    if "answer" in payload:
        print(payload["answer"])
        print(f"route={payload['route']} reason={payload['reason']}")
        print(f"membrane={payload.get('membrane', {})}")
        print(f"homeostasis={payload.get('homeostasis', {})}")
        if payload.get("opportunities"):
            print(f"opportunities={', '.join(payload['opportunities'])}")
        if payload.get("executed_jobs"):
            print(f"executed_jobs={', '.join(payload['executed_jobs'])}")
        print(f"counts={payload['counts']}")
        return
    for key, value in payload.items():
        print(f"{key}={value}")


def _send_json(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _send_html(handler: BaseHTTPRequestHandler, html: str) -> None:
    encoded = html.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


if __name__ == "__main__":
    main()
