import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sqlite3
import sys
import tempfile
import unittest
import zipfile


CLI = Path("scripts/local_assistant_os_cli.py")


class LocalAssistantOSCliMvpTests(unittest.TestCase):
    def test_cli_init_and_ask_use_seeded_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            init = _run_cli("init", "--db", str(db), "--json")
            ask = _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--json")

            self.assertEqual(init["counts"]["events"], 0)
            self.assertGreaterEqual(init["counts"]["inventories"], 8)
            self.assertEqual(ask["route"], "local_answer")
            self.assertEqual(ask["reason"], "local_story_inventory")
            self.assertEqual(ask["counts"]["events"], 1)
            self.assertEqual(ask["counts"]["membrane_decisions"], 1)
            self.assertEqual(ask["counts"]["homeostatic_snapshots"], 1)
            self.assertEqual(ask["membrane"]["boundary_crossed"], "none")
            self.assertEqual(ask["debug_parse"]["chat_frame"]["intent"], "story")

    def test_cli_opt_in_turn_exposes_integrity_and_quarantined_improvement_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            ask = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Can you explain quasar algebra to my zorbulator?",
                "--improvement-opt-in",
                "--json",
            )
            queue = _run_cli("improvement-queue", "--db", str(db), "--json")

            self.assertEqual(ask["response_integrity"]["band"], "review")
            self.assertTrue(ask["response_integrity"]["research_recommended"])
            self.assertEqual(
                ask["response_integrity"]["research_topics"],
                ["quasar", "algebra", "zorbulator"],
            )
            self.assertTrue(ask["improvement"]["consent"]["opted_in"])
            self.assertEqual(ask["improvement"]["candidate"]["status"], "queued")
            self.assertFalse(ask["improvement"]["live_router_mutated"])
            self.assertEqual(queue["candidate_count"], 1)
            self.assertFalse(queue["candidates"][0]["cloud_export_allowed"])
            self.assertEqual(
                queue["policy"]["next_stage"],
                "redact_research_evaluate_before_promotion",
            )

    def test_cli_identity_and_parse_debug_use_self_model_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            identity = _run_cli("ask", "--db", str(db), "--utterance", "Who are you?", "--json")
            parsed = _run_cli("parse-debug", "--utterance", "What is your name?", "--json")

            self.assertEqual(identity["intent"], "assistant_identity")
            self.assertEqual(identity["route"], "local_answer")
            self.assertEqual(identity["reason"], "self_model_identity")
            self.assertIn("MELM Local Assistant OS", identity["answer"])
            self.assertEqual(identity["debug_parse"]["uol"]["object"], "self_model")
            self.assertEqual(identity["debug_parse"]["chat_frame"]["intent"], "assistant_identity")
            self.assertEqual(identity["debug_parse"]["chat_frame"]["domain"], "self_model")
            self.assertTrue(identity["debug_parse"]["chat_frame"]["can_answer_locally"])
            self.assertEqual(parsed["uol"]["action"], "name_awareness")
            self.assertEqual(parsed["uol"]["object"], "self_model")
            self.assertEqual(parsed["nlp"]["primary_parse_basis"], "uol_chat_frame")
            self.assertEqual(parsed["nlp"]["compositional_parse"]["pattern"], "what_copula_possessive_name")
            self.assertEqual(parsed["secondary_meaning_hints"], [])
            self.assertNotIn("vocabulary_hits", parsed)
            self.assertNotIn("bounded_vocabulary_hits", parsed["nlp"])
            self.assertNotIn("routing_basis", parsed["chat_frame"])
            self.assertEqual(
                parsed["uol"]["slot_sources"]["object"]["source"],
                "self_model_from_identity_composition:what_copula_possessive_name",
            )
            self.assertEqual(parsed["chat_frame"]["intent"], "assistant_identity")
            self.assertEqual(parsed["chat_frame"]["capabilities"]["local_sources"], ["self_model"])
            self.assertIn("assistant_identity", parsed["nlp"]["domain_hints"])
            self.assertIn("local_sources:self_model", parsed["chat_frame"]["primary_routing_basis"])
            self.assertIn("composition:what_copula_possessive_name", parsed["chat_frame"]["primary_routing_basis"])
            self.assertEqual(parsed["chat_frame"]["secondary_debug_hints"], [])
            self.assertNotIn("secondary_routing_hints", parsed["chat_frame"])
            self.assertNotIn("vocabulary_hits:your name", parsed["chat_frame"]["primary_routing_basis"])
            by_stage = {stage["stage"]: stage["output"] for stage in parsed["mapping"]}
            self.assertNotIn("routing_basis", by_stage["chat_frame"])
            self.assertIn("identity_should_be_local_self_model", parsed["notes"])

    def test_cli_self_status_uses_ledger_debug_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli("ask", "--db", str(db), "--utterance", "Who are you?", "--json")
            status = _run_cli("ask", "--db", str(db), "--utterance", "What have you done so far?", "--json")
            next_steps = _run_cli("parse-debug", "--utterance", "What do you need next?", "--json")

            self.assertEqual(status["intent"], "assistant_status")
            self.assertEqual(status["route"], "local_answer")
            self.assertEqual(status["reason"], "self_status_ledger_summary")
            self.assertIn("local ledger has 1 event(s)", status["answer"])
            self.assertEqual(status["debug_parse"]["uol"]["object"], "runtime_status")
            self.assertEqual(status["debug_parse"]["chat_frame"]["intent"], "assistant_status")
            self.assertIn("status_should_use_local_ledger", status["debug_parse"]["notes"])
            self.assertIn("self_status.counts", status["synthesis"]["citations"])
            self.assertEqual(next_steps["uol"]["object"], "next_steps")
            self.assertEqual(next_steps["chat_frame"]["intent"], "assistant_status")

    def test_cli_parse_debug_exposes_uol_chatframe_routing_basis_for_actions(self) -> None:
        parsed = _run_cli("parse-debug", "--utterance", "I need to talk to someone", "--json")

        self.assertEqual(parsed["nlp"]["bounded_intent"], "social_contact")
        self.assertEqual(parsed["nlp"]["domain_hints"]["social_contact"], ["talk", "someone"])
        self.assertEqual(parsed["uol"]["subject"], "user")
        self.assertEqual(parsed["uol"]["action"], "call")
        self.assertEqual(parsed["uol"]["object"], "someone")
        self.assertEqual(parsed["uol"]["target"], "trusted_contact")
        self.assertEqual(parsed["uol"]["slot_sources"]["action"]["source"], "contact_action_slots")
        self.assertEqual(parsed["uol"]["slot_sources"]["object"]["source"], "requested_contact_or_trusted_contact_slot")
        self.assertEqual(parsed["chat_frame"]["domain"], "trusted_contact_action")
        self.assertTrue(parsed["chat_frame"]["capabilities"]["device_action_possible"])
        self.assertTrue(parsed["chat_frame"]["capabilities"]["requires_confirmation"])
        self.assertIn("confirmation_gate:required_before_side_effect", parsed["chat_frame"]["primary_routing_basis"])
        self.assertNotIn("secondary_meaning_hints:talk to someone", parsed["chat_frame"]["primary_routing_basis"])
        self.assertEqual(parsed["chat_frame"]["secondary_debug_hints"], ["secondary_debug_hint:talk,someone"])
        self.assertNotIn("secondary_routing_hints", parsed["chat_frame"])
        by_stage = {stage["stage"]: stage["output"] for stage in parsed["mapping"]}
        self.assertEqual(by_stage["chat_frame"]["secondary_debug_hints"], ["secondary_debug_hint:talk,someone"])
        self.assertEqual(by_stage["uol_parse"]["slot_sources"]["target"]["source"], "device_action_target")
        self.assertTrue(by_stage["chat_frame"]["can_answer_locally"])

    def test_cli_shortcut_audit_reports_behavior_and_source_boundaries(self) -> None:
        report = _run_cli("shortcut-audit", "--json")

        self.assertEqual(report["schema"], "melm.local_assistant_shortcut_audit.v1")
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        by_label = {item["label"]: item for item in report["behavior_cases"]}
        self.assertEqual(by_label["weather_concept_not_cache"]["actual"]["intent"], "open_domain")
        self.assertEqual(by_label["weather_concept_not_cache"]["actual"]["route"], "local_answer")
        self.assertEqual(by_label["weather_observation_cache"]["actual"]["intent"], "weather")
        self.assertEqual(by_label["weather_observation_cache"]["actual"]["route"], "cached_tool")
        self.assertEqual(by_label["meal_you_cook_not_advice"]["actual"]["intent"], "open_domain")
        self.assertEqual(by_label["meal_user_choice_advice"]["actual"]["intent"], "meal_suggestion")
        self.assertEqual(by_label["kernel_paraphrased_latest_event_recall"]["actual"]["intent"], "autobiographical_memory")
        self.assertEqual(by_label["kernel_statement_not_memory_recall"]["actual"]["intent"], "unknown")
        self.assertFalse(any(item["secondary_hint_in_primary_route"] for item in report["behavior_cases"]))

        by_source_id = {item["id"]: item for item in report["source_checks"]}
        self.assertTrue(by_source_id["primary_classifier_no_secondary_helpers"]["passed"])
        self.assertTrue(by_source_id["primary_frame_registry_no_legacy_composition_helpers"]["passed"])
        self.assertEqual(
            by_source_id["primary_frame_registry_no_legacy_composition_helpers"]["forbidden_present"],
            [],
        )
        self.assertTrue(by_source_id["functional_grammar_no_transcript_phrase_table"]["passed"])
        self.assertEqual(by_source_id["functional_grammar_no_transcript_phrase_table"]["forbidden_present"], [])
        self.assertTrue(by_source_id["identity_composition_no_surface_phrase_table"]["passed"])
        self.assertEqual(by_source_id["identity_composition_no_surface_phrase_table"]["forbidden_present"], [])
        self.assertTrue(by_source_id["self_status_composition_no_surface_phrase_table"]["passed"])
        self.assertEqual(by_source_id["self_status_composition_no_surface_phrase_table"]["forbidden_present"], [])
        self.assertTrue(by_source_id["autobiographical_composition_no_exact_recall_phrases"]["passed"])
        self.assertEqual(by_source_id["autobiographical_composition_no_exact_recall_phrases"]["forbidden_present"], [])
        self.assertTrue(by_source_id["kernel_autobiographical_gate_uses_shared_frame"]["passed"])
        self.assertEqual(by_source_id["kernel_autobiographical_gate_uses_shared_frame"]["forbidden_present"], [])

    def test_cli_capability_probe_reports_mvp_can_and_cannot_handle_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "capability.sqlite"
            report = _run_cli("capability-probe", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["total_cases"], 18)
            self.assertEqual(
                report["bucket_counts"],
                {"blocked": 2, "device_action": 4, "local": 12},
            )
            self.assertEqual(
                report["route_counts"],
                {"cached_tool": 1, "device_action": 4, "local_answer": 11, "reject": 2},
            )
            self.assertEqual(report["local_device_rate"], 0.889)
            self.assertEqual(report["confirmation_cases"], ["media_request", "contact_request"])
            self.assertEqual(report["expected_mismatches"], [])
            self.assertGreaterEqual(report["complexity"]["avg"], 0.49)
            self.assertIn("medium", report["complexity"]["bands"])
            self.assertGreater(report["unknown_tokens"]["max"], 0)
            self.assertEqual(len(report["unsupported_examples"]), 2)
            by_label = {item["label"]: item for item in report["cases"]}
            self.assertEqual(by_label["story"]["route"], "local_answer")
            self.assertEqual(by_label["weather"]["route"], "cached_tool")
            self.assertEqual(by_label["media_request"]["confirmation_required"], 1)
            self.assertEqual(by_label["media_confirm"]["action_execution"]["status"], "prepared")
            self.assertFalse(by_label["media_confirm"]["action_execution"]["side_effect_executed"])
            self.assertEqual(by_label["contact_confirm"]["action_execution"]["resolved_target"], "+234-000-MOM")
            self.assertEqual(by_label["open_domain_science"]["bucket"], "local")
            self.assertTrue(by_label["open_domain_science"]["can_answer_locally"])
            self.assertEqual(by_label["private_cloud"]["bucket"], "blocked")
            self.assertIn("blocked_private_facts_to_cloud", by_label["private_cloud"]["reason"])
            self.assertTrue(all(item["mapping"] == ["basic_nlp", "uol_parse", "chat_frame"] for item in report["cases"]))
            self.assertTrue(
                all(
                    item["primary_domain_evidence"].get("source") == "no_local_composition"
                    or (
                        item["primary_domain_evidence"].get("frame_registry")
                        == "melm.assistant_frame_registry.v1"
                        and item["primary_domain_evidence"].get("frame_id")
                        and item["primary_domain_evidence"].get("source_policy")
                        == "primary_uol_chatframe_only"
                    )
                    for item in report["cases"]
                )
            )
            self.assertFalse(
                any(
                    any(part.startswith("secondary_meaning_hints:") for part in item["primary_routing_basis"])
                    for item in report["cases"]
                )
            )
            self.assertEqual(report["counts"]["events"], 18)
            self.assertEqual(report["counts"]["membrane_decisions"], 18)
            self.assertEqual(report["counts"]["homeostatic_snapshots"], 18)

    def test_cli_dataset_audit_validates_seed_fixtures_and_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "dataset_audit.sqlite"
            report = _run_cli("dataset-audit", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(len(report["checks"]), 19)
            self.assertEqual(report["runtime"], "stdlib_python_sqlite_json_csv")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(set(report["files"]), {
                "seed",
                "story_metadata",
                "media_manifest",
                "weather_fixture",
                "gutenberg_catalog",
                "internet_archive_search",
                "open_traces",
                "transcript_replay",
            })
            self.assertTrue(all(item["sha256"] for item in report["files"].values()))
            self.assertEqual(report["seed"]["facts"], 6)
            self.assertEqual(report["seed"]["inventories"], 10)
            self.assertEqual(report["source_fixtures"]["story_metadata_items"], 4)
            self.assertEqual(report["source_fixtures"]["media_items"], 3)
            self.assertEqual(report["source_fixtures"]["weather_days"], 7)
            self.assertEqual(report["source_fixtures"]["open_trace_turns"], 29)
            self.assertEqual(report["source_fixtures"]["transcript_replay_user_turns"], 25)
            self.assertGreaterEqual(report["source_fixtures"]["gutenberg_story_candidates"], 2)
            self.assertGreaterEqual(report["source_fixtures"]["internet_archive_story_candidates"], 2)
            self.assertEqual(report["bootstrap"]["counts"]["user_facts"], 6)
            self.assertEqual(report["bootstrap"]["counts"]["inventories"], 10)
            self.assertEqual(report["bootstrap"]["profile"]["user_name"], "Maya")
            self.assertEqual(report["bootstrap"]["profile"]["age"], 7)
            self.assertEqual(report["bootstrap"]["profile"]["location"], "Lagos")
            self.assertEqual(report["bootstrap"]["profile"]["culture"], "Yoruba")
            self.assertEqual(report["bootstrap"]["profile"]["contacts"], ["mom"])
            self.assertIn("calm piano", report["bootstrap"]["profile"]["media_library"])

    def test_cli_v01_audit_reports_runnable_core_and_remaining_real_world_blockers(self) -> None:
        report = _run_cli("v01-audit", "--json")

        failed_checks = [k for k, v in report.get("checks", {}).items() if not v]
        failed_reqs = [c["id"] for c in report.get("core_requirements", []) if c["status"] != "met"]
        self.assertTrue(report["passed"], f"v01-audit failed: checks={failed_checks}, reqs={failed_reqs}")
        self.assertTrue(report["core_browser_cli_ready"])
        self.assertFalse(report["architecture_complete"])
        self.assertEqual(report["status"], "browser_cli_ready_with_real_world_blockers")
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["checks"]["shortcut_audit_passed"])
        self.assertTrue(report["shortcut_audit_summary"]["passed"])
        self.assertEqual(report["runtime"], "stdlib_python_sqlite_http_html_audit")
        self.assertEqual(report["dependency_class"], "stdlib_only")
        by_id = {item["id"]: item for item in report["core_requirements"]}
        self.assertEqual(by_id["authoritative_plan"]["status"], "met")
        self.assertEqual(by_id["drift_rule"]["status"], "met")
        self.assertEqual(by_id["uol_chatframe_static_shortcut_guard"]["status"], "met")
        self.assertIn(
            "tests/test_local_assistant_router_mvp.py",
            by_id["uol_chatframe_static_shortcut_guard"]["evidence"][0].replace("\\", "/"),
        )
        self.assertIn("shortcut-audit --json", by_id["uol_chatframe_static_shortcut_guard"]["evidence"][2])
        self.assertEqual(by_id["required_seed_and_source_datasets"]["status"], "met")
        self.assertEqual(by_id["required_seed_and_source_datasets"]["missing"], [])
        self.assertEqual(by_id["setup_gap_to_memory_action_loop"]["status"], "met")
        self.assertIn("setup-integration-smoke", by_id["setup_gap_to_memory_action_loop"]["evidence"][0])
        blocker_ids = {item["id"] for item in report["completion_blockers"]}
        self.assertEqual(
            blocker_ids,
            {
                "user_derived_bounded_synthesis_traces",
                "longer_live_inventory_soak",
                "planner_priority_on_user_derived_traces",
                "real_user_derived_lifecycle_traces",
                "digest_quality_and_route_threshold_calibration",
                "configured_target_device_apps",
            },
        )
        self.assertEqual(report["blocker_count"], 6)
        self.assertTrue(all(item["status"] == "remaining_blocker" for item in report["completion_blockers"]))
        self.assertIn("target-report --reset --json", report["command_evidence"]["target_report"])
        self.assertIn("shortcut-audit --json", report["command_evidence"]["shortcut_audit"])
        self.assertIn("v01-acceptance --reset --json", report["command_evidence"]["v01_acceptance"])
        self.assertIn("v01-progress --json", report["command_evidence"]["v01_progress"])
        self.assertIn("inventory-soak-matrix --reset --json", report["command_evidence"]["inventory_soak_matrix"])
        self.assertIn("inventory-soak-matrix --live", report["command_evidence"]["live_inventory_soak_matrix"])
        self.assertIn("--require-configured", report["command_evidence"]["host_app_configured"])
        self.assertIn("export-transcript-replay", report["command_evidence"]["event_transcript_export"])
        self.assertIn("calibrate-event-ledger", report["command_evidence"]["event_ledger_calibration"])
        self.assertIn("candidate-session-audit", report["command_evidence"]["candidate_session_audit"])
        self.assertIn("--session <session|all>", report["command_evidence"]["candidate_session_audit"])
        self.assertIn("write-source-attestation", report["command_evidence"]["source_attestation"])
        self.assertIn("--event-ledger-db", report["command_evidence"]["source_attestation"])
        self.assertIn("--event-ledger-session", report["command_evidence"]["source_attestation"])
        self.assertIn("--human-reviewed", report["command_evidence"]["source_attestation"])
        self.assertIn("write-host-app-attestation", report["command_evidence"]["host_app_attestation"])
        self.assertIn("--host-app-config-json", report["command_evidence"]["host_app_attestation"])
        self.assertIn("--not-demo-recorder", report["command_evidence"]["host_app_attestation"])
        self.assertIn("v01-blocker-evidence", report["command_evidence"]["blocker_evidence"])
        self.assertIn("--event-ledger-session", report["command_evidence"]["blocker_evidence"])
        self.assertIn("--source-attestation-json", report["command_evidence"]["blocker_evidence"])
        self.assertIn("--host-app-attestation-json", report["command_evidence"]["blocker_evidence"])
        self.assertIn("--auto-lifecycle", report["command_evidence"]["blocker_evidence"])
        self.assertIn("redacted_user_session", report["command_evidence"]["blocker_evidence"])
        self.assertIn("--inventory-soak-report-json", report["command_evidence"]["blocker_evidence"])
        self.assertIn("--run-host-app-probe", report["command_evidence"]["blocker_evidence"])
        self.assertIn(
            "config/safe_lifecycle_controls.example.json",
            report["command_evidence"]["event_ledger_calibration"],
        )
        self.assertIn("calibrate-transcript-replay", report["command_evidence"]["calibration"])
        self.assertIn("--controls-json", report["command_evidence"]["calibration"])
        self.assertIn("config/safe_lifecycle_controls.example.json", report["command_evidence"]["calibration"])
        self.assertIn("--min-synthesis-traces", report["command_evidence"]["calibration_synthesis"])
        self.assertIn("--controls-json", report["command_evidence"]["calibration_synthesis"])
        self.assertIn("--require-priority-signals", report["command_evidence"]["calibration_planner"])
        self.assertIn("--controls-json", report["command_evidence"]["calibration_planner"])
        self.assertIn("--require-memory-digest-quality", report["command_evidence"]["calibration_digest_baseline"])
        self.assertIn("--require-strict-baseline-win", report["command_evidence"]["calibration_digest_baseline"])
        self.assertIn("--controls-json", report["command_evidence"]["calibration_digest_baseline"])
        blockers = {item["id"]: item for item in report["completion_blockers"]}
        self.assertIn("v01-blocker-evidence", blockers["user_derived_bounded_synthesis_traces"]["next_command"])
        self.assertIn("--event-source-kind redacted_user_session", blockers["user_derived_bounded_synthesis_traces"]["next_command"])
        self.assertIn("v01-blocker-evidence", blockers["planner_priority_on_user_derived_traces"]["next_command"])
        self.assertIn("--auto-lifecycle", blockers["planner_priority_on_user_derived_traces"]["next_command"])
        self.assertIn("v01-blocker-evidence", blockers["real_user_derived_lifecycle_traces"]["next_command"])
        self.assertIn("--event-source-kind redacted_user_session", blockers["real_user_derived_lifecycle_traces"]["next_command"])
        self.assertIn(
            "v01-blocker-evidence",
            blockers["digest_quality_and_route_threshold_calibration"]["next_command"],
        )
        self.assertIn(
            "--inventory-soak-report-json",
            blockers["digest_quality_and_route_threshold_calibration"]["next_command"],
        )
        self.assertIn("--run-host-app-probe", blockers["configured_target_device_apps"]["next_command"])
        self.assertIn("test_local_assistant_router_mvp", report["command_evidence"]["anti_static_router_tests"])

    def test_cli_v01_progress_summarizes_current_state_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocker_report_path = Path(tmp) / "v01_blockers.json"
            blocker_report = _run_cli("v01-blocker-evidence", "--json")
            blocker_report_path.write_text(json.dumps(blocker_report), encoding="utf-8")

            default_progress = _run_cli("v01-progress", "--json")
            loaded_progress = _run_cli(
                "v01-progress",
                "--blocker-evidence-json",
                str(blocker_report_path),
                "--json",
            )

            for report in (default_progress, loaded_progress):
                if not report["passed"]:
                    failed_checks = [k for k, v in report.get("checks", {}).items() if not v]
                    failed_reqs = [c["id"] for c in report.get("core_requirements", []) if c["status"] != "met"]
                    self.fail(f"v01-audit failed: checks={failed_checks}, reqs={failed_reqs}")
                self.assertFalse(report["architecture_complete"])
                self.assertFalse(report["architecture_complete_claimed"])
                self.assertFalse(report["candidate_review_ready"])
                self.assertEqual(report["status"], "browser_cli_ready_with_real_world_blockers")
                self.assertTrue(report["core"]["browser_cli_ready"])
                self.assertEqual(report["blockers"]["blocker_count"], 6)
                self.assertEqual(report["blockers"]["candidate_blockers_satisfied"], 0)
                self.assertEqual(report["blockers"]["remaining_blocker_count"], 6)
                self.assertEqual(len(report["blockers"]["missing_blockers"]), 6)
                self.assertTrue(report["completion_boundary"]["must_not_claim_complete_from_development_or_fixture_evidence"])
                self.assertFalse(report["completion_boundary"]["v01_audit_architecture_complete"])
                self.assertFalse(report["completion_boundary"]["v01_blocker_evidence_architecture_complete_claimed"])
                self.assertTrue(any("v01-blocker-evidence" in command for command in report["next_commands"]))
                self.assertIn("candidate-session-audit", report["evidence_commands"]["candidate_session_audit"])
                self.assertIn("write-source-attestation", report["evidence_commands"]["source_attestation"])
                self.assertIn("--event-ledger-session", report["evidence_commands"]["source_attestation"])
                self.assertIn("write-host-app-attestation", report["evidence_commands"]["host_app_attestation"])
                self.assertIn("--event-ledger-session", report["evidence_commands"]["blocker_evidence"])
                self.assertIn("--host-app-attestation-json", report["evidence_commands"]["blocker_evidence"])

            self.assertEqual(default_progress["blocker_evidence_source"], "v01-audit_completion_blockers")
            self.assertEqual(loaded_progress["blocker_evidence_source"], str(blocker_report_path))

    def test_cli_bootstrap_runtime_creates_usable_local_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            report = _run_cli("bootstrap-runtime", "--db", str(db), "--reset", "--json",
                             _env={"MELM_BULK_MAX_ENTRIES": "2000"})
            dashboard = _run_cli("dashboard", "--db", str(db), "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(db.exists())
            self.assertEqual(report["runtime"], "stdlib_python_sqlite")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(report["media_import"]["imported_items"], 3)
            self.assertGreaterEqual(report["initial_counts"]["inventories"], 10)
            self.assertEqual(report["counts"]["events"], 3)
            self.assertEqual(report["counts"]["membrane_decisions"], 3)
            self.assertEqual(report["counts"]["homeostatic_snapshots"], 3)
            by_label = {turn["label"]: turn for turn in report["turns"]}
            self.assertEqual(by_label["story"]["route"], "local_answer")
            self.assertEqual(by_label["story"]["reason"], "local_story_inventory")
            self.assertEqual(by_label["weather"]["route"], "cached_tool")
            self.assertEqual(by_label["weather"]["reason"], "weather_cache_hit")
            self.assertEqual(by_label["safety"]["reason"], "local_common_sense_policy")
            self.assertIn("serve --db", report["next_commands"]["serve_local_api"])
            self.assertEqual(dashboard["counts"]["events"], 3)
            self.assertEqual(dashboard["safety_flags"]["unconfirmed_executed_actions"], 0)

    def test_cli_lifecycle_runs_cold_start_into_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            report = _run_cli("run-lifecycle", "--db", str(db), "--reset", "--json")
            conn = sqlite3.connect(db)
            try:
                cloud_rows = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE route='cloud_handoff'"
                ).fetchone()[0]
            finally:
                conn.close()

            self.assertEqual(report["steps"], 17)
            self.assertEqual(report["cloud_handoffs"], 3)
            self.assertEqual(report["external_fetches"], 1)
            self.assertEqual(report["blocked_offline"], 1)
            self.assertEqual(report["counts"]["events"], 17)
            self.assertEqual(report["counts"]["membrane_decisions"], 17)
            self.assertEqual(report["counts"]["homeostatic_snapshots"], 17)
            self.assertIn("build_story_inventory", report["jobs_executed"])
            self.assertEqual(cloud_rows, report["cloud_handoffs"])

    def test_cli_lifecycle_suite_emits_multi_profile_json(self) -> None:
        report = _run_cli("run-lifecycle-suite", "--json")

        self.assertEqual(report["scenarios"], 3)
        self.assertEqual(report["steps"], 34)
        self.assertEqual(report["safety_flags"]["cloud_private_inclusions"], 0)
        self.assertEqual(report["safety_flags"]["unconfirmed_executed_actions"], 0)
        self.assertIn("build_media_index", report["opportunities_by_kind"])
        self.assertIn("ask_household_memory", report["opportunities_by_kind"])
        self.assertEqual(len(report["scenario_reports"]), 3)

    def test_cli_household_week_emits_longer_lifecycle_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "household_week.sqlite"
            report = _run_cli("run-household-week", "--db", str(db), "--reset", "--json")

            self.assertEqual(report["steps"], 37)
            self.assertGreaterEqual(report["local_resolution_rate"], 0.64)
            self.assertTrue(all(report["architecture_checks"].values()))
            self.assertEqual(report["safety_flags"]["cloud_private_inclusions"], 0)
            self.assertEqual(report["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["safety_flags"]["dangling_memory_links"], 0)
            self.assertGreaterEqual(report["digest"]["session_count"], 6)
            self.assertGreaterEqual(report["digest"]["event_count"], 20)
            self.assertTrue(report["digest"]["quality"]["passed"])
            self.assertGreaterEqual(report["digest"]["quality"]["score"], report["digest"]["quality"]["floor"])
            digest_threads = {item["thread"] for item in report["digest"]["threads"]}
            self.assertIn("story_inventory", digest_threads)
            self.assertIn("media_playback", digest_threads)
            self.assertIn("boundary_control", digest_threads)
            self.assertIn(
                "story requests moved from cloud handoff to local story inventory",
                report["digest"]["capability_transitions"],
            )
            self.assertIn("private conversation and user memory stayed local-only", report["digest"]["active_limits"])
            self.assertIn("main threads:", report["digest"]["summary"])
            self.assertIn("build_media_index", report["opportunities_by_kind"])
            self.assertIn("request_trusted_contact", report["opportunities_by_kind"])

    def test_cli_confirmation_executes_persisted_pending_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            first = _run_cli("ask", "--db", str(db), "--utterance", "I need to talk to someone.", "--json")
            second = _run_cli("ask", "--db", str(db), "--utterance", "Yes, call mom.", "--json")
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT confirmation_state, executed FROM pending_actions"
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual(first["route"], "device_action")
            self.assertEqual(first["membrane"]["confirmation_required"], 1)
            self.assertEqual(second["reason"], "confirmed_device_action")
            self.assertEqual(second["membrane"]["confirmation_required"], 0)
            self.assertEqual(row[0], "confirmed")
            self.assertEqual(row[1], 1)
            self.assertEqual(second["action_execution"]["action_type"], "call_contact")
            self.assertEqual(second["action_execution"]["mode"], "dry-run")
            self.assertEqual(second["action_execution"]["status"], "prepared")
            self.assertFalse(second["action_execution"]["side_effect_executed"])

    def test_cli_confirmed_media_action_uses_dry_run_device_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli(
                "import-media",
                "--db",
                str(db),
                "--manifest",
                "benchmarks/local_media_manifest.json",
                "--cold-start",
                "--json",
            )
            first = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Play calm piano.",
                "--cold-start",
                "--json",
            )
            confirm = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Yes, play calm piano.",
                "--cold-start",
                "--json",
            )
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT confirmation_state, executed, result FROM pending_actions"
                ).fetchone()
            finally:
                conn.close()
            result = json.loads(row[2])

            self.assertEqual(first["route"], "device_action")
            self.assertEqual(first["membrane"]["confirmation_required"], 1)
            self.assertEqual(confirm["reason"], "confirmed_device_action")
            self.assertEqual(confirm["action_execution"]["action_type"], "play_media")
            self.assertEqual(confirm["action_execution"]["payload"]["item_id"], "calm piano")
            self.assertEqual(confirm["action_execution"]["payload"]["path"], "media/calm_piano.mp3")
            resolved_path = confirm["action_execution"]["payload"]["resolved_path"].replace("\\", "/")
            self.assertTrue(resolved_path.endswith("benchmarks/media/calm_piano.mp3"))
            self.assertEqual(confirm["action_execution"]["mode"], "dry-run")
            self.assertEqual(confirm["action_execution"]["status"], "prepared")
            self.assertFalse(confirm["action_execution"]["side_effect_executed"])
            self.assertEqual(row[0], "confirmed")
            self.assertEqual(row[1], 1)
            self.assertEqual(result["payload"]["item_id"], "calm piano")

    def test_cli_real_media_action_blocks_without_configured_player_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli(
                "import-media",
                "--db",
                str(db),
                "--manifest",
                "benchmarks/local_media_manifest.json",
                "--cold-start",
                "--json",
            )
            _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Play calm piano.",
                "--cold-start",
                "--json",
            )
            confirm = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Yes, play calm piano.",
                "--cold-start",
                "--action-mode",
                "real",
                "--json",
            )
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT confirmation_state, executed, result FROM pending_actions"
                ).fetchone()
            finally:
                conn.close()
            result = json.loads(row[2])

            self.assertEqual(confirm["reason"], "confirmed_device_action")
            self.assertEqual(confirm["action_execution"]["mode"], "real")
            self.assertEqual(confirm["action_execution"]["status"], "blocked")
            self.assertEqual(confirm["action_execution"]["reason"], "missing_media_player_command")
            self.assertFalse(confirm["action_execution"]["side_effect_executed"])
            self.assertEqual(row[0], "confirmed")
            self.assertEqual(row[1], 0)
            self.assertEqual(result["reason"], "missing_media_player_command")

    def test_cli_action_smoke_prepares_media_and_call_in_dry_run_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "action_smoke.sqlite"
            report = _run_cli("action-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertEqual(report["mode"], "dry-run")
            self.assertEqual(report["expected_status"], "prepared")
            self.assertEqual([item["status"] for item in report["action_results"]], ["prepared", "prepared"])
            self.assertEqual(
                [item["action_type"] for item in report["action_results"]],
                ["play_media", "call_contact"],
            )
            self.assertEqual(report["action_results"][1]["resolved_target"], "+234-000-ADA")
            self.assertFalse(any(item["side_effect_executed"] for item in report["action_results"]))
            self.assertEqual(report["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["safety_flags"]["action_without_confirmation_gate"], 0)
            self.assertEqual(report["counts"]["pending_actions"], 2)

    def test_cli_setup_integration_smoke_proves_gaps_then_user_supplied_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "setup_integration.sqlite"
            report = _run_cli("setup-integration-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(
                set(report["setup_requests_after_gaps"]),
                {"routine_memory", "household_memory", "trusted_contact"},
            )
            self.assertTrue(
                all(
                    item["requires_user_supplied_value"]
                    for item in report["setup_requests_after_gaps"].values()
                )
            )
            self.assertEqual(report["facts_after_gaps"], {})
            self.assertEqual(report["contact_inventory_after_gaps"], {})
            self.assertEqual(report["facts_after_setup"]["morning_routine"], "stretch, breakfast, then bus")
            self.assertIn("Maya and Mom", report["facts_after_setup"]["household_context"])
            self.assertEqual(report["contacts_after_setup"]["ada"], "+234-000-ADA")
            self.assertEqual(report["fact_privacy"]["facts.morning_routine"]["scope"], "routine_local")
            self.assertEqual(report["fact_privacy"]["facts.household_context"]["scope"], "household_local")
            self.assertTrue(report["fact_privacy"]["facts.morning_routine"]["local_only"])
            self.assertTrue(report["fact_privacy"]["facts.household_context"]["local_only"])

            by_label = {turn["label"]: turn for turn in report["turns"]}
            self.assertEqual(by_label["routine_gap"]["route"], "clarify")
            self.assertEqual(by_label["routine_gap"]["uol"]["object"], "routine_memory")
            self.assertEqual(by_label["household_gap"]["route"], "clarify")
            self.assertEqual(by_label["household_gap"]["uol"]["object"], "household_memory")
            self.assertEqual(by_label["contact_gap"]["route"], "clarify")
            self.assertEqual(by_label["contact_gap"]["uol"]["object"], "someone")
            self.assertEqual(by_label["routine_still_empty"]["route"], "clarify")
            self.assertEqual(by_label["routine_setup"]["reason"], "consented_routine_memory_stored")
            self.assertEqual(by_label["routine_setup"]["evidence_keys"], ["facts.morning_routine"])
            self.assertEqual(by_label["household_setup"]["reason"], "consented_household_memory_stored")
            self.assertEqual(by_label["household_setup"]["uol"]["object"], "household_memory")
            self.assertEqual(by_label["contact_setup"]["reason"], "consented_trusted_contact_stored")
            self.assertEqual(by_label["routine_recall"]["route"], "local_answer")
            self.assertEqual(by_label["routine_recall"]["reason"], "personal_memory_recall")
            self.assertEqual(by_label["household_recall"]["route"], "local_answer")
            self.assertEqual(by_label["contact_request"]["route"], "device_action")
            self.assertEqual(by_label["contact_request"]["confirmation_required"], 1)
            self.assertEqual(by_label["contact_request"]["pending_action"]["action_type"], "call_contact")
            self.assertEqual(by_label["contact_confirm"]["reason"], "confirmed_device_action")
            self.assertEqual(by_label["contact_confirm"]["action_execution"]["status"], "prepared")
            self.assertEqual(by_label["contact_confirm"]["action_execution"]["resolved_target"], "+234-000-ADA")
            self.assertFalse(by_label["contact_confirm"]["action_execution"]["side_effect_executed"])
            self.assertTrue(
                all(
                    turn["chat_frame"]["primary_routing_basis"][0].startswith("bounded_intent:")
                    for turn in report["turns"]
                )
            )
            self.assertEqual(report["pending_actions"]["pending"], 0)
            self.assertGreaterEqual(report["pending_actions"]["confirmed"], 1)
            self.assertEqual(report["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["safety_flags"]["action_without_confirmation_gate"], 0)

    def test_cli_action_smoke_executes_configured_real_media_and_call_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "action_smoke.sqlite"
            media_dir = root / "media"
            media_dir.mkdir()
            (media_dir / "Calm Piano.mp3").write_bytes(b"local smoke media")
            log = root / "actions.jsonl"

            report = _run_cli(
                "action-smoke",
                "--db",
                str(db),
                "--reset",
                "--action-mode",
                "real",
                "--media-dir",
                str(media_dir),
                "--media-player-command",
                _action_recorder_command(log, "media"),
                "--call-command",
                _action_recorder_command(log, "call"),
                "--json",
            )
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

            self.assertTrue(report["passed"])
            self.assertEqual(report["mode"], "real")
            self.assertEqual(report["expected_status"], "executed")
            self.assertEqual([item["status"] for item in report["action_results"]], ["executed", "executed"])
            self.assertEqual(
                [item["action_type"] for item in report["action_results"]],
                ["play_media", "call_contact"],
            )
            self.assertTrue(all(item["side_effect_executed"] for item in report["action_results"]))
            self.assertEqual([row["label"] for row in rows], ["media", "call"])
            self.assertTrue(rows[0]["target_exists"])
            self.assertFalse(rows[1]["target_exists"])
            self.assertEqual(rows[1]["target"], "+234-000-ADA")
            self.assertTrue(report["action_results"][0]["payload"]["path_exists"])
            self.assertEqual(report["action_results"][1]["resolved_target"], "+234-000-ADA")
            self.assertEqual(report["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["safety_flags"]["action_without_confirmation_gate"], 0)

    def test_cli_host_action_smoke_runs_real_command_mode_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "host_action.sqlite"
            work_dir = root / "host_action"
            report = _run_cli(
                "host-action-smoke",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite_subprocess")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(report["action_smoke"]["mode"], "real")
            self.assertEqual([item["status"] for item in report["action_smoke"]["action_results"]], ["executed", "executed"])
            self.assertEqual([record["label"] for record in report["records"]], ["media", "call"])
            self.assertTrue(report["records"][0]["target_exists"])
            self.assertEqual(report["records"][1]["target"], "+234-000-ADA")
            self.assertEqual(report["action_smoke"]["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["action_smoke"]["safety_flags"]["action_without_confirmation_gate"], 0)

    def test_cli_host_app_probe_reports_unconfigured_app_commands_without_pretending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "host_app.sqlite"
            work_dir = root / "host_app"
            report = _run_cli(
                "host-app-probe",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertFalse(report["configured"])
            self.assertTrue(report["skipped"])
            self.assertTrue(report["checks"]["configuration_reported"])
            self.assertFalse(report["checks"]["media_command_configured"])
            self.assertFalse(report["checks"]["call_command_configured"])
            self.assertFalse(report["checks"]["probe_executed"])
            self.assertTrue(report["checks"]["require_configured_satisfied"])
            self.assertIn("--require-configured", " ".join(report["next_steps"]))

            strict = _run_cli(
                "host-app-probe",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--reset",
                "--require-configured",
                "--json",
            )
            self.assertFalse(strict["passed"])
            self.assertFalse(strict["configured"])
            self.assertFalse(strict["checks"]["require_configured_satisfied"])

    def test_cli_host_app_probe_executes_configured_commands_through_typed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "host_app.sqlite"
            work_dir = root / "host_app"
            log = root / "host_app_actions.jsonl"
            report = _run_cli(
                "host-app-probe",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--media-player-command",
                _action_recorder_command(log, "media"),
                "--call-command",
                _action_recorder_command(log, "call"),
                "--reset",
                "--json",
            )
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

            self.assertTrue(report["passed"])
            self.assertTrue(report["configured"])
            self.assertFalse(report["skipped"])
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(report["evidence_class"]["demo_recorder_detected"])
            self.assertFalse(report["evidence_class"]["candidate_target_device_app_evidence"])
            self.assertEqual(report["action_smoke"]["mode"], "real")
            self.assertEqual([item["status"] for item in report["action_results"]], ["executed", "executed"])
            self.assertEqual([item["action_type"] for item in report["action_results"]], ["play_media", "call_contact"])
            self.assertEqual([row["label"] for row in rows], ["media", "call"])
            self.assertTrue(rows[0]["target_exists"])
            self.assertEqual(rows[1]["target"], "+234-000-ADA")
            self.assertEqual(report["action_results"][1]["resolved_target"], "+234-000-ADA")
            self.assertEqual(report["action_smoke"]["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["action_smoke"]["safety_flags"]["action_without_confirmation_gate"], 0)

    def test_cli_host_app_probe_reads_config_json_through_typed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "host_app.sqlite"
            work_dir = root / "host_app"
            log = root / "host_app_actions.jsonl"
            config = root / "host_actions.json"
            config.write_text(
                json.dumps(
                    {
                        "media_player_command": _action_recorder_command(log, "media"),
                        "call_command": _action_recorder_command(log, "call"),
                    }
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "host-app-probe",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--config-json",
                str(config),
                "--require-configured",
                "--reset",
                "--json",
            )
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

            self.assertTrue(report["passed"])
            self.assertTrue(report["configured"])
            self.assertFalse(report["skipped"])
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(report["config"]["loaded"])
            self.assertTrue(report["evidence_class"]["demo_recorder_detected"])
            self.assertFalse(report["evidence_class"]["candidate_target_device_app_evidence"])
            self.assertEqual(report["config"]["error"], "")
            self.assertEqual(
                set(report["config"]["keys"]),
                {"call_command", "media_player_command"},
            )
            self.assertIn("config:", report["command_sources"]["media"])
            self.assertIn("config:", report["command_sources"]["call"])
            self.assertEqual([row["label"] for row in rows], ["media", "call"])
            self.assertTrue(rows[0]["target_exists"])
            self.assertEqual(rows[1]["target"], "+234-000-ADA")

    def test_cli_write_host_actions_demo_config_proves_configured_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config" / "host_actions.local_recorder.json"
            log = root / "host_actions.local_recorder.jsonl"
            db = root / "host_app.sqlite"
            work_dir = root / "host_app"

            generated = _run_cli(
                "write-host-actions-demo-config",
                "--out",
                str(config),
                "--log",
                str(log),
                "--overwrite",
                "--json",
            )
            payload = json.loads(config.read_text(encoding="utf-8"))
            report = _run_cli(
                "host-app-probe",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--config-json",
                str(config),
                "--require-configured",
                "--reset",
                "--json",
            )
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

            self.assertTrue(generated["passed"])
            self.assertEqual(generated["out"], str(config))
            self.assertIn("host-action-recorder", payload["media_player_command"])
            self.assertIn("host-action-recorder", payload["call_command"])
            self.assertTrue(report["passed"])
            self.assertTrue(report["configured"])
            self.assertFalse(report["skipped"])
            self.assertTrue(report["checks"]["typed_confirmation_gate_used"])
            self.assertTrue(report["evidence_class"]["demo_recorder_detected"])
            self.assertEqual(report["evidence_class"]["evidence_kind"], "development_recorder_rehearsal")
            self.assertEqual([row["label"] for row in rows], ["media", "call"])
            self.assertTrue(rows[0]["target_exists"])
            self.assertEqual(rows[1]["target"], "+234-000-ADA")

    def test_cli_pi_smoke_runs_compact_readiness_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "pi_smoke.sqlite"
            report = _run_cli("pi-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertTrue(all(item["exists"] for item in report["datasets"]))
            self.assertTrue(report["checks"]["dataset_audit_passed"])
            self.assertTrue(report["dataset_audit"]["passed"])
            self.assertTrue(all(report["dataset_audit"]["checks"].values()))
            self.assertEqual(report["dataset_audit"]["source_fixtures"]["open_trace_turns"], 29)
            self.assertEqual(report["dataset_audit"]["source_fixtures"]["transcript_replay_user_turns"], 25)
            self.assertEqual(report["dataset_audit"]["source_fixtures"]["transcript_replay_user_turns"], 25)
            self.assertGreater(report["dataset_audit_db_bytes"], 0)
            self.assertEqual(report["ask"]["route"], "local_answer")
            self.assertEqual(report["ask"]["reason"], "local_story_inventory")
            self.assertTrue(report["ask"]["synthesis_applied"])
            self.assertEqual(report["lifecycle"]["steps"], 17)
            self.assertEqual(report["lifecycle"]["actions_executed"], 1)
            self.assertEqual(report["lifecycle"]["story_route_after_inventory"], "local_answer")
            self.assertTrue(report["action_smoke"]["passed"])
            self.assertEqual(
                [item["action_type"] for item in report["action_smoke"]["action_results"]],
                ["play_media", "call_contact"],
            )
            self.assertTrue(report["action_smoke"]["action_results"][0]["payload"]["path_exists"])
            self.assertEqual(report["action_smoke"]["action_results"][1]["resolved_target"], "+234-000-ADA")
            self.assertEqual(report["action_smoke"]["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertTrue(report["checks"]["setup_integration_smoke_passed"])
            self.assertTrue(report["setup_integration_smoke"]["passed"])
            self.assertTrue(all(report["setup_integration_smoke"]["checks"].values()))
            self.assertEqual(
                set(report["setup_integration_smoke"]["setup_requests_after_gaps"]),
                {"routine_memory", "household_memory", "trusted_contact"},
            )
            self.assertEqual(
                report["setup_integration_smoke"]["facts_after_setup"]["morning_routine"],
                "stretch, breakfast, then bus",
            )
            self.assertEqual(report["setup_integration_smoke"]["contacts_after_setup"]["ada"], "+234-000-ADA")
            self.assertEqual(report["setup_integration_smoke"]["action_execution"]["resolved_target"], "+234-000-ADA")
            self.assertTrue(report["open_traces"]["passed"])
            self.assertEqual(report["open_traces"]["turns"], 29)
            self.assertTrue(report["open_traces"]["debug_checks"]["debug_maps_present"])
            self.assertTrue(report["open_traces"]["debug_checks"]["identity_maps_to_self_model"])
            self.assertTrue(report["checks"]["open_trace_debug_gate_passed"])
            self.assertTrue(report["transcript_replay"]["passed"])
            self.assertEqual(report["transcript_replay"]["turns"], 25)
            self.assertTrue(report["transcript_replay"]["fixture_checks"]["no_static_answer_or_route_expectations"])
            self.assertTrue(report["transcript_replay"]["baseline_comparison"]["passed"])
            self.assertEqual(
                report["transcript_replay"]["baseline_comparison"]["wins"]["local_resolution_rate_gain_vs_best_baseline"],
                0.16,
            )
            self.assertTrue(report["checks"]["transcript_replay_gate_passed"])
            self.assertTrue(report["checks"]["inventory_soak_passed"])
            self.assertTrue(report["inventory_soak"]["passed"])
            self.assertTrue(all(report["inventory_soak"]["checks"].values()))
            self.assertEqual(report["inventory_soak"]["mode"], "offline_fixture")
            self.assertEqual(report["inventory_soak"]["source"], "both")
            self.assertEqual(report["inventory_soak"]["cycles_completed"], 2)
            self.assertEqual(report["inventory_soak"]["failed_import_cycles"], 0)
            self.assertEqual(report["inventory_soak"]["source_coverage"]["missing"], [])
            self.assertEqual(
                set(report["inventory_soak"]["source_coverage"]["required"]),
                {"project_gutenberg_catalog_csv", "internet_archive_search_metadata"},
            )
            self.assertTrue(report["inventory_soak"]["failure_observability"]["present"])
            self.assertGreater(report["inventory_soak"]["inventory_delta"]["story_inventory_added"], 0)
            self.assertTrue(report["checks"]["inventory_soak_matrix_passed"])
            self.assertTrue(report["inventory_soak_matrix"]["passed"])
            self.assertTrue(report["inventory_soak_matrix"]["checks"]["total_cycles_at_least_nine"])
            self.assertTrue(report["inventory_soak_matrix"]["checks"]["both_source_families_covered"])
            self.assertTrue(
                report["inventory_soak_matrix"]["checks"][
                    "future_story_routes_local_from_imported_inventory"
                ]
            )
            self.assertEqual(report["inventory_soak_matrix"]["profile_count"], 3)
            self.assertEqual(report["inventory_soak_matrix"]["total_failed_import_cycles"], 0)
            self.assertTrue(report["checks"]["inventory_diversity_smoke_passed"])
            self.assertTrue(report["inventory_diversity_smoke"]["passed"])
            self.assertTrue(report["inventory_diversity_smoke"]["checks"]["all_queries_reached_import_jobs"])
            self.assertTrue(report["inventory_diversity_smoke"]["checks"]["future_story_routes_local"])
            self.assertEqual(report["inventory_diversity_smoke"]["niche_count"], 3)
            self.assertTrue(report["checks"]["inventory_retry_smoke_passed"])
            self.assertTrue(report["inventory_retry_smoke"]["passed"])
            self.assertTrue(report["inventory_retry_smoke"]["checks"]["gutenberg_source_retried"])
            self.assertTrue(report["inventory_retry_smoke"]["checks"]["internet_archive_source_retried"])
            self.assertTrue(report["inventory_retry_smoke"]["checks"]["future_story_routes_local_after_reload"])
            self.assertEqual(report["inventory_retry_smoke"]["after_story"]["route"], "local_answer")
            self.assertGreaterEqual(report["inventory_retry_smoke"]["transient_failures"], 2)
            self.assertTrue(report["checks"]["inventory_failure_smoke_passed"])
            self.assertTrue(report["inventory_failure_smoke"]["passed"])
            self.assertTrue(report["inventory_failure_smoke"]["checks"]["no_fake_story_inventory"])
            self.assertTrue(report["inventory_failure_smoke"]["checks"]["future_story_routes_missing_inventory"])
            self.assertTrue(report["inventory_failure_smoke"]["checks"]["errors_are_observable"])
            self.assertEqual(report["inventory_failure_smoke"]["case_count"], 3)
            self.assertTrue(all(run["story_route"] == "cloud_handoff" for run in report["inventory_failure_smoke"]["runs"]))
            self.assertTrue(all(run["story_inventory_added"] == 0 for run in report["inventory_failure_smoke"]["runs"]))
            self.assertTrue(report["checks"]["synthesis_variant_smoke_passed"])
            self.assertTrue(report["synthesis_variant_smoke"]["passed"])
            self.assertEqual(report["synthesis_variant_smoke"]["variant_count"], 10)
            self.assertTrue(report["synthesis_variant_smoke"]["checks"]["story_tale_not_exact_story_phrase"])
            self.assertTrue(report["synthesis_variant_smoke"]["checks"]["health_sleep_not_exact_health_phrase"])
            self.assertTrue(report["synthesis_variant_smoke"]["checks"]["primary_uol_chatframe_not_secondary_phrase_route"])
            self.assertEqual(report["synthesis_variant_smoke"]["route_counts"], {"cached_tool": 1, "local_answer": 9})
            self.assertTrue(report["checks"]["synthesis_stress_smoke_passed"])
            self.assertTrue(report["synthesis_stress_smoke"]["passed"])
            self.assertEqual(report["synthesis_stress_smoke"]["turn_count"], 24)
            self.assertEqual(report["synthesis_stress_smoke"]["session_count"], 3)
            self.assertEqual(report["synthesis_stress_smoke"]["route_counts"], {"cached_tool": 2, "clarify": 1, "local_answer": 21})
            self.assertTrue(report["synthesis_stress_smoke"]["checks"]["autobiographical_summaries_use_events_and_digest"])
            self.assertGreaterEqual(report["synthesis_stress_smoke"]["quality"]["min"], 0.72)
            self.assertTrue(report["pi_constraints"]["no_required_network"])
            self.assertTrue(report["pi_constraints"]["no_required_vector_db"])
            self.assertTrue(report["pi_constraints"]["no_required_ml_framework"])

    def test_cli_synthesis_variant_smoke_runs_bounded_story_advice_memory_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "synthesis_variants.sqlite"
            report = _run_cli("synthesis-variant-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(report["variant_count"], 10)
            self.assertEqual(report["counts"]["events"], 10)
            self.assertEqual(report["counts"]["membrane_decisions"], 10)
            self.assertEqual(report["counts"]["homeostatic_snapshots"], 10)
            self.assertEqual(report["route_counts"], {"cached_tool": 1, "local_answer": 9})
            self.assertEqual(report["reason_counts"]["local_story_inventory"], 3)
            self.assertEqual(report["reason_counts"]["bounded_general_health_guidance"], 2)
            by_label = {turn["label"]: turn for turn in report["turns"]}
            self.assertNotIn("story", by_label["story_tale"]["tokens"])
            self.assertEqual(by_label["story_tale"]["route"], "local_answer")
            self.assertNotIn("health", by_label["health_sleep"]["tokens"])
            self.assertEqual(by_label["health_sleep"]["reason"], "bounded_general_health_guidance")
            self.assertEqual(by_label["urgent_health"]["reason"], "urgent_health_safety_escalation")
            self.assertEqual(by_label["session_summary"]["reason"], "autobiographical_session_summary")
            self.assertTrue(all(citation.startswith("events.") for citation in by_label["session_summary"]["citations"]))
            self.assertEqual(by_label["long_horizon_digest"]["citations"], ["memory_digest.long_horizon_latest"])
            self.assertTrue(all(turn["mapping"] == ["basic_nlp", "uol_parse", "chat_frame"] for turn in report["turns"]))
            self.assertTrue(all(turn["primary_parse_basis"] == "uol_chat_frame" for turn in report["turns"]))
            self.assertTrue(
                all(
                    turn["primary_domain_evidence"]["source"] == "slot_role_relation"
                    for turn in report["turns"]
                )
            )
            self.assertFalse(
                any(
                    any(part.startswith("secondary_meaning_hints:") for part in turn["primary_routing_basis"])
                    for turn in report["turns"]
                )
            )
            self.assertEqual(report["safety_flags"]["low_quality_applied_synthesis"], 0)
            self.assertEqual(report["safety_flags"]["cloud_private_inclusions"], 0)

    def test_cli_synthesis_stress_smoke_runs_longer_multisession_synthesis_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "synthesis_stress.sqlite"
            report = _run_cli("synthesis-stress-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(report["turn_count"], 24)
            self.assertEqual(report["session_count"], 3)
            self.assertEqual(report["counts"]["events"], 24)
            self.assertEqual(report["counts"]["membrane_decisions"], 24)
            self.assertEqual(report["counts"]["homeostatic_snapshots"], 24)
            self.assertEqual(report["counts"]["synthesis_traces"], 24)
            self.assertEqual(report["route_counts"], {"cached_tool": 2, "clarify": 1, "local_answer": 21})
            self.assertGreaterEqual(report["reason_counts"]["local_story_inventory"], 3)
            self.assertEqual(report["reason_counts"]["story_constraint_unmet"], 1)
            self.assertGreaterEqual(report["reason_counts"]["bounded_general_health_guidance"], 4)
            self.assertEqual(report["intent_counts"]["autobiographical_memory"], 3)
            self.assertGreaterEqual(report["quality"]["min"], 0.72)
            self.assertGreaterEqual(report["complexity"]["max"], 0.57)
            by_label = {turn["label"]: turn for turn in report["turns"]}
            self.assertEqual(by_label["last_question"]["citations"], ["events.os_e18"])
            self.assertTrue(all(citation.startswith("events.") for citation in by_label["session_summary"]["citations"]))
            self.assertEqual(by_label["long_horizon_digest"]["citations"], ["memory_digest.long_horizon_latest"])
            self.assertIn("self_status.counts", by_label["status_next"]["citations"])
            self.assertTrue(any(citation.startswith("story_models.") for citation in by_label["story_fable"]["citations"]))
            self.assertTrue(any(citation.startswith("food_inventory.") for citation in by_label["meal_dinner"]["citations"]))
            self.assertEqual(by_label["urgent_health"]["citations"], ["local_health_safety_policy"])
            self.assertEqual({turn["session"] for turn in report["turns"]}, {"session_1", "session_2", "session_3"})
            self.assertTrue(all(
                turn["synthesis_applied"] or turn.get("synthesis_refused")
                for turn in report["turns"]
            ))
            self.assertTrue(all(turn["mapping"] == ["basic_nlp", "uol_parse", "chat_frame"] for turn in report["turns"]))
            self.assertFalse(
                any(
                    any(part.startswith("secondary_meaning_hints:") for part in turn["primary_routing_basis"])
                    for turn in report["turns"]
                )
            )
            self.assertEqual(report["safety_flags"]["low_quality_applied_synthesis"], 0)
            self.assertEqual(report["safety_flags"]["cloud_private_inclusions"], 0)

    def test_cli_api_smoke_verifies_localhost_health_and_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "api_smoke.sqlite"
            report = _run_cli("api-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite_http")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertTrue(report["base_url"].startswith("http://127.0.0.1:"))
            self.assertEqual(report["initial_health"]["counts"]["events"], 0)
            self.assertGreaterEqual(report["initial_health"]["counts"]["inventories"], 8)
            self.assertEqual(report["after_parse_health"]["counts"]["events"], 0)
            self.assertEqual(report["parse_debug"]["intent"], "assistant_identity")
            self.assertEqual(report["parse_debug"]["speech_act"], "challenge")
            self.assertEqual(report["parse_debug"]["object"], "self_model")
            self.assertEqual(report["parse_debug"]["mapping"], ["basic_nlp", "uol_parse", "chat_frame"])
            self.assertEqual(report["parse_debug"]["primary_parse_basis"], "uol_chat_frame")
            self.assertEqual(report["parse_debug"]["primary_domain_evidence"]["source"], "token_role_relation")
            self.assertEqual(report["parse_debug"]["composition_pattern"], "who_copula_second_person")
            self.assertEqual(report["parse_debug"]["secondary_domain_hints"], {})
            self.assertEqual(report["parse_debug"]["secondary_meaning_hints"], [])
            self.assertEqual(report["parse_debug"]["unknown_token_count"], 0)
            self.assertEqual(report["ask"]["route"], "local_answer")
            self.assertEqual(report["ask"]["reason"], "local_story_inventory")
            self.assertTrue(report["ask"]["synthesis_applied"])
            self.assertEqual(report["ask"]["membrane"]["boundary_crossed"], "none")
            self.assertGreaterEqual(report["after_health"]["counts"]["events"], 1)
            self.assertGreaterEqual(report["after_health"]["counts"]["membrane_decisions"], 1)
            self.assertGreaterEqual(report["after_health"]["counts"]["homeostatic_snapshots"], 1)
            self.assertEqual(report["dashboard"]["counts"]["events"], report["after_health"]["counts"]["events"])
            self.assertEqual(report["event_transcript_export"]["events_exported"], 1)
            self.assertFalse(report["event_transcript_export"]["answers_routes_reasons_exported"])
            self.assertEqual(report["event_transcript_export"]["forbidden_static_fields_exported"], [])

    def test_cli_api_session_smoke_runs_realistic_local_assistant_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "api_session.sqlite"
            report = _run_cli("api-session-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite_http")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(len(report["turns"]), 11)
            self.assertEqual(report["route_counts"], {"cached_tool": 1, "device_action": 4, "local_answer": 6})
            by_label = {turn["label"]: turn for turn in report["turns"]}
            self.assertEqual(by_label["identity"]["reason"], "self_model_identity")
            self.assertEqual(by_label["identity"]["debug_parse"]["chat_frame"]["intent"], "assistant_identity")
            self.assertEqual(by_label["identity"]["debug_parse"]["uol"]["object"], "self_model")
            self.assertEqual(by_label["story"]["reason"], "local_story_inventory")
            self.assertEqual(by_label["weather"]["reason"], "weather_cache_hit")
            self.assertEqual(by_label["safety"]["reason"], "local_common_sense_policy")
            self.assertEqual(by_label["media_request"]["confirmation_required"], 1)
            self.assertEqual(by_label["media_confirm"]["reason"], "confirmed_device_action")
            self.assertEqual(by_label["health"]["reason"], "bounded_general_health_guidance")
            self.assertEqual(by_label["profile_memory"]["reason"], "personal_memory_summary")
            self.assertEqual(by_label["meal"]["reason"], "memory_plus_weather_cache")
            self.assertEqual(by_label["call_request"]["confirmation_required"], 1)
            self.assertEqual(by_label["call_confirm"]["reason"], "confirmed_device_action")
            self.assertEqual(
                [item["action_type"] for item in report["action_results"]],
                ["play_media", "call_contact"],
            )
            self.assertEqual([item["status"] for item in report["action_results"]], ["prepared", "prepared"])
            self.assertFalse(any(item["side_effect_executed"] for item in report["action_results"]))
            self.assertEqual(report["action_results"][1]["resolved_target"], "+234-000-MOM")
            self.assertEqual(report["after_health"]["counts"]["events"], 11)
            self.assertEqual(report["after_health"]["counts"]["membrane_decisions"], 11)
            self.assertEqual(report["after_health"]["counts"]["homeostatic_snapshots"], 11)
            self.assertEqual(report["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["safety_flags"]["action_without_confirmation_gate"], 0)
            self.assertEqual(report["event_transcript_export"]["events_exported"], 11)
            self.assertFalse(report["event_transcript_export"]["answers_routes_reasons_exported"])
            self.assertEqual(report["event_transcript_export"]["forbidden_static_fields_exported"], [])
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["capture_surface_counts"],
                {"browser_api": 11},
            )
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["capture_source_counts"],
                {"scripted_api_smoke": 11},
            )
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["scripted_capture_source_counts"],
                {"scripted_api_smoke": 11},
            )
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["candidate_capture_source_count"],
                0,
            )
            self.assertTrue(report["event_transcript_export"]["capture_provenance"]["all_turns_scripted"])
            self.assertTrue(report["event_ledger_calibration"]["passed"])
            self.assertEqual(report["event_ledger_calibration"]["events_exported"], 11)
            self.assertEqual(report["event_ledger_calibration"]["turns_replayed"], 11)
            self.assertEqual(
                report["event_ledger_calibration"]["capture_provenance"]["capture_source_counts"],
                {"scripted_api_smoke": 11},
            )
            self.assertEqual(
                report["event_ledger_calibration"]["capture_provenance"]["candidate_capture_source_count"],
                0,
            )
            self.assertTrue(report["event_ledger_calibration"]["capture_provenance"]["all_turns_scripted"])
            self.assertGreaterEqual(report["event_ledger_calibration"]["local_resolution_rate"], 0.8)
            self.assertGreaterEqual(report["event_ledger_calibration"]["intent_kinds"], 8)
            self.assertTrue(Path(report["event_ledger_calibration"]["transcript_jsonl"]).exists())

    def test_cli_api_session_smoke_can_execute_configured_real_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "api_session.sqlite"
            log = Path(tmp) / "api_actions.jsonl"
            report = _run_cli(
                "api-session-smoke",
                "--db",
                str(db),
                "--reset",
                "--action-mode",
                "real",
                "--media-player-command",
                _action_recorder_command(log, "media"),
                "--call-command",
                _action_recorder_command(log, "call"),
                "--json",
            )
            log_rows = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertTrue(report["passed"])
            self.assertTrue(report["checks"]["media_gated_then_confirmed"])
            self.assertTrue(report["checks"]["call_gated_then_confirmed"])
            self.assertTrue(report["checks"]["host_action_gate_mode_respected"])
            self.assertEqual(report["host_actions"]["mode"], "real")
            self.assertTrue(report["host_actions"]["configured"])
            self.assertTrue(report["media_import"]["generated_for_real_action_smoke"])
            self.assertEqual(report["media_import"]["imported_items"], 1)
            self.assertEqual([item["status"] for item in report["action_results"]], ["executed", "executed"])
            self.assertTrue(all(item["side_effect_executed"] for item in report["action_results"]))
            self.assertTrue(report["action_results"][0]["payload"]["path_exists"])
            self.assertEqual(report["action_results"][1]["resolved_target"], "+234-000-MOM")
            self.assertEqual([row["label"] for row in log_rows], ["media", "call"])
            self.assertTrue(log_rows[0]["target_exists"])
            self.assertFalse(log_rows[1]["target_exists"])

    def test_cli_chat_runs_scripted_cross_platform_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "chat.sqlite"
            report = _run_cli(
                "chat",
                "--db",
                str(db),
                "--reset",
                "--turn",
                "Tell me a story.",
                "--turn",
                "What is the weather today?",
                "--turn",
                "Should I go to school dressed naked?",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["mode"], "scripted")
            self.assertEqual(len(report["turns"]), 3)
            self.assertEqual([turn["route"] for turn in report["turns"]], ["local_answer", "cached_tool", "local_answer"])
            self.assertEqual(report["turns"][0]["reason"], "local_story_inventory")
            self.assertEqual(report["turns"][1]["reason"], "weather_cache_hit")
            self.assertEqual(report["turns"][2]["reason"], "local_common_sense_policy")
            self.assertTrue(
                all(turn["capture_provenance"] == {"surface": "cli_chat", "source": "scripted_cli_turn"}
                    for turn in report["turns"])
            )
            self.assertEqual(report["counts"]["events"], 3)
            self.assertEqual(report["counts"]["membrane_decisions"], 3)
            self.assertEqual(report["counts"]["homeostatic_snapshots"], 3)

    def test_cli_ui_smoke_verifies_browser_chat_shell_and_api_turn(self) -> None:
        root = Path("artifacts/local_assistant_os/test_tmp/ui_smoke_case")
        db = root / "ui_smoke.sqlite"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            report = _run_cli("ui-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite_http_html")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertTrue(report["base_url"].startswith("http://127.0.0.1:"))
            self.assertEqual(report["ui"]["path"], "/")
            self.assertGreater(report["ui"]["bytes"], 1000)
            self.assertTrue(report["ui"]["form_present"])
            self.assertTrue(report["ui"]["dependency_free"])
            self.assertTrue(report["checks"]["browser_ui_capture_token_wired"])
            self.assertTrue(report["checks"]["browser_ui_capture_source_requires_token"])
            self.assertEqual(report["browser_ui_capture_token_rejection"]["status"], 400)
            self.assertEqual(report["parse_debug"]["intent"], "assistant_identity")
            self.assertEqual(report["parse_debug"]["speech_act"], "challenge")
            self.assertEqual(report["parse_debug"]["object"], "self_model")
            self.assertEqual(report["parse_debug"]["mapping"], ["basic_nlp", "uol_parse", "chat_frame"])
            self.assertEqual(report["parse_debug"]["primary_parse_basis"], "uol_chat_frame")
            self.assertEqual(report["parse_debug"]["primary_domain_evidence"]["source"], "token_role_relation")
            self.assertEqual(report["parse_debug"]["composition_pattern"], "who_copula_second_person")
            self.assertEqual(report["parse_debug"]["secondary_meaning_hints"], [])
            self.assertTrue(report["checks"]["functional_grammar_debug_wired"])
            self.assertTrue(report["checks"]["functional_grammar_parse_exposed"])
            self.assertEqual(report["functional_parse_debug"]["intent"], "personal_goal_advice")
            self.assertEqual(report["functional_parse_debug"]["source"], "weighted_functional_relation")
            self.assertEqual(report["functional_parse_debug"]["subject"], "assistant")
            self.assertEqual(report["functional_parse_debug"]["action"], "help")
            self.assertEqual(report["functional_parse_debug"]["object"], "career")
            self.assertEqual(report["functional_parse_debug"]["complement_action"], "grow")
            self.assertEqual(report["functional_parse_debug"]["indirect_object"], "user")
            self.assertGreaterEqual(report["functional_parse_debug"]["parse_score"], 0.9)
            self.assertTrue(report["functional_parse_debug"]["relations"])
            self.assertEqual(report["identity"]["intent"], "assistant_identity")
            self.assertEqual(report["identity"]["route"], "local_answer")
            self.assertEqual(report["identity"]["reason"], "self_model_identity")
            self.assertEqual(report["identity"]["debug_parse"]["uol"]["object"], "self_model")
            self.assertEqual(report["status"]["intent"], "assistant_status")
            self.assertEqual(report["status"]["route"], "local_answer")
            self.assertEqual(report["status"]["reason"], "self_status_ledger_summary")
            self.assertEqual(report["status"]["debug_parse"]["uol"]["object"], "runtime_status")
            self.assertIn("self_status.counts", report["status"]["citations"])
            self.assertEqual(report["ask"]["route"], "local_answer")
            self.assertEqual(report["ask"]["reason"], "local_story_inventory")
            self.assertTrue(report["ask"]["synthesis_applied"])
            self.assertEqual(report["action"]["request_route"], "device_action")
            self.assertEqual(report["action"]["request_reason"], "local_media_action")
            self.assertEqual(report["action"]["confirmation_required"], 1)
            self.assertEqual(report["action"]["pending_action"]["action_type"], "play_media")
            self.assertEqual(report["action"]["confirm_reason"], "confirmed_device_action")
            self.assertEqual(report["action"]["execution"]["status"], "prepared")
            self.assertFalse(report["action"]["execution"]["side_effect_executed"])
            self.assertGreaterEqual(report["after_health"]["counts"]["events"], 5)
            self.assertEqual(report["event_transcript_export"]["events_exported"], 5)
            self.assertFalse(report["event_transcript_export"]["answers_routes_reasons_exported"])
            self.assertEqual(report["event_transcript_export"]["forbidden_static_fields_exported"], [])
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["capture_source_counts"],
                {"scripted_ui_smoke": 5},
            )
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["candidate_capture_source_count"],
                0,
            )
            self.assertTrue(report["event_transcript_export"]["capture_provenance"]["all_turns_scripted"])
            self.assertTrue(report["event_ledger_calibration"]["passed"])
            self.assertGreaterEqual(report["event_ledger_calibration"]["turns_replayed"], 5)
            self.assertGreaterEqual(report["event_ledger_calibration"]["local_resolution_rate"], 0.8)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_cli_target_report_collects_environment_and_runs_smokes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "target_report"
            report = _run_cli("target-report", "--db-dir", str(db_dir), "--reset", "--json")

            failed_checks = [k for k, v in report.get("checks", {}).items() if not v]
            self.assertTrue(report["passed"], f"target-report failed: checks={failed_checks}")
            self.assertTrue(report["checks"]["python_supported"])
            self.assertTrue(report["checks"]["sqlite_available"])
            self.assertTrue(report["checks"]["db_dir_writable"])
            self.assertTrue(report["checks"]["dataset_audit_passed"])
            self.assertTrue(report["checks"]["pi_smoke_passed"])
            self.assertTrue(report["checks"]["inventory_soak_matrix_passed"])
            self.assertTrue(report["checks"]["autoimmune_smoke_passed"])
            self.assertTrue(report["checks"]["synthesis_variant_smoke_passed"])
            self.assertTrue(report["checks"]["synthesis_stress_smoke_passed"])
            self.assertTrue(report["checks"]["setup_integration_smoke_passed"])
            self.assertTrue(report["checks"]["host_action_smoke_passed"])
            self.assertTrue(report["checks"]["host_app_probe_reported"])
            self.assertTrue(report["checks"]["host_app_requirement_satisfied"])
            self.assertTrue(report["checks"]["capability_probe_passed"])
            self.assertTrue(report["checks"]["v01_audit_passed"])
            self.assertTrue(report["checks"]["api_smoke_passed"])
            self.assertTrue(report["checks"]["api_session_smoke_passed"])
            self.assertTrue(report["checks"]["ui_smoke_passed"])
            self.assertTrue(report["checks"]["bootstrap_runtime_passed"])
            self.assertTrue(report["checks"]["open_traces_passed"])
            self.assertTrue(report["checks"]["transcript_replay_passed"])
            self.assertTrue(report["checks"]["transcript_calibration_passed"])
            self.assertTrue(report["checks"]["localhost_api"])
            self.assertTrue(report["checks"]["stdlib_only"])
            self.assertTrue(report["checks"]["raspberry_pi_requirement_satisfied"])
            self.assertNotIn("raspberry_pi_required", report["checks"])
            self.assertIn("raspberry_pi_detected", report["hardware"])
            self.assertFalse(report["hardware_policy"]["raspberry_pi_required"])
            self.assertTrue(report["hardware_policy"]["raspberry_pi_hardware_optional_for_v01"])
            self.assertTrue(report["hardware_policy"]["raspberry_pi_requirement_satisfied"])
            self.assertIn("python", report["runtime"])
            self.assertIn("sqlite", report["runtime"])
            self.assertGreater(report["resources"]["disk_total_bytes"], 0)
            self.assertGreaterEqual(report["resources"]["disk_free_bytes"], 0)
            self.assertTrue(report["smokes"]["dataset_audit"]["passed"])
            self.assertEqual(report["smokes"]["dataset_audit"]["source_fixtures"]["weather_days"], 7)
            self.assertEqual(report["smokes"]["dataset_audit"]["source_fixtures"]["transcript_replay_user_turns"], 25)
            self.assertTrue(report["smokes"]["pi_smoke"]["passed"])
            self.assertTrue(report["smokes"]["pi_smoke"]["checks"]["inventory_soak_passed"])
            self.assertTrue(report["smokes"]["pi_smoke"]["checks"]["inventory_soak_matrix_passed"])
            self.assertTrue(report["smokes"]["pi_smoke"]["checks"]["inventory_diversity_smoke_passed"])
            self.assertTrue(report["smokes"]["pi_smoke"]["checks"]["inventory_retry_smoke_passed"])
            self.assertTrue(report["smokes"]["pi_smoke"]["checks"]["inventory_failure_smoke_passed"])
            self.assertEqual(report["smokes"]["pi_smoke"]["inventory_soak"]["source"], "both")
            self.assertTrue(report["smokes"]["pi_smoke"]["inventory_soak"]["failure_observability"]["present"])
            self.assertEqual(report["smokes"]["pi_smoke"]["inventory_soak_matrix"]["profile_count"], 3)
            self.assertEqual(
                report["smokes"]["pi_smoke"]["inventory_soak_matrix"]["total_failed_import_cycles"],
                0,
            )
            self.assertTrue(
                report["smokes"]["pi_smoke"]["inventory_soak_matrix"]["checks"][
                    "future_story_routes_local_from_imported_inventory"
                ]
            )
            self.assertEqual(report["smokes"]["pi_smoke"]["inventory_diversity_smoke"]["niche_count"], 3)
            self.assertTrue(report["smokes"]["pi_smoke"]["inventory_diversity_smoke"]["checks"]["future_story_routes_local"])
            self.assertTrue(report["smokes"]["pi_smoke"]["inventory_retry_smoke"]["checks"]["gutenberg_source_retried"])
            self.assertEqual(report["smokes"]["pi_smoke"]["inventory_retry_smoke"]["after_story"]["route"], "local_answer")
            self.assertEqual(report["smokes"]["pi_smoke"]["inventory_failure_smoke"]["case_count"], 3)
            self.assertTrue(report["smokes"]["pi_smoke"]["inventory_failure_smoke"]["checks"]["no_fake_story_inventory"])
            self.assertTrue(report["smokes"]["autoimmune_smoke"]["passed"])
            self.assertTrue(report["smokes"]["autoimmune_smoke"]["checks"]["conversation_export_blocked"])
            self.assertEqual(report["smokes"]["autoimmune_smoke"]["pending_actions"]["executed"], 0)
            self.assertTrue(report["smokes"]["synthesis_variant_smoke"]["passed"])
            self.assertTrue(
                report["smokes"]["synthesis_variant_smoke"]["checks"][
                    "primary_uol_chatframe_not_secondary_phrase_route"
                ]
            )
            self.assertEqual(report["smokes"]["synthesis_variant_smoke"]["variant_count"], 10)
            self.assertTrue(report["smokes"]["synthesis_stress_smoke"]["passed"])
            self.assertEqual(report["smokes"]["synthesis_stress_smoke"]["turn_count"], 24)
            self.assertEqual(report["smokes"]["synthesis_stress_smoke"]["session_count"], 3)
            self.assertTrue(
                report["smokes"]["synthesis_stress_smoke"]["checks"]["quality_clean_under_longer_trace"]
            )
            self.assertTrue(
                report["smokes"]["synthesis_stress_smoke"]["checks"][
                    "autobiographical_summaries_use_events_and_digest"
                ]
            )
            self.assertTrue(report["smokes"]["setup_integration_smoke"]["passed"])
            self.assertTrue(all(report["smokes"]["setup_integration_smoke"]["checks"].values()))
            self.assertEqual(
                set(report["smokes"]["setup_integration_smoke"]["setup_requests_after_gaps"]),
                {"routine_memory", "household_memory", "trusted_contact"},
            )
            self.assertEqual(
                report["smokes"]["setup_integration_smoke"]["facts_after_setup"]["morning_routine"],
                "stretch, breakfast, then bus",
            )
            self.assertEqual(
                report["smokes"]["setup_integration_smoke"]["action_execution"]["resolved_target"],
                "+234-000-ADA",
            )
            self.assertTrue(report["smokes"]["host_action_smoke"]["passed"])
            self.assertTrue(report["smokes"]["host_action_smoke"]["checks"]["media_command_received_existing_file"])
            self.assertEqual(report["smokes"]["host_action_smoke"]["records"][1]["target"], "+234-000-ADA")
            self.assertTrue(report["smokes"]["host_app_probe"]["passed"])
            self.assertFalse(report["smokes"]["host_app_probe"]["configured"])
            self.assertTrue(report["smokes"]["host_app_probe"]["skipped"])
            self.assertTrue(report["smokes"]["host_app_probe"]["checks"]["configuration_reported"])
            self.assertFalse(report["smokes"]["host_app_probe"]["checks"]["probe_executed"])
            self.assertTrue(report["smokes"]["capability_probe"]["passed"])
            self.assertEqual(report["smokes"]["capability_probe"]["total_cases"], 18)
            self.assertEqual(report["smokes"]["capability_probe"]["bucket_counts"]["local"], 12)
            self.assertEqual(report["smokes"]["capability_probe"]["bucket_counts"]["device_action"], 4)
            self.assertEqual(report["smokes"]["capability_probe"]["bucket_counts"]["blocked"], 2)
            self.assertGreaterEqual(report["smokes"]["capability_probe"]["local_device_rate"], 0.88)
            self.assertTrue(report["smokes"]["v01_audit"]["passed"])
            self.assertFalse(report["smokes"]["v01_audit"]["architecture_complete"])
            self.assertEqual(report["smokes"]["v01_audit"]["blocker_count"], 6)
            self.assertTrue(report["smokes"]["api_smoke"]["passed"])
            self.assertEqual(report["smokes"]["api_smoke"]["parse_debug"]["intent"], "assistant_identity")
            self.assertEqual(report["smokes"]["api_smoke"]["ask"]["route"], "local_answer")
            self.assertEqual(report["smokes"]["api_smoke"]["event_transcript_export"]["events_exported"], 1)
            self.assertFalse(
                report["smokes"]["api_smoke"]["event_transcript_export"]["answers_routes_reasons_exported"]
            )
            self.assertTrue(report["smokes"]["api_session_smoke"]["passed"])
            self.assertEqual(report["smokes"]["api_session_smoke"]["route_counts"]["device_action"], 4)
            self.assertTrue(report["smokes"]["api_session_smoke"]["event_ledger_calibration"]["passed"])
            self.assertEqual(report["smokes"]["api_session_smoke"]["event_ledger_calibration"]["turns_replayed"], 11)
            self.assertTrue(report["smokes"]["ui_smoke"]["passed"])
            self.assertTrue(report["smokes"]["ui_smoke"]["ui"]["dependency_free"])
            self.assertEqual(report["smokes"]["ui_smoke"]["ask"]["route"], "local_answer")
            self.assertTrue(report["smokes"]["ui_smoke"]["event_ledger_calibration"]["passed"])
            self.assertTrue(report["smokes"]["bootstrap_runtime"]["passed"])
            self.assertEqual(report["smokes"]["bootstrap_runtime"]["counts"]["events"], 3)
            self.assertTrue(report["smokes"]["open_traces"]["passed"])
            self.assertEqual(report["smokes"]["open_traces"]["turns"], 29)
            self.assertTrue(report["smokes"]["open_traces"]["debug_checks"]["identity_maps_to_self_model"])
            self.assertTrue(report["smokes"]["transcript_replay"]["passed"])
            self.assertEqual(report["smokes"]["transcript_replay"]["turns"], 25)
            self.assertTrue(report["smokes"]["transcript_replay"]["fixture_checks"]["memory_digest_quality_passed"])
            self.assertTrue(report["smokes"]["transcript_replay"]["baseline_comparison"]["passed"])
            self.assertGreaterEqual(
                report["smokes"]["transcript_replay"]["baseline_comparison"]["best_baseline"]["local_or_device_resolved"],
                12,
            )
            self.assertTrue(report["smokes"]["transcript_calibration"]["passed"])
            calibration = report["smokes"]["transcript_calibration"]["aggregate"]
            self.assertEqual(calibration["turns_imported"], 4)
            self.assertTrue(all(calibration["checks"].values()))
            self.assertEqual(calibration["thresholds"]["min_total_turns"], 4)
            self.assertEqual(calibration["thresholds"]["min_local_resolution_rate"], 0.2)
            self.assertEqual(calibration["thresholds"]["min_route_kinds"], 2)
            self.assertEqual(calibration["thresholds"]["min_intent_kinds"], 3)
            self.assertTrue(calibration["thresholds"]["require_redaction"])
            self.assertTrue(calibration["thresholds"]["require_static_drop"])
            self.assertTrue(calibration["debug_mapping_passed"])
            self.assertEqual(calibration["baseline_required_replays"], 0)
            self.assertEqual(calibration["redaction_counts"]["email"], 1)
            self.assertEqual(calibration["redaction_counts"]["phone"], 1)
            self.assertEqual(calibration["redaction_counts"]["manual_rule_1"], 1)
            self.assertTrue((db_dir / "pi_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "dataset_audit_target.sqlite").exists())
            self.assertTrue((db_dir / "autoimmune_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "synthesis_variant_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "synthesis_stress_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "host_action_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "host_action_target_smoke" / "actions.jsonl").exists())
            self.assertTrue((db_dir / "host_app_probe").exists())
            self.assertTrue((db_dir / "capability_probe_target.sqlite").exists())
            self.assertTrue((db_dir / "api_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "api_session_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "ui_target_smoke.sqlite").exists())
            self.assertTrue((db_dir / "bootstrap_runtime.sqlite").exists())
            self.assertTrue((db_dir / "open_traces_target").exists())
            self.assertTrue((db_dir / "transcript_replay_target").exists())
            self.assertTrue((db_dir / "transcript_calibration_target").exists())

    def test_cli_v01_acceptance_reports_release_candidate_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "v01_acceptance"
            report = _run_cli("v01-acceptance", "--db-dir", str(db_dir), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(report["release_candidate"])
            self.assertFalse(report["architecture_complete"])
            self.assertEqual(report["blocker_count"], 6)
            self.assertEqual(report["schema"], "melm.local_assistant_v01_acceptance.v1")
            self.assertEqual(report["runtime"], "stdlib_python_sqlite_http_html_acceptance")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            requirements = {item["id"]: item for item in report["requirements"]}
            for requirement_id, requirement in requirements.items():
                if requirement_id == "portable_bundle":
                    self.assertEqual(requirement["status"], "skipped")
                    self.assertFalse(requirement["passed"])
                else:
                    self.assertEqual(requirement["status"], "met", requirement_id)
                    self.assertTrue(requirement["passed"], requirement_id)
                    self.assertTrue(report["checks"][requirement_id])
            self.assertEqual(
                set(requirements),
                {
                    "target_report_smokes",
                    "initial_datasets_and_runtime_db",
                    "readiness_pi_smoke",
                    "cross_platform_cli_chat",
                    "browser_api_surface",
                    "memory_synthesis_transcript_gates",
                    "action_and_setup_gates",
                    "anti_static_uol_chatframe_guard",
                    "portable_bundle",
                    "completion_boundary_explicit",
                },
            )
            self.assertIn("shortcut-audit --json", requirements["anti_static_uol_chatframe_guard"]["evidence"])
            self.assertTrue(report["target_report"]["passed"])
            self.assertTrue(report["target_report"]["checks"]["inventory_soak_matrix_passed"])
            self.assertEqual(report["target_report"]["smokes"]["inventory_soak_matrix"]["profile_count"], 3)
            self.assertEqual(
                report["target_report"]["smokes"]["inventory_soak_matrix"]["total_failed_import_cycles"],
                0,
            )
            self.assertTrue(
                report["target_report"]["smokes"]["inventory_soak_matrix"]["checks"][
                    "future_story_routes_local_from_imported_inventory"
                ]
            )
            self.assertEqual(report["chat"]["turn_count"], 3)
            self.assertEqual(report["chat"]["routes"], ["local_answer", "cached_tool", "local_answer"])
            self.assertEqual(
                report["chat"]["reasons"],
                ["local_story_inventory", "weather_cache_hit", "local_common_sense_policy"],
            )
            self.assertEqual(report["chat"]["counts"]["events"], 3)
            self.assertTrue(report["v01_audit"]["core_browser_cli_ready"])
            self.assertFalse(report["v01_audit"]["architecture_complete"])
            self.assertEqual(report["v01_audit"]["blocker_count"], 6)
            self.assertTrue(report["bundle"]["skipped"])
            self.assertTrue((db_dir / "target_report").exists())
            self.assertTrue((db_dir / "cli_chat.sqlite").exists())

    def test_cli_v01_acceptance_threads_host_app_config_json_to_target_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_dir = root / "v01_acceptance"
            log = root / "host_app_actions.jsonl"
            config = root / "host_actions.json"
            config.write_text(
                json.dumps(
                    {
                        "media_player_command": _action_recorder_command(log, "media"),
                        "call_command": _action_recorder_command(log, "call"),
                    }
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "v01-acceptance",
                "--db-dir",
                str(db_dir),
                "--host-app-config-json",
                str(config),
                "--require-host-app-configured",
                "--reset",
                "--json",
            )
            requirements = {item["id"]: item for item in report["requirements"]}
            action_gate = requirements["action_and_setup_gates"]
            host_app = report["target_report"]["smokes"]["host_app_probe"]
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

            self.assertTrue(report["passed"])
            self.assertTrue(report["release_candidate"])
            self.assertEqual(action_gate["status"], "met")
            self.assertIn("--config-json", " ".join(action_gate["evidence"]))
            self.assertEqual(action_gate["details"]["host_app_config_source"], "config_json")
            self.assertTrue(action_gate["details"]["host_app_required"])
            self.assertTrue(action_gate["details"]["host_app_configured"])
            self.assertFalse(action_gate["details"]["host_app_skipped"])
            self.assertTrue(host_app["passed"])
            self.assertTrue(host_app["configured"])
            self.assertFalse(host_app["skipped"])
            self.assertTrue(host_app["checks"]["typed_confirmation_gate_used"])
            self.assertIn("config:", host_app["command_sources"]["media"])
            self.assertIn("config:", host_app["command_sources"]["call"])
            self.assertEqual([row["label"] for row in rows], ["media", "call"])

    def test_cli_pi_bundle_builds_portable_self_checked_bundle(self) -> None:
        root = Path("artifacts/local_assistant_os/test_tmp/pi_bundle_case")
        archive = root.with_suffix(".zip")
        archive_extract = root.parent / "pi_bundle_archive_extract"
        for target in (root, archive, archive_extract):
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        try:
            out = root
            report = _run_cli("pi-bundle", "--out", str(out), "--reset", "--zip", "--json")

            manifest_path = Path(report["manifest"])
            runbook_path = Path(report["runbook"])
            self_check_path = Path(report["self_check"])
            archive_path = Path(report["archive"]["path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self_check = json.loads(self_check_path.read_text(encoding="utf-8"))
            bundled_paths = {item["path"] for item in manifest["files"]}

            self.assertTrue(report["passed"])
            self.assertEqual(runbook_path.name, "RUN_PORTABLE_APP.md")
            self.assertFalse(report["smoke_skipped"])
            self.assertTrue(report["dataset_audit"]["passed"])
            self.assertTrue(all(report["dataset_audit"]["checks"].values()))
            self.assertEqual(report["dataset_audit"]["source_fixtures"]["open_trace_turns"], 29)
            self.assertTrue(report["smoke"]["passed"])
            self.assertTrue(all(report["smoke"]["checks"].values()))
            self.assertTrue(report["smoke"]["checks"]["inventory_soak_passed"])
            self.assertTrue(report["smoke"]["checks"]["inventory_soak_matrix_passed"])
            self.assertTrue(report["smoke"]["checks"]["inventory_diversity_smoke_passed"])
            self.assertTrue(report["smoke"]["checks"]["inventory_retry_smoke_passed"])
            self.assertTrue(report["smoke"]["checks"]["inventory_failure_smoke_passed"])
            self.assertTrue(report["smoke"]["checks"]["setup_integration_smoke_passed"])
            self.assertEqual(report["smoke"]["inventory_soak"]["source"], "both")
            self.assertEqual(report["smoke"]["inventory_soak"]["source_coverage"]["missing"], [])
            self.assertEqual(report["smoke"]["inventory_soak_matrix"]["profile_count"], 3)
            self.assertEqual(report["smoke"]["inventory_soak_matrix"]["total_failed_import_cycles"], 0)
            self.assertTrue(
                report["smoke"]["inventory_soak_matrix"]["checks"][
                    "future_story_routes_local_from_imported_inventory"
                ]
            )
            self.assertEqual(report["smoke"]["inventory_diversity_smoke"]["niche_count"], 3)
            self.assertTrue(report["smoke"]["inventory_retry_smoke"]["checks"]["internet_archive_source_retried"])
            self.assertEqual(report["smoke"]["inventory_retry_smoke"]["after_story"]["route"], "local_answer")
            self.assertEqual(report["smoke"]["inventory_failure_smoke"]["case_count"], 3)
            self.assertTrue(report["smoke"]["inventory_failure_smoke"]["checks"]["future_story_routes_missing_inventory"])
            self.assertEqual(report["smoke"]["runtime"], "stdlib_python_sqlite")
            self.assertEqual(report["smoke"]["dependency_class"], "stdlib_only")
            self.assertTrue(report["autoimmune_smoke"]["passed"])
            self.assertTrue(all(report["autoimmune_smoke"]["checks"].values()))
            self.assertEqual(report["autoimmune_smoke"]["pending_actions"]["executed"], 0)
            self.assertTrue(report["synthesis_variant_smoke"]["passed"])
            self.assertTrue(all(report["synthesis_variant_smoke"]["checks"].values()))
            self.assertEqual(report["synthesis_variant_smoke"]["variant_count"], 10)
            self.assertTrue(
                report["synthesis_variant_smoke"]["checks"]["primary_uol_chatframe_not_secondary_phrase_route"]
            )
            self.assertTrue(report["synthesis_stress_smoke"]["passed"])
            self.assertTrue(all(report["synthesis_stress_smoke"]["checks"].values()))
            self.assertEqual(report["synthesis_stress_smoke"]["turn_count"], 24)
            self.assertEqual(report["synthesis_stress_smoke"]["session_count"], 3)
            self.assertTrue(report["setup_integration_smoke"]["passed"])
            self.assertTrue(all(report["setup_integration_smoke"]["checks"].values()))
            self.assertEqual(
                set(report["setup_integration_smoke"]["setup_requests_after_gaps"]),
                {"routine_memory", "household_memory", "trusted_contact"},
            )
            self.assertEqual(
                report["setup_integration_smoke"]["facts_after_setup"]["morning_routine"],
                "stretch, breakfast, then bus",
            )
            self.assertEqual(report["setup_integration_smoke"]["contacts_after_setup"]["ada"], "+234-000-ADA")
            self.assertEqual(report["setup_integration_smoke"]["action_execution"]["resolved_target"], "+234-000-ADA")
            self.assertTrue(report["host_action_smoke"]["passed"])
            self.assertTrue(all(report["host_action_smoke"]["checks"].values()))
            self.assertEqual(report["host_action_smoke"]["records"][1]["target"], "+234-000-ADA")
            self.assertTrue(report["host_app_probe"]["passed"])
            self.assertFalse(report["host_app_probe"]["configured"])
            self.assertTrue(report["host_app_probe"]["skipped"])
            self.assertTrue(report["host_app_probe"]["checks"]["configuration_reported"])
            self.assertTrue(report["capability_probe"]["passed"])
            self.assertTrue(all(report["capability_probe"]["checks"].values()))
            self.assertEqual(report["capability_probe"]["total_cases"], 18)
            self.assertEqual(report["capability_probe"]["bucket_counts"]["local"], 12)
            self.assertEqual(report["capability_probe"]["bucket_counts"]["device_action"], 4)
            self.assertEqual(report["capability_probe"]["bucket_counts"]["blocked"], 2)
            self.assertTrue(report["shortcut_audit"]["passed"])
            self.assertTrue(all(report["shortcut_audit"]["checks"].values()))
            self.assertEqual(report["shortcut_audit"]["behavior_case_count"], 11)
            self.assertGreaterEqual(report["shortcut_audit"]["source_check_count"], 6)
            self.assertTrue(report["shortcut_audit"]["checks"]["no_debug_hit_labels"])
            self.assertTrue(report["v01_audit"]["passed"])
            self.assertFalse(report["v01_audit"]["architecture_complete"])
            self.assertEqual(report["v01_audit"]["blocker_count"], 6)
            self.assertTrue(report["v01_progress"]["passed"])
            self.assertFalse(report["v01_progress"]["architecture_complete"])
            self.assertEqual(report["v01_progress"]["remaining_blocker_count"], 6)
            self.assertTrue(report["api_smoke"]["passed"])
            self.assertTrue(all(report["api_smoke"]["checks"].values()))
            self.assertEqual(report["api_smoke"]["parse_debug"]["mapping"], ["basic_nlp", "uol_parse", "chat_frame"])
            self.assertEqual(report["api_smoke"]["ask"]["route"], "local_answer")
            self.assertEqual(report["api_smoke"]["ask"]["reason"], "local_story_inventory")
            self.assertTrue(report["api_session_smoke"]["passed"])
            self.assertTrue(all(report["api_session_smoke"]["checks"].values()))
            self.assertEqual(report["api_session_smoke"]["route_counts"]["device_action"], 4)
            self.assertTrue(report["ui_smoke"]["passed"])
            self.assertTrue(all(report["ui_smoke"]["checks"].values()))
            self.assertTrue(report["ui_smoke"]["ui"]["dependency_free"])
            self.assertEqual(report["ui_smoke"]["ask"]["route"], "local_answer")
            self.assertTrue(report["bootstrap_runtime"]["passed"])
            self.assertTrue(all(report["bootstrap_runtime"]["checks"].values()))
            self.assertEqual(report["bootstrap_runtime"]["counts"]["events"], 3)
            self.assertTrue(report["launcher_smoke"]["passed"])
            self.assertTrue(all(report["launcher_smoke"]["checks"].values()))
            self.assertIn("start_app", report["launcher_smoke"]["platform_launcher"])
            self.assertIn("health_check", report["launcher_smoke"]["health_launcher"])
            self.assertEqual(report["launcher_smoke"]["health"]["counts"]["events"], 3)
            self.assertTrue(report["launcher_smoke"]["checks"]["launcher_shutdown_endpoint_ok"])
            self.assertTrue(report["launcher_smoke"]["checks"]["launcher_process_stopped"])
            self.assertTrue(report["open_traces"]["passed"])
            self.assertEqual(report["open_traces"]["turns"], 29)
            self.assertTrue(report["open_traces"]["debug_checks"]["debug_maps_present"])
            self.assertTrue(report["transcript_replay"]["passed"])
            self.assertEqual(report["transcript_replay"]["turns"], 25)
            self.assertTrue(report["transcript_replay"]["fixture_checks"]["no_static_answer_or_route_expectations"])
            self.assertFalse(report["bundle"]["required_network"])
            self.assertFalse(report["bundle"]["required_vector_db"])
            self.assertFalse(report["bundle"]["required_ml_framework"])
            self.assertGreaterEqual(report["bundle"]["file_count"], 20)
            self.assertEqual(manifest["entrypoint"], "scripts/local_assistant_os_cli.py")
            self.assertEqual(manifest["bundle_name"], "melm_local_assistant_os_v01_portable_bundle")
            self.assertEqual(manifest["runtime"], "stdlib_python_sqlite_http_html")
            self.assertEqual(manifest["portable_chat_command"], "python3 scripts/local_assistant_os_cli.py chat")
            self.assertEqual(
                manifest["portable_api_session_command"],
                "python3 scripts/local_assistant_os_cli.py api-session-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_dataset_audit_command"],
                "python3 scripts/local_assistant_os_cli.py dataset-audit --reset --json",
            )
            self.assertEqual(
                manifest["portable_autoimmune_command"],
                "python3 scripts/local_assistant_os_cli.py autoimmune-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_synthesis_variant_command"],
                "python3 scripts/local_assistant_os_cli.py synthesis-variant-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_synthesis_stress_command"],
                "python3 scripts/local_assistant_os_cli.py synthesis-stress-smoke --reset --json",
            )
            self.assertEqual(
                manifest["setup_integration_smoke_command"],
                "python scripts/local_assistant_os_cli.py setup-integration-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_setup_integration_command"],
                "python3 scripts/local_assistant_os_cli.py setup-integration-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_host_action_command"],
                "python3 scripts/local_assistant_os_cli.py host-action-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_host_actions_demo_config_command"],
                "python3 scripts/local_assistant_os_cli.py write-host-actions-demo-config --out config/host_actions.local_recorder.json --overwrite --json",
            )
            self.assertEqual(
                manifest["portable_host_app_probe_command"],
                "python3 scripts/local_assistant_os_cli.py host-app-probe --reset --json",
            )
            self.assertEqual(
                manifest["portable_host_app_configured_probe_command"],
                "python3 scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.json --require-configured --json",
            )
            self.assertEqual(
                manifest["portable_host_app_demo_config_probe_command"],
                "python3 scripts/local_assistant_os_cli.py host-app-probe --config-json config/host_actions.local_recorder.json --require-configured --json",
            )
            self.assertEqual(
                manifest["portable_capability_probe_command"],
                "python3 scripts/local_assistant_os_cli.py capability-probe --reset --json",
            )
            self.assertEqual(
                manifest["shortcut_audit_command"],
                "python scripts/local_assistant_os_cli.py shortcut-audit --json",
            )
            self.assertEqual(
                manifest["portable_shortcut_audit_command"],
                "python3 scripts/local_assistant_os_cli.py shortcut-audit --json",
            )
            self.assertEqual(
                manifest["v01_audit_command"],
                "python scripts/local_assistant_os_cli.py v01-audit --json",
            )
            self.assertEqual(
                manifest["portable_v01_audit_command"],
                "python3 scripts/local_assistant_os_cli.py v01-audit --json",
            )
            self.assertEqual(
                manifest["v01_progress_command"],
                "python scripts/local_assistant_os_cli.py v01-progress --json",
            )
            self.assertEqual(
                manifest["portable_v01_progress_command"],
                "python3 scripts/local_assistant_os_cli.py v01-progress --json",
            )
            self.assertIn(
                "python scripts/local_assistant_os_cli.py v01-evidence-pack",
                manifest["v01_evidence_pack_command"],
            )
            self.assertIn(
                "--db artifacts/local_assistant_os/assistant_v01.sqlite",
                manifest["v01_evidence_pack_command"],
            )
            self.assertIn(
                "python3 scripts/local_assistant_os_cli.py v01-evidence-pack",
                manifest["portable_v01_evidence_pack_command"],
            )
            self.assertIn(
                "--auto-lifecycle",
                manifest["portable_v01_evidence_pack_command"],
            )
            self.assertIn(
                "python scripts/local_assistant_os_cli.py v01-blocker-evidence",
                manifest["v01_blocker_evidence_command"],
            )
            self.assertIn(
                "python scripts/local_assistant_os_cli.py candidate-session-audit",
                manifest["candidate_session_audit_command"],
            )
            self.assertIn(
                "--session all",
                manifest["candidate_session_audit_command"],
            )
            self.assertIn(
                "python scripts/local_assistant_os_cli.py write-source-attestation",
                manifest["source_attestation_command"],
            )
            self.assertIn(
                "--event-ledger-session all",
                manifest["source_attestation_command"],
            )
            self.assertIn(
                "python scripts/local_assistant_os_cli.py write-host-app-attestation",
                manifest["host_app_attestation_command"],
            )
            self.assertIn(
                "--not-demo-recorder",
                manifest["host_app_attestation_command"],
            )
            self.assertIn(
                "--source-attestation-json artifacts/local_assistant_os/source_attestation.json",
                manifest["v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--event-ledger-session all",
                manifest["v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--transcript-calibration-report-json artifacts/local_assistant_os/user_transcript_calibration.json",
                manifest["v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json",
                manifest["v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--event-source-kind redacted_user_session",
                manifest["v01_blocker_evidence_command"],
            )
            self.assertIn(
                "python3 scripts/local_assistant_os_cli.py write-source-attestation",
                manifest["portable_source_attestation_command"],
            )
            self.assertIn(
                "python3 scripts/local_assistant_os_cli.py candidate-session-audit",
                manifest["portable_candidate_session_audit_command"],
            )
            self.assertIn(
                "--session all",
                manifest["portable_candidate_session_audit_command"],
            )
            self.assertIn(
                "--event-ledger-session all",
                manifest["portable_source_attestation_command"],
            )
            self.assertIn(
                "python3 scripts/local_assistant_os_cli.py write-host-app-attestation",
                manifest["portable_host_app_attestation_command"],
            )
            self.assertIn(
                "python3 scripts/local_assistant_os_cli.py v01-blocker-evidence",
                manifest["portable_v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--source-attestation-json artifacts/local_assistant_os/source_attestation.json",
                manifest["portable_v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--event-ledger-session all",
                manifest["portable_v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--transcript-calibration-report-json artifacts/local_assistant_os/user_transcript_calibration.json",
                manifest["portable_v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--host-app-attestation-json artifacts/local_assistant_os/host_app_attestation.json",
                manifest["portable_v01_blocker_evidence_command"],
            )
            self.assertIn(
                "--event-source-kind redacted_user_session",
                manifest["portable_v01_blocker_evidence_command"],
            )
            self.assertEqual(
                manifest["v01_acceptance_command"],
                "python scripts/local_assistant_os_cli.py v01-acceptance --reset --json",
            )
            self.assertEqual(
                manifest["portable_v01_acceptance_command"],
                "python3 scripts/local_assistant_os_cli.py v01-acceptance --reset --json",
            )
            self.assertEqual(
                manifest["portable_v01_acceptance_configured_host_app_command"],
                "python3 scripts/local_assistant_os_cli.py v01-acceptance --host-app-config-json config/host_actions.json --require-host-app-configured --json",
            )
            self.assertEqual(
                manifest["portable_ui_command"],
                "python3 scripts/local_assistant_os_cli.py ui-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_bootstrap_runtime_command"],
                "python3 scripts/local_assistant_os_cli.py bootstrap-runtime --reset --json",
            )
            self.assertEqual(
                manifest["portable_open_traces_command"],
                "python3 scripts/local_assistant_os_cli.py run-open-traces --reset --json",
            )
            self.assertEqual(
                manifest["portable_transcript_replay_command"],
                "python3 scripts/local_assistant_os_cli.py run-transcript-replay --reset --json",
            )
            self.assertEqual(
                manifest["portable_transcript_calibration_command"],
                'python3 scripts/local_assistant_os_cli.py calibrate-transcript-replay --input benchmarks/sample_local_assistant_raw_transcript.jsonl --replace "Maya=<person_1>" --min-total-turns 4 --min-local-resolution-rate 0.2 --min-route-kinds 3 --min-intent-kinds 3 --require-redaction --require-static-drop --out artifacts/local_assistant_os/sample_transcript_calibration.json --reset --json',
            )
            self.assertIn("export-transcript-replay", manifest["portable_event_transcript_export_command"])
            self.assertIn("event_ledger_transcript_replay.jsonl", manifest["portable_event_transcript_export_command"])
            self.assertIn("calibrate-event-ledger", manifest["portable_event_ledger_calibration_command"])
            self.assertIn("event_ledger_calibration", manifest["portable_event_ledger_calibration_command"])
            self.assertIn(
                "config/safe_lifecycle_controls.example.json",
                manifest["portable_event_ledger_calibration_command"],
            )
            self.assertEqual(
                manifest["portable_safe_lifecycle_controls_template"],
                "config/safe_lifecycle_controls.example.json",
            )
            self.assertIn("--controls-json", manifest["portable_user_transcript_import_command"])
            self.assertIn(
                "config/safe_lifecycle_controls.example.json",
                manifest["portable_user_transcript_import_command"],
            )
            self.assertIn("import-transcript-replay", manifest["portable_user_transcript_import_command"])
            self.assertIn("--controls-json", manifest["portable_user_transcript_calibration_command"])
            self.assertIn(
                "config/safe_lifecycle_controls.example.json",
                manifest["portable_user_transcript_calibration_command"],
            )
            self.assertIn("--require-priority-signals", manifest["portable_user_transcript_calibration_command"])
            self.assertIn("--require-strict-baseline-win", manifest["portable_user_transcript_calibration_command"])
            self.assertIn(
                "--out artifacts/local_assistant_os/user_transcript_calibration.json",
                manifest["portable_user_transcript_calibration_command"],
            )
            self.assertEqual(
                manifest["portable_refresh_weather_command"],
                "python3 scripts/local_assistant_os_cli.py refresh-weather --offline-json benchmarks/sample_open_meteo_forecast.json --json",
            )
            self.assertEqual(
                manifest["portable_verify_command"],
                "python3 scripts/local_assistant_os_cli.py verify-bundle --json",
            )
            self.assertEqual(
                manifest["portable_launcher_smoke_command"],
                "python3 scripts/local_assistant_os_cli.py launcher-smoke --reset --json",
            )
            self.assertEqual(
                manifest["portable_first_run_smoke_command"],
                "python3 scripts/local_assistant_os_cli.py first-run-smoke --json",
            )
            self.assertEqual(
                manifest["portable_target_report_command"],
                "python3 scripts/local_assistant_os_cli.py target-report --reset --json",
            )
            self.assertTrue(manifest["self_check"]["transcript_baseline_checks"]["same_user_turns_compared"])
            self.assertTrue(
                manifest["self_check"]["transcript_baseline_checks"][
                    "current_beats_best_baseline_local_resolution"
                ]
            )
            self.assertTrue(manifest["self_check"]["transcript_calibration_checks"]["passed"])
            self.assertTrue(all(manifest["self_check"]["transcript_calibration_checks"]["checks"].values()))
            self.assertTrue(manifest["self_check"]["transcript_calibration_checks"]["thresholds"]["require_redaction"])
            self.assertTrue(manifest["self_check"]["transcript_calibration_checks"]["thresholds"]["require_static_drop"])
            self.assertEqual(
                manifest["inventory_soak_matrix_command"],
                "python scripts/local_assistant_os_cli.py inventory-soak-matrix --reset --json",
            )
            self.assertTrue(manifest["self_check"]["inventory_soak_matrix_checks"]["both_source_families_covered"])
            self.assertTrue(
                manifest["self_check"]["inventory_soak_matrix_checks"][
                    "future_story_routes_local_from_imported_inventory"
                ]
            )
            self.assertTrue(manifest["self_check"]["host_app_probe_checks"]["configuration_reported"])
            self.assertFalse(manifest["self_check"]["host_app_probe_checks"]["probe_executed"])
            self.assertTrue(manifest["self_check"]["setup_integration_checks"]["future_memory_routes_change_after_setup"])
            self.assertTrue(manifest["self_check"]["setup_integration_checks"]["trusted_contact_action_uses_confirmation_gate"])
            self.assertTrue(manifest["self_check"]["shortcut_audit_checks"]["behavior_probes_passed"])
            self.assertTrue(manifest["self_check"]["shortcut_audit_checks"]["source_boundaries_passed"])
            self.assertTrue(manifest["self_check"]["v01_audit_checks"]["completion_blockers_explicit"])
            self.assertTrue(manifest["self_check"]["v01_progress_checks"]["audit_passed"])
            self.assertEqual(manifest["self_check"]["v01_progress_remaining_blockers"], 6)
            self.assertFalse(manifest["self_check"]["architecture_complete"])
            self.assertEqual(manifest["self_check"]["completion_blocker_count"], 6)
            self.assertEqual(manifest["self_check"]["transcript_calibration_checks"]["turns_imported"], 4)
            self.assertEqual(manifest["self_check"]["transcript_calibration_checks"]["redaction_counts"]["email"], 1)
            self.assertEqual(manifest["self_check"]["transcript_calibration_checks"]["redaction_counts"]["phone"], 1)
            self.assertEqual(manifest["self_check"]["transcript_calibration_checks"]["redaction_counts"]["manual_rule_1"], 1)
            self.assertEqual(manifest["portable_first_run_script"], "sh bin/first_run.sh")
            self.assertEqual(
                manifest["portable_inventory_soak_matrix_command"],
                "python3 scripts/local_assistant_os_cli.py inventory-soak-matrix --reset --json",
            )
            self.assertEqual(manifest["portable_start_app_command"], "sh bin/start_app.sh or bin\\start_app.cmd")
            self.assertEqual(manifest["portable_windows_first_run"], "bin\\first_run.cmd")
            self.assertEqual(manifest["portable_windows_start_app"], "bin\\start_app.cmd")
            self.assertEqual(manifest["portable_raspberry_first_run_script"], "sh bin/first_run_on_raspberry_pi.sh")
            self.assertEqual(manifest["portable_browser_url"], "http://127.0.0.1:8771/")
            self.assertEqual(manifest["portable_start_api_command"], "sh bin/start_api.sh")
            self.assertEqual(manifest["portable_health_check_command"], "sh bin/health_check.sh")
            self.assertEqual(
                manifest["systemd_service_example"],
                "systemd/melm-local-assistant.service.example",
            )
            self.assertEqual(manifest["self_check"]["passed"], True)
            self.assertEqual(self_check["passed"], True)
            self.assertTrue(self_check["dataset_audit"]["passed"])
            self.assertTrue(self_check["pi_smoke"]["passed"])
            self.assertTrue(self_check["pi_smoke"]["checks"]["inventory_soak_passed"])
            self.assertTrue(self_check["pi_smoke"]["checks"]["inventory_soak_matrix_passed"])
            self.assertTrue(self_check["pi_smoke"]["checks"]["inventory_retry_smoke_passed"])
            self.assertTrue(self_check["pi_smoke"]["checks"]["inventory_failure_smoke_passed"])
            self.assertTrue(self_check["pi_smoke"]["inventory_soak"]["failure_observability"]["present"])
            self.assertTrue(
                self_check["pi_smoke"]["inventory_soak_matrix"]["checks"][
                    "future_story_routes_local_from_imported_inventory"
                ]
            )
            self.assertTrue(self_check["pi_smoke"]["inventory_retry_smoke"]["checks"]["future_story_routes_local_after_reload"])
            self.assertTrue(self_check["pi_smoke"]["inventory_failure_smoke"]["checks"]["errors_are_observable"])
            self.assertTrue(self_check["autoimmune_smoke"]["passed"])
            self.assertTrue(self_check["synthesis_variant_smoke"]["passed"])
            self.assertTrue(
                self_check["synthesis_variant_smoke"]["checks"]["story_tale_not_exact_story_phrase"]
            )
            self.assertTrue(self_check["synthesis_stress_smoke"]["passed"])
            self.assertTrue(
                self_check["synthesis_stress_smoke"]["checks"]["autobiographical_summaries_use_events_and_digest"]
            )
            self.assertTrue(self_check["setup_integration_smoke"]["passed"])
            self.assertTrue(
                self_check["setup_integration_smoke"]["checks"]["setup_requests_recorded_without_fake_facts"]
            )
            self.assertEqual(
                self_check["setup_integration_smoke"]["facts_after_setup"]["morning_routine"],
                "stretch, breakfast, then bus",
            )
            self.assertTrue(self_check["host_action_smoke"]["passed"])
            self.assertTrue(self_check["host_app_probe"]["passed"])
            self.assertTrue(self_check["host_app_probe"]["skipped"])
            self.assertTrue(self_check["capability_probe"]["passed"])
            self.assertTrue(self_check["shortcut_audit"]["passed"])
            self.assertTrue(self_check["shortcut_audit"]["checks"]["behavior_probes_passed"])
            self.assertTrue(self_check["shortcut_audit"]["checks"]["source_boundaries_passed"])
            self.assertTrue(self_check["v01_audit"]["passed"])
            self.assertFalse(self_check["v01_audit"]["architecture_complete"])
            self.assertEqual(self_check["v01_audit"]["blocker_count"], 6)
            self.assertTrue(self_check["v01_progress"]["passed"])
            self.assertFalse(self_check["v01_progress"]["architecture_complete"])
            self.assertEqual(self_check["v01_progress"]["remaining_blocker_count"], 6)
            self.assertTrue(self_check["api_smoke"]["passed"])
            self.assertTrue(self_check["api_session_smoke"]["passed"])
            self.assertTrue(self_check["ui_smoke"]["passed"])
            self.assertTrue(self_check["bootstrap_runtime"]["passed"])
            self.assertTrue(self_check["launcher_smoke"]["passed"])
            self.assertTrue(self_check["open_traces"]["passed"])
            self.assertTrue(self_check["transcript_replay"]["passed"])
            self.assertTrue(self_check["transcript_replay"]["baseline_comparison"]["passed"])
            self.assertTrue(self_check["transcript_calibration"]["passed"])
            self.assertEqual(self_check["transcript_calibration"]["aggregate"]["turns_imported"], 4)
            self.assertTrue(all(self_check["transcript_calibration"]["aggregate"]["checks"].values()))
            self.assertEqual(
                self_check["transcript_replay"]["baseline_comparison"]["wins"]["cloud_handoff_reduction_vs_best_baseline"],
                7,
            )
            self.assertIn("scripts/local_assistant_os_cli.py", bundled_paths)
            self.assertIn("config/host_actions.example.json", bundled_paths)
            self.assertIn("config/safe_lifecycle_controls.example.json", bundled_paths)
            self.assertIn("tests/test_local_assistant_router_mvp.py", bundled_paths)
            self.assertIn("melm/appliance/assistant_os_kernel.py", bundled_paths)
            self.assertIn("benchmarks/local_assistant_os_seed.json", bundled_paths)
            self.assertIn("benchmarks/local_media_manifest.json", bundled_paths)
            self.assertIn("benchmarks/sample_open_meteo_forecast.json", bundled_paths)
            self.assertIn("benchmarks/local_assistant_open_traces.json", bundled_paths)
            self.assertIn("benchmarks/local_assistant_transcript_replay.jsonl", bundled_paths)
            self.assertIn("benchmarks/sample_local_assistant_raw_transcript.jsonl", bundled_paths)
            self.assertIn("bin/first_run.sh", bundled_paths)
            self.assertIn("bin/start_app.sh", bundled_paths)
            self.assertIn("bin/first_run_on_raspberry_pi.sh", bundled_paths)
            self.assertIn("bin/start_api.sh", bundled_paths)
            self.assertIn("bin/health_check.sh", bundled_paths)
            self.assertIn("bin/first_run.ps1", bundled_paths)
            self.assertIn("bin/start_app.ps1", bundled_paths)
            self.assertIn("bin/health_check.ps1", bundled_paths)
            self.assertIn("bin/first_run.cmd", bundled_paths)
            self.assertIn("bin/start_app.cmd", bundled_paths)
            self.assertIn("bin/health_check.cmd", bundled_paths)
            self.assertIn("systemd/melm-local-assistant.service.example", bundled_paths)
            runbook = runbook_path.read_text(encoding="utf-8")
            first_run = (out / "bin" / "first_run.sh").read_text(encoding="utf-8")
            start_app = (out / "bin" / "start_app.sh").read_text(encoding="utf-8")
            start_script = (out / "bin" / "start_api.sh").read_text(encoding="utf-8")
            first_run_script = (out / "bin" / "first_run_on_raspberry_pi.sh").read_text(encoding="utf-8")
            first_run_ps1 = (out / "bin" / "first_run.ps1").read_text(encoding="utf-8")
            start_app_cmd = (out / "bin" / "start_app.cmd").read_text(encoding="utf-8")
            service = (out / "systemd" / "melm-local-assistant.service.example").read_text(encoding="utf-8")
            self.assertTrue(runbook.startswith("# MELM Local Assistant OS"))
            self.assertIn("verify-bundle --json", runbook)
            self.assertIn("dataset-audit --reset --json", runbook)
            self.assertIn("target-report --reset --json", runbook)
            self.assertIn("bootstrap-runtime --reset --json", runbook)
            self.assertIn("api-session-smoke --reset --json", runbook)
            self.assertIn("autoimmune-smoke --reset --json", runbook)
            self.assertIn("synthesis-variant-smoke --reset --json", runbook)
            self.assertIn("synthesis-stress-smoke --reset --json", runbook)
            self.assertIn("setup-integration-smoke --reset --json", runbook)
            self.assertIn("host-action-smoke --reset --json", runbook)
            self.assertIn("write-host-actions-demo-config --out config/host_actions.local_recorder.json", runbook)
            self.assertIn("host-app-probe --reset --json", runbook)
            self.assertIn("host-app-probe --config-json config/host_actions.local_recorder.json --require-configured --json", runbook)
            self.assertIn("capability-probe --reset --json", runbook)
            self.assertIn("shortcut-audit --json", runbook)
            self.assertIn("v01-audit --json", runbook)
            self.assertIn("v01-progress --json", runbook)
            self.assertIn("v01-evidence-pack --db artifacts/local_assistant_os/assistant_v01.sqlite", runbook)
            self.assertIn("write-source-attestation", runbook)
            self.assertIn("write-host-app-attestation", runbook)
            self.assertIn("v01-blocker-evidence", runbook)
            self.assertIn("--source-attestation-json", runbook)
            self.assertIn("--host-app-attestation-json", runbook)
            self.assertIn("recorder config is development evidence only", runbook)
            self.assertIn("v01-acceptance --reset --json", runbook)
            self.assertIn("ui-smoke --reset --json", runbook)
            self.assertIn("launcher-smoke --reset --json", runbook)
            self.assertIn("first-run-smoke --json", runbook)
            self.assertIn("run-open-traces --reset --json", runbook)
            self.assertIn("run-transcript-replay --reset --json", runbook)
            self.assertIn("export-transcript-replay --db artifacts/local_assistant_os/assistant_v01.sqlite", runbook)
            self.assertIn("calibrate-event-ledger --db artifacts/local_assistant_os/assistant_v01.sqlite", runbook)
            self.assertIn("inventory-soak-matrix --reset --json", runbook)
            self.assertIn("calibrate-transcript-replay --input benchmarks/sample_local_assistant_raw_transcript.jsonl", runbook)
            self.assertIn("import-transcript-replay --input <raw_chat.jsonl>", runbook)
            self.assertIn("--controls-json config/safe_lifecycle_controls.example.json", runbook)
            self.assertIn("--require-strict-baseline-win", runbook)
            self.assertIn("api-session-smoke --action-mode real --host-app-config-json config/host_actions.json", runbook)
            self.assertIn("serve --action-mode real --host-app-config-json config/host_actions.json", runbook)
            self.assertIn("http://127.0.0.1:8771/", runbook)
            self.assertIn("sh bin/start_app.sh", runbook)
            self.assertIn("bin\\start_app.cmd", runbook)
            self.assertIn("chat --turn", runbook)
            self.assertIn("target-report --reset --json", first_run)
            self.assertIn("dataset-audit --reset --json", first_run)
            self.assertIn("ui-smoke --reset --json", first_run)
            self.assertIn("launcher-smoke --reset --json", first_run)
            self.assertIn("serve --db", start_app)
            self.assertIn("serve --db", start_script)
            self.assertIn("dataset-audit\", \"--reset\", \"--json", first_run_ps1)
            self.assertIn("target-report --reset --require-raspberry-pi --json", first_run_script)
            self.assertIn("target-report\", \"--reset\", \"--json", first_run_ps1)
            self.assertIn("powershell -NoProfile -ExecutionPolicy Bypass", start_app_cmd)
            self.assertIn("ExecStart=/bin/sh", service)
            self.assertIn("melm_local_assistant_os_v01_portable_bundle", service)
            verify = _run_cli("verify-bundle", "--bundle-root", str(out), "--json")
            self.assertTrue(verify["passed"])
            self.assertTrue(all(verify["checks"].values()))
            self.assertTrue(verify["self_check_summary"]["autoimmune_smoke_passed"])
            self.assertTrue(verify["self_check_summary"]["synthesis_variant_smoke_passed"])
            self.assertTrue(verify["self_check_summary"]["synthesis_stress_smoke_passed"])
            self.assertTrue(verify["self_check_summary"]["setup_integration_smoke_passed"])
            self.assertTrue(verify["self_check_summary"]["dataset_audit_passed"])
            self.assertTrue(verify["self_check_summary"]["pi_smoke_inventory_soak_matrix_passed"])
            self.assertTrue(verify["self_check_summary"]["pi_smoke_inventory_failure_passed"])
            self.assertTrue(verify["self_check_summary"]["pi_smoke_inventory_retry_passed"])
            self.assertTrue(verify["self_check_summary"]["host_action_smoke_passed"])
            self.assertTrue(verify["self_check_summary"]["host_app_probe_passed"])
            self.assertFalse(verify["self_check_summary"]["host_app_probe_configured"])
            self.assertTrue(verify["self_check_summary"]["host_app_probe_skipped"])
            self.assertTrue(verify["self_check_summary"]["transcript_calibration_passed"])
            self.assertTrue(verify["self_check_summary"]["capability_probe_passed"])
            self.assertTrue(verify["self_check_summary"]["shortcut_audit_passed"])
            self.assertTrue(verify["self_check_summary"]["v01_audit_passed"])
            self.assertTrue(verify["self_check_summary"]["v01_progress_passed"])
            self.assertEqual(verify["self_check_summary"]["v01_progress_remaining_blockers"], 6)
            self.assertFalse(verify["self_check_summary"]["v01_audit_architecture_complete"])
            self.assertEqual(verify["self_check_summary"]["v01_audit_blocker_count"], 6)
            self.assertTrue(verify["self_check_summary"]["launcher_smoke_passed"])
            self.assertTrue(verify["self_check_summary"]["open_traces_passed"])
            self.assertTrue(verify["self_check_summary"]["transcript_replay_passed"])
            self.assertEqual(verify["verified_files"], manifest["file_count"])
            self.assertEqual(verify["missing_files"], [])
            self.assertEqual(verify["sha256_mismatches"], [])
            self.assertEqual(verify["missing_launcher_files"], [])
            self.assertFalse((out / "artifacts").exists())
            first_run_smoke = _run_cli("first-run-smoke", "--bundle-root", str(out), "--json")
            self.assertTrue(first_run_smoke["passed"])
            self.assertTrue(all(first_run_smoke["checks"].values()))
            self.assertEqual(first_run_smoke["json_reports"], 6)
            self.assertEqual(first_run_smoke["reports"]["verify_bundle"]["verified_files"], manifest["file_count"])
            self.assertTrue(first_run_smoke["reports"]["dataset_audit"]["passed"])
            self.assertEqual(first_run_smoke["reports"]["dataset_audit"]["source_fixtures"]["open_trace_turns"], 29)
            self.assertEqual(first_run_smoke["reports"]["dataset_audit"]["source_fixtures"]["transcript_replay_user_turns"], 25)
            self.assertTrue(first_run_smoke["reports"]["target_report"]["passed"])
            self.assertTrue(first_run_smoke["reports"]["bootstrap_runtime"]["passed"])
            self.assertEqual(first_run_smoke["reports"]["bootstrap_runtime"]["counts"]["events"], 3)
            self.assertTrue(first_run_smoke["reports"]["ui_smoke"]["passed"])
            self.assertTrue(first_run_smoke["reports"]["launcher_smoke"]["passed"])
            self.assertTrue(Path(first_run_smoke["artifacts"]["runtime_db"]).exists())
            self.assertTrue(Path(first_run_smoke["artifacts"]["target_report_dir"]).exists())
            self.assertIn("first_run", first_run_smoke["first_run_launcher"])
            archive_smoke = _run_cli(
                "archive-smoke",
                "--archive",
                str(archive_path),
                "--work-dir",
                str(archive_extract),
                "--reset",
                "--json",
            )
            self.assertTrue(archive_smoke["passed"])
            self.assertTrue(all(archive_smoke["checks"].values()))
            self.assertEqual(archive_smoke["top_level_roots"], [out.name])
            self.assertEqual(Path(archive_smoke["extracted_bundle_root"]).name, out.name)
            self.assertEqual(archive_smoke["manifest"]["file_count"], manifest["file_count"])
            self.assertFalse(archive_smoke["manifest"]["required_network"])
            self.assertFalse(archive_smoke["manifest"]["required_vector_db"])
            self.assertFalse(archive_smoke["manifest"]["required_ml_framework"])
            self.assertEqual(archive_smoke["verify_bundle"]["verified_files"], manifest["file_count"])
            self.assertEqual(archive_smoke["verify_bundle"]["sha256_mismatches"], [])
            self.assertTrue(archive_smoke["first_run_smoke"]["passed"])
            self.assertEqual(archive_smoke["first_run_smoke"]["json_reports"], 6)
            self.assertTrue(Path(archive_smoke["first_run_smoke"]["artifacts"]["runtime_db"]).exists())
            (out / "README.md").write_text("tampered bundle file\n", encoding="utf-8")
            tampered = _run_cli("verify-bundle", "--bundle-root", str(out), "--json")
            self.assertFalse(tampered["passed"])
            self.assertFalse(tampered["checks"]["byte_counts_match"])
            self.assertFalse(tampered["checks"]["sha256_match"])
            self.assertEqual(tampered["sha256_mismatches"][0]["path"], "README.md")
            self.assertTrue((out / "artifacts").exists())
            self.assertTrue(archive_path.exists())
            archive_prefix = f"{out.name}/"
            with zipfile.ZipFile(archive_path) as archive_file:
                names = set(archive_file.namelist())
            self.assertIn(f"{archive_prefix}scripts/local_assistant_os_cli.py", names)
            self.assertIn(f"{archive_prefix}bundle_manifest.json", names)
            self.assertIn(f"{archive_prefix}bin/start_app.sh", names)
            self.assertIn(f"{archive_prefix}bin/start_app.cmd", names)
            self.assertIn(f"{archive_prefix}bin/start_api.sh", names)
            self.assertIn(f"{archive_prefix}systemd/melm-local-assistant.service.example", names)
        finally:
            for target in (root, archive, archive_extract):
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()

    def test_cli_blocks_confirmation_replay_without_pending_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            replay = _run_cli("ask", "--db", str(db), "--utterance", "Yes, call mom.", "--json")
            dashboard = _run_cli("dashboard", "--db", str(db), "--json")
            conn = sqlite3.connect(db)
            try:
                pending_count = conn.execute("SELECT COUNT(*) FROM pending_actions").fetchone()[0]
            finally:
                conn.close()

            self.assertEqual(replay["route"], "clarify")
            self.assertEqual(replay["reason"], "no_pending_action_to_confirm")
            self.assertFalse(replay["device_action"])
            self.assertEqual(replay["membrane"]["confirmation_required"], 0)
            self.assertEqual(pending_count, 0)
            self.assertEqual(dashboard["safety_flags"]["action_replay_blocks"], 1)
            self.assertEqual(dashboard["safety_flags"]["unconfirmed_executed_actions"], 0)

    def test_cli_cancel_pending_action_prevents_later_replay_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            first = _run_cli("ask", "--db", str(db), "--utterance", "I need to talk to someone.", "--json")
            cancel = _run_cli("ask", "--db", str(db), "--utterance", "Cancel that call.", "--json")
            replay = _run_cli("ask", "--db", str(db), "--utterance", "Yes, call mom.", "--json")
            dashboard = _run_cli("dashboard", "--db", str(db), "--json")
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT confirmation_state, executed, result FROM pending_actions"
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual(first["route"], "device_action")
            self.assertEqual(cancel["route"], "local_answer")
            self.assertEqual(cancel["reason"], "cancelled_pending_action")
            self.assertEqual(replay["reason"], "no_pending_action_to_confirm")
            self.assertEqual(row[0], "cancelled")
            self.assertEqual(row[1], 0)
            self.assertIn("Cancelled", row[2])
            self.assertEqual(dashboard["pending_actions"]["cancelled"], 1)
            self.assertEqual(dashboard["pending_actions"]["executed"], 0)
            self.assertEqual(dashboard["safety_flags"]["cancelled_pending_actions"], 1)
            self.assertEqual(dashboard["safety_flags"]["action_replay_blocks"], 1)

    def test_cli_confirmation_target_switch_does_not_execute_pending_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            first = _run_cli("ask", "--db", str(db), "--utterance", "I need to talk to someone.", "--json")
            mismatch = _run_cli("ask", "--db", str(db), "--utterance", "Yes, call dad.", "--json")
            dashboard = _run_cli("dashboard", "--db", str(db), "--json")
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT confirmation_state, executed FROM pending_actions"
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual(first["route"], "device_action")
            self.assertEqual(mismatch["route"], "clarify")
            self.assertEqual(mismatch["reason"], "confirmation_target_mismatch")
            self.assertEqual(row[0], "pending")
            self.assertEqual(row[1], 0)
            self.assertEqual(dashboard["safety_flags"]["confirmation_target_mismatches"], 1)

    def test_cli_forget_revokes_memory_and_stale_weather_fetches_instead_of_answering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli("init", "--db", str(db), "--json")
            forget = _run_cli("ask", "--db", str(db), "--utterance", "Forget my favorite color.", "--json")
            memory = _run_cli("ask", "--db", str(db), "--utterance", "Tell me something about myself.", "--json")
            conn = sqlite3.connect(db)
            try:
                conn.execute(
                    """
                    UPDATE inventories
                    SET payload_json=?
                    WHERE kind='weather' AND item_id='today'
                    """,
                    (json.dumps({"forecast": "warm with afternoon rain", "stale": True}),),
                )
                conn.commit()
            finally:
                conn.close()
            weather = _run_cli("ask", "--db", str(db), "--utterance", "What is the weather today?", "--json")
            dashboard = _run_cli("dashboard", "--db", str(db), "--json")

            self.assertEqual(forget["reason"], "consent_revoked_user_fact")
            self.assertNotIn("green", memory["answer"].lower())
            self.assertEqual(weather["route"], "external_fetch")
            self.assertEqual(weather["reason"], "weather_cache_miss")
            self.assertEqual(weather["homeostasis"]["cache_freshness"], 0.0)
            self.assertEqual(dashboard["safety_flags"]["consent_revocations"], 1)

    def test_cli_autoimmune_smoke_runs_privacy_action_cache_boundary_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "autoimmune.sqlite"
            report = _run_cli("autoimmune-smoke", "--db", str(db), "--reset", "--json")

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["runtime"], "stdlib_python_sqlite")
            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertEqual(len(report["turns"]), 26)
            by_label = {turn["label"]: turn for turn in report["turns"]}
            self.assertEqual(by_label["private_cloud_block"]["reason"], "blocked_private_facts_to_cloud")
            self.assertIn(by_label["generic_cloud_allowed"]["boundary_crossed"], {"local", "none"})
            self.assertEqual(by_label["public_profile_cloud_allowed"]["route"], "cloud_handoff")
            self.assertEqual(by_label["public_profile_cloud_allowed"]["boundary_crossed"], "cloud")
            self.assertIn(
                "facts.public_profile",
                by_label["public_profile_cloud_allowed"]["personal_facts_included"],
            )
            self.assertNotIn(
                "facts.public_profile",
                by_label["public_profile_cloud_allowed"]["personal_facts_excluded"],
            )
            self.assertEqual(by_label["household_setup"]["reason"], "consented_household_memory_stored")
            self.assertIn("facts.household_context", by_label["household_private_cloud"]["evidence_keys"])
            self.assertEqual(by_label["household_private_cloud"]["reason"], "blocked_private_facts_to_cloud")
            self.assertEqual(by_label["mixed_public_household_cloud"]["reason"], "blocked_private_facts_to_cloud")
            self.assertIn("facts.public_profile", by_label["mixed_public_household_cloud"]["evidence_keys"])
            self.assertIn("facts.household_context", by_label["mixed_public_household_cloud"]["evidence_keys"])
            self.assertIn(
                "facts.household_context",
                by_label["mixed_public_household_cloud"]["personal_facts_excluded"],
            )
            self.assertEqual(by_label["mixed_public_household_cloud"]["personal_facts_included"], [])
            self.assertEqual(by_label["household_consent_revoke"]["reason"], "consent_revoked_user_fact")
            self.assertEqual(by_label["household_after_revoke"]["reason"], "personal_memory_empty")
            self.assertNotIn("facts.household_context", by_label["household_after_revoke"]["evidence_keys"])
            self.assertEqual(by_label["consent_revoke"]["reason"], "consent_revoked_user_fact")
            self.assertNotIn("facts.favorite_color", by_label["memory_after_revoke"]["evidence_keys"])
            self.assertEqual(by_label["child_age_setup"]["evidence_keys"], ["facts.child_age"])
            self.assertEqual(by_label["child_school_setup"]["evidence_keys"], ["facts.child_school"])
            self.assertEqual(by_label["child_school_consent_revoke"]["reason"], "consent_revoked_user_fact")
            self.assertEqual(by_label["child_school_consent_revoke"]["evidence_keys"], ["facts.child_school"])
            self.assertEqual(by_label["child_school_after_revoke"]["reason"], "personal_memory_empty")
            self.assertNotIn("facts.child_school", by_label["child_school_after_revoke"]["evidence_keys"])
            self.assertNotIn("facts.school", by_label["child_school_after_revoke"]["evidence_keys"])
            self.assertEqual(by_label["child_age_after_school_revoke"]["evidence_keys"], ["facts.child_age"])
            self.assertEqual(by_label["stale_weather_cache"]["route"], "external_fetch")
            self.assertEqual(by_label["stale_weather_cache"]["cache_freshness"], 0.0)
            self.assertEqual(by_label["action_request"]["confirmation_required"], 1)
            self.assertEqual(by_label["invented_target"]["reason"], "confirmation_target_mismatch")
            self.assertEqual(by_label["replay_after_cancel"]["reason"], "no_pending_action_to_confirm")
            self.assertEqual(by_label["parent_child_private_cloud"]["reason"], "blocked_private_facts_to_cloud")
            self.assertIn("facts.child_age", by_label["parent_child_private_cloud"]["evidence_keys"])
            self.assertIn("facts.child_school", by_label["parent_child_private_cloud"]["evidence_keys"])
            self.assertNotIn("profile.age", by_label["parent_child_private_cloud"]["evidence_keys"])
            self.assertNotIn("facts.school", by_label["parent_child_private_cloud"]["evidence_keys"])
            self.assertEqual(by_label["child_location_private_cloud"]["reason"], "blocked_private_facts_to_cloud")
            self.assertEqual(by_label["child_location_private_cloud"]["evidence_keys"], ["facts.child_location"])
            self.assertEqual(by_label["media_action_request"]["confirmation_required"], 1)
            self.assertEqual(by_label["media_action_request"]["reason"], "local_media_action")
            self.assertEqual(by_label["invented_media_target"]["reason"], "confirmation_target_mismatch")
            self.assertEqual(by_label["cancel_media_pending"]["reason"], "cancelled_pending_action")
            self.assertEqual(by_label["conversation_export_block"]["reason"], "blocked_private_facts_to_cloud")
            self.assertTrue(report["checks"]["mixed_public_household_excludes_private_without_partial_cloud"])
            self.assertTrue(report["checks"]["final_pending_actions_do_not_linger"])
            self.assertEqual(report["pending_actions"]["cancelled"], 2)
            self.assertEqual(report["pending_actions"]["pending"], 0)
            self.assertEqual(report["pending_actions"]["executed"], 0)
            self.assertEqual(report["safety_flags"]["cloud_private_inclusions"], 0)
            self.assertEqual(report["safety_flags"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(report["safety_flags"]["confirmation_target_mismatches"], 2)

    def test_cli_refresh_weather_turns_cold_fetch_gap_into_cached_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            miss = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is the weather today?",
                "--cold-start",
                "--json",
            )
            refresh = _run_cli(
                "refresh-weather",
                "--db",
                str(db),
                "--cold-start",
                "--offline-json",
                "benchmarks/sample_open_meteo_forecast.json",
                "--json",
            )
            hit = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is the weather today?",
                "--cold-start",
                "--json",
            )

            self.assertEqual(miss["route"], "external_fetch")
            self.assertEqual(miss["reason"], "weather_cache_miss")
            self.assertFalse(miss["self_observation"]["cache_health"]["weather_cache_ready"])
            self.assertGreaterEqual(miss["self_observation"]["history_summary"]["points"], 2)
            self.assertFalse(refresh["result"]["network_used"])
            self.assertGreaterEqual(refresh["result"]["weather_days"], 7)
            self.assertIn("today", refresh["weekly_weather"])
            self.assertTrue(refresh["self_observation"]["cache_health"]["weather_cache_ready"])
            self.assertTrue(refresh["self_observation"]["history_summary"]["weather_cache_became_ready"])
            self.assertGreaterEqual(refresh["self_observation"]["cache_health"]["weather_days"], 7)
            self.assertEqual(hit["route"], "cached_tool")
            self.assertEqual(hit["reason"], "weather_cache_hit")
            self.assertTrue(hit["self_observation"]["cache_health"]["weather_cache_ready"])
            self.assertIn("rain", hit["answer"].lower())

            status = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What have you done so far?",
                "--cold-start",
                "--json",
            )
            self.assertIn("weather_cache=ready", status["answer"])
            self.assertIn("weather_cache_transition=ready", status["answer"])
            self.assertIn("self_status.self_observation", status["synthesis"]["citations"])

    def test_cli_run_jobs_executes_weather_refresh_job_from_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is the weather today?",
                "--cold-start",
                "--json",
            )
            schedule = _run_cli(
                "schedule-refreshes",
                "--db",
                str(db),
                "--cold-start",
                "--offline-samples",
                "--json",
            )
            jobs = _run_cli(
                "run-jobs",
                "--db",
                str(db),
                "--cold-start",
                "--limit",
                "2",
                "--weather-json",
                "benchmarks/sample_open_meteo_forecast.json",
                "--json",
            )
            hit = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is the weather today?",
                "--cold-start",
                "--json",
            )

            self.assertIn("refresh_weather_cache", [item["kind"] for item in schedule["recommendations"]])
            weather_job = next(item for item in jobs["executed"] if item["kind"] == "refresh_weather_cache")
            self.assertFalse(weather_job["network_used"])
            self.assertGreaterEqual(weather_job["weather_days"], 7)
            self.assertTrue(jobs["self_observation"]["cache_health"]["weather_cache_ready"])
            self.assertGreaterEqual(jobs["self_observation"]["job_health"]["completed"], 1)
            self.assertEqual(hit["route"], "cached_tool")
            self.assertEqual(hit["reason"], "weather_cache_hit")

    def test_cli_memory_replay_queries_persisted_event_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--cold-start", "--json")
            _run_cli("ask", "--db", str(db), "--utterance", "What is the weather today?", "--cold-start", "--json")
            _run_cli("ask", "--db", str(db), "--utterance", "I need to talk to someone.", "--json")

            story = _run_cli("memory-replay", "--db", str(db), "--query", "story", "--json")
            latest = _run_cli("memory-replay", "--db", str(db), "--session", "latest", "--json")
            contact = _run_cli(
                "memory-replay",
                "--db",
                str(db),
                "--intent",
                "social_contact",
                "--route",
                "device_action",
                "--json",
            )

            self.assertTrue(story["local_only"])
            self.assertEqual(story["matches"], 1)
            self.assertEqual(story["events"][0]["intent"], "story")
            self.assertIn("story", story["events"][0]["utterance"].lower())
            self.assertEqual(contact["matches"], 1)
            self.assertTrue(contact["events"][0]["device_action"])
            self.assertEqual(latest["memory"]["events"], 3)
            self.assertEqual(latest["memory"]["dangling_previous"], 0)
            self.assertEqual(latest["memory"]["dangling_next"], 0)
            self.assertGreaterEqual(len(latest["memory"]["recent_sessions"]), 1)

    def test_cli_ask_can_recall_conversation_memory_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--cold-start", "--json")
            _run_cli("ask", "--db", str(db), "--utterance", "What is the weather today?", "--cold-start", "--json")

            recall = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What did we talk about earlier?",
                "--cold-start",
                "--json",
            )
            blocked = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Send our previous conversation to the cloud.",
                "--cold-start",
                "--json",
            )

            self.assertEqual(recall["route"], "local_answer")
            self.assertEqual(recall["intent"], "autobiographical_memory")
            self.assertEqual(recall["reason"], "autobiographical_memory_summary")
            self.assertIn("conversation memory", recall["answer"].lower())
            self.assertTrue(recall["synthesis"]["applied"])
            self.assertTrue(any(key.startswith("events.") for key in recall["evidence_keys"]))
            self.assertEqual(blocked["route"], "reject")
            self.assertEqual(blocked["reason"], "blocked_private_facts_to_cloud")
            self.assertIn("events.local_conversation", blocked["evidence_keys"])
            self.assertTrue(blocked["synthesis"]["refused"])

    def test_cli_surfaces_future_media_routine_household_opportunities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            media = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Play a song for me.",
                "--cold-start",
                "--json",
            )
            routine = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is my morning routine?",
                "--cold-start",
                "--json",
            )
            household = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What do you know about this household?",
                "--cold-start",
                "--json",
            )

            self.assertEqual(media["reason"], "empty_media_library")
            self.assertIn("build_media_index", media["opportunities"])
            self.assertEqual(routine["reason"], "personal_memory_empty")
            self.assertIn("ask_routine_memory", routine["opportunities"])
            self.assertEqual(household["reason"], "personal_memory_empty")
            self.assertIn("ask_household_memory", household["opportunities"])

    def test_cli_execute_jobs_records_setup_request_without_fabricating_routine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            first = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is my morning routine?",
                "--cold-start",
                "--execute-jobs",
                "--json",
            )
            second = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is my morning routine?",
                "--cold-start",
                "--json",
            )
            supplied = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "My morning routine is stretch, breakfast, then bus.",
                "--cold-start",
                "--json",
            )
            recall = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What is my morning routine?",
                "--cold-start",
                "--json",
            )
            conn = sqlite3.connect(db)
            try:
                setup_row = conn.execute(
                    """
                    SELECT payload_json
                    FROM inventories
                    WHERE kind='setup_request' AND item_id='routine_memory'
                    """
                ).fetchone()
                fact_row = conn.execute(
                    "SELECT value FROM user_facts WHERE key='facts.morning_routine'"
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual(first["reason"], "personal_memory_empty")
            self.assertIn("ask_routine_memory", first["executed_jobs"])
            self.assertEqual(second["reason"], "personal_memory_empty")
            self.assertEqual(supplied["reason"], "consented_routine_memory_stored")
            self.assertEqual(recall["route"], "local_answer")
            self.assertIn("stretch", recall["answer"])
            self.assertIsNotNone(setup_row)
            self.assertIsNotNone(fact_row)
            self.assertIn("stretch", fact_row[0])

    def test_cli_memory_replay_and_chat_summarize_recent_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--cold-start", "--json")
            _run_cli("ask", "--db", str(db), "--utterance", "What is the weather today?", "--cold-start", "--json")
            _run_cli("ask", "--db", str(db), "--utterance", "I need to talk to someone.", "--json")

            replay = _run_cli(
                "memory-replay",
                "--db",
                str(db),
                "--sessions",
                "3",
                "--events-per-session",
                "1",
                "--json",
            )
            recall = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Summarize our recent sessions.",
                "--json",
            )

            self.assertEqual(replay["session_count"], 3)
            self.assertEqual(replay["matches"], 3)
            self.assertEqual(replay["session_ids"], ["session_1", "session_2", "session_3"])
            self.assertEqual(recall["route"], "local_answer")
            self.assertEqual(recall["intent"], "autobiographical_memory")
            self.assertEqual(recall["reason"], "autobiographical_session_summary")
            self.assertIn("session 1 (session_1)", recall["answer"])
            self.assertIn("session 3 (session_3)", recall["answer"])
            self.assertIn("Open local gaps:", recall["answer"])
            self.assertIn("story inventory was missing", recall["answer"])
            self.assertIn("weather cache was missing", recall["answer"])
            self.assertIn("Action state:", recall["answer"])
            self.assertTrue(recall["synthesis"]["applied"])
            self.assertTrue(all(key.startswith("events.") for key in recall["evidence_keys"]))

    def test_cli_memory_digest_compacts_and_chat_uses_long_horizon_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--cold-start", "--json")
            _run_cli("ask", "--db", str(db), "--utterance", "What is the weather today?", "--cold-start", "--json")
            _run_cli("ask", "--db", str(db), "--utterance", "I need to talk to someone.", "--json")

            digest = _run_cli(
                "memory-digest",
                "--db",
                str(db),
                "--sessions",
                "3",
                "--events-per-session",
                "2",
                "--json",
            )
            recall = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "What happened over the last few days?",
                "--json",
            )
            dashboard = _run_cli("dashboard", "--db", str(db), "--json")

            self.assertEqual(digest["digest"]["digest_id"], "long_horizon_latest")
            self.assertTrue(digest["digest"]["local_only"])
            self.assertGreaterEqual(digest["digest"]["session_count"], 3)
            self.assertGreaterEqual(digest["digest"]["event_count"], 3)
            self.assertTrue(digest["digest"]["quality"]["passed"])
            self.assertGreaterEqual(digest["digest"]["quality"]["score"], digest["digest"]["quality"]["floor"])
            self.assertIn("story_inventory", {item["thread"] for item in digest["digest"]["threads"]})
            self.assertIn("weather_cache", {item["thread"] for item in digest["digest"]["threads"]})
            self.assertIn("main threads:", digest["digest"]["summary"])
            self.assertIn("limits/open loops:", digest["digest"]["summary"])
            self.assertEqual(recall["reason"], "autobiographical_memory_digest")
            self.assertEqual(recall["evidence_keys"], ["memory_digest.long_horizon_latest"])
            self.assertIn("long-horizon memory digest", recall["answer"])
            self.assertIn("main threads:", recall["answer"])
            self.assertIn("story inventory", recall["answer"])
            self.assertTrue(recall["synthesis"]["applied"])
            self.assertGreaterEqual(dashboard["memory"]["memory_digests"], 1)
            self.assertTrue(dashboard["memory"]["latest_memory_digest"]["quality_passed"])
            self.assertGreaterEqual(
                dashboard["memory"]["latest_memory_digest"]["quality_score"],
                dashboard["memory"]["latest_memory_digest"]["quality_floor"],
            )

    def test_cli_inventory_soak_runs_repeated_replayable_refresh_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            report = _run_cli(
                "inventory-soak",
                "--db",
                str(db),
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

            trends = report["dashboard"]["jobs"]["importer_trends"]
            story_quality = report["dashboard"]["inventories"]["story_quality"]
            source_coverage = report["source_coverage"]
            failure_observability = report["failure_observability"]
            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["mode"], "offline_fixture")
            self.assertFalse(report["network_used"])
            self.assertEqual(report["source"], "both")
            self.assertEqual(report["cycles_requested"], 2)
            self.assertEqual(report["cycles_completed"], 2)
            self.assertEqual(report["successful_import_cycles"], 2)
            self.assertEqual(report["failed_import_cycles"], 0)
            self.assertGreaterEqual(report["inventory_delta"]["story_inventory_added"], 2)
            self.assertEqual(
                set(source_coverage["required"]),
                {"project_gutenberg_catalog_csv", "internet_archive_search_metadata"},
            )
            self.assertEqual(source_coverage["missing"], [])
            self.assertTrue(source_coverage["covered"])
            self.assertGreaterEqual(source_coverage["observed"]["project_gutenberg_catalog_csv"], 1)
            self.assertGreaterEqual(source_coverage["observed"]["internet_archive_search_metadata"], 1)
            self.assertTrue(failure_observability["present"])
            self.assertEqual(failure_observability["recent_cycle_count"], 2)
            self.assertEqual(failure_observability["failed_import_jobs"], 0)
            self.assertEqual(failure_observability["byte_budget_exhausted_results"], 0)
            self.assertEqual(trends["completed_cycles"], 2)
            self.assertEqual(trends["latest_completed_cycle"]["refresh_cycle"], 2)
            self.assertTrue(trends["latest_completed_cycle"]["job_id"].endswith(":cycle_2"))
            self.assertGreaterEqual(trends["imported_items_total"], 4)
            self.assertGreaterEqual(story_quality["with_quality_scores"], 2)
            self.assertEqual(story_quality["below_metadata_quality_floor"], 0)
            self.assertEqual(len(report["cycles"]), 2)
            for cycle in report["cycles"]:
                self.assertTrue(cycle["source_coverage"]["covered"])
                self.assertGreaterEqual(
                    cycle["story_inventory_count_after"],
                    cycle["story_inventory_count_before"],
                )

    def test_cli_inventory_diversity_smoke_runs_multi_niche_source_growth_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_cli(
                "inventory-diversity-smoke",
                "--db-dir",
                str(Path(tmp) / "inventory_diversity"),
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["schema"], "melm.inventory_diversity_smoke.v1")
            self.assertEqual(report["mode"], "offline_fixture")
            self.assertFalse(report["network_used"])
            self.assertEqual(report["niche_count"], 3)
            queries = [item["query"] for item in report["niches"]]
            self.assertEqual(len(set(queries)), 3)
            for run in report["runs"]:
                self.assertTrue(run["story_local"])
                self.assertEqual(run["story_route"], "local_answer")
                self.assertEqual(run["story_reason"], "local_story_inventory")
                self.assertIn(run["query"], run["executed_import_queries"])
                self.assertTrue(run["soak"]["source_coverage_ok"])
                self.assertGreater(run["soak"]["story_inventory_added"], 0)
                self.assertEqual(run["soak"]["below_quality_floor"], 0)

    def test_cli_inventory_soak_matrix_runs_cold_start_multi_source_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "inventory_matrix_report.json"
            report = _run_cli(
                "inventory-soak-matrix",
                "--db-dir",
                str(Path(tmp) / "inventory_matrix"),
                "--out",
                str(report_path),
                "--reset",
                "--json",
            )
            written_report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertTrue(report["passed"])
            self.assertEqual(written_report, report)
            self.assertEqual(report["report_path"], str(report_path))
            self.assertTrue(report["report_written"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["schema"], "melm.inventory_soak_matrix.v1")
            self.assertEqual(report["mode"], "offline_fixture")
            self.assertFalse(report["network_used"])
            self.assertEqual(report["profile_count"], 3)
            self.assertGreaterEqual(report["total_cycles_completed"], 9)
            self.assertEqual(report["total_failed_import_cycles"], 0)
            self.assertGreaterEqual(report["total_story_inventory_added"], 6)
            self.assertEqual(
                set(report["source_families_observed"]),
                {"project_gutenberg_catalog_csv", "internet_archive_search_metadata"},
            )
            for run in report["runs"]:
                self.assertTrue(run["story_local"])
                self.assertTrue(run["story_synthesis_applied"])
                self.assertTrue(run["story_primary_uol_ok"])
                self.assertEqual(run["story_route"], "local_answer")
                self.assertEqual(run["story_reason"], "local_story_inventory")
                self.assertTrue(run["soak"]["passed"])
                self.assertEqual(run["soak"]["initial_story_inventory_count"], 0)
                self.assertGreater(run["soak"]["story_inventory_added"], 0)
                self.assertTrue(run["soak"]["source_coverage_ok"])
                self.assertTrue(run["soak"]["failure_observability_present"])
                self.assertEqual(run["soak"]["failed_import_cycles"], 0)
                self.assertEqual(run["soak"]["below_quality_floor"], 0)

    def test_cli_inventory_retry_smoke_proves_localhost_retry_changes_future_story_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_cli(
                "inventory-retry-smoke",
                "--db",
                str(Path(tmp) / "inventory_retry.sqlite"),
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["schema"], "melm.inventory_retry_smoke.v1")
            self.assertTrue(report["network_used"])
            self.assertFalse(report["external_network_used"])
            self.assertEqual(report["before_story"]["route"], "cloud_handoff")
            self.assertEqual(report["before_story"]["reason"], "missing_story_model")
            self.assertEqual(report["after_story"]["route"], "local_answer")
            self.assertEqual(report["after_story"]["reason"], "local_story_inventory")
            self.assertGreaterEqual(report["inventory_delta"]["story_inventory_added"], 2)
            self.assertEqual(report["attempts_by_path"]["/gutenberg.csv"], 2)
            self.assertEqual(report["attempts_by_path"]["/ia/scrape"], 2)
            self.assertEqual(report["gutenberg"]["fetch_attempts"], 2)
            self.assertEqual(report["internet_archive"]["fetch_attempts_total"], 2)
            self.assertGreaterEqual(report["dashboard"]["jobs"]["importer_health"]["fetch_attempts_total"], 4)

    def test_cli_inventory_failure_smoke_proves_source_failures_do_not_fabricate_story(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_cli(
                "inventory-failure-smoke",
                "--work-dir",
                str(Path(tmp) / "inventory_failures"),
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["schema"], "melm.inventory_failure_smoke.v1")
            self.assertEqual(report["case_count"], 3)
            labels = {run["label"]: run for run in report["runs"]}
            self.assertEqual(labels["malformed_internet_archive_json"]["failed_import_jobs"], 1)
            self.assertEqual(labels["internet_archive_byte_budget_exceeded"]["failed_import_jobs"], 1)
            self.assertEqual(labels["empty_sources_no_fake_story"]["completed_import_jobs"], 1)
            for run in report["runs"]:
                self.assertEqual(run["story_inventory_added"], 0)
                self.assertEqual(run["story_route"], "cloud_handoff")
                self.assertEqual(run["story_reason"], "missing_story_model")
                self.assertFalse(run["story_synthesis_applied"])

    def test_cli_run_open_traces_emits_realistic_debuggable_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_cli(
                "run-open-traces",
                "--db-dir",
                str(Path(tmp) / "open_traces"),
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], "melm.local_assistant_open_trace_report.v1")
            self.assertEqual(report["scenarios"], 2)
            self.assertEqual(report["turns"], 29)
            self.assertGreaterEqual(report["local_resolution_rate"], 0.65)
            self.assertEqual(report["safety_totals"]["cloud_private_inclusions"], 0)
            child = report["scenario_reports"][0]
            identity = next(turn for turn in child["routes"] if turn["label"] == "identity")
            self.assertEqual(identity["debug_parse"]["mapping"][0]["stage"], "basic_nlp")
            self.assertEqual(identity["debug_parse"]["uol"]["object"], "self_model")
            self.assertTrue(child["checks"]["primary_uol_chatframe_not_secondary_phrase_route"])

    def test_cli_run_transcript_replay_emits_non_static_debuggable_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_cli(
                "run-transcript-replay",
                "--db-dir",
                str(Path(tmp) / "transcript_replay"),
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], "melm.local_assistant_transcript_replay_report.v1")
            self.assertEqual(report["source_type"], "authored_transcript_fixture")
            self.assertEqual(report["turns"], 25)
            self.assertGreaterEqual(report["local_resolution_rate"], 0.72)
            self.assertTrue(report["fixture_checks"]["no_static_answer_or_route_expectations"])
            self.assertTrue(report["fixture_checks"]["memory_digest_quality_passed"])
            self.assertEqual(report["reason_counts"]["profile_update"], 2)
            self.assertGreater(report["complexity"]["unknown_tokens_total"], 0)
            self.assertEqual(report["debug_mapping"]["stages"], ["basic_nlp", "uol_parse", "chat_frame"])
            baseline = report["baseline_comparison"]
            self.assertTrue(baseline["passed"])
            self.assertEqual(baseline["current"]["local_or_device_resolved"], 18)
            self.assertEqual(baseline["best_baseline"]["local_or_device_resolved"], 14)
            self.assertEqual(baseline["wins"]["local_resolution_rate_gain_vs_best_baseline"], 0.16)
            self.assertEqual(baseline["wins"]["cloud_handoff_reduction_vs_best_baseline"], 2)
            self.assertEqual(baseline["wins"]["clarification_reduction_vs_best_baseline"], 2)
            self.assertTrue(all(baseline["checks"].values()))
            scenario = report["scenario_reports"][0]
            by_label = {turn["label"]: turn for turn in scenario["routes"]}
            self.assertEqual(by_label["age_fact"]["reason"], "profile_update")
            self.assertEqual(by_label["age_fact"]["debug_parse"]["chat_frame"]["intent"], "personal_memory")
            self.assertEqual(by_label["long_horizon_digest"]["reason"], "autobiographical_memory_digest")
            self.assertTrue(scenario["memory_digest"]["quality_passed"])
            self.assertTrue(scenario["checks"]["primary_uol_chatframe_not_secondary_phrase_route"])

    def test_cli_import_transcript_replay_redacts_and_outputs_calibration_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            out = Path(tmp) / "imported_transcript.jsonl"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {
                            "role": "user",
                            "content": "Maya email maya@example.com wants calm piano.",
                            "expected_answer": "bad",
                        },
                        {
                            "role": "assistant",
                            "content": "This answer must be skipped.",
                            "expected_route": "bad",
                        },
                        {
                            "role": "user",
                            "content": "Call +234-555-0101 tomorrow.",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "import-transcript-replay",
                "--input",
                str(raw),
                "--out",
                str(out),
                "--replace",
                "Maya=<person_1>",
                "--json",
            )
            output = out.read_text(encoding="utf-8")

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], "melm.local_assistant_transcript_import_report.v1")
            self.assertEqual(report["turns_written"], 2)
            self.assertEqual(report["assistant_rows_skipped"], 1)
            self.assertEqual(report["redaction_counts"]["email"], 1)
            self.assertEqual(report["redaction_counts"]["phone"], 1)
            self.assertEqual(report["redaction_counts"]["manual_rule_1"], 1)
            self.assertEqual(report["static_expectation_fields_dropped"]["expected_answer"], 1)
            self.assertEqual(report["static_expectation_fields_dropped"]["expected_route"], 1)
            self.assertNotIn("maya@example.com", output)
            self.assertNotIn("+234-555-0101", output)
            self.assertNotIn("Maya", output)
            self.assertIn('"source_type": "redacted_user_transcript_import"', output)

            replay = _run_cli(
                "run-transcript-replay",
                "--transcript-jsonl",
                str(out),
                "--db-dir",
                str(Path(tmp) / "db"),
                "--reset",
                "--json",
            )
            self.assertTrue(replay["passed"])
            self.assertEqual(replay["source_type"], "redacted_user_transcript_import")
            self.assertFalse(replay["baseline_comparison"]["required"])

    def test_cli_export_transcript_replay_from_event_ledger_without_static_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            out = Path(tmp) / "event_ledger_replay.jsonl"
            chat = _run_cli(
                "chat",
                "--db",
                str(db),
                "--reset",
                "--turn",
                "Who are you?",
                "--turn",
                "Tell me a story.",
                "--turn",
                "What is the weather today?",
                "--json",
            )

            report = _run_cli(
                "export-transcript-replay",
                "--db",
                str(db),
                "--out",
                str(out),
                "--json",
            )
            raw_path = Path(report["raw_event_jsonl"])
            raw_text = raw_path.read_text(encoding="utf-8")
            replay_text = out.read_text(encoding="utf-8")

            self.assertEqual(chat["counts"]["events"], 3)
            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], "melm.local_assistant_event_transcript_export_report.v1")
            self.assertEqual(report["source_type"], "event_ledger_transcript_export")
            self.assertEqual(report["events_exported"], 3)
            self.assertEqual(report["forbidden_static_fields_exported"], [])
            self.assertFalse(report["answers_routes_reasons_exported"])
            self.assertEqual(report["import"]["source_type"], "event_ledger_transcript_export")
            self.assertEqual(report["import"]["turns_written"], 3)
            self.assertNotIn('"route"', raw_text)
            self.assertNotIn('"reason"', raw_text)
            self.assertNotIn('"answer"', raw_text)
            self.assertNotIn("expected_route", raw_text)
            self.assertNotIn("expected_answer", replay_text)
            self.assertIn('"source_type": "event_ledger_transcript_export"', replay_text)

            replay = _run_cli(
                "run-transcript-replay",
                "--transcript-jsonl",
                str(out),
                "--db-dir",
                str(Path(tmp) / "db"),
                "--reset",
                "--json",
            )
            self.assertTrue(replay["passed"])
            self.assertEqual(replay["source_type"], "event_ledger_transcript_export")
            self.assertTrue(replay["fixture_checks"]["no_static_answer_or_route_expectations"])
            self.assertFalse(replay["baseline_comparison"]["required"])

    def test_cli_calibrate_event_ledger_exports_replays_and_aggregates_runtime_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            work_dir = Path(tmp) / "event_calibration"
            _run_cli(
                "chat",
                "--db",
                str(db),
                "--reset",
                "--turn",
                "Who are you?",
                "--turn",
                "Tell me a story.",
                "--turn",
                "What is the weather today?",
                "--turn",
                "What should I eat today?",
                "--json",
            )

            report = _run_cli(
                "calibrate-event-ledger",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--min-total-turns",
                "4",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "4",
                "--min-local-resolution-rate",
                "0.75",
                "--reset",
                "--json",
            )
            raw_text = Path(report["raw_event_jsonl"]).read_text(encoding="utf-8")
            replay_text = Path(report["transcript_jsonl"]).read_text(encoding="utf-8")
            aggregate = report["aggregate"]

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], "melm.local_assistant_event_ledger_calibration_report.v1")
            self.assertEqual(report["source_type"], "event_ledger_transcript_export")
            self.assertEqual(report["events_exported"], 4)
            self.assertEqual(report["capture_provenance"]["capture_surface_counts"], {"cli_chat": 4})
            self.assertEqual(report["capture_provenance"]["capture_source_counts"], {"scripted_cli_turn": 4})
            self.assertTrue(report["capture_provenance"]["has_capture_provenance"])
            self.assertTrue(report["capture_provenance"]["all_turns_scripted"])
            self.assertFalse(report["answers_routes_reasons_exported"])
            self.assertEqual(report["forbidden_static_fields_exported"], [])
            self.assertEqual(report["item"]["import"]["turns_written"], 4)
            self.assertEqual(report["item"]["replay"]["source_type"], "event_ledger_transcript_export")
            self.assertTrue(report["item"]["replay"]["debug_checks"]["debug_maps_present"])
            self.assertTrue(
                report["item"]["replay"]["debug_checks"]["primary_uol_chatframe_not_secondary_phrase_route"]
            )
            self.assertTrue(aggregate["passed"])
            self.assertEqual(aggregate["turns_imported"], 4)
            self.assertEqual(aggregate["turns_replayed"], 4)
            self.assertGreaterEqual(aggregate["local_resolution_rate"], 0.75)
            self.assertGreaterEqual(aggregate["intent_kinds"], 4)
            self.assertNotIn('"answer"', raw_text)
            self.assertNotIn('"route"', raw_text)
            self.assertNotIn('"reason"', raw_text)
            self.assertIn('"capture_source": "scripted_cli_turn"', raw_text)
            self.assertNotIn("expected_route", replay_text)

    def test_cli_v01_evidence_pack_packages_real_ledger_without_promoting_development(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            work_dir = Path(tmp) / "evidence_pack"
            _run_cli(
                "chat",
                "--db",
                str(db),
                "--reset",
                "--turn",
                "Who are you?",
                "--turn",
                "Tell me a story.",
                "--turn",
                "What is the weather today?",
                "--turn",
                "What should I eat today?",
                "--turn",
                "What have you done so far?",
                "--json",
            )

            report = _run_cli(
                "v01-evidence-pack",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--min-total-turns",
                "5",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "5",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "0",
                "--auto-lifecycle",
                "--reset",
                "--json",
            )
            artifacts = {key: Path(value) for key, value in report["artifact_paths"].items() if value}
            written_pack = json.loads(artifacts["evidence_pack_report"].read_text(encoding="utf-8"))

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], "melm.local_assistant_v01_evidence_pack.v1")
            self.assertEqual(written_pack, report)
            self.assertTrue(all(path.exists() for path in artifacts.values()))
            self.assertFalse(report["architecture_complete"])
            self.assertFalse(report["candidate_review_ready"])
            self.assertEqual(report["event_transcript_export"]["events_exported"], 5)
            self.assertTrue(report["checks"]["event_capture_provenance_present"])
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["capture_source_counts"],
                {"scripted_cli_turn": 5},
            )
            self.assertTrue(report["event_transcript_export"]["capture_provenance"]["all_turns_scripted"])
            self.assertFalse(report["event_transcript_export"]["answers_routes_reasons_exported"])
            self.assertEqual(report["event_transcript_export"]["forbidden_static_fields_exported"], [])
            self.assertTrue(report["event_ledger_calibration"]["passed"])
            self.assertEqual(
                report["event_ledger_calibration"]["capture_provenance"]["capture_source_counts"],
                {"scripted_cli_turn": 5},
            )
            self.assertEqual(report["event_ledger_calibration"]["turns_replayed"], 5)
            self.assertEqual(report["blocker_evidence"]["candidate_blockers_satisfied"], 0)
            self.assertEqual(report["progress"]["remaining_blocker_count"], 6)
            self.assertTrue(report["development_source_note"]["candidate_evidence_allowed"] is False)
            self.assertEqual(
                report["blocker_evidence"]["statuses"]["real_user_derived_lifecycle_traces"],
                "development_evidence_present",
            )
            self.assertEqual(
                report["blocker_evidence"]["statuses"]["user_derived_bounded_synthesis_traces"],
                "development_evidence_present",
            )

    def test_cli_v01_evidence_pack_rejects_scripted_cli_as_user_candidate_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            work_dir = Path(tmp) / "evidence_pack"
            _run_cli(
                "chat",
                "--db",
                str(db),
                "--reset",
                "--turn",
                "Who are you?",
                "--turn",
                "Tell me a story.",
                "--turn",
                "What is the weather today?",
                "--turn",
                "What should I eat today?",
                "--turn",
                "What have you done so far?",
                "--json",
            )

            report = _run_cli(
                "v01-evidence-pack",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--event-source-kind",
                "redacted_user_session",
                "--write-source-attestation",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--min-total-turns",
                "5",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "5",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "0",
                "--auto-lifecycle",
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertTrue(Path(report["artifact_paths"]["source_attestation"]).exists())
            self.assertFalse(report["source_attestation_write"]["passed"])
            self.assertFalse(report["source_attestation"]["valid"])
            self.assertEqual(
                report["source_attestation_write"]["validation"]["event_capture_provenance"]["capture_source_counts"],
                {"scripted_cli_turn": 5},
            )
            self.assertFalse(
                report["source_attestation_write"]["validation"]["checks"]["candidate_capture_not_all_scripted"]
            )
            self.assertIn(
                "candidate user evidence must come entirely from imported redacted, interactive CLI, browser UI, or target-device capture; scripted CLI/API/UI smokes stay development evidence",
                report["source_attestation_write"]["validation"]["missing"],
            )
            self.assertEqual(
                report["source_attestation"]["event_capture_provenance"]["capture_source_counts"],
                {"scripted_cli_turn": 5},
            )
            self.assertEqual(report["blocker_evidence"]["candidate_blockers_satisfied"], 0)
            self.assertEqual(
                report["blocker_evidence"]["statuses"]["real_user_derived_lifecycle_traces"],
                "unattested_user_evidence_present",
            )
            self.assertEqual(
                report["blocker_evidence"]["statuses"]["user_derived_bounded_synthesis_traces"],
                "unattested_user_evidence_present",
            )
            self.assertFalse(report["architecture_complete"])

    def test_cli_v01_evidence_pack_rejects_scripted_api_smoke_as_user_candidate_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            work_dir = Path(tmp) / "evidence_pack"
            _run_cli("api-session-smoke", "--db", str(db), "--reset", "--json")

            report = _run_cli(
                "v01-evidence-pack",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--event-source-kind",
                "redacted_user_session",
                "--capture-surface",
                "browser_api",
                "--write-source-attestation",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--min-total-turns",
                "8",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "6",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "0",
                "--auto-lifecycle",
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertTrue(Path(report["artifact_paths"]["source_attestation"]).exists())
            self.assertFalse(report["source_attestation_write"]["passed"])
            self.assertFalse(report["source_attestation"]["valid"])
            self.assertEqual(
                report["source_attestation_write"]["validation"]["event_capture_provenance"]["capture_surface_counts"],
                {"browser_api": 11},
            )
            self.assertEqual(
                report["source_attestation_write"]["validation"]["event_capture_provenance"]["capture_source_counts"],
                {"scripted_api_smoke": 11},
            )
            self.assertTrue(
                report["source_attestation_write"]["validation"]["event_capture_provenance"]["all_turns_scripted"]
            )
            self.assertFalse(
                report["source_attestation_write"]["validation"]["checks"]["candidate_capture_not_all_scripted"]
            )
            self.assertEqual(
                report["source_attestation_write"]["validation"]["event_capture_provenance"][
                    "candidate_capture_source_count"
                ],
                0,
            )
            self.assertIn(
                "candidate user evidence must come entirely from imported redacted, interactive CLI, browser UI, or target-device capture; scripted CLI/API/UI smokes stay development evidence",
                report["source_attestation_write"]["validation"]["missing"],
            )
            self.assertEqual(report["blocker_evidence"]["candidate_blockers_satisfied"], 0)
            self.assertEqual(
                report["blocker_evidence"]["statuses"]["real_user_derived_lifecycle_traces"],
                "unattested_user_evidence_present",
            )
            self.assertEqual(
                report["blocker_evidence"]["statuses"]["user_derived_bounded_synthesis_traces"],
                "unattested_user_evidence_present",
            )
            self.assertFalse(report["architecture_complete"])

    def test_cli_source_attestation_rejects_mixed_scripted_and_browser_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            attestation = Path(tmp) / "source_attestation.json"
            _run_cli("api-session-smoke", "--db", str(db), "--reset", "--json")

            connection = sqlite3.connect(db)
            try:
                connection.execute(
                    """
                    UPDATE events
                    SET capture_source = 'browser_ui'
                    WHERE rowid = (SELECT MIN(rowid) FROM events)
                    """
                )
                connection.commit()
            finally:
                connection.close()

            written = _run_cli(
                "write-source-attestation",
                "--out",
                str(attestation),
                "--event-ledger-db",
                str(db),
                "--source-kind",
                "redacted_user_session",
                "--capture-surface",
                "browser_api",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--overwrite",
                "--json",
            )

            validation = written["validation"]
            self.assertFalse(written["passed"])
            self.assertFalse(validation["valid"])
            self.assertTrue(validation["checks"]["capture_surface_matches_events"])
            self.assertFalse(validation["checks"]["candidate_capture_not_all_scripted"])
            self.assertFalse(validation["checks"]["candidate_or_imported_capture_complete"])
            self.assertFalse(validation["checks"]["candidate_capture_covers_events"])
            self.assertFalse(validation["checks"]["imported_capture_covers_events"])
            self.assertEqual(validation["candidate_capture_source_count"], 1)
            self.assertEqual(validation["event_capture_provenance"]["turn_count"], 11)
            self.assertEqual(
                validation["event_capture_provenance"]["capture_source_counts"],
                {"browser_ui": 1, "scripted_api_smoke": 10},
            )
            self.assertIn(
                "candidate user evidence must come entirely from imported redacted, interactive CLI, browser UI, or target-device capture; scripted CLI/API/UI smokes stay development evidence",
                validation["missing"],
            )

    def test_cli_candidate_session_audit_rejects_scripted_api_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            work_dir = Path(tmp) / "candidate_audit"
            _run_cli("api-session-smoke", "--db", str(db), "--reset", "--json")

            report = _run_cli(
                "candidate-session-audit",
                "--db",
                str(db),
                "--work-dir",
                str(work_dir),
                "--event-source-kind",
                "redacted_user_session",
                "--capture-surface",
                "browser_api",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--min-total-turns",
                "8",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "6",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "0",
                "--reset",
                "--json",
            )

            self.assertTrue(report["passed"])
            self.assertFalse(report["candidate_session_ready"])
            self.assertFalse(report["ready_for_source_attestation_write"])
            self.assertTrue(report["checks"]["event_calibration_passed"])
            self.assertFalse(report["checks"]["source_attestation_preview_valid"])
            self.assertEqual(
                report["event_transcript_export"]["capture_provenance"]["capture_source_counts"],
                {"scripted_api_smoke": 11},
            )
            self.assertEqual(
                report["source_attestation_preview"]["event_capture_provenance"]["candidate_capture_source_count"],
                0,
            )
            projection = report["blocker_projection"]["after_writing_source_attestation"]
            self.assertEqual(projection["candidate_blockers_satisfied"], 0)
            self.assertEqual(projection["candidate_blockers"], [])
            self.assertIn(
                "--event-ledger-session all",
                report["next_commands"]["write_source_attestation"],
            )

    def test_cli_candidate_session_audit_is_session_scoped_for_imported_redacted_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw_chat.jsonl"
            calibration_dir = root / "calibration"
            calibration_report = root / "calibration_report.json"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {"role": "user", "session_id": "real_1", "content": "Maya emailed maya@example.com."},
                        {"role": "user", "session_id": "real_1", "content": "who are you"},
                        {"role": "user", "session_id": "real_1", "content": "tell me a story"},
                    ]
                ),
                encoding="utf-8",
            )
            replay_calibration = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(calibration_dir),
                "--out",
                str(calibration_report),
                "--replace",
                "Maya=<person_1>",
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--require-redaction",
                "--reset",
                "--json",
            )
            db = Path(replay_calibration["items"][0]["replay_event_ledger_db"])
            connection = sqlite3.connect(db)
            try:
                imported_session = connection.execute(
                    """
                    SELECT session_id
                    FROM events
                    WHERE capture_surface='imported_redacted_transcript'
                    ORDER BY rowid
                    LIMIT 1
                    """
                ).fetchone()[0]
            finally:
                connection.close()

            artifact_scoped = _run_cli(
                "candidate-session-audit",
                "--db",
                str(db),
                "--session",
                str(imported_session),
                "--work-dir",
                str(root / "artifact_scoped_audit"),
                "--event-source-kind",
                "redacted_user_session",
                "--capture-surface",
                "imported_redacted_transcript",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--transcript-calibration-report-json",
                str(calibration_report),
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--min-synthesis-traces",
                "0",
                "--min-priority-signal-samples",
                "0",
                "--reset",
                "--json",
            )

            _run_cli("ask", "--db", str(db), "--utterance", "Who are you?", "--json")

            scoped = _run_cli(
                "candidate-session-audit",
                "--db",
                str(db),
                "--session",
                str(imported_session),
                "--work-dir",
                str(root / "scoped_audit"),
                "--event-source-kind",
                "redacted_user_session",
                "--capture-surface",
                "imported_redacted_transcript",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--min-synthesis-traces",
                "0",
                "--min-priority-signal-samples",
                "0",
                "--reset",
                "--json",
            )
            all_sessions = _run_cli(
                "candidate-session-audit",
                "--db",
                str(db),
                "--session",
                "all",
                "--work-dir",
                str(root / "all_audit"),
                "--event-source-kind",
                "redacted_user_session",
                "--capture-surface",
                "imported_redacted_transcript",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--min-synthesis-traces",
                "0",
                "--min-priority-signal-samples",
                "0",
                "--reset",
                "--json",
            )

            self.assertTrue(scoped["passed"])
            self.assertTrue(scoped["ready_for_source_attestation_write"])
            self.assertTrue(scoped["ready_for_v01_evidence_pack_with_write"])
            self.assertTrue(scoped["source_attestation_preview"]["valid"])
            self.assertEqual(scoped["source_attestation_preview"]["event_ledger_session"], str(imported_session))
            self.assertEqual(
                scoped["event_transcript_export"]["capture_provenance"]["capture_surface_counts"],
                {"imported_redacted_transcript": 3},
            )
            self.assertTrue(
                scoped["source_attestation_preview"]["checks"]["imported_capture_covers_events"]
            )
            scoped_projection = scoped["blocker_projection"]["after_writing_source_attestation"]
            self.assertEqual(scoped_projection["candidate_blockers_satisfied"], 1)
            self.assertEqual(scoped_projection["candidate_blockers"], ["real_user_derived_lifecycle_traces"])
            self.assertEqual(scoped_projection["status_counts"]["candidate_evidence_present"], 1)
            artifact_inputs = artifact_scoped["blocker_projection"]["artifact_inputs"]
            self.assertTrue(artifact_inputs["transcript_calibration_report"]["present"])
            self.assertTrue(
                artifact_inputs["transcript_calibration_report"]["event_ledger_binding"]["path_matched"]
            )
            self.assertTrue(
                artifact_inputs["transcript_calibration_report"]["event_ledger_binding"]["hash_matched"]
            )
            digest_row = next(
                row
                for row in artifact_scoped["blocker_projection"]["after_writing_source_attestation"]["rows"]
                if row["id"] == "digest_quality_and_route_threshold_calibration"
            )
            self.assertEqual(digest_row["status"], "missing_evidence")
            self.assertTrue(
                digest_row["evidence"]["transcript_calibration"]["event_ledger_binding"]["path_matched"]
            )
            self.assertNotIn(
                "transcript calibration replay DB SHA-256 matches the current event-ledger DB",
                digest_row["missing"],
            )
            self.assertTrue(all_sessions["passed"])
            self.assertFalse(all_sessions["ready_for_source_attestation_write"])
            self.assertFalse(all_sessions["source_attestation_preview"]["valid"])
            self.assertEqual(
                all_sessions["blocker_projection"]["after_writing_source_attestation"]["candidate_blockers_satisfied"],
                0,
            )
            self.assertEqual(
                all_sessions["source_attestation_preview"]["capture_surface_turn_count"],
                3,
            )
            self.assertEqual(
                all_sessions["source_attestation_preview"]["event_capture_provenance"]["turn_count"],
                4,
            )

    def test_cli_v01_blocker_evidence_maps_development_session_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            work_dir = Path(tmp) / "blocker_evidence"
            _run_cli(
                "chat",
                "--db",
                str(db),
                "--reset",
                "--turn",
                "Who are you?",
                "--turn",
                "Tell me a story.",
                "--turn",
                "What is the weather today?",
                "--turn",
                "What should I eat today?",
                "--turn",
                "What have you done so far?",
                "--json",
            )

            report = _run_cli(
                "v01-blocker-evidence",
                "--event-ledger-db",
                str(db),
                "--event-ledger-work-dir",
                str(work_dir),
                "--min-total-turns",
                "5",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "5",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "0",
                "--auto-lifecycle",
                "--reset",
                "--json",
            )
            blockers = {item["id"]: item for item in report["blockers"]}

            self.assertTrue(report["passed"])
            self.assertTrue(report["report_valid"])
            self.assertFalse(report["candidate_evidence_complete"])
            self.assertEqual(report["remaining_blocker_count"], 6)
            self.assertFalse(report["architecture_complete_claimed"])
            self.assertEqual(report["candidate_blockers_satisfied"], 0)
            self.assertEqual(report["blocker_count"], 6)
            self.assertTrue(report["event_ledger_calibration"]["passed"])
            self.assertEqual(report["event_ledger_calibration"]["events_exported"], 5)
            self.assertFalse(report["event_ledger_calibration"]["answers_routes_reasons_exported"])
            self.assertEqual(
                blockers["real_user_derived_lifecycle_traces"]["status"],
                "development_evidence_present",
            )
            self.assertFalse(
                blockers["real_user_derived_lifecycle_traces"]["candidate_for_architecture_review"]
            )
            self.assertEqual(
                blockers["user_derived_bounded_synthesis_traces"]["status"],
                "development_evidence_present",
            )
            self.assertIn(
                "redacted user-derived or target-device source attestation",
                blockers["user_derived_bounded_synthesis_traces"]["missing"],
            )
            self.assertEqual(
                blockers["configured_target_device_apps"]["status"],
                "missing_configured_host_app_probe",
            )

    def test_cli_v01_blocker_rehearsal_runs_real_ledger_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "blocker_rehearsal"

            report = _run_cli(
                "v01-blocker-rehearsal",
                "--work-dir",
                str(work_dir),
                "--reset",
                "--json",
            )
            artifacts = {key: Path(value) for key, value in report["artifact_paths"].items()}
            source_note = json.loads(artifacts["development_source_note"].read_text(encoding="utf-8"))
            next_commands = "\n".join(report["next_candidate_commands"])
            statuses = report["blocker_statuses"]

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], "melm.local_assistant_v01_blocker_rehearsal.v1")
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(all(path.exists() for path in artifacts.values()))
            self.assertEqual(source_note["schema"], "melm.local_assistant_development_source_note.v1")
            self.assertEqual(source_note["source_kind"], "development_session")
            self.assertFalse(source_note["candidate_evidence_allowed"])
            self.assertFalse(source_note["accepted_by_write_source_attestation"])
            self.assertTrue(source_note["static_expectations_absent"])
            self.assertTrue(source_note["answers_routes_reasons_absent"])
            self.assertEqual(report["development_boundary"]["candidate_blockers_satisfied"], 0)
            self.assertFalse(report["development_boundary"]["architecture_complete_claimed"])
            self.assertFalse(report["progress"]["architecture_complete"])
            self.assertFalse(report["progress"]["candidate_review_ready"])
            self.assertEqual(report["progress"]["remaining_blocker_count"], 6)
            self.assertEqual(
                statuses["user_derived_bounded_synthesis_traces"],
                "development_evidence_present",
            )
            self.assertEqual(
                statuses["planner_priority_on_user_derived_traces"],
                "missing_evidence",
            )
            self.assertEqual(
                statuses["real_user_derived_lifecycle_traces"],
                "development_evidence_present",
            )
            self.assertEqual(
                statuses["digest_quality_and_route_threshold_calibration"],
                "missing_evidence",
            )
            self.assertEqual(statuses["longer_live_inventory_soak"], "missing_live_inventory_soak")
            self.assertEqual(statuses["configured_target_device_apps"], "missing_configured_host_app_probe")
            self.assertFalse(report["event_ledger_calibration"]["answers_routes_reasons_exported"])
            self.assertEqual(report["event_ledger_calibration"]["forbidden_static_fields_exported"], [])
            self.assertGreaterEqual(report["event_ledger_calibration"]["events_exported"], 6)
            self.assertGreaterEqual(report["event_ledger_calibration"]["synthesis_traces"], 2)
            self.assertGreaterEqual(report["event_ledger_calibration"]["priority_signal_sample_count"], 0)
            self.assertIn("calibrate-transcript-replay", next_commands)
            self.assertIn("write-source-attestation", next_commands)
            self.assertIn("v01-blocker-evidence", next_commands)
            self.assertIn("v01-progress", next_commands)

    def test_cli_v01_blocker_evidence_rejects_forged_live_inventory_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forged_report = root / "forged_live_inventory.json"
            blocker_report = root / "blocker_evidence.json"
            required_checks = {
                "profiles_exercised": True,
                "total_cycles_at_least_nine": True,
                "cycles_completed": True,
                "all_soaks_passed": True,
                "all_source_coverage_ok": True,
                "both_source_families_covered": True,
                "all_story_inventory_grew_from_cold_start": True,
                "future_story_routes_local_from_imported_inventory": True,
                "future_story_synthesis_applied": True,
                "future_story_primary_uol_not_secondary_phrase_route": True,
                "all_failure_observability_present": True,
                "no_failed_import_cycles": True,
                "story_quality_floor_clean": True,
                "bounded_resource_budget": True,
            }
            forged_report.write_text(
                json.dumps(
                    {
                        "schema": "melm.inventory_soak_matrix.v1",
                        "passed": True,
                        "mode": "live_metadata",
                        "network_used": True,
                        "profile_count": 3,
                        "total_cycles_completed": 9,
                        "total_failed_import_cycles": 0,
                        "total_story_inventory_added": 9,
                        "source_families_observed": [
                            "project_gutenberg_catalog_csv",
                            "internet_archive_search_metadata",
                        ],
                        "checks": required_checks,
                        "db_dir": str(root / "missing_inventory_matrix"),
                        "runs": [
                            {
                                "label": "forged",
                                "db": str(root / f"missing_{index}.sqlite"),
                                "db_sha256": "0" * 64,
                                "story_route": "local_answer",
                                "story_reason": "local_story_inventory",
                                "story_local": True,
                                "story_synthesis_applied": True,
                                "story_primary_uol_ok": True,
                                "soak": {
                                    "passed": True,
                                    "mode": "live_metadata",
                                    "network_used": True,
                                    "source": "both",
                                    "cycles_completed": 3,
                                    "failed_import_cycles": 0,
                                    "story_inventory_added": 3,
                                    "source_coverage_ok": True,
                                    "network_used_results": 2,
                                    "fetch_attempts_total": 2,
                                },
                            }
                            for index in range(3)
                        ],
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "v01-blocker-evidence",
                "--inventory-soak-report-json",
                str(forged_report),
                "--out",
                str(blocker_report),
                "--json",
            )
            written_blocker_report = json.loads(blocker_report.read_text(encoding="utf-8"))
            progress = _run_cli(
                "v01-progress",
                "--blocker-evidence-json",
                str(blocker_report),
                "--json",
            )
            blocker = {item["id"]: item for item in report["blockers"]}["longer_live_inventory_soak"]
            summary = report["inventory_soak_report"]

            self.assertTrue(report["passed"])
            self.assertEqual(written_blocker_report, report)
            self.assertEqual(report["report_path"], str(blocker_report))
            self.assertTrue(report["report_written"])
            self.assertTrue(progress["passed"])
            self.assertEqual(progress["candidate_blockers_satisfied"], 0)
            self.assertEqual(progress["remaining_blocker_count"], 6)
            self.assertTrue(report["report_valid"])
            self.assertFalse(report["candidate_evidence_complete"])
            self.assertGreater(report["remaining_blocker_count"], 0)
            self.assertEqual(blocker["status"], "missing_live_inventory_soak")
            self.assertFalse(blocker["candidate_for_architecture_review"])
            self.assertFalse(summary["candidate_live_soak_passed"])
            self.assertFalse(summary["report_binding"]["db_dir_exists"])
            self.assertIn(
                "inventory-soak-matrix db_dir artifact directory",
                blocker["missing"],
            )
            self.assertIn(
                "inventory-soak-matrix run DB artifacts with matching hashes",
                blocker["missing"],
            )
            self.assertIn(
                "future story route verified inside every run DB",
                blocker["missing"],
            )

    def test_cli_v01_blocker_evidence_requires_source_attestation_for_user_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            work_dir = Path(tmp) / "blocker_evidence"
            attestation = Path(tmp) / "source_attestation.json"
            _run_cli(
                "chat",
                "--db",
                str(db),
                "--reset",
                "--turn",
                "Who are you?",
                "--turn",
                "Tell me a story.",
                "--turn",
                "What is the weather today?",
                "--turn",
                "What should I eat today?",
                "--turn",
                "What have you done so far?",
                "--json",
            )

            unattested = _run_cli(
                "v01-blocker-evidence",
                "--event-ledger-db",
                str(db),
                "--event-ledger-work-dir",
                str(work_dir),
                "--event-source-kind",
                "redacted_user_session",
                "--min-total-turns",
                "5",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "5",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "0",
                "--auto-lifecycle",
                "--reset",
                "--json",
            )
            unattested_blockers = {item["id"]: item for item in unattested["blockers"]}

            self.assertTrue(unattested["passed"])
            self.assertFalse(unattested["source_attestation"]["present"])
            self.assertEqual(unattested["candidate_blockers_satisfied"], 0)
            self.assertEqual(
                unattested_blockers["real_user_derived_lifecycle_traces"]["status"],
                "unattested_user_evidence_present",
            )
            self.assertIn(
                "valid source attestation JSON",
                unattested_blockers["real_user_derived_lifecycle_traces"]["missing"],
            )

            written = _run_cli(
                "write-source-attestation",
                "--out",
                str(attestation),
                "--event-ledger-db",
                str(db),
                "--source-kind",
                "redacted_user_session",
                "--capture-surface",
                "cli_chat",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--overwrite",
                "--json",
            )

            self.assertFalse(written["passed"])
            self.assertTrue(attestation.exists())
            self.assertTrue(written["validation"]["checks"]["event_ledger_db_sha256_matches"])
            self.assertFalse(written["validation"]["checks"]["candidate_capture_not_all_scripted"])
            self.assertEqual(
                written["validation"]["event_capture_provenance"]["capture_source_counts"],
                {"scripted_cli_turn": 5},
            )
            self.assertIn(
                "candidate user evidence must come entirely from imported redacted, interactive CLI, browser UI, or target-device capture; scripted CLI/API/UI smokes stay development evidence",
                written["validation"]["missing"],
            )

            attested = _run_cli(
                "v01-blocker-evidence",
                "--event-ledger-db",
                str(db),
                "--event-ledger-work-dir",
                str(work_dir),
                "--event-source-kind",
                "redacted_user_session",
                "--source-attestation-json",
                str(attestation),
                "--min-total-turns",
                "5",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "5",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "0",
                "--auto-lifecycle",
                "--reset",
                "--json",
            )
            attested_blockers = {item["id"]: item for item in attested["blockers"]}

            self.assertTrue(attested["passed"])
            self.assertFalse(attested["source_attestation"]["valid"])
            self.assertEqual(attested["candidate_blockers_satisfied"], 0)
            self.assertEqual(
                attested_blockers["real_user_derived_lifecycle_traces"]["status"],
                "unattested_user_evidence_present",
            )
            self.assertEqual(
                attested_blockers["user_derived_bounded_synthesis_traces"]["status"],
                "unattested_user_evidence_present",
            )
            self.assertFalse(
                attested_blockers["real_user_derived_lifecycle_traces"]["candidate_for_architecture_review"]
            )
            self.assertIn(
                "valid source attestation JSON",
                attested_blockers["real_user_derived_lifecycle_traces"]["missing"],
            )

    def test_cli_v01_blocker_evidence_does_not_count_recorder_as_target_device_app(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config" / "host_actions.local_recorder.json"
            log = root / "host_actions.local_recorder.jsonl"
            db = root / "host_app.sqlite"
            work_dir = root / "host_app"
            attestation = root / "host_app_attestation.json"
            _run_cli(
                "write-host-actions-demo-config",
                "--out",
                str(config),
                "--log",
                str(log),
                "--overwrite",
                "--json",
            )

            written = _run_cli(
                "write-host-app-attestation",
                "--out",
                str(attestation),
                "--host-app-config-json",
                str(config),
                "--capture-surface",
                "target_device_cli",
                "--media-app-configured",
                "--call-app-configured",
                "--not-demo-recorder",
                "--real-app-commands-acknowledged",
                "--human-reviewed",
                "--overwrite",
                "--json",
            )
            report = _run_cli(
                "v01-blocker-evidence",
                "--host-app-config-json",
                str(config),
                "--host-app-attestation-json",
                str(attestation),
                "--host-app-db",
                str(db),
                "--host-app-work-dir",
                str(work_dir),
                "--run-host-app-probe",
                "--reset",
                "--json",
            )
            blocker = {item["id"]: item for item in report["blockers"]}["configured_target_device_apps"]

            self.assertFalse(written["passed"])
            self.assertTrue(written["validation"]["checks"]["not_demo_recorder_asserted"])
            self.assertFalse(written["validation"]["checks"]["config_not_demo_recorder"])
            self.assertIn(
                "host app config does not use recorder/demo commands",
                written["validation"]["missing"],
            )
            self.assertTrue(report["passed"])
            self.assertTrue(report["host_app_probe"]["passed"])
            self.assertTrue(report["host_app_probe"]["configured"])
            self.assertTrue(report["host_app_probe"]["evidence_class"]["demo_recorder_detected"])
            self.assertFalse(report["host_app_attestation"]["valid"])
            self.assertEqual(blocker["status"], "development_host_app_probe_present")
            self.assertFalse(blocker["candidate_for_architecture_review"])
            self.assertTrue(blocker["evidence"]["demo_recorder_detected"])
            self.assertFalse(blocker["evidence"]["candidate_target_device_app_evidence"])
            self.assertIn(
                "replace recorder/demo commands with target-device media and call app commands",
                blocker["missing"],
            )
            self.assertIn(
                "valid host app attestation JSON bound to the target app config hash",
                blocker["missing"],
            )

    def test_cli_calibrate_transcript_replay_imports_and_aggregates_raw_chats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            work_dir = Path(tmp) / "calibration"
            report_path = Path(tmp) / "calibration_report.json"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {
                            "role": "user",
                            "content": "Maya email is maya@example.com.",
                            "expected_answer": "bad",
                        },
                        {
                            "role": "assistant",
                            "content": "This row must not become replay evidence.",
                            "expected_route": "bad",
                        },
                        {
                            "role": "user",
                            "content": "who are you",
                        },
                        {
                            "role": "user",
                            "content": "Call +234-555-0101 tomorrow.",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(work_dir),
                "--out",
                str(report_path),
                "--replace",
                "Maya=<person_1>",
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "1",
                "--min-intent-kinds",
                "2",
                "--require-redaction",
                "--require-static-drop",
                "--reset",
                "--json",
            )
            aggregate = report["aggregate"]
            item = report["items"][0]
            imported = Path(item["imported_transcript_jsonl"])
            replay_db = Path(item["replay_event_ledger_db"])
            imported_text = imported.read_text(encoding="utf-8")
            written_report = json.loads(report_path.read_text(encoding="utf-8"))
            item_commands = item["next_candidate_commands"]

            self.assertTrue(report["passed"])
            self.assertTrue(report_path.exists())
            self.assertEqual(written_report, report)
            self.assertEqual(report["report_path"], str(report_path))
            self.assertEqual(report["schema"], "melm.local_assistant_transcript_calibration_report.v1")
            self.assertEqual(report["input_count"], 1)
            self.assertEqual(aggregate["imports_passed"], 1)
            self.assertEqual(aggregate["replays_passed"], 1)
            self.assertEqual(aggregate["turns_imported"], 3)
            self.assertEqual(aggregate["turns_replayed"], 3)
            self.assertTrue(all(aggregate["checks"].values()))
            self.assertEqual(aggregate["thresholds"]["min_total_turns"], 3)
            self.assertEqual(aggregate["thresholds"]["min_route_kinds"], 1)
            self.assertEqual(aggregate["thresholds"]["min_intent_kinds"], 2)
            self.assertEqual(aggregate["thresholds"]["min_synthesis_traces"], 0)
            self.assertEqual(aggregate["thresholds"]["min_priority_signal_samples"], 0)
            self.assertFalse(aggregate["thresholds"]["require_priority_signals"])
            self.assertFalse(aggregate["thresholds"]["require_memory_digest_quality"])
            self.assertFalse(aggregate["thresholds"]["require_strict_baseline_win"])
            self.assertTrue(aggregate["thresholds"]["require_redaction"])
            self.assertTrue(aggregate["thresholds"]["require_static_drop"])
            self.assertGreaterEqual(aggregate["route_kinds"], 1)
            self.assertGreaterEqual(aggregate["intent_kinds"], 2)
            self.assertGreaterEqual(aggregate["synthesis_traces"], 1)
            self.assertIn("priority_signal_sample_count", aggregate)
            self.assertIn("memory_digest_quality", aggregate)
            self.assertTrue(replay_db.exists())
            self.assertEqual(report["candidate_event_ledger_dbs"], [str(replay_db)])
            self.assertTrue(item["replay_event_ledger_db_sha256"])
            self.assertIn(str(replay_db), item_commands["candidate_session_audit"])
            self.assertIn("candidate-session-audit", item_commands["candidate_session_audit"])
            self.assertIn("--session all", item_commands["candidate_session_audit"])
            self.assertIn("--capture-surface imported_redacted_transcript", item_commands["candidate_session_audit"])
            self.assertIn("--min-synthesis-traces 1", item_commands["candidate_session_audit"])
            self.assertIn("--min-priority-signal-samples 1", item_commands["candidate_session_audit"])
            self.assertIn(str(report_path), item_commands["candidate_session_audit"])
            self.assertIn("--transcript-calibration-report-json", item_commands["candidate_session_audit"])
            self.assertIn(str(replay_db), item_commands["write_source_attestation"])
            self.assertIn("--event-ledger-session all", item_commands["write_source_attestation"])
            self.assertIn("--capture-surface imported_redacted_transcript", item_commands["write_source_attestation"])
            self.assertIn("--static-expectations-absent", item_commands["write_source_attestation"])
            self.assertIn(str(replay_db), item_commands["v01_evidence_pack"])
            self.assertIn("--session all", item_commands["v01_evidence_pack"])
            self.assertIn(str(report_path), item_commands["v01_evidence_pack"])
            self.assertIn(str(replay_db), item_commands["v01_blocker_evidence"])
            self.assertIn("--event-ledger-session all", item_commands["v01_blocker_evidence"])
            self.assertIn(str(report_path), item_commands["v01_blocker_evidence"])
            self.assertIn("--event-source-kind redacted_user_session", item_commands["v01_blocker_evidence"])
            self.assertIn("--min-synthesis-traces 1", item_commands["v01_blocker_evidence"])
            self.assertIn("--min-priority-signal-samples 1", item_commands["v01_blocker_evidence"])
            self.assertIn("--auto-lifecycle", item_commands["v01_blocker_evidence"])
            self.assertIn("candidate-session-audit first", item_commands["note"])
            self.assertIn("positive trace and priority-signal thresholds", item_commands["note"])
            self.assertEqual(report["next_candidate_commands"][0]["label"], item["label"])
            self.assertEqual(
                report["next_candidate_commands"][0]["candidate_session_audit"],
                item_commands["candidate_session_audit"],
            )
            self.assertEqual(
                report["next_candidate_commands"][0]["write_source_attestation"],
                item_commands["write_source_attestation"],
            )
            self.assertEqual(
                report["next_candidate_commands"][0]["v01_evidence_pack"],
                item_commands["v01_evidence_pack"],
            )
            self.assertEqual(
                report["next_candidate_commands"][0]["v01_blocker_evidence"],
                item_commands["v01_blocker_evidence"],
            )
            self.assertEqual(
                item["replay"]["capture_provenance"]["capture_surface_counts"],
                {"imported_redacted_transcript": 3},
            )
            self.assertEqual(
                aggregate["capture_provenance"]["capture_source_counts"],
                {"redacted_user_transcript_import": 3},
            )
            self.assertTrue(aggregate["capture_provenance"]["has_capture_provenance"])
            self.assertEqual(aggregate["capture_provenance"]["imported_turn_count"], 3)
            self.assertTrue(aggregate["debug_mapping_passed"])
            self.assertTrue(aggregate["primary_uol_chatframe_not_secondary_phrase_route"])
            self.assertTrue(aggregate["checks"]["primary_uol_chatframe_not_secondary_phrase_route"])
            self.assertEqual(aggregate["baseline_required_replays"], 0)
            self.assertEqual(aggregate["redaction_counts"]["email"], 1)
            self.assertEqual(aggregate["redaction_counts"]["phone"], 1)
            self.assertEqual(aggregate["redaction_counts"]["manual_rule_1"], 1)
            self.assertEqual(aggregate["redaction_total"], 3)
            self.assertEqual(aggregate["static_expectation_fields_dropped"]["expected_answer"], 1)
            self.assertEqual(aggregate["static_expectation_fields_dropped"]["expected_route"], 1)
            self.assertEqual(aggregate["static_expectation_fields_dropped_total"], 2)
            self.assertIn("local_answer", aggregate["route_counts"])
            self.assertNotIn("maya@example.com", imported_text)
            self.assertNotIn("+234-555-0101", imported_text)
            self.assertNotIn("Maya", imported_text)
            self.assertIn("<person_1>", imported_text)

            synthesis_floor = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(Path(tmp) / "synthesis_floor_calibration"),
                "--replace",
                "Maya=<person_1>",
                "--min-synthesis-traces",
                "1",
                "--reset",
                "--json",
            )
            self.assertTrue(synthesis_floor["passed"])
            self.assertTrue(synthesis_floor["aggregate"]["checks"]["synthesis_trace_floor"])

            strict = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(Path(tmp) / "strict_calibration"),
                "--replace",
                "Maya=<person_1>",
                "--min-local-resolution-rate",
                "1.01",
                "--min-synthesis-traces",
                "99",
                "--require-memory-digest-quality",
                "--require-strict-baseline-win",
                "--reset",
                "--json",
            )
            self.assertFalse(strict["passed"])
            self.assertFalse(strict["aggregate"]["checks"]["local_resolution_floor"])
            self.assertFalse(strict["aggregate"]["checks"]["synthesis_trace_floor"])
            self.assertFalse(strict["aggregate"]["checks"]["memory_digest_quality_required_met"])
            self.assertFalse(strict["aggregate"]["checks"]["strict_baseline_required_met"])

    def test_cli_v01_blocker_evidence_ingests_transcript_calibration_without_static_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            work_dir = Path(tmp) / "calibration"
            calibration_report = Path(tmp) / "calibration_report.json"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {
                            "role": "user",
                            "content": "Maya emailed maya@example.com about stories.",
                            "expected_answer": "bad",
                        },
                        {"role": "user", "content": "who are you"},
                        {"role": "user", "content": "Call +234-555-0101 tomorrow."},
                    ]
                ),
                encoding="utf-8",
            )

            calibration = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(work_dir),
                "--out",
                str(calibration_report),
                "--replace",
                "Maya=<person_1>",
                "--require-redaction",
                "--require-static-drop",
                "--require-memory-digest-quality",
                "--require-strict-baseline-win",
                "--reset",
                "--json",
            )
            report = _run_cli(
                "v01-blocker-evidence",
                "--transcript-calibration-report-json",
                str(calibration_report),
                "--json",
            )
            blockers = {item["id"]: item for item in report["blockers"]}
            digest = blockers["digest_quality_and_route_threshold_calibration"]
            summary = report["transcript_calibration_report"]

            self.assertFalse(calibration["passed"])
            self.assertTrue(calibration_report.exists())
            self.assertTrue(report["passed"])
            self.assertEqual(report["candidate_blockers_satisfied"], 0)
            self.assertTrue(summary["present"])
            self.assertFalse(summary["strict_digest_route_calibration_passed"])
            self.assertTrue(summary["strict_checks"]["redaction_required"])
            self.assertTrue(summary["strict_checks"]["static_drop_required"])
            self.assertFalse(summary["strict_checks"]["memory_digest_quality_met"])
            self.assertFalse(summary["strict_checks"]["strict_baseline_met"])
            self.assertEqual(digest["status"], "missing_evidence")
            self.assertFalse(digest["candidate_for_architecture_review"])
            self.assertIn("strict digest quality, route threshold, and baseline-win calibration gates", digest["missing"])
            self.assertFalse(digest["evidence"]["transcript_calibration"]["strict_digest_route_calibration_passed"])

    def test_cli_v01_blocker_evidence_rejects_zero_threshold_candidate_shortcuts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            calibration_dir = Path(tmp) / "calibration"
            calibration_report = Path(tmp) / "calibration_report.json"
            attestation = Path(tmp) / "source_attestation.json"
            blocker_dir = Path(tmp) / "blocker_evidence"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {"role": "user", "content": "Maya emailed maya@example.com about stories."},
                        {"role": "user", "content": "who are you"},
                        {"role": "user", "content": "Call +234-555-0101 tomorrow."},
                    ]
                ),
                encoding="utf-8",
            )
            calibration = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(calibration_dir),
                "--out",
                str(calibration_report),
                "--replace",
                "Maya=<person_1>",
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--require-redaction",
                "--reset",
                "--json",
            )
            replay_db = Path(calibration["items"][0]["replay_event_ledger_db"])

            written = _run_cli(
                "write-source-attestation",
                "--out",
                str(attestation),
                "--event-ledger-db",
                str(replay_db),
                "--source-kind",
                "redacted_user_session",
                "--capture-surface",
                "imported_redacted_transcript",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--overwrite",
                "--json",
            )
            zero_floor = _run_cli(
                "v01-blocker-evidence",
                "--event-ledger-db",
                str(replay_db),
                "--event-ledger-work-dir",
                str(blocker_dir),
                "--event-source-kind",
                "redacted_user_session",
                "--source-attestation-json",
                str(attestation),
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--min-synthesis-traces",
                "0",
                "--min-priority-signal-samples",
                "0",
                "--reset",
                "--json",
            )
            positive_synthesis_floor = _run_cli(
                "v01-blocker-evidence",
                "--event-ledger-db",
                str(replay_db),
                "--event-ledger-work-dir",
                str(blocker_dir),
                "--event-source-kind",
                "redacted_user_session",
                "--source-attestation-json",
                str(attestation),
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--min-synthesis-traces",
                "1",
                "--min-priority-signal-samples",
                "0",
                "--reset",
                "--json",
            )

            zero_blockers = {item["id"]: item for item in zero_floor["blockers"]}
            positive_blockers = {item["id"]: item for item in positive_synthesis_floor["blockers"]}

            self.assertTrue(calibration["passed"])
            self.assertTrue(written["passed"])
            self.assertTrue(zero_floor["source_attestation"]["valid"])
            self.assertEqual(zero_floor["candidate_blockers_satisfied"], 1)
            self.assertEqual(
                zero_blockers["real_user_derived_lifecycle_traces"]["status"],
                "candidate_evidence_present",
            )
            self.assertEqual(
                zero_blockers["user_derived_bounded_synthesis_traces"]["status"],
                "missing_evidence",
            )
            self.assertFalse(
                zero_blockers["user_derived_bounded_synthesis_traces"]["evidence"][
                    "positive_threshold_configured"
                ]
            )
            self.assertIn(
                "positive --min-synthesis-traces threshold for candidate synthesis evidence",
                zero_blockers["user_derived_bounded_synthesis_traces"]["missing"],
            )
            self.assertEqual(
                zero_blockers["planner_priority_on_user_derived_traces"]["status"],
                "missing_evidence",
            )
            self.assertFalse(
                zero_blockers["planner_priority_on_user_derived_traces"]["evidence"][
                    "positive_threshold_configured"
                ]
            )
            self.assertIn(
                "positive --min-priority-signal-samples threshold for candidate planner evidence",
                zero_blockers["planner_priority_on_user_derived_traces"]["missing"],
            )
            self.assertEqual(
                positive_blockers["user_derived_bounded_synthesis_traces"]["status"],
                "candidate_evidence_present",
            )
            self.assertEqual(
                positive_blockers["planner_priority_on_user_derived_traces"]["status"],
                "missing_evidence",
            )

    def test_cli_v01_blocker_evidence_keeps_strict_calibration_unattested_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration_report = Path(tmp) / "strict_calibration_report.json"
            calibration_report.write_text(
                json.dumps(
                    {
                        "schema": "melm.local_assistant_transcript_calibration_report.v1",
                        "passed": True,
                        "input_count": 1,
                        "aggregate": {
                            "checks": {
                                "redaction_required_met": True,
                                "static_drop_required_met": True,
                                "memory_digest_quality_required_met": True,
                                "strict_baseline_required_met": True,
                                "route_diversity_floor": True,
                                "local_resolution_floor": True,
                                "debug_mapping_passed": True,
                                "primary_uol_chatframe_not_secondary_phrase_route": True,
                                "critical_safety_clean": True,
                            },
                            "thresholds": {
                                "require_redaction": True,
                                "require_static_drop": True,
                                "require_memory_digest_quality": True,
                                "require_strict_baseline_win": True,
                            },
                            "turns_replayed": 12,
                            "local_resolution_rate": 0.75,
                            "route_kinds": 4,
                            "intent_kinds": 6,
                            "route_counts": {"local_answer": 7, "cached_tool": 2, "device_action": 1, "cloud_handoff": 2},
                            "memory_digest_quality": {"passed_replays": 1, "all_required_passed": True},
                            "baseline_required_replays": 1,
                            "strict_baseline_passed_replays": 1,
                        },
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "v01-blocker-evidence",
                "--event-source-kind",
                "redacted_user_session",
                "--transcript-calibration-report-json",
                str(calibration_report),
                "--json",
            )
            digest = {item["id"]: item for item in report["blockers"]}[
                "digest_quality_and_route_threshold_calibration"
            ]

            self.assertTrue(report["passed"])
            self.assertFalse(report["source_attestation"]["present"])
            self.assertEqual(report["candidate_blockers_satisfied"], 0)
            self.assertTrue(report["transcript_calibration_report"]["strict_digest_route_calibration_passed"])
            self.assertFalse(report["transcript_calibration_report"]["candidate_digest_route_calibration_passed"])
            self.assertFalse(report["transcript_calibration_report"]["event_ledger_binding"]["passed"])
            self.assertEqual(digest["status"], "missing_evidence")
            self.assertFalse(digest["candidate_for_architecture_review"])
            self.assertIn("valid source attestation JSON", digest["missing"])
            self.assertIn(
                "event-ledger DB supplied for transcript calibration binding",
                digest["missing"],
            )

    def test_cli_v01_blocker_evidence_rejects_debug_map_without_primary_uol_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration_report = Path(tmp) / "weak_calibration_report.json"
            calibration_report.write_text(
                json.dumps(
                    {
                        "schema": "melm.local_assistant_transcript_calibration_report.v1",
                        "passed": True,
                        "input_count": 1,
                        "aggregate": {
                            "checks": {
                                "redaction_required_met": True,
                                "static_drop_required_met": True,
                                "memory_digest_quality_required_met": True,
                                "strict_baseline_required_met": True,
                                "route_diversity_floor": True,
                                "local_resolution_floor": True,
                                "debug_mapping_passed": True,
                                "primary_uol_chatframe_not_secondary_phrase_route": False,
                                "critical_safety_clean": True,
                            },
                            "thresholds": {
                                "require_redaction": True,
                                "require_static_drop": True,
                                "require_memory_digest_quality": True,
                                "require_strict_baseline_win": True,
                            },
                            "turns_replayed": 12,
                            "local_resolution_rate": 0.75,
                            "route_kinds": 4,
                            "intent_kinds": 6,
                            "route_counts": {
                                "local_answer": 7,
                                "cached_tool": 2,
                                "device_action": 1,
                                "cloud_handoff": 2,
                            },
                            "memory_digest_quality": {"passed_replays": 1, "all_required_passed": True},
                            "baseline_required_replays": 1,
                            "strict_baseline_passed_replays": 1,
                        },
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "v01-blocker-evidence",
                "--event-source-kind",
                "redacted_user_session",
                "--transcript-calibration-report-json",
                str(calibration_report),
                "--json",
            )
            summary = report["transcript_calibration_report"]
            digest = {item["id"]: item for item in report["blockers"]}[
                "digest_quality_and_route_threshold_calibration"
            ]

            self.assertTrue(report["passed"])
            self.assertFalse(summary["strict_digest_route_calibration_passed"])
            self.assertTrue(summary["strict_checks"]["debug_mapping_passed"])
            self.assertFalse(summary["strict_checks"]["primary_uol_chatframe_not_secondary_phrase_route"])
            self.assertEqual(digest["status"], "missing_evidence")
            self.assertIn(
                "primary UOL/ChatFrame routing evidence without secondary phrase primary routes",
                digest["missing"],
            )

    def test_cli_v01_blocker_evidence_rejects_unbound_strict_digest_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw_chat.jsonl"
            calibration_dir = root / "calibration"
            replay_calibration_report = root / "replay_calibration_report.json"
            attestation = root / "source_attestation.json"
            calibration_report = root / "strict_calibration_report.json"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {"role": "user", "content": "Maya emailed maya@example.com about stories."},
                        {"role": "user", "content": "who are you"},
                        {"role": "user", "content": "tell me a story"},
                    ]
                ),
                encoding="utf-8",
            )
            replay_calibration = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(calibration_dir),
                "--out",
                str(replay_calibration_report),
                "--replace",
                "Maya=<person_1>",
                "--min-total-turns",
                "3",
                "--min-local-resolution-rate",
                "0.2",
                "--min-route-kinds",
                "2",
                "--min-intent-kinds",
                "2",
                "--require-redaction",
                "--reset",
                "--json",
            )
            db = Path(replay_calibration["items"][0]["replay_event_ledger_db"])
            _run_cli(
                "write-source-attestation",
                "--out",
                str(attestation),
                "--event-ledger-db",
                str(db),
                "--source-kind",
                "redacted_user_session",
                "--capture-surface",
                "imported_redacted_transcript",
                "--redaction-applied",
                "--static-expectations-absent",
                "--answers-routes-reasons-absent",
                "--human-reviewed",
                "--overwrite",
                "--json",
            )
            calibration_report.write_text(
                json.dumps(
                    {
                        "schema": "melm.local_assistant_transcript_calibration_report.v1",
                        "passed": True,
                        "input_count": 1,
                        "candidate_event_ledger_dbs": [str(root / "other.sqlite")],
                        "items": [
                            {
                                "label": "unbound",
                                "replay_event_ledger_db": str(root / "other.sqlite"),
                                "replay_event_ledger_db_sha256": "0" * 64,
                            }
                        ],
                        "aggregate": {
                            "checks": {
                                "redaction_required_met": True,
                                "static_drop_required_met": True,
                                "memory_digest_quality_required_met": True,
                                "strict_baseline_required_met": True,
                                "route_diversity_floor": True,
                                "local_resolution_floor": True,
                                "debug_mapping_passed": True,
                                "primary_uol_chatframe_not_secondary_phrase_route": True,
                                "critical_safety_clean": True,
                            },
                            "thresholds": {
                                "require_redaction": True,
                                "require_static_drop": True,
                                "require_memory_digest_quality": True,
                                "require_strict_baseline_win": True,
                            },
                            "turns_replayed": 12,
                            "local_resolution_rate": 0.75,
                            "route_kinds": 4,
                            "intent_kinds": 6,
                            "route_counts": {
                                "local_answer": 7,
                                "cached_tool": 2,
                                "device_action": 1,
                                "cloud_handoff": 2,
                            },
                            "capture_provenance": {
                                "turn_count": 12,
                                "capture_surface_counts": {"imported_redacted_transcript": 12},
                                "capture_source_counts": {"redacted_user_transcript_import": 12},
                                "missing_field_count": 0,
                                "imported_turn_count": 12,
                                "has_capture_provenance": True,
                            },
                            "memory_digest_quality": {"passed_replays": 1, "all_required_passed": True},
                            "baseline_required_replays": 1,
                            "strict_baseline_passed_replays": 1,
                        },
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "v01-blocker-evidence",
                "--event-ledger-db",
                str(db),
                "--event-ledger-work-dir",
                str(root / "blocker_evidence"),
                "--event-source-kind",
                "redacted_user_session",
                "--source-attestation-json",
                str(attestation),
                "--transcript-calibration-report-json",
                str(calibration_report),
                "--min-total-turns",
                "2",
                "--min-route-kinds",
                "1",
                "--min-intent-kinds",
                "1",
                "--reset",
                "--json",
            )
            digest = {item["id"]: item for item in report["blockers"]}[
                "digest_quality_and_route_threshold_calibration"
            ]
            binding = report["transcript_calibration_report"]["event_ledger_binding"]

            self.assertTrue(report["passed"])
            self.assertTrue(report["source_attestation"]["valid"])
            self.assertTrue(report["transcript_calibration_report"]["strict_digest_route_calibration_passed"])
            self.assertFalse(report["transcript_calibration_report"]["candidate_digest_route_calibration_passed"])
            self.assertFalse(binding["passed"])
            self.assertFalse(binding["path_matched"])
            self.assertFalse(binding["hash_matched"])
            self.assertEqual(digest["status"], "missing_evidence")
            self.assertFalse(digest["candidate_for_architecture_review"])
            self.assertIn(
                "transcript calibration report lists the current event-ledger DB path",
                digest["missing"],
            )
            self.assertIn(
                "transcript calibration replay DB SHA-256 matches the current event-ledger DB",
                digest["missing"],
            )

    def test_cli_calibrate_transcript_replay_uses_safe_controls_for_priority_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            controls = Path(tmp) / "controls.json"
            work_dir = Path(tmp) / "controlled_calibration"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {"role": "user", "content": "Tell me a story."},
                        {"role": "user", "content": "What is the weather today?"},
                    ]
                ),
                encoding="utf-8",
            )
            controls.write_text(
                json.dumps(
                    {
                        "expectations": {"required_priority_signals": True},
                        "turns": {
                            "1": {"schedule_refreshes": True},
                            "2": {"schedule_refreshes": True, "execute_jobs": True},
                        },
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(work_dir),
                "--controls-json",
                str(controls),
                "--require-priority-signals",
                "--min-priority-signal-samples",
                "1",
                "--min-total-turns",
                "2",
                "--reset",
                "--json",
            )
            aggregate = report["aggregate"]
            item = report["items"][0]
            imported_text = Path(item["imported_transcript_jsonl"]).read_text(encoding="utf-8")

            self.assertTrue(report["passed"])
            self.assertTrue(aggregate["checks"]["priority_signals_required_met"])
            self.assertTrue(aggregate["checks"]["priority_signal_sample_floor"])
            self.assertGreaterEqual(aggregate["priority_signal_sample_count"], 1)
            self.assertEqual(item["import"]["control_fields_applied"]["schedule_refreshes"], 2)
            self.assertEqual(item["import"]["control_fields_applied"]["execute_jobs"], 1)
            self.assertIn('"schedule_refreshes": true', imported_text)
            self.assertIn('"execute_jobs": true', imported_text)
            self.assertNotIn("expected_route", imported_text)

            bad_controls = Path(tmp) / "bad_controls.json"
            bad_controls.write_text(
                json.dumps({"turns": {"1": {"expected_route": "local_answer"}}}, ensure_ascii=True),
                encoding="utf-8",
            )
            bad = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "calibrate-transcript-replay",
                    "--input",
                    str(raw),
                    "--work-dir",
                    str(Path(tmp) / "bad_calibration"),
                    "--controls-json",
                    str(bad_controls),
                    "--json",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(bad.returncode, 0)
            bad_report = json.loads(bad.stdout)
            self.assertFalse(bad_report["passed"])
            self.assertIn("static expectation", bad_report["items"][0]["error"])

    def test_cli_calibrate_transcript_replay_auto_lifecycle_without_turn_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            work_dir = Path(tmp) / "auto_lifecycle_calibration"
            raw.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True) + "\n"
                    for record in [
                        {"role": "user", "content": "Tell me a story.", "session_id": "s1"},
                        {"role": "user", "content": "What is the weather today?", "session_id": "s1"},
                        {"role": "user", "content": "What should I eat today?", "session_id": "s1"},
                        {"role": "user", "content": "What did we talk about earlier?", "session_id": "s2"},
                        {"role": "user", "content": "What have you done so far?", "session_id": "s2"},
                    ]
                ),
                encoding="utf-8",
            )

            report = _run_cli(
                "calibrate-transcript-replay",
                "--input",
                str(raw),
                "--work-dir",
                str(work_dir),
                "--min-total-turns",
                "5",
                "--min-route-kinds",
                "3",
                "--min-intent-kinds",
                "5",
                "--min-synthesis-traces",
                "2",
                "--min-priority-signal-samples",
                "1",
                "--require-priority-signals",
                "--require-memory-digest-quality",
                "--auto-lifecycle",
                "--reset",
                "--json",
            )
            aggregate = report["aggregate"]
            item = report["items"][0]
            imported_text = Path(item["imported_transcript_jsonl"]).read_text(encoding="utf-8")

            self.assertTrue(report["passed"])
            self.assertTrue(report["auto_lifecycle"])
            self.assertEqual(item["import"]["control_fields_applied"], {"new_session": 0})
            self.assertNotIn("schedule_refreshes", imported_text)
            self.assertNotIn("execute_jobs", imported_text)
            self.assertGreaterEqual(aggregate["priority_signal_sample_count"], 1)
            self.assertTrue(aggregate["checks"]["priority_signals_required_met"])
            self.assertTrue(aggregate["checks"]["memory_digest_quality_required_met"])
            self.assertTrue(aggregate["memory_digest_quality"]["all_required_passed"])
            self.assertGreaterEqual(aggregate["synthesis_traces"], 2)
            self.assertEqual(aggregate["turns_replayed"], 5)
            self.assertGreaterEqual(aggregate["local_resolution_rate"], 0.6)
            self.assertEqual(item["replay"]["counts"]["jobs"], 2)
            self.assertGreaterEqual(item["replay"]["counts"]["inventories"], 10)
            self.assertTrue(item["replay"]["priority_signal_samples"])


def _run_cli(*args: str, _env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    # Limit bulk entries for all CLI tests (avoids 25-min ingestion)
    env.setdefault("MELM_BULK_MAX_ENTRIES", "2000")
    if _env:
        env.update(_env)
    result = subprocess.run(
        [sys.executable, str(CLI), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def _action_recorder_command(log: Path, label: str) -> str:
    script = Path("tests/fixtures/action_recorder.py").resolve()
    parts = (sys.executable, str(script), "--log", str(log), "--label", label)
    return " ".join(shlex.quote(part) for part in parts)


if __name__ == "__main__":
    unittest.main()
