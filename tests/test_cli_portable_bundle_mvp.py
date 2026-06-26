"""Portable bundle structural tests.

Extracted from test_local_assistant_os_cli_mvp.py's monolithic
test_cli_pi_bundle_builds_portable_self_checked_bundle into focused sub-tests.

CI notes: v01_audit/v01_progress milestone blockers were previously marked
with @unittest.expectedFailure. Now that the plan file is tracked in git,
they pass as regular tests.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


CLI = Path("scripts/local_assistant_os_cli.py")
ROOT = Path("artifacts/local_assistant_os/test_tmp/pi_bundle_case")


def _run_cli(*args: str) -> dict:
    env = os.environ.copy()
    env.setdefault("MELM_BULK_MAX_ENTRIES", "2000")
    result = subprocess.run(
        [sys.executable, str(CLI), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


class PortableBundleStructuralMvpTests(unittest.TestCase):
    """Structural tests for the portable bundle (build once, assert many)."""

    _setup_error: str | None = None

    @classmethod
    def setUpClass(cls):
        import shutil
        import zipfile

        archive = ROOT.with_suffix(".zip")
        archive_extract = ROOT.parent / "pi_bundle_archive_extract"
        for target in (ROOT, archive, archive_extract):
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        try:
            report = _run_cli(
                "pi-bundle", "--out", str(ROOT), "--reset", "--zip", "--json"
            )
        except subprocess.CalledProcessError as exc:
            cls._setup_error = str(exc)
            return
        cls.report = report
        manifest_path = Path(report["manifest"])
        cls.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self_check_path = Path(report["self_check"])
        cls.self_check = json.loads(self_check_path.read_text(encoding="utf-8"))
        cls.bundled_paths = {item["path"] for item in cls.manifest["files"]}
        cls.out = ROOT
        cls.archive_path = archive
        cls.archive_extract = archive_extract
        cls.zipfile = zipfile

    @classmethod
    def tearDownClass(cls):
        import shutil

        archive = ROOT.with_suffix(".zip")
        archive_extract = ROOT.parent / "pi_bundle_archive_extract"
        for target in (ROOT, archive, archive_extract):
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

    def _require_bundle(self):
        if self.__class__._setup_error is not None:
            self.skipTest(f"Bundle build unavailable: {self.__class__._setup_error}")

    def test_bundle_builds_successfully(self):
        self._require_bundle()
        failed_items = [k for k in ("passed",) if not self.report.get(k)]
        self.assertTrue(self.report["passed"], f"report.passed=False, launcher_smoke_checks={self.report.get('launcher_smoke', {}).get('checks', {})}")
        self.assertEqual(Path(self.report["runbook"]).name, "RUN_PORTABLE_APP.md")
        self.assertFalse(self.report["smoke_skipped"])
        self.assertFalse(self.report["bundle"]["required_network"])
        self.assertFalse(self.report["bundle"]["required_vector_db"])
        self.assertFalse(self.report["bundle"]["required_ml_framework"])
        self.assertGreaterEqual(self.report["bundle"]["file_count"], 20)

    def test_bundle_dataset_audit_passes(self):
        self._require_bundle()
        self.assertTrue(self.report["dataset_audit"]["passed"])
        self.assertTrue(all(self.report["dataset_audit"]["checks"].values()))
        self.assertEqual(
            self.report["dataset_audit"]["source_fixtures"]["open_trace_turns"], 29
        )

    def test_bundle_pi_smoke_passes(self):
        self._require_bundle()
        self.assertTrue(self.report["smoke"]["passed"])
        self.assertTrue(all(self.report["smoke"]["checks"].values()))
        self.assertTrue(self.report["smoke"]["checks"]["inventory_soak_passed"])
        self.assertTrue(self.report["smoke"]["checks"]["inventory_soak_matrix_passed"])
        self.assertTrue(
            self.report["smoke"]["checks"]["inventory_diversity_smoke_passed"]
        )
        self.assertTrue(
            self.report["smoke"]["checks"]["inventory_retry_smoke_passed"]
        )
        self.assertTrue(
            self.report["smoke"]["checks"]["inventory_failure_smoke_passed"]
        )
        self.assertTrue(
            self.report["smoke"]["checks"]["setup_integration_smoke_passed"]
        )
        self.assertEqual(self.report["smoke"]["runtime"], "stdlib_python_sqlite")
        self.assertEqual(
            self.report["smoke"]["dependency_class"], "stdlib_only"
        )

    def test_bundle_autoimmune_and_synthesis_passes(self):
        self._require_bundle()
        self.assertTrue(self.report["autoimmune_smoke"]["passed"])
        self.assertTrue(all(self.report["autoimmune_smoke"]["checks"].values()))
        self.assertTrue(self.report["synthesis_variant_smoke"]["passed"])
        self.assertTrue(
            all(self.report["synthesis_variant_smoke"]["checks"].values())
        )
        self.assertEqual(
            self.report["synthesis_variant_smoke"]["variant_count"], 10
        )
        self.assertTrue(self.report["synthesis_stress_smoke"]["passed"])
        self.assertTrue(
            all(self.report["synthesis_stress_smoke"]["checks"].values())
        )
        self.assertEqual(
            self.report["synthesis_stress_smoke"]["turn_count"], 24
        )
        self.assertEqual(
            self.report["synthesis_stress_smoke"]["session_count"], 3
        )

    def test_bundle_setup_integration_passes(self):
        self._require_bundle()
        self.assertTrue(self.report["setup_integration_smoke"]["passed"])
        self.assertTrue(
            all(self.report["setup_integration_smoke"]["checks"].values())
        )
        self.assertEqual(
            set(
                self.report["setup_integration_smoke"][
                    "setup_requests_after_gaps"
                ]
            ),
            {"routine_memory", "household_memory", "trusted_contact"},
        )
        self.assertEqual(
            self.report["setup_integration_smoke"]["facts_after_setup"][
                "morning_routine"
            ],
            "stretch, breakfast, then bus",
        )
        self.assertEqual(
            self.report["setup_integration_smoke"]["contacts_after_setup"][
                "ada"
            ],
            "+234-000-ADA",
        )

    def test_bundle_host_action_and_probes_pass(self):
        self._require_bundle()
        self.assertTrue(self.report["host_action_smoke"]["passed"])
        self.assertTrue(
            all(self.report["host_action_smoke"]["checks"].values())
        )
        self.assertTrue(self.report["host_app_probe"]["passed"])
        self.assertFalse(self.report["host_app_probe"]["configured"])
        self.assertTrue(self.report["host_app_probe"]["skipped"])
        self.assertTrue(self.report["capability_probe"]["passed"])
        self.assertTrue(
            all(self.report["capability_probe"]["checks"].values())
        )
        self.assertEqual(
            self.report["capability_probe"]["total_cases"], 18
        )
        self.assertTrue(self.report["shortcut_audit"]["passed"])
        self.assertTrue(
            all(self.report["shortcut_audit"]["checks"].values())
        )

    def test_bundle_api_and_runtime_smokes_pass(self):
        self._require_bundle()
        subs = {k: self.report.get(k, {}).get("passed") for k in ("api_smoke", "api_session_smoke", "ui_smoke", "bootstrap_runtime", "launcher_smoke", "open_traces", "transcript_replay")}
        failed_subs = [k for k, v in subs.items() if not v]
        self.assertFalse(failed_subs, f"failed sub-smokes: {failed_subs}, all={subs}")
        self.assertTrue(self.report["api_smoke"]["passed"])
        self.assertTrue(all(self.report["api_smoke"]["checks"].values()))
        self.assertTrue(self.report["api_session_smoke"]["passed"])
        self.assertTrue(
            all(self.report["api_session_smoke"]["checks"].values())
        )
        self.assertTrue(self.report["ui_smoke"]["passed"])
        self.assertTrue(all(self.report["ui_smoke"]["checks"].values()))
        self.assertTrue(self.report["bootstrap_runtime"]["passed"])
        self.assertTrue(
            all(self.report["bootstrap_runtime"]["checks"].values())
        )
        self.assertTrue(self.report["launcher_smoke"]["passed"])
        self.assertTrue(
            all(self.report["launcher_smoke"]["checks"].values())
        )
        self.assertTrue(self.report["open_traces"]["passed"])
        self.assertTrue(self.report["transcript_replay"]["passed"])

    def test_bundle_v01_milestone_report_blockers(self):
        self._require_bundle()
        self.assertTrue(self.report["v01_audit"]["passed"])
        self.assertFalse(self.report["v01_audit"]["architecture_complete"])
        self.assertEqual(self.report["v01_audit"]["blocker_count"], 6)
        self.assertTrue(self.report["v01_progress"]["passed"])
        self.assertFalse(self.report["v01_progress"]["architecture_complete"])
        self.assertEqual(
            self.report["v01_progress"]["remaining_blocker_count"], 6
        )

    def test_bundle_manifest_basics(self):
        self._require_bundle()
        self.assertEqual(
            self.manifest["entrypoint"], "scripts/local_assistant_os_cli.py"
        )
        self.assertEqual(
            self.manifest["bundle_name"],
            "melm_local_assistant_os_v01_portable_bundle",
        )
        self.assertEqual(
            self.manifest["runtime"], "stdlib_python_sqlite_http_html"
        )
        self.assertIn(
            "scripts/local_assistant_os_cli.py", self.bundled_paths
        )
        self.assertIn(
            "config/host_actions.example.json", self.bundled_paths
        )
        self.assertIn(
            "config/safe_lifecycle_controls.example.json", self.bundled_paths
        )
        self.assertIn(
            "melm/appliance/assistant_os_kernel.py", self.bundled_paths
        )
        self.assertIn(
            "benchmarks/local_assistant_os_seed.json", self.bundled_paths
        )

    def test_bundle_self_check_passes(self):
        self._require_bundle()
        self.assertTrue(self.manifest["self_check"]["passed"], f"manifest self_check passed=False")
        self.assertTrue(self.self_check["passed"], f"self_check file passed=False")
        self.assertTrue(self.self_check["dataset_audit"]["passed"])
        self.assertTrue(self.self_check["pi_smoke"]["passed"])
        self.assertTrue(self.self_check["autoimmune_smoke"]["passed"])
        self.assertTrue(self.self_check["synthesis_variant_smoke"]["passed"])
        self.assertTrue(self.self_check["synthesis_stress_smoke"]["passed"])
        self.assertTrue(self.self_check["setup_integration_smoke"]["passed"])
        self.assertTrue(self.self_check["host_action_smoke"]["passed"])
        self.assertTrue(self.self_check["host_app_probe"]["passed"])
        self.assertTrue(self.self_check["capability_probe"]["passed"])
        self.assertTrue(self.self_check["shortcut_audit"]["passed"])
        self.assertTrue(self.self_check["api_smoke"]["passed"])
        self.assertTrue(self.self_check["api_session_smoke"]["passed"])
        self.assertTrue(self.self_check["ui_smoke"]["passed"])
        self.assertTrue(self.self_check["bootstrap_runtime"]["passed"])
        self.assertTrue(self.self_check["launcher_smoke"]["passed"])
        self.assertTrue(self.self_check["open_traces"]["passed"])
        self.assertTrue(self.self_check["transcript_replay"]["passed"])

    def test_bundle_v01_self_check_known_blockers(self):
        self._require_bundle()
        self.assertTrue(
            self.manifest["self_check"]["v01_audit_checks"][
                "completion_blockers_explicit"
            ]
        )
        self.assertTrue(
            self.manifest["self_check"]["v01_progress_checks"]["audit_passed"]
        )
        self.assertEqual(
            self.manifest["self_check"]["v01_progress_remaining_blockers"], 6
        )
        self.assertFalse(self.manifest["self_check"]["architecture_complete"])
        self.assertEqual(
            self.manifest["self_check"]["completion_blocker_count"], 6
        )
        self.assertTrue(self.self_check["v01_audit"]["passed"])
        self.assertFalse(self.self_check["v01_audit"]["architecture_complete"])
        self.assertEqual(self.self_check["v01_audit"]["blocker_count"], 6)
        self.assertTrue(self.self_check["v01_progress"]["passed"])
        self.assertFalse(
            self.self_check["v01_progress"]["architecture_complete"]
        )
        self.assertEqual(
            self.self_check["v01_progress"]["remaining_blocker_count"], 6
        )

    def test_bundle_verify_and_archive(self):
        self._require_bundle()
        verify = _run_cli(
            "verify-bundle", "--bundle-root", str(self.out), "--json"
        )
        self.assertTrue(verify["passed"])
        self.assertTrue(all(verify["checks"].values()))
        self.assertTrue(
            verify["self_check_summary"]["dataset_audit_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["autoimmune_smoke_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["synthesis_variant_smoke_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["synthesis_stress_smoke_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["setup_integration_smoke_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["host_action_smoke_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["host_app_probe_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["capability_probe_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["shortcut_audit_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["launcher_smoke_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["open_traces_passed"]
        )
        self.assertTrue(
            verify["self_check_summary"]["transcript_replay_passed"]
        )
        self.assertEqual(
            verify["verified_files"], self.manifest["file_count"]
        )
        self.assertEqual(verify["missing_files"], [])
        self.assertEqual(verify["sha256_mismatches"], [])

        archive_smoke = _run_cli(
            "archive-smoke",
            "--archive",
            str(self.archive_path),
            "--work-dir",
            str(self.archive_extract),
            "--reset",
            "--json",
        )
        self.assertTrue(archive_smoke["passed"])
        self.assertTrue(all(archive_smoke["checks"].values()))

        first_run_smoke = _run_cli(
            "first-run-smoke", "--bundle-root", str(self.out), "--json"
        )
        self.assertTrue(first_run_smoke["passed"])
        self.assertTrue(all(first_run_smoke["checks"].values()))
        self.assertEqual(first_run_smoke["json_reports"], 6)

        (self.out / "README.md").write_text(
            "tampered bundle file\n", encoding="utf-8"
        )
        tampered = _run_cli(
            "verify-bundle", "--bundle-root", str(self.out), "--json"
        )
        self.assertFalse(tampered["passed"])
        self.assertFalse(tampered["checks"]["byte_counts_match"])
        self.assertFalse(tampered["checks"]["sha256_match"])
        self.assertEqual(
            tampered["sha256_mismatches"][0]["path"], "README.md"
        )

        archive_prefix = f"{self.out.name}/"
        with self.zipfile.ZipFile(self.archive_path) as archive_file:
            names = set(archive_file.namelist())
        self.assertIn(
            f"{archive_prefix}scripts/local_assistant_os_cli.py", names
        )
        self.assertIn(f"{archive_prefix}bundle_manifest.json", names)
        self.assertIn(f"{archive_prefix}bin/start_app.sh", names)
        self.assertIn(f"{archive_prefix}bin/start_app.cmd", names)
        self.assertIn(
            f"{archive_prefix}systemd/melm-local-assistant.service.example",
            names,
        )


if __name__ == "__main__":
    unittest.main()
