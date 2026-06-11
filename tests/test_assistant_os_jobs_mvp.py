import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from melm.appliance import (
    AssistantDecision,
    AssistantOSKernel,
    AssistantOSStore,
    LocalAssistantProfile,
    MembranePolicy,
    OnDeviceAssistantRouter,
    OpenMeteoWeatherAdapter,
    PublicDomainStoryMetadataAdapter,
    build_assistant_os_dashboard,
    schedule_inventory_refreshes,
    weather_items_to_inventory_rows,
)


CLI = Path("scripts/local_assistant_os_cli.py")
WEATHER_SAMPLE = Path("benchmarks/sample_open_meteo_forecast.json")


class AssistantOSJobsMvpTests(unittest.TestCase):
    def test_story_inventory_builder_uses_metadata_records(self) -> None:
        profile = LocalAssistantProfile(
            age=7,
            location="Lagos",
            culture="Yoruba",
            story_models={},
            preferences={"story_theme": "folktale bedtime rain"},
        )

        result = PublicDomainStoryMetadataAdapter().build_story_inventory(profile)

        self.assertEqual(result.source_count, 4)
        self.assertEqual(len(result.story_models), 3)
        self.assertIn("pg_moon_drum_lagos_age7", result.story_models)
        self.assertIn("talking drum", result.story_models["pg_moon_drum_lagos_age7"])
        self.assertIn("project_gutenberg_catalog_metadata", {item.source for item in result.selected_items})
        self.assertIn("internet_archive_item_search_and_metadata", {item.source for item in result.selected_items})

    def test_weather_adapter_replays_open_meteo_fixture_into_inventory_rows(self) -> None:
        profile = LocalAssistantProfile(location="Lagos", weekly_weather={})

        result = OpenMeteoWeatherAdapter().refresh(profile, offline_json=WEATHER_SAMPLE)
        rows = weather_items_to_inventory_rows(result)
        today = next(row for row in rows if row["item_id"] == "today")

        self.assertFalse(result.network_used)
        self.assertEqual(result.source, "open_meteo_offline_fixture")
        self.assertGreaterEqual(result.weather_days, 7)
        self.assertEqual(today["kind"], "weather")
        self.assertEqual(today["source"], "open_meteo_offline_fixture")
        self.assertIn("rain", today["payload"]["forecast"])
        self.assertEqual(today["payload"]["location"], "Lagos, Lagos State, Nigeria")

    def test_reflection_creates_persistent_resource_budgeted_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}),
                store=store,
            )
            for _ in range(3):
                kernel.handle("Tell me a story.")

            opportunities = kernel.reflect()
            jobs = store.load_jobs(status="queued")

            self.assertEqual(opportunities[0].kind, "build_story_inventory")
            self.assertEqual(jobs[0].kind, "build_story_inventory")
            self.assertEqual(jobs[0].resource_budget["cpu_class"], "raspberry_pi")
            self.assertEqual(jobs[0].resource_budget["network"], "metadata_only")
            store.close()

    def test_inventory_refresh_scheduler_queues_pi_budgeted_idempotent_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            store.upsert_inventory(
                "weather",
                "today",
                {"forecast": "warm with afternoon rain", "stale": True},
                source="weather_cache",
                license="local_cache",
                tags=("weather",),
            )
            store.connection.commit()
            profile = store.load_profile(
                LocalAssistantProfile(story_models={}, weekly_weather={"today": "warm"})
            )

            first = schedule_inventory_refreshes(
                store,
                profile,
                min_story_models=3,
                story_limit=4,
                use_offline_samples=True,
            )
            second = schedule_inventory_refreshes(
                store,
                profile,
                min_story_models=3,
                story_limit=4,
                use_offline_samples=True,
            )
            jobs = store.load_jobs(status="queued")
            kinds = {job.kind for job in jobs}
            budgets = {job.kind: job.resource_budget for job in jobs}

            self.assertEqual(first.story_inventory_count, 0)
            self.assertFalse(first.weather_today_cached)
            self.assertEqual(first.queued_jobs, 2)
            self.assertEqual(second.queued_jobs, 2)
            self.assertEqual(kinds, {"import_story_metadata", "refresh_weather_cache"})
            self.assertEqual(budgets["import_story_metadata"]["network"], "offline_fixture")
            self.assertEqual(budgets["import_story_metadata"]["cpu_class"], "raspberry_pi")
            self.assertEqual(budgets["import_story_metadata"]["quality_floor"], 0.5)
            self.assertEqual(budgets["refresh_weather_cache"]["cache_policy"], "refresh_stale_or_missing_today")
            store.close()

    def test_refresh_scheduler_prioritizes_from_homeostatic_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}),
                store=store,
            )
            for _ in range(3):
                kernel.handle("Tell me a story.")
            profile = store.load_profile(LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}))

            report = schedule_inventory_refreshes(
                store,
                profile,
                min_story_models=4,
                story_limit=3,
                use_offline_samples=True,
            )
            story = next(item for item in report.recommendations if item.kind == "import_story_metadata")
            story_job = next(job for job in store.load_jobs(status="queued") if job.kind == "import_story_metadata")

            self.assertEqual(story.priority_signals["recent_story_cloud_handoffs"], 3)
            self.assertGreater(story.priority_signals["avg_cloud_dependence"], 0.0)
            self.assertGreater(story.priority, 0.88)
            self.assertEqual(story_job.payload["priority_signals"]["recent_story_cloud_handoffs"], 3)
            store.close()

    def test_refresh_scheduler_uses_self_observation_history_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}),
                    store=store,
                )
                kernel.handle("What is the weather today?")
                kernel.reflect()
                profile = store.load_profile(LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}))

                report = schedule_inventory_refreshes(
                    store,
                    profile,
                    min_story_models=1,
                    story_limit=1,
                    use_offline_samples=True,
                )
                weather = next(item for item in report.recommendations if item.kind == "refresh_weather_cache")
                weather_job = next(job for job in store.load_jobs(status="queued") if job.kind == "refresh_weather_cache")

                self.assertGreaterEqual(weather.priority_signals["self_observation_points"], 2)
                self.assertEqual(weather.priority_signals["weather_cache_gap_persistence"], 1.0)
                self.assertGreater(weather.priority, 0.82)
                self.assertEqual(
                    weather_job.payload["priority_signals"]["weather_cache_gap_persistence"],
                    1.0,
                )
            finally:
                store.close()

    def test_refresh_scheduler_persists_internet_archive_pagination_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            profile = LocalAssistantProfile(story_models={}, weekly_weather={"today": "warm"})

            schedule_inventory_refreshes(
                store,
                profile,
                min_story_models=3,
                story_limit=4,
                source="internet-archive",
                internet_archive_page_size=125,
                internet_archive_max_pages=4,
                internet_archive_cursor="tail-cursor",
                internet_archive_rate_limit_delay_seconds=0.125,
            )
            story_job = next(job for job in store.load_jobs(status="queued") if job.kind == "import_story_metadata")

            self.assertEqual(story_job.payload["internet_archive_page_size"], 125)
            self.assertEqual(story_job.payload["internet_archive_max_pages"], 4)
            self.assertEqual(story_job.payload["internet_archive_cursor"], "tail-cursor")
            self.assertEqual(story_job.payload["internet_archive_rate_limit_delay_seconds"], 0.125)
            self.assertEqual(story_job.resource_budget["internet_archive_page_size"], 125)
            self.assertEqual(story_job.resource_budget["internet_archive_max_pages"], 4)
            self.assertEqual(story_job.resource_budget["internet_archive_rate_limit_delay_seconds"], 0.125)
            store.close()

    def test_refresh_scheduler_keeps_completed_cycles_and_dedupes_active_followups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            profile = LocalAssistantProfile(story_models={}, weekly_weather={"today": "warm"})

            first = schedule_inventory_refreshes(
                store,
                profile,
                min_story_models=3,
                story_limit=4,
                use_offline_samples=True,
            )
            first_job = next(job for job in store.load_jobs(status="queued") if job.kind == "import_story_metadata")
            started = store.start_next_job(kinds=("import_story_metadata",))
            self.assertIsNotNone(started)
            store.complete_job(first_job.job_id, result={"executed": True, "imported_items": 0, "results": []})

            second = schedule_inventory_refreshes(
                store,
                profile,
                min_story_models=3,
                story_limit=4,
                use_offline_samples=True,
            )
            third = schedule_inventory_refreshes(
                store,
                profile,
                min_story_models=3,
                story_limit=4,
                use_offline_samples=True,
            )
            import_jobs = [job for job in store.load_jobs() if job.kind == "import_story_metadata"]
            queued = [job for job in import_jobs if job.status == "queued"]
            completed = [job for job in import_jobs if job.status == "completed"]

            self.assertEqual(first.queued_jobs, 1)
            self.assertEqual(second.queued_jobs, 1)
            self.assertEqual(third.queued_jobs, 1)
            self.assertEqual(len(completed), 1)
            self.assertEqual(len(queued), 1)
            self.assertEqual(completed[0].job_id, "import_story_metadata:inventory_scheduler_story_model_thin")
            self.assertEqual(queued[0].job_id, "import_story_metadata:inventory_scheduler_story_model_thin:cycle_2")
            self.assertEqual(queued[0].payload["opportunity_id"], "import_story_metadata:inventory_scheduler_story_model_thin")
            self.assertEqual(queued[0].payload["refresh_cycle"], 2)
            self.assertEqual(queued[0].resource_budget["refresh_cycle"], 2)
            store.close()

    def test_cli_schedule_refreshes_then_run_jobs_uses_replayable_importers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            scheduled = _run_cli(
                "schedule-refreshes",
                "--db",
                str(db),
                "--cold-start",
                "--offline-samples",
                "--min-story-models",
                "4",
                "--story-limit",
                "3",
                "--json",
            )
            executed = _run_cli("run-jobs", "--db", str(db), "--cold-start", "--json")
            story = _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--json")

            self.assertEqual(
                {item["kind"] for item in scheduled["recommendations"]},
                {"import_story_metadata", "refresh_weather_cache"},
            )
            self.assertGreaterEqual(executed["completed"], 2)
            self.assertIn("import_story_metadata", {item["kind"] for item in executed["executed"]})
            self.assertEqual(story["route"], "local_answer")
            self.assertTrue(story["synthesis"]["applied"])

    def test_dashboard_reports_importer_health_and_story_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli(
                "schedule-refreshes",
                "--db",
                str(db),
                "--cold-start",
                "--offline-samples",
                "--min-story-models",
                "4",
                "--story-limit",
                "3",
                "--json",
            )
            _run_cli("run-jobs", "--db", str(db), "--cold-start", "--json")
            store = AssistantOSStore(db)
            try:
                dashboard = build_assistant_os_dashboard(store).to_dict()
            finally:
                store.close()

            self.assertEqual(dashboard["jobs"]["importer_health"]["completed_import_jobs"], 1)
            self.assertEqual(dashboard["jobs"]["importer_health"]["imported_items"], 4)
            self.assertEqual(dashboard["jobs"]["importer_health"]["quality_rejected_items"], 0)
            self.assertIn(
                "project_gutenberg_catalog_csv",
                dashboard["jobs"]["importer_health"]["sources"],
            )
            self.assertGreaterEqual(dashboard["inventories"]["story_quality"]["with_quality_scores"], 4)
            self.assertEqual(dashboard["inventories"]["story_quality"]["below_metadata_quality_floor"], 0)

    def test_dashboard_reports_import_pagination_and_rate_limit_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            job_id = store.enqueue_job(
                kind="import_story_metadata",
                payload={"opportunity_id": "import_story_metadata:test"},
                priority=0.9,
                resource_budget={"network": "metadata_only"},
            )
            store.start_next_job(kinds=("import_story_metadata",))
            store.complete_job(
                job_id,
                result={
                    "executed": True,
                    "source": "internet-archive",
                    "imported_items": 2,
                    "results": [
                        {
                            "source": "internet_archive_search_metadata",
                            "selected_count": 2,
                            "rejected_count": 1,
                            "network_used": True,
                            "observability": {
                                "page_count": 2,
                                "fetch_attempts_total": 3,
                                "rate_limit_sleep_count": 1,
                                "rate_limit_delay_total_seconds": 0.25,
                                "max_pages": 3,
                                "next_cursor": "tail",
                                "quality_rejected_count": 1,
                                "duplicate_rejected_count": 1,
                                "byte_budget_exhausted": True,
                            },
                        }
                    ],
                },
            )
            try:
                dashboard = build_assistant_os_dashboard(store).to_dict()
            finally:
                store.close()

            health = dashboard["jobs"]["importer_health"]
            self.assertEqual(health["completed_import_jobs"], 1)
            self.assertEqual(health["pages_fetched"], 2)
            self.assertEqual(health["fetch_attempts_total"], 3)
            self.assertEqual(health["rate_limit_sleep_count"], 1)
            self.assertEqual(health["rate_limit_delay_total_seconds"], 0.25)
            self.assertEqual(health["max_pages_requested"], 3)
            self.assertEqual(health["last_next_cursor"], "tail")
            self.assertEqual(health["quality_rejected_items"], 1)
            self.assertEqual(health["duplicate_rejected_items"], 1)
            self.assertEqual(health["byte_budget_exhausted_results"], 1)

    def test_dashboard_reports_import_quality_trends_across_completed_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            base_job_id = "import_story_metadata:inventory_scheduler_story_model_thin"
            first_job_id = store.enqueue_job(
                kind="import_story_metadata",
                payload={"opportunity_id": base_job_id, "refresh_cycle": 1},
                priority=0.9,
                resource_budget={"network": "metadata_only", "refresh_cycle": 1},
            )
            store.start_next_job(kinds=("import_story_metadata",))
            store.complete_job(
                first_job_id,
                result=_import_result(
                    imported_items=2,
                    selected_count=2,
                    metadata_quality=0.7,
                    pages=1,
                    fetch_attempts=1,
                ),
            )
            second_job_id = store.enqueue_job(
                kind="import_story_metadata",
                payload={"opportunity_id": base_job_id, "refresh_cycle": 2},
                priority=0.9,
                resource_budget={"network": "metadata_only", "refresh_cycle": 2},
                job_id=f"{base_job_id}:cycle_2",
            )
            store.start_next_job(kinds=("import_story_metadata",))
            store.complete_job(
                second_job_id,
                result=_import_result(
                    imported_items=3,
                    selected_count=3,
                    metadata_quality=0.9,
                    pages=2,
                    fetch_attempts=3,
                ),
            )
            try:
                dashboard = build_assistant_os_dashboard(store).to_dict()
            finally:
                store.close()

            trends = dashboard["jobs"]["importer_trends"]
            self.assertEqual(trends["cycles"], 2)
            self.assertEqual(trends["completed_cycles"], 2)
            self.assertEqual(trends["imported_items_total"], 5)
            self.assertEqual(trends["selected_items_total"], 5)
            self.assertEqual(trends["pages_fetched_total"], 3)
            self.assertEqual(trends["fetch_attempts_total"], 4)
            self.assertAlmostEqual(trends["avg_metadata_quality"], 0.8)
            self.assertAlmostEqual(trends["quality_delta"], 0.2)
            self.assertEqual(trends["latest_completed_cycle"]["refresh_cycle"], 2)
            self.assertEqual(trends["recent_cycles"][-1]["job_id"], f"{base_job_id}:cycle_2")

    def test_cli_run_jobs_executes_queued_story_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            AssistantOSStore(db).close()
            for _ in range(3):
                _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--cold-start", "--no-auto-execute", "--json")
            queued = _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--cold-start", "--no-auto-execute", "--json")
            executed = _run_cli("run-jobs", "--db", str(db), "--cold-start", "--json")
            after = _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--json")

            self.assertIn("build_story_inventory", queued["opportunities"])
            self.assertGreaterEqual(executed["completed"], 1)
            self.assertEqual(after["route"], "local_answer")
            self.assertIn("story_models.", after["evidence_keys"][0])

    def test_cli_resource_report_is_stdlib_and_pi_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_cli(
                "resource-report",
                "--db",
                str(Path(tmp) / "assistant.sqlite"),
                "--lifecycle-db",
                str(Path(tmp) / "lifecycle.sqlite"),
                "--reset",
                "--json",
            )

            self.assertEqual(report["dependency_class"], "stdlib_only")
            self.assertTrue(report["pi_constraints"]["sqlite_indexes"])
            self.assertGreater(report["db_bytes"], 0)
            self.assertGreater(report["lifecycle_db_bytes"], 0)
            self.assertEqual(report["lifecycle"]["counts"]["membrane_decisions"], 17)
            self.assertLess(report["lifecycle_ms"], 5000)


class AssistantOSAutoimmuneFailureTests(unittest.TestCase):
    def test_private_facts_are_excluded_from_cloud_membrane(self) -> None:
        decision = AssistantDecision(
            utterance="Tell the big model my favorite color and ask for a poem.",
            intent="unknown",
            route="cloud_handoff",
            answer="Hand off to a larger model.",
            evidence_keys=("facts.favorite_color", "contacts.mom"),
            cloud_needed=True,
            privacy_exposure=True,
            confidence=0.4,
            reason="cloud_first_general_language",
        )

        membrane = MembranePolicy().evaluate(decision)

        self.assertFalse(membrane.allowed)
        self.assertEqual(membrane.boundary_crossed, "blocked_cloud")
        self.assertEqual(membrane.personal_facts_included, ())
        self.assertIn("facts.favorite_color", membrane.personal_facts_excluded)
        self.assertIn("contacts.mom", membrane.personal_facts_excluded)

    def test_user_approved_cloud_eligible_fact_can_cross_membrane(self) -> None:
        decision = AssistantDecision(
            utterance="Send my public profile to the cloud.",
            intent="personal_memory",
            route="cloud_handoff",
            answer="Hand off to a larger model.",
            evidence_keys=("facts.public_profile",),
            cloud_needed=True,
            privacy_exposure=True,
            confidence=0.68,
            reason="memory_cloud_request",
        )

        membrane = MembranePolicy().evaluate(
            decision,
            fact_privacy={
                "facts.public_profile": {
                    "consent": True,
                    "local_only": False,
                    "cloud_eligible": True,
                    "scope": "shareable_profile",
                }
            },
        )

        self.assertTrue(membrane.allowed)
        self.assertEqual(membrane.boundary_crossed, "cloud")
        self.assertEqual(membrane.personal_facts_included, ("facts.public_profile",))
        self.assertEqual(membrane.personal_facts_excluded, ())
        self.assertEqual(membrane.reason, "cloud_allowed_with_user_eligible_facts")

    def test_kernel_rejects_private_cloud_request_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            kernel = AssistantOSKernel(store=store)

            decision = kernel.handle("Send my favorite color and mom contact to the cloud.")
            membrane = store.connection.execute(
                "SELECT allowed, boundary_crossed, reason FROM membrane_decisions"
            ).fetchone()

            self.assertEqual(decision.route, "reject")
            self.assertEqual(decision.reason, "blocked_private_facts_to_cloud")
            self.assertEqual(membrane["allowed"], 0)
            self.assertEqual(membrane["boundary_crossed"], "blocked")
            store.close()

    def test_kernel_allows_explicitly_shareable_profile_fact_to_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            store.upsert_user_fact(
                "facts.public_profile",
                "Public science fair helper profile.",
                source="user_approved",
                confidence=0.91,
                consent=True,
                local_only=False,
                cloud_eligible=True,
                scope="shareable_profile",
            )
            kernel = AssistantOSKernel(store=store)

            decision = kernel.handle("Send my public profile to the cloud.")
            membrane = store.connection.execute(
                """
                SELECT allowed, boundary_crossed, personal_facts_included_json,
                       personal_facts_excluded_json, reason
                FROM membrane_decisions
                """
            ).fetchone()

            self.assertEqual(decision.route, "cloud_handoff")
            self.assertEqual(decision.reason, "private_memory_cloud_request")
            self.assertEqual(membrane["allowed"], 1)
            self.assertEqual(membrane["boundary_crossed"], "cloud")
            self.assertIn("facts.public_profile", membrane["personal_facts_included_json"])
            self.assertEqual(membrane["personal_facts_excluded_json"], "[]")
            self.assertEqual(membrane["reason"], "cloud_allowed_with_user_eligible_facts")
            store.close()

    def test_empty_personal_memory_does_not_invent_user_fact(self) -> None:
        decision = OnDeviceAssistantRouter(LocalAssistantProfile(facts={})).handle(
            "Tell me something about myself."
        )

        self.assertEqual(decision.route, "clarify")
        self.assertEqual(decision.reason, "personal_memory_empty")
        self.assertNotIn("favorite", decision.answer.lower())

    def test_device_action_stays_pending_until_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            kernel = AssistantOSKernel(store=store)

            kernel.handle("I need to talk to someone.")
            row = store.connection.execute(
                "SELECT confirmation_state, executed FROM pending_actions"
            ).fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(row["confirmation_state"], "pending")
            self.assertEqual(row["executed"], 0)
            store.close()


def _run_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(CLI), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _import_result(
    *,
    imported_items: int,
    selected_count: int,
    metadata_quality: float,
    pages: int,
    fetch_attempts: int,
) -> dict:
    return {
        "executed": True,
        "source": "internet-archive",
        "imported_items": imported_items,
        "results": [
            {
                "source": "internet_archive_search_metadata",
                "selected_count": selected_count,
                "rejected_count": 0,
                "network_used": True,
                "observability": {
                    "selected_avg_metadata_quality": metadata_quality,
                    "page_count": pages,
                    "fetch_attempts_total": fetch_attempts,
                    "rate_limit_sleep_count": max(0, pages - 1),
                    "quality_rejected_count": 0,
                    "duplicate_rejected_count": 0,
                },
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
