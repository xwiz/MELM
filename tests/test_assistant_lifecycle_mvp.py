import unittest

from melm.appliance import (
    run_household_week_lifecycle_probe,
    realistic_lifecycle_steps,
    run_multi_profile_lifecycle_suite,
    run_realistic_assistant_lifecycle_probe,
)


class AssistantLifecycleMvpTests(unittest.TestCase):
    def test_realistic_lifecycle_turns_repeated_cloud_story_into_local_inventory(self) -> None:
        report = run_realistic_assistant_lifecycle_probe()

        self.assertEqual(report.steps, 17)
        self.assertEqual(report.story_route_before_inventory, "cloud_handoff")
        self.assertEqual(report.story_route_after_inventory, "local_answer")
        self.assertEqual(report.cloud_story_handoffs_before_inventory, 3)
        self.assertEqual(report.story_inventory_count, 3)
        self.assertIn("build_story_inventory", report.opportunities_created)
        self.assertIn("build_story_inventory", report.jobs_executed)
        self.assertEqual(report.cloud_handoffs, 3)

    def test_realistic_lifecycle_covers_cache_policy_action_and_offline_behavior(self) -> None:
        report = run_realistic_assistant_lifecycle_probe()

        self.assertEqual(report.external_fetches, 1)
        self.assertEqual(report.weather_cache_days, 3)
        self.assertEqual(report.contact_count, 1)
        self.assertEqual(report.confirmations_required, 1)
        self.assertEqual(report.actions_executed, 1)
        self.assertEqual(report.blocked_offline, 1)
        self.assertGreaterEqual(report.local_resolution_rate, 0.70)
        self.assertEqual(report.route_counts["external_fetch"], 1)
        self.assertEqual(report.route_counts["cloud_handoff"], 3)
        self.assertEqual(report.route_counts["clarify"], 1)

    def test_realistic_lifecycle_order_exercises_end_to_end_state_changes(self) -> None:
        steps = realistic_lifecycle_steps()
        report = run_realistic_assistant_lifecycle_probe()

        self.assertEqual(len(steps), report.steps)
        self.assertEqual(report.results[0].reason, "profile_update")
        self.assertEqual(report.results[4].reason, "weather_cache_miss")
        self.assertEqual(report.results[5].reason, "school_clothing_weather_policy")
        self.assertEqual(report.results[9].executed_jobs, ("build_story_inventory",))
        self.assertEqual(report.results[10].reason, "local_story_inventory")
        self.assertEqual(report.results[11].reason, "personal_memory_summary")
        self.assertTrue(report.results[13].confirmation_required)
        self.assertTrue(report.results[14].action_executed)
        self.assertEqual(report.results[-1].reason, "cloud_unavailable")
        self.assertTrue(report.results[-1].blocked_offline)

    def test_multi_profile_lifecycle_suite_covers_longer_planner_paths(self) -> None:
        suite = run_multi_profile_lifecycle_suite()
        payload = suite.to_dict()

        self.assertEqual(suite.scenario_count, 3)
        self.assertEqual(suite.steps, 34)
        self.assertGreaterEqual(suite.local_resolution_rate, 0.55)
        self.assertEqual(payload["safety_flags"]["cloud_private_inclusions"], 0)
        self.assertEqual(payload["safety_flags"]["unconfirmed_executed_actions"], 0)
        self.assertEqual(payload["safety_flags"]["fake_latest_news_local_answers"], 0)
        self.assertEqual(payload["safety_flags"]["dangling_memory_links"], 0)
        self.assertIn("build_story_inventory", payload["opportunities_by_kind"])
        self.assertIn("build_media_index", payload["opportunities_by_kind"])
        self.assertIn("ask_routine_memory", payload["opportunities_by_kind"])
        self.assertIn("ask_household_memory", payload["opportunities_by_kind"])
        self.assertIn("request_trusted_contact", payload["opportunities_by_kind"])
        self.assertGreaterEqual(payload["blocked_offline"], 2)
        self.assertGreaterEqual(payload["actions_executed"], 2)

    def test_household_week_lifecycle_exercises_full_architecture_loop(self) -> None:
        report = run_household_week_lifecycle_probe()
        payload = report.to_dict()

        self.assertEqual(report.steps, 37)
        self.assertGreaterEqual(report.local_resolution_rate, 0.60)
        self.assertEqual(payload["safety_flags"]["cloud_private_inclusions"], 0)
        self.assertEqual(payload["safety_flags"]["unconfirmed_executed_actions"], 0)
        self.assertEqual(payload["safety_flags"]["dangling_memory_links"], 0)
        self.assertEqual(payload["safety_flags"]["low_quality_applied_synthesis"], 0)
        self.assertTrue(all(payload["architecture_checks"].values()))
        self.assertGreaterEqual(payload["digest"]["session_count"], 6)
        self.assertGreaterEqual(payload["digest"]["event_count"], 20)
        self.assertTrue(payload["digest"]["quality"]["passed"])
        self.assertGreaterEqual(payload["digest"]["quality"]["score"], payload["digest"]["quality"]["floor"])
        self.assertEqual(payload["digest"]["quality"]["warnings"], [])
        digest_threads = {item["thread"] for item in payload["digest"]["threads"]}
        self.assertTrue(
            {
                "story_inventory",
                "weather_cache",
                "household_memory",
                "routine_memory",
                "media_playback",
                "trusted_contact",
                "boundary_control",
            }.issubset(digest_threads)
        )
        self.assertIn(
            "story requests moved from cloud handoff to local story inventory",
            payload["digest"]["capability_transitions"],
        )
        self.assertIn(
            "weather moved from cache miss to cached local forecast",
            payload["digest"]["capability_transitions"],
        )
        self.assertIn("private conversation and user memory stayed local-only", payload["digest"]["active_limits"])
        self.assertIn("main threads:", payload["digest"]["summary"])
        self.assertNotIn("1 sessions", payload["digest"]["summary"])
        self.assertIn("build_story_inventory", payload["opportunities_by_kind"])
        self.assertIn("build_media_index", payload["opportunities_by_kind"])
        self.assertIn("request_trusted_contact", payload["opportunities_by_kind"])
        self.assertIn("ask_routine_memory", payload["opportunities_by_kind"])
        self.assertIn("ask_household_memory", payload["opportunities_by_kind"])


if __name__ == "__main__":
    unittest.main()
