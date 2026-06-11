import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace

from melm.appliance import (
    load_open_trace_scenarios,
    load_transcript_replay_scenarios,
    parse_assistant_debug_frame,
    run_open_trace_suite,
    run_transcript_replay_suite,
)
from melm.appliance.assistant_open_traces import (
    _primary_uol_debug_maps_are_not_secondary_phrase_routes,
)


class AssistantOpenTraceMvpTests(unittest.TestCase):
    def test_open_trace_suite_exercises_real_kernel_growth_and_debug_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_open_trace_suite(db_dir=tmp, reset=True)
            payload = report.to_dict()

            self.assertTrue(report.passed)
            self.assertEqual(report.scenario_count, 2)
            self.assertEqual(report.turns, 29)
            self.assertGreaterEqual(report.local_resolution_rate, 0.65)
            self.assertEqual(payload["route_counts"]["cloud_handoff"], 3)
            self.assertEqual(payload["route_counts"]["external_fetch"], 1)
            self.assertEqual(payload["route_counts"]["reject"], 1)
            self.assertEqual(payload["safety_totals"]["cloud_private_inclusions"], 0)
            self.assertEqual(payload["safety_totals"]["unconfirmed_executed_actions"], 0)
            self.assertEqual(payload["safety_totals"]["fake_latest_news_local_answers"], 0)

            child = next(item for item in report.scenarios if item.name == "child_local_first_capability_growth")
            child_reasons = [turn.reason for turn in child.turns]
            self.assertLess(child_reasons.index("weather_cache_miss"), child_reasons.index("weather_cache_hit"))
            self.assertLess(child_reasons.index("missing_story_model"), child_reasons.index("local_story_inventory"))
            self.assertEqual(child.inventory["weather_days"], 8)
            self.assertEqual(child.inventory["story_models"], 3)
            self.assertEqual(child.self_observation["history_summary"]["weather_cache_became_ready"], True)
            self.assertGreaterEqual(child.self_observation["history_summary"]["points"], 18)

            story_pressure = next(
                sample
                for sample in child.priority_signal_samples
                if sample["kind"] == "import_story_metadata"
            )
            self.assertEqual(story_pressure["signals"]["story_inventory_gap_persistence"], 1.0)
            self.assertEqual(story_pressure["signals"]["recent_story_cloud_handoffs"], 3)

            identity = next(turn for turn in child.turns if turn.label == "identity_challenge")
            self.assertEqual(identity.debug_parse["uol"]["object"], "self_model")
            self.assertEqual(
                [stage["stage"] for stage in identity.debug_parse["mapping"]],
                ["basic_nlp", "uol_parse", "chat_frame"],
            )

            adult = next(item for item in report.scenarios if item.name == "adult_household_memory_and_offline_boundaries")
            meal = next(turn for turn in adult.turns if turn.label == "meal")
            self.assertNotIn("You could eat You could eat", meal.answer)
            self.assertEqual(next(turn for turn in adult.turns if turn.label == "offline_latest_news").reason, "cloud_unavailable")

    def test_parse_debug_maps_memory_domains_to_specific_uol_objects(self) -> None:
        household = parse_assistant_debug_frame("What do you know about this household?").to_dict()
        routine = parse_assistant_debug_frame("What is my morning routine?").to_dict()
        sessions = parse_assistant_debug_frame("Summarize our recent sessions.").to_dict()

        self.assertEqual(household["uol"]["object"], "household_memory")
        self.assertEqual(routine["uol"]["object"], "routine_memory")
        self.assertEqual(sessions["chat_frame"]["intent"], "autobiographical_memory")
        self.assertEqual(sessions["uol"]["object"], "conversation_events")
        self.assertEqual(sessions["nlp"]["bounded_intent"], "autobiographical_memory")
        self.assertEqual(
            [stage["stage"] for stage in sessions["mapping"]],
            ["basic_nlp", "uol_parse", "chat_frame"],
        )

    def test_open_trace_fixture_is_transcript_like_not_static_answers(self) -> None:
        scenarios = load_open_trace_scenarios()
        turns = [turn for scenario in scenarios for turn in scenario.turns]

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(len(turns), 29)
        self.assertTrue(any(turn.schedule_refreshes and turn.execute_jobs for turn in turns))
        self.assertTrue(any(turn.execute_opportunities for turn in turns))
        self.assertTrue(any(not turn.network_available for turn in turns))

    def test_transcript_replay_gate_scores_messy_chat_without_static_answers(self) -> None:
        scenarios = load_transcript_replay_scenarios()
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(len(scenarios[0].turns), 25)

        with tempfile.TemporaryDirectory() as tmp:
            report = run_transcript_replay_suite(db_dir=tmp, reset=True)
            payload = report.to_dict()

            self.assertTrue(report.passed)
            self.assertEqual(payload["schema"], "melm.local_assistant_transcript_replay_report.v1")
            self.assertEqual(payload["source_type"], "authored_transcript_fixture")
            self.assertTrue(payload["fixture_checks"]["no_static_answer_or_route_expectations"])
            self.assertTrue(payload["fixture_checks"]["memory_digest_quality_passed"])
            self.assertGreaterEqual(payload["local_resolution_rate"], 0.62)
            self.assertEqual(payload["reason_counts"]["profile_update"], 2)
            self.assertGreaterEqual(payload["complexity"]["turns_scored"], 25)
            self.assertGreater(payload["complexity"]["unknown_tokens_total"], 0)
            self.assertEqual(payload["debug_mapping"]["stages"], ["basic_nlp", "uol_parse", "chat_frame"])
            baseline = payload["baseline_comparison"]
            self.assertTrue(baseline["passed"])
            self.assertEqual(baseline["schema"], "melm.local_assistant_transcript_baseline_comparison.v1")
            self.assertEqual(baseline["current"]["local_or_device_resolved"], 17)
            self.assertEqual(baseline["best_baseline"]["strategy"], "local_state_router_no_lifecycle")
            self.assertEqual(baseline["best_baseline"]["local_or_device_resolved"], 7)
            self.assertEqual(baseline["wins"]["local_resolution_rate_gain_vs_best_baseline"], 0.4)
            self.assertEqual(baseline["wins"]["cloud_handoff_reduction_vs_best_baseline"], 7)
            self.assertEqual(baseline["wins"]["capability_advantages"]["profile_updates_vs_best_baseline"], 2)
            self.assertEqual(baseline["wins"]["capability_advantages"]["private_cloud_blocks_vs_best_baseline"], 1)
            self.assertTrue(all(baseline["checks"].values()))

            scenario = report.open_trace_report.scenarios[0]
            age = next(turn for turn in scenario.turns if turn.label == "age_fact")
            digest = next(turn for turn in scenario.turns if turn.label == "long_horizon_digest")
            self.assertEqual(age.intent, "personal_memory")
            self.assertEqual(age.route, "local_answer")
            self.assertEqual(age.reason, "profile_update")
            self.assertEqual(age.debug_parse["uol"]["object"], "user_profile")
            self.assertTrue(scenario.memory_digest["quality_passed"])
            self.assertEqual(digest.reason, "autobiographical_memory_digest")

    def test_primary_route_gate_uses_frame_contract_not_debug_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_transcript_replay_suite(db_dir=tmp, reset=True)

        turn = next(
            turn
            for scenario in report.open_trace_report.scenarios
            for turn in scenario.turns
            if turn.debug_parse.get("nlp", {}).get("primary_domain_evidence", {}).get("frame_id")
        )
        renamed_debug = deepcopy(turn.debug_parse)
        renamed_debug["nlp"]["primary_domain_evidence"]["source"] = "renamed_debug_label"
        renamed = replace(turn, debug_parse=renamed_debug)
        self.assertTrue(_primary_uol_debug_maps_are_not_secondary_phrase_routes([renamed]))

        unowned_debug = deepcopy(renamed_debug)
        unowned_debug["nlp"]["primary_domain_evidence"]["frame_registry"] = ""
        unowned = replace(turn, debug_parse=unowned_debug)
        self.assertFalse(_primary_uol_debug_maps_are_not_secondary_phrase_routes([unowned]))


if __name__ == "__main__":
    unittest.main()
