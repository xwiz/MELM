import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from melm.appliance import (
    AssistantOSKernel,
    AssistantOSStore,
    LocalAssistantProfile,
    SYNTHESIS_QUALITY_FLOOR,
    initialize_assistant_os_database,
)


CLI = Path("scripts/local_assistant_os_cli.py")
SEED = Path("benchmarks/local_assistant_os_seed.json")


class AssistantSynthesisMvpTests(unittest.TestCase):
    def test_seeded_local_story_synthesis_cites_inventory_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = initialize_assistant_os_database(Path(tmp) / "assistant.sqlite", seed_path=SEED)
            try:
                kernel = AssistantOSKernel(store=store)
                decision = kernel.handle("Tell me a story.")
                synthesis = kernel.last_synthesis
            finally:
                store.close()

            self.assertEqual(decision.route, "local_answer")
            self.assertIsNotNone(synthesis)
            self.assertTrue(synthesis.applied)
            self.assertEqual(synthesis.route, "local_answer")
            self.assertIn("story_models.public_domain_folktale_lagos_age7", synthesis.citations)
            self.assertIn("profile.location", synthesis.citations)
            sources = {item.source for item in synthesis.evidence}
            self.assertIn("local_seed_public_domain_story_frame", sources)
            self.assertEqual(synthesis.boundary_crossed, "none")
            self.assertGreaterEqual(synthesis.admitted_evidence_count, 2)
            self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)
            self.assertEqual(synthesis.quality["citation_coverage"], 1.0)
            self.assertEqual(synthesis.quality["warnings"], [])

    def test_health_synthesis_cites_goals_and_local_policy(self) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                health_goals=("sleep earlier", "walk after school"),
                story_models={},
                weekly_weather={},
                contacts={},
            )
        )

        decision = kernel.handle("What do you think I should do to improve my health?")
        synthesis = kernel.last_synthesis

        self.assertEqual(decision.reason, "bounded_general_health_guidance")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("health_goals.0", synthesis.citations)
        self.assertIn("health_goals.1", synthesis.citations)
        self.assertIn("local_health_safety_policy", synthesis.citations)
        self.assertEqual(
            {item.kind for item in synthesis.evidence},
            {"health_goal", "policy"},
        )
        self.assertIn("one small goal", synthesis.answer)
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)
        self.assertEqual(synthesis.quality["local_privacy_discipline"], 1.0)

    def test_personal_memory_summary_cites_multiple_local_facts(self) -> None:
        kernel = AssistantOSKernel()

        decision = kernel.handle("What do you know about me?")
        synthesis = kernel.last_synthesis

        self.assertEqual(decision.route, "local_answer")
        self.assertEqual(decision.reason, "personal_memory_summary")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("profile.age", synthesis.citations)
        self.assertIn("profile.location", synthesis.citations)
        self.assertIn("facts.favorite_color", synthesis.citations)
        self.assertIn("preferences.music", synthesis.citations)
        self.assertGreaterEqual(synthesis.admitted_evidence_count, 5)
        self.assertIn("local memory", synthesis.answer)
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)
        self.assertEqual(synthesis.quality["citation_coverage"], 1.0)

    def test_meal_synthesis_mentions_local_inventory_without_generic_warning(self) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                food_inventory=("oatmeal", "eggs", "fruit"),
                weekly_weather={},
                story_models={},
                contacts={},
            )
        )

        decision = kernel.handle("What should I have for lunch?")
        synthesis = kernel.last_synthesis

        self.assertEqual(decision.reason, "memory_plus_weather_cache")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("food_inventory.oatmeal", synthesis.citations)
        self.assertIn("food_inventory.eggs", synthesis.citations)
        self.assertIn("local food inventory", synthesis.answer)
        self.assertIn("oatmeal", synthesis.answer)
        self.assertIn("eggs", synthesis.answer)
        self.assertEqual(synthesis.quality["warnings"], [])

    def test_contact_cancel_synthesis_clears_call_without_generic_warning(self) -> None:
        store = AssistantOSStore(":memory:")
        try:
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(
                    contacts={"mom": "+234-000-MOM"},
                    story_models={},
                    weekly_weather={},
                    media_library=(),
                ),
                store=store,
            )
            kernel.handle("Call mom.")
            cancel = kernel.handle("Cancel that.")
            synthesis = kernel.last_synthesis
        finally:
            store.close()

        self.assertEqual(cancel.reason, "cancelled_pending_action")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("contacts.mom", synthesis.citations)
        self.assertIn("trusted-contact action is cleared", synthesis.answer)
        self.assertIn("no call will run", synthesis.answer)
        self.assertEqual(synthesis.quality["warnings"], [])

    def test_trusted_contact_setup_synthesis_is_specific_without_generic_warning(self) -> None:
        store = AssistantOSStore(":memory:")
        try:
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(
                    contacts={},
                    story_models={},
                    weekly_weather={},
                    media_library=(),
                ),
                store=store,
            )
            decision = kernel.handle("Ada is my trusted contact at +234-000-ADA.")
            synthesis = kernel.last_synthesis
        finally:
            store.close()

        self.assertEqual(decision.reason, "consented_trusted_contact_stored")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("trusted contact", synthesis.answer)
        self.assertIn("contacts.ada", synthesis.citations)
        self.assertEqual(synthesis.quality["warnings"], [])
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)

    def test_assistant_identity_synthesis_cites_self_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)
                decision = kernel.handle("Who are you?")
                synthesis = kernel.last_synthesis
                row = store.connection.execute(
                    "SELECT intent, route, reason FROM events WHERE event_id=?",
                    (kernel.events[-1].event_id,),
                ).fetchone()
            finally:
                store.close()

        self.assertEqual(decision.intent, "assistant_identity")
        self.assertEqual(decision.route, "local_answer")
        self.assertEqual(decision.reason, "self_model_identity")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("MELM Local Assistant OS", decision.answer)
        self.assertIn("self_model.name", synthesis.citations)
        self.assertIn("self_model.purpose", synthesis.citations)
        self.assertIn("self_model.local_capabilities", synthesis.citations)
        self.assertEqual(synthesis.quality["warnings"], [])
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)
        self.assertIsNotNone(row)
        self.assertEqual(row["intent"], "assistant_identity")

    def test_assistant_status_synthesis_uses_local_ledger_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)
                kernel.handle("Who are you?")
                kernel.handle("Tell me a story.")
                status = kernel.handle("What have you done so far?")
                synthesis = kernel.last_synthesis
            finally:
                store.close()

        self.assertEqual(status.intent, "assistant_status")
        self.assertEqual(status.route, "local_answer")
        self.assertEqual(status.reason, "self_status_ledger_summary")
        self.assertIn("local ledger has 2 event(s)", status.answer)
        self.assertIn("safety dashboard is clean", status.answer)
        self.assertIn("persisted self-observation", status.answer)
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("self_status.counts", synthesis.citations)
        self.assertIn("self_status.routes", synthesis.citations)
        self.assertIn("self_status.safety_flags", synthesis.citations)
        self.assertIn("self_status.self_observation", synthesis.citations)
        self.assertIn("self_status.next_steps", synthesis.citations)
        self.assertEqual(synthesis.quality["warnings"], [])
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)

    def test_consent_revocation_synthesis_does_not_echo_revoked_fact_value(self) -> None:
        kernel = AssistantOSKernel()

        decision = kernel.handle("Forget my favorite color.")
        synthesis = kernel.last_synthesis

        self.assertEqual(decision.reason, "consent_revoked_user_fact")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertNotIn("green", synthesis.answer.lower())
        self.assertIn("active local memory", decision.answer)
        self.assertEqual(synthesis.citations, ("local_privacy_policy.consent_revocation",))
        self.assertEqual({item.kind for item in synthesis.evidence}, {"policy"})
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)
        self.assertEqual(synthesis.quality["warnings"], [])

    def test_safety_weather_and_cancel_synthesis_are_specific_not_generic(self) -> None:
        store = AssistantOSStore(":memory:")
        try:
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(
                    location="Austin",
                    weekly_weather={"today": "hot and dry"},
                    media_library=("calm piano",),
                    preferences={"music": "calm piano"},
                    story_models={},
                    contacts={},
                ),
                store=store,
            )
            safety = kernel.handle("Should I walk outside naked?")
            safety_synthesis = kernel.last_synthesis
            weather = kernel.handle("What is the weather today?")
            weather_synthesis = kernel.last_synthesis
            kernel.handle("Play some music.")
            cancel = kernel.handle("Cancel that.")
            cancel_synthesis = kernel.last_synthesis
        finally:
            store.close()

        self.assertEqual(safety.reason, "local_common_sense_policy")
        self.assertIsNotNone(safety_synthesis)
        self.assertEqual(safety_synthesis.quality["warnings"], [])
        self.assertIn("proper clothes", safety.answer)
        self.assertEqual(weather.route, "cached_tool")
        self.assertIsNotNone(weather_synthesis)
        self.assertEqual(weather_synthesis.quality["warnings"], [])
        self.assertIn("cached local forecast", weather.answer)
        self.assertEqual(cancel.reason, "cancelled_pending_action")
        self.assertIsNotNone(cancel_synthesis)
        self.assertEqual(cancel_synthesis.quality["warnings"], [])
        self.assertIn("no pending media action", cancel.answer)

    def test_urgent_health_synthesis_is_specific_and_still_bounded(self) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                story_models={},
                weekly_weather={},
                contacts={},
            )
        )

        decision = kernel.handle("I swallowed poison.")
        synthesis = kernel.last_synthesis

        self.assertEqual(decision.reason, "urgent_health_safety_escalation")
        self.assertIsNotNone(synthesis)
        self.assertEqual(synthesis.quality["warnings"], [])
        self.assertIn("poison", decision.answer.lower())
        self.assertIn("cannot diagnose", decision.answer.lower())
        self.assertIn("emergency", decision.answer.lower())

    def test_recent_session_synthesis_extracts_transitions_and_open_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                cold = LocalAssistantProfile(
                    story_models={},
                    weekly_weather={},
                    contacts={"mom": "+234-000-MOM"},
                    media_library=(),
                )
                kernel = AssistantOSKernel(profile=cold, store=store)
                kernel.handle("Tell me a story.")
                store.start_new_session()
                kernel.handle("What is the weather today?")
                store.start_new_session()
                kernel.handle("I need to talk to someone.")
                store.start_new_session()
                recall_kernel = AssistantOSKernel(profile=cold, store=store)
                recall = recall_kernel.handle("Summarize our recent sessions.")
                synthesis = recall_kernel.last_synthesis
            finally:
                store.close()

        self.assertEqual(recall.reason, "autobiographical_session_summary")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.applied)
        self.assertIn("session 1 (session_1)", synthesis.answer)
        self.assertIn("Open local gaps:", synthesis.answer)
        self.assertIn("story inventory was missing", synthesis.answer)
        self.assertIn("weather cache was missing", synthesis.answer)
        self.assertIn("Action state:", synthesis.answer)
        self.assertIn("device action was prepared behind confirmation", synthesis.answer)
        self.assertTrue(all(key.startswith("events.") for key in synthesis.citations))
        self.assertEqual(synthesis.quality["warnings"], [])
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)

    def test_private_cloud_block_refuses_local_synthesis(self) -> None:
        kernel = AssistantOSKernel()

        decision = kernel.handle("Send my favorite color and mom contact to the cloud.")
        synthesis = kernel.last_synthesis

        self.assertEqual(decision.route, "reject")
        self.assertEqual(decision.reason, "blocked_private_facts_to_cloud")
        self.assertIsNotNone(synthesis)
        self.assertTrue(synthesis.refused)
        self.assertFalse(synthesis.applied)
        self.assertEqual(synthesis.reason, "membrane_blocked")
        self.assertEqual(synthesis.citations, ())
        self.assertTrue(synthesis.quality["expected_refusal"])
        self.assertGreaterEqual(synthesis.quality["score"], SYNTHESIS_QUALITY_FLOOR)

    def test_imported_story_synthesis_uses_metadata_shape_not_static_catalog_cheat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            _run_cli(
                "import-stories",
                "--db",
                str(db),
                "--cold-start",
                "--source",
                "both",
                "--gutenberg-csv",
                "benchmarks/sample_gutenberg_catalog.csv",
                "--internet-archive-json",
                "benchmarks/sample_internet_archive_search.json",
                "--limit",
                "3",
                "--json",
            )
            payload = _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--json")
            store = AssistantOSStore(db)
            try:
                story_payloads = store.load_inventory("story_model")
            finally:
                store.close()

            self.assertEqual(payload["route"], "local_answer")
            self.assertTrue(payload["synthesis"]["applied"])
            self.assertIn("I picked", payload["answer"])
            self.assertIn("The Rain Map Bedtime Story", payload["answer"])
            self.assertNotIn("Using the public-domain catalog entry", payload["answer"])
            first_story = story_payloads["ia_rainmapbedtimestory00test"]
            self.assertIn("topics", first_story)
            self.assertIn("cultures", first_story)
            self.assertIn("narrative_frame", first_story)
            self.assertNotIn("template", first_story)
            self.assertGreater(first_story["quality_score"], 0)
            self.assertGreater(first_story["metadata_quality"], 0)
            self.assertGreaterEqual(payload["synthesis"]["quality"]["score"], SYNTHESIS_QUALITY_FLOOR)

    def test_cli_ask_returns_machine_readable_synthesis_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            payload = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Tell me a story.",
                "--json",
            )

            self.assertEqual(payload["route"], "local_answer")
            self.assertTrue(payload["synthesis"]["applied"])
            self.assertFalse(payload["synthesis"]["refused"])
            self.assertIn(
                "story_models.public_domain_folktale_lagos_age7",
                payload["synthesis"]["citations"],
            )
            self.assertGreaterEqual(payload["synthesis"]["admitted_evidence_count"], 2)
            self.assertGreaterEqual(payload["synthesis"]["quality"]["score"], SYNTHESIS_QUALITY_FLOOR)
            self.assertEqual(payload["counts"]["synthesis_traces"], 1)


def _run_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(CLI), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


if __name__ == "__main__":
    unittest.main()
