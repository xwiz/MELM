import json
import tempfile
from pathlib import Path
import unittest

from melm.appliance import (
    AssistantOSKernel,
    AssistantOSStore,
    LocalAssistantProfile,
    initialize_assistant_os_database,
)


SEED = Path("benchmarks/local_assistant_os_seed.json")


class AssistantOSStoreMvpTests(unittest.TestCase):
    def test_seed_database_loads_profile_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = initialize_assistant_os_database(db, seed_path=SEED)
            profile = store.load_profile(LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}))

            self.assertEqual(profile.user_name, "Maya")
            self.assertEqual(profile.age, 7)
            self.assertEqual(profile.location, "Lagos")
            self.assertIn("public_domain_folktale_lagos_age7", profile.story_models)
            self.assertEqual(profile.weekly_weather["today"], "warm with afternoon rain")
            self.assertEqual(profile.contacts["mom"], "+234-000-MOM")
            self.assertGreaterEqual(store.count("user_facts"), 6)
            self.assertGreaterEqual(store.count("inventories"), 8)
            privacy = store.load_user_fact_privacy_index()
            self.assertIn("facts.favorite_color", privacy)
            self.assertTrue(privacy["facts.favorite_color"]["local_only"])
            self.assertFalse(privacy["facts.favorite_color"]["cloud_eligible"])
            self.assertEqual(privacy["facts.favorite_color"]["scope"], "private_local")
            store.close()

    def test_story_inventory_prefers_narrative_frame_but_loads_legacy_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            store.upsert_inventory(
                "story_model",
                "legacy_story",
                {"title": "Legacy Story", "template": "{name} noticed rain in {location}."},
                source="legacy_fixture",
                license="public_domain_story_frame",
                tags=("story",),
            )
            store.upsert_inventory(
                "story_model",
                "current_story",
                {"title": "Current Story", "narrative_frame": "{name} asked a careful question in {location}."},
                source="current_fixture",
                license="public_domain_story_frame",
                tags=("story",),
            )
            profile = store.load_profile(LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}))

            self.assertEqual(profile.story_models["legacy_story"], "{name} noticed rain in {location}.")
            self.assertEqual(profile.story_models["current_story"], "{name} asked a careful question in {location}.")
            store.close()

    def test_user_fact_policy_metadata_can_mark_explicit_cloud_eligible_facts(self) -> None:
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
            store.upsert_user_fact(
                "facts.local_note",
                "Private note.",
                source="user_profile",
                confidence=0.9,
                consent=True,
                local_only=True,
                cloud_eligible=True,
                scope="shareable_profile",
            )
            privacy = store.load_user_fact_privacy_index()

            self.assertFalse(privacy["facts.public_profile"]["local_only"])
            self.assertTrue(privacy["facts.public_profile"]["cloud_eligible"])
            self.assertEqual(privacy["facts.public_profile"]["scope"], "shareable_profile")
            self.assertTrue(privacy["facts.local_note"]["local_only"])
            self.assertFalse(privacy["facts.local_note"]["cloud_eligible"])
            self.assertEqual(privacy["facts.local_note"]["scope"], "private_local")
            store.close()

    def test_profile_sync_preserves_existing_fact_policy_metadata(self) -> None:
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
            store.save_profile(
                LocalAssistantProfile(
                    facts={"public_profile": "Public science fair helper profile."}
                )
            )
            privacy = store.load_user_fact_privacy_index()

            self.assertFalse(privacy["facts.public_profile"]["local_only"])
            self.assertTrue(privacy["facts.public_profile"]["cloud_eligible"])
            self.assertEqual(privacy["facts.public_profile"]["scope"], "shareable_profile")
            store.close()

    def test_kernel_persists_membrane_homeostasis_and_reloadable_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(db)
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}),
                store=store,
            )

            for _ in range(3):
                kernel.handle("Tell me a story.")
            opportunities = kernel.reflect()
            kernel.execute(opportunities[0])
            after = kernel.handle("Tell me a story.")
            counts = store.table_counts()
            store.close()

            reloaded = AssistantOSStore(db)
            reloaded_kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(story_models={}, weekly_weather={}, contacts={}),
                store=reloaded,
            )

            self.assertEqual(after.route, "local_answer")
            self.assertEqual(counts["events"], 4)
            self.assertEqual(counts["membrane_decisions"], 4)
            self.assertEqual(counts["homeostatic_snapshots"], 4)
            self.assertEqual(counts["synthesis_traces"], 4)
            self.assertGreaterEqual(counts["opportunities"], 1)
            self.assertEqual(reloaded_kernel.events[-1].event_id, "os_e4")
            self.assertEqual(reloaded_kernel.self_model.inventory_counts["story_models"], 3)
            self.assertIn("build_story_inventory", reloaded_kernel.executed_jobs)
            next_decision = reloaded_kernel.handle("Tell me a story.")
            event_rows = reloaded.connection.execute(
                """
                SELECT event_id, session_id, previous_event_id, next_event_id
                FROM events
                ORDER BY rowid
                """
            ).fetchall()
            self.assertEqual(next_decision.route, "local_answer")
            self.assertEqual(reloaded.count("events"), 5)
            self.assertEqual(event_rows[0]["previous_event_id"], "")
            self.assertEqual(event_rows[0]["next_event_id"], "os_e2")
            self.assertEqual(event_rows[3]["previous_event_id"], "os_e3")
            self.assertEqual(event_rows[3]["next_event_id"], "os_e5")
            self.assertEqual(event_rows[4]["previous_event_id"], "os_e4")
            self.assertEqual(event_rows[4]["next_event_id"], "")
            self.assertNotEqual(event_rows[3]["session_id"], event_rows[4]["session_id"])
            reloaded.close()

    def test_response_integrity_queues_only_opted_in_low_confidence_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                store.use_session("browser_integrity_test")
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(),
                    store=store,
                    capture_surface="browser_api",
                    capture_source="browser_ui",
                    improvement_opt_in=True,
                )

                known = kernel.handle("Who are you?")
                known_integrity = kernel.last_response_integrity
                unknown = kernel.handle("Can you explain quasar algebra to my zorbulator?")
                unknown_integrity = kernel.last_response_integrity
                queue = store.improvement_queue(session_id="browser_integrity_test")

                self.assertEqual(known.route, "local_answer")
                self.assertIsNotNone(known_integrity)
                self.assertEqual(known_integrity.band, "reliable")
                self.assertFalse(known_integrity.research_recommended)
                self.assertEqual(unknown.route, "local_answer")
                self.assertIsNotNone(unknown_integrity)
                self.assertEqual(unknown_integrity.band, "review")
                self.assertTrue(unknown_integrity.research_recommended)
                self.assertEqual(
                    unknown_integrity.research_topics,
                    ("quasar", "algebra", "zorbulator"),
                )
                self.assertEqual(store.count("response_integrity"), 2)
                self.assertEqual(queue["candidate_count"], 1)
                self.assertFalse(queue["candidates"][0]["cloud_export_allowed"])
                self.assertFalse(queue["policy"]["live_router_mutation"])

                store.set_session_improvement_consent(
                    "browser_integrity_test",
                    opted_in=False,
                )
                revoked = store.improvement_queue(session_id="browser_integrity_test")
                self.assertEqual(revoked["status_counts"], {"consent_revoked": 1})
            finally:
                store.close()

    def test_low_confidence_turn_without_opt_in_is_scored_but_not_queued(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            store.use_session("browser_no_consent")
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(),
                store=store,
                improvement_opt_in=False,
            )

            kernel.handle("Can you explain quasar algebra to my zorbulator?")

            self.assertEqual(store.count("response_integrity"), 1)
            self.assertEqual(store.count("improvement_candidates"), 0)
            self.assertFalse(
                store.session_improvement_consent("browser_no_consent")["opted_in"]
            )
            store.close()

    def test_event_memory_query_replays_linked_autobiographical_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            kernel = AssistantOSKernel(
                profile=LocalAssistantProfile(
                    story_models={},
                    weekly_weather={},
                    contacts={"mom": "+234-000-MOM"},
                ),
                store=store,
            )

            kernel.handle("Tell me a story.")
            kernel.handle("What is the weather today?")
            kernel.handle("Call mom.")

            story = store.query_event_memory(query="story", limit=5)
            actions = store.query_event_memory(intent="social_contact", route="device_action", limit=5)
            latest_session = store.query_event_memory(session_id="latest", limit=10)
            sessions = store.memory_session_summaries(limit=3)

            self.assertEqual(story["matches"], 1)
            self.assertEqual(story["events"][0]["intent"], "story")
            self.assertEqual(story["events"][0]["previous_link_valid"], True)
            self.assertEqual(actions["matches"], 1)
            self.assertTrue(actions["events"][0]["device_action"])
            self.assertEqual(latest_session["matches"], 3)
            self.assertEqual(latest_session["chain"]["dangling_previous"], 0)
            self.assertEqual(latest_session["chain"]["dangling_next"], 0)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["event_count"], 3)
            self.assertEqual(sessions[0]["intent_counts"]["story"], 1)
            store.close()

    def test_recent_session_memory_returns_bounded_events_across_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            first_store = AssistantOSStore(db)
            try:
                first_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(story_models={}, weekly_weather={}),
                    store=first_store,
                )
                first_kernel.handle("Tell me a story.")
                first_kernel.handle("What is the weather today?")
            finally:
                first_store.close()

            second_store = AssistantOSStore(db)
            try:
                second_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(contacts={"mom": "+234-000-MOM"}),
                    store=second_store,
                )
                second_kernel.handle("I need to talk to someone.")
                second_kernel.handle("Yes, call mom.")

                replay = second_store.query_recent_session_memory(
                    session_limit=2,
                    events_per_session=2,
                )
            finally:
                second_store.close()

            self.assertEqual(replay["session_count"], 2)
            self.assertEqual(replay["matches"], 4)
            self.assertEqual(replay["session_ids"], ["session_1", "session_2"])
            self.assertEqual(replay["events"][0]["session_id"], "session_1")
            self.assertEqual(replay["events"][-1]["session_id"], "session_2")
            self.assertEqual(replay["chain"]["dangling_previous"], 0)
            self.assertEqual(replay["chain"]["dangling_next"], 0)
            self.assertEqual(len(replay["sessions"]), 2)

    def test_memory_digest_compacts_multi_session_event_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            first_store = AssistantOSStore(db)
            try:
                first_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(story_models={}, weekly_weather={}),
                    store=first_store,
                )
                first_kernel.handle("Tell me a story.")
                first_kernel.handle("What is the weather today?")
            finally:
                first_store.close()

            second_store = AssistantOSStore(db)
            try:
                second_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(contacts={"mom": "+234-000-MOM"}),
                    store=second_store,
                )
                second_kernel.handle("I need to talk to someone.")
                second_kernel.handle("Yes, call mom.")
                digest = second_store.build_memory_digest(
                    session_limit=2,
                    events_per_session=2,
                )
                stored = second_store.load_memory_digest()
            finally:
                second_store.close()

            self.assertEqual(digest["digest_id"], "long_horizon_latest")
            self.assertTrue(digest["local_only"])
            self.assertEqual(digest["session_count"], 2)
            self.assertEqual(digest["event_count"], 4)
            self.assertEqual(digest["session_ids"], ["session_1", "session_2"])
            self.assertIn("story", digest["intent_counts"])
            self.assertIn("device_action", digest["route_counts"])
            self.assertLessEqual(len(digest["summary"]), 900)
            self.assertGreaterEqual(digest["quality"]["score"], digest["quality"]["floor"])
            self.assertTrue(digest["quality"]["passed"])
            self.assertEqual(digest["quality"]["warnings"], [])
            self.assertGreaterEqual(digest["quality"]["components"]["thread_coverage"], 0.75)
            self.assertEqual(stored["event_count"], 4)
            self.assertTrue(stored["quality"]["passed"])

    def test_memory_digest_quality_flags_too_thin_long_horizon_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(story_models={}, weekly_weather={}),
                    store=store,
                )
                kernel.handle("Tell me a story.")
                digest = store.build_memory_digest(
                    session_limit=3,
                    events_per_session=2,
                )
            finally:
                store.close()

            self.assertEqual(digest["event_count"], 1)
            self.assertLess(digest["quality"]["score"], digest["quality"]["floor"])
            self.assertFalse(digest["quality"]["passed"])
            self.assertIn("insufficient_long_horizon", digest["quality"]["warnings"])

    def test_device_action_creates_pending_plan_and_confirmation_executes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)

                action = kernel.handle("I need to talk to someone.")
                confirmed = action.__class__(
                    utterance="Yes, call mom.",
                    intent=action.intent,
                    route="device_action",
                    answer=f"Confirmed: {action.answer}",
                    evidence_keys=action.evidence_keys,
                    local_memory_used=True,
                    device_action=True,
                    confidence=0.96,
                    reason="confirmed_device_action",
                )
                kernel.remember(confirmed)

                row = store.connection.execute(
                    "SELECT confirmation_state, executed, result FROM pending_actions"
                ).fetchone()
                result = json.loads(row["result"])
                self.assertEqual(action.route, "device_action")
                self.assertIsNotNone(row)
                self.assertEqual(row["confirmation_state"], "confirmed")
                self.assertEqual(row["executed"], 1)
                self.assertEqual(result["action_type"], "call_contact")
                self.assertEqual(result["mode"], "dry-run")
                self.assertEqual(result["status"], "prepared")
                self.assertFalse(result["side_effect_executed"])
            finally:
                store.close()

    def test_revoked_facts_and_stale_weather_do_not_reload_as_usable_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            store.set_user_fact_consent("facts.favorite_color", consent=False)
            store.upsert_inventory(
                "weather",
                "today",
                {"forecast": "sunny", "stale": True},
                source="weather_cache",
                license="local_cache",
                tags=("weather",),
            )
            store.upsert_inventory(
                "weather",
                "tomorrow",
                {"forecast": "mild", "expires_at": "2999-01-01T00:00:00+00:00"},
                source="weather_cache",
                license="local_cache",
                tags=("weather",),
            )
            store.connection.commit()

            profile = store.load_profile(
                LocalAssistantProfile(
                    facts={"favorite_color": "green"},
                    weekly_weather={"today": "warm with afternoon rain"},
                    story_models={},
                    contacts={},
                )
            )

            self.assertNotIn("favorite_color", profile.facts)
            self.assertNotIn("today", profile.weekly_weather)
            self.assertEqual(profile.weekly_weather["tomorrow"], "mild")
            store.close()


if __name__ == "__main__":
    unittest.main()
