import inspect
import json
import tempfile
import unittest
from pathlib import Path

import melm.appliance.assistant_os_kernel as kernel_module
from melm.appliance import (
    AssistantOSKernel,
    AssistantOSStore,
    LocalAssistantProfile,
    lexicon_ingest,
    lookup_lexical_senses,
    run_assistant_kernel_learning_probe,
)
from melm.appliance.assistant_os_kernel import _requested_confirmation_target


class AssistantOSKernelMvpTests(unittest.TestCase):
    def test_kernel_remembers_cloud_story_gap_and_builds_local_inventory(self) -> None:
        report = run_assistant_kernel_learning_probe()

        self.assertEqual(report.before_route, "cloud_handoff")
        self.assertEqual(report.after_route, "local_answer")
        self.assertEqual(report.cloud_handoffs_before, 3)
        self.assertEqual(report.cloud_handoffs_after, 0)
        self.assertEqual(report.executed_jobs, ("build_story_inventory",))
        self.assertEqual(report.story_inventory_count, 3)
        self.assertEqual(report.remembered_events, 4)
        self.assertEqual(report.opportunities[0].kind, "build_story_inventory")
        self.assertEqual(report.opportunities[0].priority, 0.95)
        self.assertEqual(report.opportunities[0].expected_cloud_reduction, 3)
        self.assertIn(
            "project_gutenberg_catalog_metadata",
            report.opportunities[0].source_candidates,
        )
        self.assertIn(
            "internet_archive_item_search_and_metadata",
            report.opportunities[0].source_candidates,
        )

    def test_kernel_reflection_separates_story_weather_memory_and_contact_gaps(
        self,
    ) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                story_models={},
                weekly_weather={},
                facts={},
                contacts={},
            )
        )

        for _ in range(3):
            kernel.handle("Tell me a story.")
        kernel.handle("What is the weather today?")
        kernel.handle("Tell me something about myself.")
        kernel.handle("I need to talk to someone.")

        opportunities = kernel.reflect()
        kinds = [opportunity.kind for opportunity in opportunities]

        self.assertEqual(kinds[0], "build_story_inventory")
        self.assertIn("refresh_weather_cache", kinds)
        self.assertIn("ask_profile_memory", kinds)
        self.assertIn("request_trusted_contact", kinds)
        self.assertEqual(len(kernel.events), 6)
        self.assertEqual(kernel.events[0].event_id, "os_e1")
        self.assertEqual(kernel.events[-1].reason, "missing_contact")

    def test_kernel_reflection_prioritizes_non_story_gaps_from_pressure_signals(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        story_models={},
                        weekly_weather={"today": "warm"},
                        facts={},
                        contacts={},
                    ),
                    store=store,
                )

                kernel.handle("Tell me something about myself.")
                kernel.handle("I need to talk to someone.")
                kernel.handle("I need help talking to someone.")
                opportunities = {item.kind: item for item in kernel.reflect()}

                contact = opportunities["request_trusted_contact"]
                profile = opportunities["ask_profile_memory"]
                row = store.connection.execute(
                    "SELECT priority FROM opportunities WHERE kind='request_trusted_contact'"
                ).fetchone()

                self.assertGreater(contact.priority, profile.priority)
                self.assertEqual(contact.priority_signals["trusted_contact_misses"], 2)
                self.assertEqual(profile.priority_signals["profile_memory_misses"], 1)
                self.assertGreater(contact.priority_signals["avg_uncertainty"], 0.0)
                self.assertIsNotNone(row)
                self.assertAlmostEqual(row["priority"], contact.priority)
            finally:
                store.close()

    def test_kernel_reflection_creates_media_routine_and_household_opportunities(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        story_models={"seed": "{name} rested in {location}."},
                        weekly_weather={"today": "warm"},
                        facts={},
                        contacts={"mom": "+234-000-MOM"},
                        media_library=(),
                    ),
                    store=store,
                )

                media = kernel.handle("Play a song for me.")
                routine = kernel.handle("What is my morning routine?")
                household = kernel.handle("What do you know about this household?")
                opportunities = {item.kind: item for item in kernel.reflect()}

                self.assertEqual(media.reason, "empty_media_library")
                self.assertEqual(routine.reason, "personal_memory_empty")
                self.assertEqual(household.reason, "personal_memory_empty")
                self.assertIn("build_media_index", opportunities)
                self.assertIn("ask_routine_memory", opportunities)
                self.assertIn("ask_household_memory", opportunities)
                self.assertEqual(
                    opportunities["build_media_index"].priority_signals["media_misses"],
                    1,
                )
                self.assertEqual(
                    opportunities["ask_routine_memory"].priority_signals[
                        "routine_memory_misses"
                    ],
                    1,
                )
                self.assertEqual(
                    opportunities["ask_household_memory"].priority_signals[
                        "household_memory_misses"
                    ],
                    1,
                )
                self.assertIn(
                    "local_consent_policy",
                    opportunities["ask_household_memory"].source_candidates,
                )
            finally:
                store.close()

    def test_story_request_with_unmatched_constraints_does_not_fake_local_answer(
        self,
    ) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                story_models={
                    "moon_drum_walk": "Maya found a talking drum in Lagos and came home before the rain."
                },
                weekly_weather={"today": "warm"},
            )
        )

        decision = kernel.handle("Tell me a story about a dragon and a robot")

        self.assertEqual(decision.intent, "story")
        self.assertEqual(decision.route, "clarify")
        self.assertEqual(decision.reason, "story_constraint_unmet")
        self.assertIn("dragon", decision.answer.lower())
        self.assertIn("robot", decision.answer.lower())
        self.assertNotIn("Moon Drum Walk", decision.answer)

    def test_kernel_stores_story_preference_without_lifecycle_shortcut(self) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                facts={},
                preferences={},
                story_models={},
                weekly_weather={},
                contacts={},
            )
        )

        stored = kernel.handle("I like dinosaur stories and Yoruba folktales.")
        recall = kernel.handle("Tell me something about myself.")

        self.assertEqual(stored.reason, "profile_update")
        self.assertEqual(stored.evidence_keys, ("preferences.story_theme",))
        self.assertEqual(
            kernel.profile.preferences["story_theme"],
            "dinosaur stories and Yoruba folktales",
        )
        self.assertEqual(kernel.profile.culture, "Yoruba")
        self.assertNotIn("favorite_story_theme", kernel.profile.facts)
        self.assertEqual(recall.reason, "personal_memory_summary")
        self.assertIn("preferences.story_theme", recall.evidence_keys)

    def test_confirmation_requires_confirmation_token_not_prefix_substring(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        contacts={"mom": "+234-000-MOM"},
                        weekly_weather={"today": "warm"},
                    ),
                    store=store,
                )

                request = kernel.handle("Call mom.")
                not_confirmation = kernel.handle("Yesterday I saw mom.")
                pending = store.latest_pending_action()
                self.assertIsNotNone(pending)
                state = store.connection.execute(
                    "SELECT confirmation_state, executed FROM pending_actions WHERE action_id=?",
                    (pending["action_id"],),
                ).fetchone()

                self.assertEqual(request.reason, "trusted_contact_action")
                self.assertEqual(not_confirmation.reason, "unknown_intent")
                self.assertEqual(state["confirmation_state"], "pending")
                self.assertEqual(state["executed"], 0)
            finally:
                store.close()

    def test_kernel_executes_media_index_and_changes_future_media_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        story_models={},
                        weekly_weather={"today": "warm"},
                        preferences={},
                        contacts={},
                        media_library=(),
                    ),
                    store=store,
                )

                before = kernel.handle("Play a song for me.")
                media_opportunity = next(
                    item
                    for item in kernel.reflect()
                    if item.kind == "build_media_index"
                )
                kernel.execute(media_opportunity)
                after = kernel.handle("Play a song for me.")
                persisted_media = store.load_inventory("media")
                media_source = store.connection.execute(
                    """
                    SELECT source
                    FROM inventories
                    WHERE kind='media' AND item_id='calm piano'
                    """
                ).fetchone()

                self.assertEqual(before.route, "clarify")
                self.assertEqual(before.reason, "empty_media_library")
                self.assertEqual(media_opportunity.priority_signals["media_misses"], 1)
                self.assertEqual(after.route, "device_action")
                self.assertEqual(after.reason, "local_media_action")
                self.assertEqual(kernel.self_model.inventory_counts["media_items"], 3)
                self.assertIn("build_media_index", kernel.executed_jobs)
                self.assertIn("calm piano", persisted_media)
                self.assertIn("rain sounds", persisted_media)
                self.assertIsNotNone(media_source)
                self.assertEqual(media_source["source"], "local_media_manifest")
                self.assertNotIn("music", kernel.profile.preferences)
            finally:
                store.close()

    def test_kernel_executes_setup_requests_without_fabricating_user_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        story_models={"seed": "{name} rested in {location}."},
                        weekly_weather={"today": "warm"},
                        facts={},
                        contacts={},
                        media_library=("calm piano",),
                    ),
                    store=store,
                )

                kernel.handle("What is my morning routine?")
                kernel.handle("What do you know about this household?")
                kernel.handle("I need to talk to someone.")
                opportunities = {item.kind: item for item in kernel.reflect()}

                kernel.execute(opportunities["ask_routine_memory"])
                kernel.execute(opportunities["ask_household_memory"])
                kernel.execute(opportunities["request_trusted_contact"])
                still_empty_routine = kernel.handle("What is my morning routine?")
                still_missing_contact = kernel.handle("I need to talk to someone.")
                setup_requests = store.load_inventory("setup_request")

                self.assertEqual(still_empty_routine.route, "clarify")
                self.assertEqual(still_empty_routine.reason, "personal_memory_empty")
                self.assertEqual(still_missing_contact.route, "clarify")
                self.assertEqual(still_missing_contact.reason, "missing_contact")
                self.assertEqual(kernel.self_model.inventory_counts["routine_facts"], 0)
                self.assertEqual(
                    kernel.self_model.inventory_counts["household_facts"], 0
                )
                self.assertEqual(kernel.self_model.inventory_counts["contacts"], 0)
                self.assertIn("routine_memory", setup_requests)
                self.assertIn("household_memory", setup_requests)
                self.assertIn("trusted_contact", setup_requests)
                self.assertTrue(
                    setup_requests["routine_memory"]["requires_user_supplied_value"]
                )
            finally:
                store.close()

    def test_kernel_stores_user_supplied_routine_household_and_contact_setup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        facts={}, contacts={}, weekly_weather={"today": "warm"}
                    ),
                    store=store,
                )

                routine = kernel.handle(
                    "My morning routine is stretch, breakfast, then bus."
                )
                household = kernel.handle(
                    "Our household includes Maya and Mom, and memory stays local."
                )
                contact = kernel.handle("Ada is my trusted contact at +234-000-ADA.")
                routine_recall = kernel.handle("What is my morning routine?")
                household_recall = kernel.handle(
                    "What do you know about this household?"
                )
                call = kernel.handle("I need to talk to someone.")

                self.assertEqual(routine.reason, "consented_routine_memory_stored")
                self.assertEqual(household.reason, "consented_household_memory_stored")
                self.assertEqual(contact.reason, "consented_trusted_contact_stored")
                self.assertEqual(routine_recall.route, "local_answer")
                self.assertEqual(routine_recall.reason, "personal_memory_recall")
                self.assertIn("stretch", routine_recall.answer)
                self.assertEqual(household_recall.route, "local_answer")
                self.assertIn("Maya and Mom", household_recall.answer)
                self.assertEqual(call.route, "device_action")
                self.assertEqual(call.reason, "trusted_contact_action")
                self.assertEqual(kernel.self_model.inventory_counts["routine_facts"], 1)
                self.assertEqual(
                    kernel.self_model.inventory_counts["household_facts"], 1
                )
                self.assertEqual(kernel.self_model.inventory_counts["contacts"], 1)
            finally:
                store.close()

            reloaded = AssistantOSStore(db)
            try:
                reloaded_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        facts={}, contacts={}, weekly_weather={"today": "warm"}
                    ),
                    store=reloaded,
                )
                self.assertIn("morning_routine", reloaded_kernel.profile.facts)
                self.assertIn("household_context", reloaded_kernel.profile.facts)
                self.assertIn("ada", reloaded_kernel.profile.contacts)
            finally:
                reloaded.close()

    def test_kernel_stores_child_memory_as_owned_local_facts_not_generic_profile_shortcuts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        facts={}, contacts={}, weekly_weather={}
                    ),
                    store=store,
                )

                age = kernel.handle("My child is 8 years old.")
                school = kernel.handle("My child's school is Bright School.")
                recall = kernel.handle("What is my child's school?")
                cloud = kernel.handle("Send my child's age and school to the cloud.")
                privacy = store.load_user_fact_privacy_index()

                self.assertEqual(age.reason, "consented_child_memory_stored")
                self.assertEqual(school.reason, "consented_child_memory_stored")
                self.assertEqual(age.evidence_keys, ("facts.child_age",))
                self.assertEqual(school.evidence_keys, ("facts.child_school",))
                self.assertEqual(recall.route, "local_answer")
                self.assertEqual(recall.reason, "personal_memory_recall")
                self.assertEqual(recall.evidence_keys, ("facts.child_school",))
                self.assertIn("Bright School", recall.answer)
                self.assertEqual(cloud.route, "reject")
                self.assertEqual(cloud.reason, "blocked_private_facts_to_cloud")
                self.assertIn("facts.child_age", cloud.evidence_keys)
                self.assertIn("facts.child_school", cloud.evidence_keys)
                self.assertNotIn("profile.age", cloud.evidence_keys)
                self.assertNotIn("facts.school", cloud.evidence_keys)
                self.assertEqual(privacy["facts.child_age"]["scope"], "child_local")
                self.assertEqual(privacy["facts.child_school"]["scope"], "child_local")
                self.assertFalse(privacy["facts.child_age"]["cloud_eligible"])
                self.assertEqual(
                    kernel.self_model.inventory_counts["household_facts"], 2
                )
            finally:
                store.close()

            reloaded = AssistantOSStore(db)
            try:
                reloaded_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        facts={}, contacts={}, weekly_weather={}
                    ),
                    store=reloaded,
                )
                self.assertEqual(reloaded_kernel.profile.facts["child_age"], "8")
                self.assertEqual(
                    reloaded_kernel.profile.facts["child_school"], "Bright School"
                )
            finally:
                reloaded.close()

    def test_kernel_revoked_child_school_does_not_fall_back_to_generic_school(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        facts={"school": "you usually go to school on weekdays"},
                        contacts={},
                        weekly_weather={},
                    ),
                    store=store,
                )

                kernel.handle("My child is 8 years old.")
                kernel.handle("My child's school is Bright School.")
                forget = kernel.handle("Forget my child's school.")
            finally:
                store.close()

            reloaded = AssistantOSStore(db)
            try:
                reloaded_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        facts={"school": "you usually go to school on weekdays"},
                        contacts={},
                        weekly_weather={},
                    ),
                    store=reloaded,
                )
                child_school = reloaded_kernel.handle("What is my child's school?")
                child_age = reloaded_kernel.handle("How old is my child?")
                privacy = reloaded.load_user_fact_privacy_index()

                self.assertEqual(forget.reason, "consent_revoked_user_fact")
                self.assertEqual(forget.evidence_keys, ("facts.child_school",))
                self.assertEqual(child_school.route, "clarify")
                self.assertEqual(child_school.reason, "personal_memory_empty")
                self.assertEqual(child_school.evidence_keys, ())
                self.assertNotIn("weekdays", child_school.answer)
                self.assertEqual(child_age.route, "local_answer")
                self.assertEqual(child_age.evidence_keys, ("facts.child_age",))
                self.assertFalse(privacy["facts.child_school"]["consent"])
                self.assertEqual(privacy["facts.child_school"]["scope"], "child_local")
            finally:
                reloaded.close()

    def test_self_model_tracks_inventory_after_background_jobs(self) -> None:
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                story_models={}, weekly_weather={}, contacts={}
            )
        )
        for _ in range(3):
            kernel.handle("Tell me a story.")
        story_opportunity = kernel.reflect()[0]

        kernel.execute(story_opportunity)
        decision = kernel.handle("Tell me a story.")

        self.assertEqual(decision.route, "local_answer")
        self.assertEqual(kernel.self_model.inventory_counts["story_models"], 3)
        self.assertEqual(kernel.executed_jobs, ["build_story_inventory"])
        self.assertIn(
            "prepare local inventories from public-domain sources",
            kernel.self_model.strengths,
        )

    def test_self_model_persists_runtime_health_trends_from_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(db)
            try:
                kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        story_models={},
                        weekly_weather={},
                        contacts={},
                        media_library=(),
                    ),
                    store=store,
                )

                miss = kernel.handle("What is the weather today?")
                kernel.reflect()
                state = store.load_self_state()
                trends = state["runtime_health_trends"]
                history = state["runtime_health_history"]
                status = kernel.handle("What have you done so far?")
                synthesis = kernel.last_synthesis
            finally:
                store.close()

            self.assertEqual(miss.route, "external_fetch")
            self.assertEqual(trends["routing"]["external_fetches"], 1)
            self.assertFalse(trends["cache_health"]["weather_cache_ready"])
            self.assertEqual(trends["job_health"]["queued"], 1)
            self.assertGreaterEqual(len(history), 2)
            self.assertEqual(
                trends["history_summary"]["weather_cache_gap_persistence"], 1.0
            )
            self.assertEqual(
                trends["history_summary"]["story_inventory_gap_persistence"], 1.0
            )
            self.assertIn(
                "refresh the local weather cache", trends["next_observed_needs"]
            )
            self.assertIn("weather_cache=missing", status.answer)
            self.assertIn("history_points", status.answer)
            self.assertIsNotNone(synthesis)
            self.assertIn("self_status.self_observation", synthesis.citations)

            reloaded = AssistantOSStore(db)
            try:
                persisted = reloaded.load_self_state()["runtime_health_trends"]
                persisted_history = reloaded.load_self_state()["runtime_health_history"]
            finally:
                reloaded.close()
            self.assertEqual(persisted["schema"], "melm.assistant_self_observation.v1")
            self.assertFalse(persisted["cache_health"]["weather_cache_ready"])
            self.assertGreaterEqual(len(persisted_history), 2)

    def test_kernel_self_status_gate_reuses_uol_composition_not_phrase_table(
        self,
    ) -> None:
        status_source = inspect.getsource(kernel_module._is_assistant_status_request)
        next_source = inspect.getsource(kernel_module._is_assistant_next_step_request)

        self.assertIn("compose_assistant_status_frame", status_source)
        self.assertIn("compose_assistant_status_frame", next_source)
        self.assertNotIn("_has_any_marker", status_source)
        self.assertNotIn("_has_any_marker", next_source)

        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)

                ledger = kernel.handle("Show your memory ledger.")
                cloud = kernel.handle("Are you using cloud?")
                advice = kernel.handle(
                    "What do you think I should do to improve my health?"
                )
            finally:
                store.close()

        self.assertEqual(ledger.intent, "assistant_status")
        self.assertEqual(ledger.reason, "self_status_ledger_summary")
        self.assertEqual(cloud.intent, "assistant_status")
        self.assertEqual(advice.intent, "health_advice")

    def test_kernel_blocks_action_confirmation_replay_without_pending_action(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)
                decision = kernel.handle("Yes, call mom.")

                self.assertEqual(decision.route, "clarify")
                self.assertEqual(decision.reason, "no_pending_action_to_confirm")
                self.assertEqual(store.count("pending_actions"), 0)
                self.assertEqual(store.count("events"), 1)
            finally:
                store.close()

    def test_kernel_cancelled_pending_action_cannot_be_confirmed_later(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)

                first = kernel.handle("I need to talk to someone.")
                cancel = kernel.handle("Do not call mom.")
                replay = kernel.handle("Yes, call mom.")
                row = store.connection.execute(
                    "SELECT confirmation_state, executed FROM pending_actions"
                ).fetchone()

                self.assertEqual(first.route, "device_action")
                self.assertEqual(cancel.reason, "cancelled_pending_action")
                self.assertEqual(replay.reason, "no_pending_action_to_confirm")
                self.assertEqual(row["confirmation_state"], "cancelled")
                self.assertEqual(row["executed"], 0)
            finally:
                store.close()

    def test_kernel_blocks_confirmation_target_switch_without_new_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)

                first = kernel.handle("I need to talk to someone.")
                mismatch = kernel.handle("Yes, call dad.")
                confirm = kernel.handle("Yes, call mom.")
                row = store.connection.execute(
                    "SELECT confirmation_state, executed FROM pending_actions"
                ).fetchone()

                self.assertEqual(first.route, "device_action")
                self.assertEqual(mismatch.route, "clarify")
                self.assertEqual(mismatch.reason, "confirmation_target_mismatch")
                self.assertEqual(confirm.reason, "confirmed_device_action")
                self.assertEqual(row["confirmation_state"], "confirmed")
                self.assertEqual(row["executed"], 1)
            finally:
                store.close()

    def test_kernel_confirmation_target_extraction_is_token_bounded(self) -> None:
        self.assertEqual(
            _requested_confirmation_target("Yes, play calm piano."), "calm piano"
        )
        self.assertEqual(_requested_confirmation_target("Yes, call to mom."), "mom")
        self.assertEqual(_requested_confirmation_target("Yes, phone my mom."), "mom")
        self.assertEqual(_requested_confirmation_target("Yes, replay calm piano."), "")
        self.assertEqual(_requested_confirmation_target("Yes, display dad."), "")

    def test_kernel_revokes_fact_consent_and_reload_excludes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(db)
            try:
                profile = LocalAssistantProfile(
                    facts={"favorite_color": "green"},
                    weekly_weather={},
                    story_models={},
                    contacts={},
                )
                kernel = AssistantOSKernel(profile=profile, store=store)
                forget = kernel.handle("Forget my favorite color.")
                after = kernel.handle("Tell me something about myself.")

                self.assertEqual(forget.reason, "consent_revoked_user_fact")
                self.assertEqual(after.route, "clarify")
                self.assertNotIn("favorite_color", kernel.profile.facts)
            finally:
                store.close()

            reloaded = AssistantOSStore(db)
            try:
                reloaded_kernel = AssistantOSKernel(
                    profile=LocalAssistantProfile(
                        facts={"favorite_color": "green"},
                        weekly_weather={},
                        story_models={},
                        contacts={},
                    ),
                    store=reloaded,
                )
                self.assertNotIn("favorite_color", reloaded_kernel.profile.facts)
            finally:
                reloaded.close()

    def test_kernel_answers_autobiographical_recall_from_event_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)

                kernel.handle("Tell me a story.")
                kernel.handle("What is the weather today?")
                recall = kernel.handle("What did we talk about earlier?")
                row = store.connection.execute(
                    """
                    SELECT applied, citation_count, evidence_count
                    FROM synthesis_traces
                    WHERE event_id=?
                    """,
                    (kernel.events[-1].event_id,),
                ).fetchone()

                self.assertEqual(recall.route, "local_answer")
                self.assertEqual(recall.intent, "autobiographical_memory")
                self.assertEqual(recall.reason, "autobiographical_memory_summary")
                self.assertIn("conversation memory", recall.answer.lower())
                self.assertIn("Tell me a story", recall.answer)
                self.assertIn("What is the weather today", recall.answer)
                self.assertTrue(
                    any(key.startswith("events.") for key in recall.evidence_keys)
                )
                self.assertIsNotNone(kernel.last_synthesis)
                self.assertTrue(kernel.last_synthesis.applied)
                self.assertEqual(store.count("events"), 3)
                self.assertIsNotNone(row)
                self.assertEqual(row["applied"], 1)
                self.assertGreaterEqual(row["citation_count"], 2)
                self.assertGreaterEqual(row["evidence_count"], 2)
            finally:
                store.close()

    def test_kernel_last_question_recall_uses_bounded_latest_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)

                kernel.handle("Tell me a story.")
                kernel.handle("What is the weather today?")
                recall = kernel.handle("What was my last question?")

                self.assertEqual(recall.route, "local_answer")
                self.assertEqual(recall.reason, "autobiographical_memory_summary")
                self.assertEqual(len(recall.evidence_keys), 1)
                self.assertIn("What is the weather today", recall.answer)
                self.assertNotIn("Tell me a story", recall.answer)
            finally:
                store.close()

    def test_kernel_autobiographical_gate_uses_chatframe_scope_not_exact_markers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)

                kernel.handle("Tell me a story.")
                kernel.handle("What is the weather today?")
                paraphrase = kernel.handle("What was the last thing I asked you?")
                statement = kernel.handle("I dropped the last thing yesterday.")
                gate_source = inspect.getsource(
                    kernel_module._is_autobiographical_recall_request
                )

                self.assertEqual(paraphrase.route, "local_answer")
                self.assertEqual(paraphrase.intent, "autobiographical_memory")
                self.assertEqual(paraphrase.reason, "autobiographical_memory_summary")
                self.assertEqual(len(paraphrase.evidence_keys), 1)
                self.assertIn("What is the weather today", paraphrase.answer)
                self.assertEqual(statement.intent, "unknown")
                self.assertEqual(statement.route, "cloud_handoff")
                self.assertIn("compose_autobiographical_memory_frame", gate_source)
                self.assertNotIn("recall_markers", gate_source)
                self.assertNotIn("what was my last question", gate_source)
                self.assertNotIn("_has_any_marker", gate_source)
            finally:
                store.close()

    def test_kernel_blocks_cloud_export_of_conversation_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)
                kernel.handle("Tell me a story.")
                blocked = kernel.handle("Send our previous conversation to the cloud.")

                self.assertEqual(blocked.route, "reject")
                self.assertEqual(blocked.reason, "blocked_private_facts_to_cloud")
                self.assertIn("events.local_conversation", blocked.evidence_keys)
                self.assertIsNotNone(kernel.last_synthesis)
                self.assertTrue(kernel.last_synthesis.refused)
            finally:
                store.close()

    def test_kernel_summarizes_recent_sessions_from_linked_memory(self) -> None:
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
                recall = second_kernel.handle("Summarize our recent sessions.")

                self.assertEqual(recall.route, "local_answer")
                self.assertEqual(recall.intent, "autobiographical_memory")
                self.assertEqual(recall.reason, "autobiographical_session_summary")
                self.assertIn("session 1 (session_1)", recall.answer)
                self.assertIn("session 2 (session_2)", recall.answer)
                self.assertIn("Tell me a story", recall.answer)
                self.assertIn("I need to talk to someone", recall.answer)
                self.assertTrue(
                    all(key.startswith("events.") for key in recall.evidence_keys)
                )
                self.assertEqual(len(recall.evidence_keys), 4)
                self.assertIsNotNone(second_kernel.last_synthesis)
                self.assertTrue(second_kernel.last_synthesis.applied)
            finally:
                second_store.close()

    def test_kernel_builds_long_horizon_digest_for_multi_day_recall(self) -> None:
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
            finally:
                second_store.close()

            third_store = AssistantOSStore(db)
            try:
                third_kernel = AssistantOSKernel(store=third_store)
                recall = third_kernel.handle("What happened over the last few days?")
                digest = third_store.load_memory_digest()

                self.assertEqual(recall.route, "local_answer")
                self.assertEqual(recall.reason, "autobiographical_memory_digest")
                self.assertEqual(
                    recall.evidence_keys, ("memory_digest.long_horizon_latest",)
                )
                self.assertIn("long-horizon memory digest", recall.answer)
                self.assertIn("Tell me a story", recall.answer)
                self.assertIn("I need to talk to someone", recall.answer)
                self.assertGreaterEqual(digest["session_count"], 2)
                self.assertGreaterEqual(digest["event_count"], 4)
                self.assertIsNotNone(third_kernel.last_synthesis)
                self.assertTrue(third_kernel.last_synthesis.applied)
            finally:
                third_store.close()


class AssistantOSKernelAcquisitionTests(unittest.TestCase):
    """M3 runtime acquisition loop — automated lookup triggered by unknown tokens."""

    def setUp(self):
        from melm.appliance.local_assistant_router import _IN_MEMORY_LEXICON
        self._saved_lexicon = dict(_IN_MEMORY_LEXICON)

    def tearDown(self):
        from melm.appliance.local_assistant_router import _IN_MEMORY_LEXICON, replace_in_memory_lexicon
        replace_in_memory_lexicon(self._saved_lexicon)

    def _seed_genus(self, store, term="instrument", class_id="physical_object.instrument"):
        candidate = {
            "schema_id": "melm.sense_candidate.v1",
            "batch_id": "test_batch",
            "lemma": term,
            "language": "en",
            "pos": "noun",
            "source": {
                "provenance": "seed_authored",
                "source_ref": f"test:{term}:{class_id}",
                "license": "test-license",
            },
            "definition": f"a test {term}",
            "semantic_class_candidates": [
                {"class_id": class_id, "method": "seed_authored", "confidence": 0.95},
            ],
            "forms": [],
            "relations": [],
            "safety": {"reserved_conflict": False, "policy_term_overlap": False},
            "suggested_status": "active",
            "confidence_prior": 0.95,
        }
        lexicon_ingest(store, candidate, expected_provenance="seed_authored")

    def _write_jsonl(self, tmp: Path, entries: list[dict]) -> Path:
        path = tmp / "dictionary.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return path

    def test_acquisition_offline_lookup_quarantines_unknown_word(self) -> None:
        """Unknown word in utterance triggers offline lookup; result is quarantined."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AssistantOSStore(tmp_path / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    tmp_path,
                    [{"lemma": "xylophone", "definition": "a musical instrument"}],
                )
                kernel = AssistantOSKernel(
                    store=store,
                    offline_dictionary_path=str(dict_path),
                )
                kernel.handle("What is a xylophone?")
                senses = lookup_lexical_senses(
                    store, "xylophone", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 1)
                self.assertEqual(senses[0]["status"], "quarantined")
                self.assertEqual(senses[0]["lemma"], "xylophone")
            finally:
                store.close()

    def test_acquisition_skips_known_word(self) -> None:
        """Known word does not trigger offline lookup — no quarantined entry created."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AssistantOSStore(tmp_path / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    tmp_path,
                    [{"lemma": "weather", "definition": "the state of the atmosphere"}],
                )
                kernel = AssistantOSKernel(
                    store=store,
                    offline_dictionary_path=str(dict_path),
                )
                kernel.handle("What is the weather today?")
                senses = lookup_lexical_senses(
                    store, "weather", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 0)
            finally:
                store.close()

    def test_acquisition_no_store_no_crash(self) -> None:
        """Kernel without store skips acquisition gracefully."""
        kernel = AssistantOSKernel(
            offline_dictionary_path="nonexistent.jsonl",
        )
        decision = kernel.handle("What is a xylophone?")
        self.assertIsNotNone(decision)

    def test_acquisition_no_dictionary_no_entry_created(self) -> None:
        """Kernel without dictionary_path does not create lexical entry for unknown word."""
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                kernel = AssistantOSKernel(store=store)
                kernel.handle("What is a xylophone?")
                senses = lookup_lexical_senses(
                    store, "xylophone", statuses=("quarantined", "active", "dormant"),
                )
                self.assertEqual(len(senses), 0)
            finally:
                store.close()

    def test_acquisition_no_matching_dictionary_entry_no_entry_created(self) -> None:
        """Dictionary file without matching lemma does not create lexical entry."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AssistantOSStore(tmp_path / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    tmp_path,
                    [{"lemma": "kalimba", "definition": "a small thumb piano"}],
                )
                kernel = AssistantOSKernel(
                    store=store,
                    offline_dictionary_path=str(dict_path),
                )
                kernel.handle("What is a xylophone?")
                senses = lookup_lexical_senses(
                    store, "xylophone", statuses=("quarantined", "active", "dormant"),
                )
                self.assertEqual(len(senses), 0)
            finally:
                store.close()

    def test_acquisition_missing_dictionary_file_no_crash(self) -> None:
        """Non-existent dictionary file does not crash the acquisition pass."""
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                kernel = AssistantOSKernel(
                    store=store,
                    offline_dictionary_path=str(Path(tmp) / "missing.jsonl"),
                )
                kernel.handle("What is a xylophone?")
                senses = lookup_lexical_senses(
                    store, "xylophone", statuses=("quarantined", "active", "dormant"),
                )
                self.assertEqual(len(senses), 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
