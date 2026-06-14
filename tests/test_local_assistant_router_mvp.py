import unittest
import inspect

from melm.appliance import (
    LocalAssistantProfile,
    OnDeviceAssistantRouter,
    compare_assistant_mvp_directions,
    compare_assistant_strategy_reports_for_utterances,
    parse_assistant_debug_frame,
)
from melm.appliance import local_assistant_router as router_module
from melm.appliance import functional_grammar as grammar_module


class LocalAssistantRouterMvpTests(unittest.TestCase):
    def test_weighted_functional_grammar_understands_real_transcript_without_phrase_rules(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        cases = {
            "hi": ("social_greeting", "local_answer", "user", "greet", "", "", ""),
            "do you always tell people the same thing?": (
                "assistant_behavior",
                "local_answer",
                "assistant",
                "tell",
                "thing",
                "",
                "people",
            ),
            "do you like repeating yourself?": (
                "assistant_behavior",
                "local_answer",
                "assistant",
                "like",
                "assistant",
                "repeat",
                "",
            ),
            "I want to grow in my career": (
                "personal_goal_advice",
                "cloud_handoff",
                "user",
                "want",
                "career",
                "grow",
                "",
            ),
            "can you help me grow in my career?": (
                "personal_goal_advice",
                "cloud_handoff",
                "assistant",
                "help",
                "career",
                "grow",
                "user",
            ),
        }

        for utterance, expected in cases.items():
            with self.subTest(utterance=utterance):
                intent, route, subject, action, object_value, complement, indirect = expected
                decision = router.handle(utterance)
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual((decision.intent, decision.route), (intent, route))
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "weighted_functional_relation")
                self.assertEqual(parsed["uol"]["subject"], subject)
                self.assertEqual(parsed["uol"]["action"], action)
                self.assertEqual(parsed["uol"]["object"], object_value)
                self.assertEqual(parsed["uol"]["complement_action"], complement)
                self.assertEqual(parsed["uol"]["indirect_object"], indirect)
                self.assertGreaterEqual(parsed["uol"]["parse_score"], 0.9)
                self.assertNotEqual(decision.reason, "unknown_intent")

    def test_weighted_functional_grammar_generalizes_to_held_out_relations(self) -> None:
        cases = {
            "hello there": ("social_greeting", "greet", ""),
            "Do you usually say identical answers?": ("assistant_behavior", "say", "answer"),
            "I would like to improve at work.": ("personal_goal_advice", "like", "work"),
            "Could you help me improve at work?": ("personal_goal_advice", "help", "work"),
            "How do airplanes fly?": ("open_domain", "fly", ""),
        }

        for utterance, (intent, action, object_value) in cases.items():
            with self.subTest(utterance=utterance):
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(parsed["chat_frame"]["intent"], intent)
                self.assertEqual(parsed["uol"]["action"], action)
                self.assertEqual(parsed["uol"]["object"], object_value)
                self.assertEqual(parsed["nlp"]["compositional_parse"]["source"], "weighted_functional_relation")
                self.assertTrue(parsed["nlp"]["candidate_parses"])
                self.assertTrue(parsed["uol"]["relations"])

    def test_functional_relations_do_not_turn_topic_mentions_into_local_shortcuts(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        declarative_story = router.handle("The same people tell stories.")
        career_definition = router.handle("Career is a board game.")
        greeting_fragment = router.handle("hi-fi audio")
        bare_action = router.handle("Play")

        self.assertEqual((declarative_story.intent, declarative_story.route), ("open_domain", "cloud_handoff"))
        self.assertEqual((career_definition.intent, career_definition.route), ("open_domain", "cloud_handoff"))
        self.assertNotEqual(declarative_story.intent, "story")
        self.assertNotEqual(career_definition.intent, "personal_goal_advice")
        self.assertEqual(greeting_fragment.intent, "unknown")
        self.assertEqual(bare_action.intent, "unknown")

    def test_functional_grammar_source_contains_no_transcript_phrase_table(self) -> None:
        source = inspect.getsource(grammar_module)

        self.assertNotIn("do you always tell people the same thing", source)
        self.assertNotIn("do you like repeating yourself", source)
        self.assertNotIn("can you help me grow in my career", source)
        self.assertNotIn("i want to grow in my career", source)

    def test_memory_centric_route_handles_realistic_examples_without_cloud(self) -> None:
        reports = {
            report.strategy: report
            for report in compare_assistant_mvp_directions()
        }
        memory = reports["memory_centric_local_triage"]

        self.assertEqual(memory.cases, 8)
        self.assertEqual(memory.local_or_device_resolved, 8)
        self.assertEqual(memory.cloud_handoffs, 0)
        self.assertEqual(memory.external_fetches, 0)
        self.assertEqual(memory.clarifications, 0)
        self.assertEqual(memory.privacy_exposures, 0)
        self.assertEqual(memory.memory_uses, 7)
        self.assertEqual(memory.local_resolution_rate, 1.0)
        self.assertEqual(
            [decision.route for decision in memory.decisions],
            [
                "local_answer",
                "cached_tool",
                "local_answer",
                "device_action",
                "local_answer",
                "local_answer",
                "local_answer",
                "device_action",
            ],
        )

    def test_alternative_mvp_directions_lose_on_same_assistant_examples(self) -> None:
        reports = {
            report.strategy: report
            for report in compare_assistant_mvp_directions()
        }

        self.assertEqual(reports["thin_tools_plus_cloud"].local_or_device_resolved, 4)
        self.assertEqual(reports["thin_tools_plus_cloud"].cloud_handoffs, 3)
        self.assertEqual(reports["thin_tools_plus_cloud"].clarifications, 1)
        self.assertEqual(reports["cloud_first_assistant"].local_or_device_resolved, 2)
        self.assertEqual(reports["cloud_first_assistant"].cloud_handoffs, 5)
        self.assertEqual(reports["cloud_first_assistant"].external_fetches, 1)
        self.assertEqual(reports["secondary_lexical_baseline"].local_or_device_resolved, 1)
        self.assertEqual(reports["secondary_lexical_baseline"].cloud_handoffs, 6)

    def test_cache_and_memory_presence_changes_route_without_retraining(self) -> None:
        thin_profile = LocalAssistantProfile(
            weekly_weather={},
            story_models={},
            facts={},
            contacts={},
            media_library=(),
        )
        router = OnDeviceAssistantRouter(thin_profile)

        weather = router.handle("What is the weather today?")
        story = router.handle("Tell me a story.")
        about_me = router.handle("Tell me something about myself.")
        contact = router.handle("I need to talk to someone.")
        song = router.handle("Play a song for me.")

        self.assertEqual(weather.route, "external_fetch")
        self.assertEqual(weather.reason, "weather_cache_miss")
        self.assertEqual(story.route, "cloud_handoff")
        self.assertEqual(story.reason, "missing_story_model")
        self.assertEqual(about_me.route, "clarify")
        self.assertEqual(about_me.reason, "personal_memory_empty")
        self.assertEqual(contact.route, "clarify")
        self.assertEqual(contact.reason, "missing_contact")
        self.assertEqual(song.route, "clarify")
        self.assertEqual(song.reason, "empty_media_library")

    def test_history_question_is_not_misrouted_as_story_inventory(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        decision = router.handle("Give me a detailed history of ancient Rome.")

        self.assertEqual(decision.route, "cloud_handoff")
        self.assertEqual(decision.intent, "open_domain")
        self.assertEqual(decision.reason, "understood_open_domain")

    def test_identity_questions_are_local_self_model_not_cloud_unknowns(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        for utterance in ("Who are you?", "What is your name?", "You don't know who you are?"):
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)

                self.assertEqual(decision.intent, "assistant_identity")
                self.assertEqual(decision.route, "local_answer")
                self.assertEqual(decision.reason, "self_model_identity")
                self.assertFalse(decision.cloud_needed)
                self.assertIn("self_model.name", decision.evidence_keys)

    def test_identity_debug_uses_uol_composition_not_phrase_shortcut(self) -> None:
        parsed = parse_assistant_debug_frame("who are you").to_dict()

        self.assertEqual(parsed["tokens"], ["who", "are", "you"])
        self.assertEqual(parsed["uol"]["subject"], "assistant")
        self.assertEqual(parsed["uol"]["action"], "identify")
        self.assertEqual(parsed["uol"]["object"], "self_model")
        self.assertEqual(parsed["secondary_meaning_hints"], [])
        self.assertNotIn("vocabulary_hits", parsed)
        self.assertNotIn("bounded_vocabulary_hits", parsed["nlp"])
        self.assertNotIn("routing_basis", parsed["chat_frame"])
        self.assertEqual(parsed["nlp"]["primary_parse_basis"], "uol_chat_frame")
        self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "token_role_relation")
        self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["pattern"], "who_copula_second_person")
        self.assertEqual(
            parsed["nlp"]["primary_domain_evidence"]["frame_registry"],
            "melm.assistant_frame_registry.v1",
        )
        self.assertEqual(
            parsed["nlp"]["primary_domain_evidence"]["frame_id"],
            "assistant_identity.who_copula_second_person",
        )
        self.assertEqual(
            parsed["nlp"]["primary_domain_evidence"]["source_policy"],
            "primary_uol_chatframe_only",
        )
        self.assertEqual(parsed["nlp"]["compositional_parse"]["pattern"], "who_copula_second_person")
        self.assertEqual(
            parsed["nlp"]["compositional_parse"]["frame_registry"],
            "melm.assistant_frame_registry.v1",
        )
        self.assertEqual(parsed["nlp"]["compositional_parse"]["frame_match"]["source"], "token_role_relation")
        self.assertEqual(parsed["chat_frame"]["frame_registry"], "melm.assistant_frame_registry.v1")
        self.assertEqual(parsed["chat_frame"]["frame_id"], "assistant_identity.who_copula_second_person")
        self.assertEqual(parsed["chat_frame"]["frame_source_policy"], "primary_uol_chatframe_only")
        self.assertEqual(
            [item["role"] for item in parsed["nlp"]["token_roles"]],
            ["interrogative", "relation", "deictic_pronoun"],
        )
        self.assertEqual(
            parsed["uol"]["slot_sources"]["subject"]["source"],
            "second_person_deixis_resolved_to_assistant",
        )
        self.assertEqual(
            parsed["uol"]["slot_sources"]["action"]["source"],
            "identity_composition:who_copula_second_person",
        )
        self.assertEqual(
            parsed["uol"]["slot_sources"]["object"]["source"],
            "self_model_from_identity_composition:who_copula_second_person",
        )
        self.assertIn("composition:who_copula_second_person", parsed["chat_frame"]["primary_routing_basis"])
        self.assertIn(
            "frame_registry:melm.assistant_frame_registry.v1",
            parsed["chat_frame"]["primary_routing_basis"],
        )
        self.assertIn(
            "frame_id:assistant_identity.who_copula_second_person",
            parsed["chat_frame"]["primary_routing_basis"],
        )
        self.assertIn("source_policy:primary_uol_chatframe_only", parsed["chat_frame"]["primary_routing_basis"])
        self.assertIn("token_role:who:interrogative_identity", parsed["chat_frame"]["primary_routing_basis"])
        self.assertEqual(parsed["chat_frame"]["secondary_debug_hints"], [])
        self.assertIn("question_mapped_by_semantic_parse_not_question_mark", parsed["notes"])
        self.assertNotIn("statement_mapped_by_semantic_parse_not_question_mark", parsed["notes"])
        self.assertNotIn("secondary_routing_hints", parsed["chat_frame"])
        self.assertNotIn("vocabulary_hits:who are you", parsed["chat_frame"]["primary_routing_basis"])
        self.assertNotIn("secondary_meaning_hints:who are you", parsed["chat_frame"]["primary_routing_basis"])
        by_stage = {stage["stage"]: stage["output"] for stage in parsed["mapping"]}
        self.assertNotIn("routing_basis", by_stage["chat_frame"])
        self.assertEqual(by_stage["chat_frame"]["frame_id"], "assistant_identity.who_copula_second_person")
        self.assertEqual(by_stage["chat_frame"]["frame_source_policy"], "primary_uol_chatframe_only")

    def test_identity_challenge_uses_nested_clause_composition(self) -> None:
        parsed = parse_assistant_debug_frame("wow you don't know who you are?").to_dict()

        self.assertEqual(parsed["chat_frame"]["intent"], "assistant_identity")
        self.assertEqual(parsed["uol"]["speech_act"], "challenge")
        self.assertEqual(parsed["uol"]["object"], "self_model")
        self.assertEqual(parsed["nlp"]["compositional_parse"]["pattern"], "who_copula_second_person")
        self.assertIn("identity_challenge_detected", parsed["uol"]["notes"])
        self.assertEqual(parsed["nlp"]["unknown_tokens"], [])

    def test_identity_purpose_variants_report_actual_token_role_composition(self) -> None:
        cases = {
            "Why are you here?": (
                "why_copula_second_person_here",
                (
                    "token_role:why:purpose_question",
                    "token_role:are:state_relation",
                    "token_role:you:assistant_deixis",
                    "token_role:here:runtime_purpose_context",
                ),
            ),
            "What is your purpose?": (
                "what_copula_possessive_purpose",
                (
                    "token_role:what:attribute_question",
                    "token_role:is:state_relation",
                    "token_role:your:assistant_possessive",
                    "token_role:purpose:self_model_attribute",
                ),
            ),
        }
        for utterance, (pattern, basis) in cases.items():
            with self.subTest(utterance=utterance):
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(parsed["chat_frame"]["intent"], "assistant_identity")
                self.assertEqual(parsed["uol"]["action"], "describe_purpose")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "token_role_relation")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["pattern"], pattern)
                self.assertIn(f"composition:{pattern}", parsed["chat_frame"]["primary_routing_basis"])
                for item in basis:
                    self.assertIn(item, parsed["chat_frame"]["primary_routing_basis"])
                self.assertNotIn("secondary_routing_hints", parsed["chat_frame"])
                self.assertFalse(
                    any(
                        item.startswith("secondary_meaning_hints:")
                        or item.startswith("vocabulary_hits:")
                        for item in parsed["chat_frame"]["primary_routing_basis"]
                    )
                )

    def test_identity_frames_accept_modifiers_without_becoming_task_shortcuts(self) -> None:
        identity_cases = {
            "Who exactly are you on this device?": ("who_copula_second_person", "identify"),
            "What kind of assistant are you?": ("what_copula_second_person", "identify"),
            "What can you help me with?": ("modal_second_person_capability", "describe_capabilities"),
        }
        task_cases = (
            "Who are you calling?",
            "What are you doing about dinner?",
            "What can you do about dinner?",
        )
        for utterance, (pattern, action) in identity_cases.items():
            with self.subTest(utterance=utterance):
                decision = OnDeviceAssistantRouter(LocalAssistantProfile()).handle(utterance)
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(decision.intent, "assistant_identity")
                self.assertEqual(decision.route, "local_answer")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "token_role_relation")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["pattern"], pattern)
                self.assertEqual(parsed["uol"]["action"], action)
                self.assertEqual(parsed["nlp"]["unknown_tokens"], [])
                self.assertFalse(
                    any(
                        item.startswith("secondary_meaning_hints:")
                        or item.startswith("vocabulary_hits:")
                        for item in parsed["chat_frame"]["primary_routing_basis"]
                    )
                )
        for utterance in task_cases:
            with self.subTest(utterance=utterance):
                decision = OnDeviceAssistantRouter(LocalAssistantProfile()).handle(utterance)
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertNotEqual(decision.intent, "assistant_identity")
                self.assertNotEqual(parsed["chat_frame"]["intent"], "assistant_identity")
                self.assertNotEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "token_role_relation")

    def test_secondary_meaning_hints_use_token_boundaries_not_substrings(self) -> None:
        parsed = parse_assistant_debug_frame("What is the weather today?").to_dict()

        self.assertEqual(parsed["chat_frame"]["intent"], "weather")
        self.assertEqual(parsed["secondary_meaning_hints"], ["weather"])
        self.assertEqual(parsed["nlp"]["compositional_parse"]["source"], "slot_role_relation")
        self.assertEqual(parsed["nlp"]["compositional_parse"]["pattern"], "question_weather_cache")
        self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")
        self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["pattern"], "question_weather_cache")
        self.assertEqual(parsed["nlp"]["secondary_domain_hints"], parsed["nlp"]["domain_hints"])
        self.assertIn("composition:question_weather_cache", parsed["chat_frame"]["primary_routing_basis"])
        self.assertEqual(parsed["nlp"]["secondary_lexical_evidence"][0]["basis"], "secondary_token_sequence")
        self.assertEqual(parsed["chat_frame"]["secondary_debug_hints"], ["secondary_debug_hint:weather"])
        self.assertNotIn("secondary_routing_hints", parsed["chat_frame"])
        self.assertNotIn("meal_suggestion", parsed["nlp"]["domain_hints"])
        self.assertNotIn("eat", parsed["secondary_meaning_hints"])
        self.assertNotIn("secondary_meaning_hints:weather", parsed["chat_frame"]["primary_routing_basis"])
        self.assertEqual(parsed["uol"]["slot_sources"]["object"]["source"], "weather_question_slots")

    def test_non_identity_debug_uses_slot_composition_not_primary_phrase_hits(self) -> None:
        cases = {
            "Show your memory ledger.": ("assistant_status", "self_status_ledger_question"),
            "Tell me a story.": ("story", "request_story_inventory"),
            "Play a song for me.": ("media_playback", "command_media_playback"),
            "I need to talk to someone.": ("social_contact", "command_trusted_contact"),
            "What should I eat today?": ("meal_suggestion", "request_meal_suggestion"),
        }
        for utterance, (intent, pattern) in cases.items():
            with self.subTest(utterance=utterance):
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(parsed["chat_frame"]["intent"], intent)
                self.assertEqual(parsed["nlp"]["compositional_parse"]["source"], "slot_role_relation")
                self.assertEqual(parsed["nlp"]["compositional_parse"]["pattern"], pattern)
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["pattern"], pattern)
                self.assertEqual(
                    parsed["nlp"]["primary_domain_evidence"]["frame_registry"],
                    "melm.assistant_frame_registry.v1",
                )
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["frame_id"], f"{intent}.{pattern}")
                self.assertEqual(parsed["chat_frame"]["frame_id"], f"{intent}.{pattern}")
                self.assertIn(f"composition:{pattern}", parsed["chat_frame"]["primary_routing_basis"])
                self.assertIn(f"frame_id:{intent}.{pattern}", parsed["chat_frame"]["primary_routing_basis"])
                self.assertTrue(parsed["uol"]["slot_sources"]["object"]["source"])
                self.assertFalse(
                    any(
                        item.startswith("secondary_meaning_hints:")
                        for item in parsed["chat_frame"]["primary_routing_basis"]
                    )
                )

    def test_all_current_handled_intents_expose_primary_uol_evidence_not_secondary_routes(self) -> None:
        cases = {
            "Who are you?": ("assistant_identity", "token_role_relation"),
            "What have you done so far?": ("assistant_status", "slot_role_relation"),
            "Tell me a story.": ("story", "slot_role_relation"),
            "What is the weather today?": ("weather", "slot_role_relation"),
            "Should I go to school naked?": ("common_sense_safety", "slot_role_relation"),
            "Play a song for me.": ("media_playback", "slot_role_relation"),
            "What should I do to improve my health?": ("health_advice", "slot_role_relation"),
            "Tell me something about myself.": ("personal_memory", "slot_role_relation"),
            "What did we talk about earlier?": ("autobiographical_memory", "slot_role_relation"),
            "What should I eat today?": ("meal_suggestion", "slot_role_relation"),
            "I need to talk to someone.": ("social_contact", "slot_role_relation"),
        }
        for utterance, (intent, source) in cases.items():
            with self.subTest(utterance=utterance):
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(parsed["chat_frame"]["intent"], intent)
                self.assertEqual(parsed["nlp"]["primary_parse_basis"], "uol_chat_frame")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], source)
                self.assertTrue(parsed["nlp"]["primary_domain_evidence"]["pattern"])
                self.assertEqual(parsed["chat_frame"]["frame_registry"], "melm.assistant_frame_registry.v1")
                self.assertTrue(parsed["chat_frame"]["frame_id"])
                self.assertEqual(parsed["chat_frame"]["frame_source_policy"], "primary_uol_chatframe_only")
                self.assertFalse(
                    any(
                        item.startswith("secondary_meaning_hints:")
                        or item.startswith("vocabulary_hits:")
                        for item in parsed["chat_frame"]["primary_routing_basis"]
                    )
                )
                if intent == "assistant_identity":
                    self.assertEqual(parsed["nlp"]["secondary_domain_hints"], {})
                else:
                    self.assertEqual(parsed["nlp"]["secondary_domain_hints"], parsed["nlp"]["domain_hints"])

    def test_primary_intent_classifier_does_not_call_secondary_phrase_tables(self) -> None:
        source = inspect.getsource(router_module._classify_intent_from_uol_slots)

        self.assertNotIn("_has_any_marker", source)
        self.assertNotIn("_has_any_word", source)
        self.assertNotIn("_secondary_meaning", source)
        self.assertNotIn("_secondary_debug", source)
        self.assertNotIn("calm piano", source)
        self.assertNotIn("rain sounds", source)

    def test_primary_compositions_are_owned_by_frame_registry(self) -> None:
        source = inspect.getsource(router_module.AssistantFrameRegistry)

        self.assertIn("class AssistantFrameRegistry", source)
        self.assertIn("AssistantFrameMatch", source)
        self.assertIn("primary_uol_chatframe_only", source)
        self.assertIn("_identity_composition", source)
        self.assertIn("_self_status_composition", source)
        self.assertIn("_semantic_slot_composition", source)
        self.assertNotIn("_secondary_meaning", source)
        self.assertNotIn("_secondary_debug", source)
        self.assertNotIn("_has_marker", source)

    def test_secondary_hint_groups_do_not_contain_identity_routes(self) -> None:
        hints = router_module._secondary_meaning_hint_groups()

        self.assertNotIn("assistant_identity", hints)
        self.assertNotIn("who are you", "\n".join(",".join(values) for values in hints.values()))
        self.assertNotIn("what is your name", "\n".join(",".join(values) for values in hints.values()))

    def test_secondary_hint_groups_are_not_request_surface_phrase_tables(self) -> None:
        hints = router_module._secondary_meaning_hint_groups()
        request_shaped_phrases = {
            "what have you done",
            "what do you need",
            "what are you missing",
            "show your ledger",
            "about me",
            "about myself",
            "who am i",
            "talk to someone",
            "reach caregiver",
        }
        flattened = [marker for markers in hints.values() for marker in markers]

        self.assertTrue(flattened)
        self.assertFalse([marker for marker in flattened if " " in marker])
        for phrase in request_shaped_phrases:
            self.assertNotIn(phrase, flattened)

    def test_debug_output_does_not_label_composition_as_phrase_hit(self) -> None:
        parsed = parse_assistant_debug_frame("Who are you?").to_dict()
        serialized_keys = "\n".join(
            key
            for section in (parsed, parsed["nlp"], parsed["chat_frame"])
            for key in section.keys()
        )

        self.assertNotIn("hit", serialized_keys.lower())
        self.assertEqual(parsed["nlp"]["secondary_domain_hints"], {})
        self.assertEqual(parsed["chat_frame"]["secondary_debug_hints"], [])
        self.assertNotIn("secondary_routing_hints", parsed["chat_frame"])

    def test_primary_intent_helpers_do_not_call_phrase_table_helpers(self) -> None:
        helpers = (
            router_module._is_story_request,
            router_module._story_request_question,
            router_module._is_weather_request,
            router_module._is_common_sense_safety_request,
            router_module._is_media_request,
            router_module._is_health_advice_request,
            router_module._is_social_contact_request,
            router_module._is_personal_memory_frame,
            router_module._is_autobiographical_debug_request,
            router_module._is_meal_suggestion_request,
        )
        source = "\n".join(inspect.getsource(helper) for helper in helpers)

        self.assertNotIn("_has_marker", source)
        self.assertNotIn("_has_any_marker", source)
        self.assertNotIn("_has_any_word", source)
        self.assertNotIn("_secondary_meaning", source)
        self.assertNotIn("_secondary_debug", source)

    def test_post_route_slot_helpers_do_not_smuggle_marker_shortcuts(self) -> None:
        helpers = (
            router_module.OnDeviceAssistantRouter._safety,
            router_module.OnDeviceAssistantRouter._health,
            router_module.OnDeviceAssistantRouter._personal_memory,
            router_module._media_object_from_request_tokens,
            router_module._is_broad_personal_memory_request,
            router_module._is_routine_memory_request,
            router_module._is_household_memory_request,
            router_module._is_child_memory_request,
            router_module._personal_memory_object_from_text,
            router_module._object_source,
        )
        source = "\n".join(inspect.getsource(helper) for helper in helpers)

        self.assertNotIn("_has_marker", source)
        self.assertNotIn("_has_any_marker", source)
        self.assertNotIn("_has_any_word", source)
        self.assertNotIn("_secondary_meaning", source)
        self.assertNotIn("_secondary_debug", source)

    def test_self_status_debug_uses_composition_not_static_phrase_list(self) -> None:
        cases = {
            "Show your memory ledger.": ("self_status_ledger_question", "runtime_status"),
            "How much memory do you have?": ("self_status_count_question", "runtime_status"),
            "Are you using cloud?": ("self_status_boundary_question", "runtime_status"),
            "What should you build next?": ("self_status_planning_question", "next_steps"),
        }
        for utterance, (pattern, object_value) in cases.items():
            with self.subTest(utterance=utterance):
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(parsed["chat_frame"]["intent"], "assistant_status")
                self.assertEqual(parsed["uol"]["object"], object_value)
                self.assertEqual(parsed["nlp"]["compositional_parse"]["schema"], "melm.self_status_uol_composition.v1")
                if utterance.startswith("Show"):
                    self.assertTrue(parsed["nlp"]["imperative_like"])
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["pattern"], pattern)
                self.assertIn(f"composition:{pattern}", parsed["chat_frame"]["primary_routing_basis"])
                self.assertTrue(parsed["nlp"]["token_roles"])
                self.assertFalse(
                    any(
                        item.startswith("secondary_meaning_hints:")
                        or item.startswith("vocabulary_hits:")
                        for item in parsed["chat_frame"]["primary_routing_basis"]
                    )
                )

    def test_self_status_composition_is_not_exact_surface_phrase_table(self) -> None:
        source = inspect.getsource(router_module._self_status_composition)

        self.assertNotIn("what have you done", source)
        self.assertNotIn("what do you need next", source)
        self.assertNotIn("show your ledger", source)
        self.assertNotIn("_has_any_marker", source)

    def test_secondary_lexical_baseline_does_not_borrow_uol_classifier(self) -> None:
        source = inspect.getsource(router_module._secondary_lexical_baseline_decision)

        self.assertNotIn("_classify_intent(_normalize", source)
        self.assertIn("_classify_intent_for_secondary_lexical_baseline", source)

    def test_private_cloud_policy_runs_after_uol_personal_memory_frame(self) -> None:
        source = inspect.getsource(router_module.OnDeviceAssistantRouter._route_impl)

        self.assertLess(
            source.index("_classify_intent_from_uol_slots"),
            source.index("_is_private_cloud_export_request"),
        )
        self.assertIn('intent == "personal_memory"', source)

    def test_identity_composition_is_not_exact_surface_phrase_table(self) -> None:
        who_source = inspect.getsource(router_module._matches_who_identity_frame)
        name_source = inspect.getsource(router_module._matches_name_identity_frame)
        kind_source = inspect.getsource(router_module._matches_kind_identity_frame)
        purpose_source = inspect.getsource(router_module._purpose_identity_frame)
        self_description_source = inspect.getsource(router_module._matches_self_description_frame)
        bare_name = parse_assistant_debug_frame("Your name.").to_dict()
        question_name = parse_assistant_debug_frame("Your name?").to_dict()
        describe_self = parse_assistant_debug_frame("Describe yourself.").to_dict()

        self.assertNotIn('("who", "are", "you")', who_source)
        self.assertNotIn('("who", "you", "are")', who_source)
        self.assertNotIn('("your", "name")', name_source)
        self.assertNotIn('("what", "are", "you")', kind_source)
        self.assertNotIn('("what", "is", "you")', kind_source)
        self.assertNotIn('("what", "is", "your", "purpose")', purpose_source)
        self.assertNotIn('("why", "are", "you", "here")', purpose_source)
        self.assertNotIn("tell me about yourself", self_description_source)
        self.assertNotIn("tell_about_reflexive_second_person", inspect.getsource(router_module._identity_composition))
        self.assertFalse(hasattr(router_module, "_has_who_identity_clause"))
        self.assertFalse(hasattr(router_module, "_has_name_identity_clause"))
        self.assertFalse(hasattr(router_module, "_has_kind_identity_clause"))
        self.assertEqual(bare_name["chat_frame"]["intent"], "unknown")
        self.assertEqual(bare_name["nlp"]["primary_domain_evidence"]["source"], "no_local_composition")
        self.assertEqual(question_name["chat_frame"]["intent"], "assistant_identity")
        self.assertEqual(question_name["nlp"]["primary_domain_evidence"]["source"], "token_role_relation")
        self.assertEqual(describe_self["chat_frame"]["intent"], "assistant_identity")
        self.assertTrue(describe_self["nlp"]["imperative_like"])
        self.assertEqual(describe_self["uol"]["speech_act"], "request")
        self.assertEqual(describe_self["nlp"]["compositional_parse"]["uol_projection"]["speech_act"], "request")
        self.assertEqual(describe_self["nlp"]["primary_domain_evidence"]["pattern"], "request_reflexive_second_person_description")
        self.assertIn("token_role:describe:request", describe_self["chat_frame"]["primary_routing_basis"])
        self.assertIn("request_mapped_by_semantic_parse_not_question_mark", describe_self["notes"])
        self.assertNotIn("statement_mapped_by_semantic_parse_not_question_mark", describe_self["notes"])

    def test_topic_nouns_do_not_route_as_tool_or_action_shortcuts(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        music_theory = router.handle("Can you explain music theory?")
        bought_phone = router.handle("I bought a phone yesterday.")
        latest_news = router.handle("Tell me the latest news about Mars.")
        fun_volcanoes = router.handle("Tell me something fun about volcanoes.")
        valid_phone_action = router.handle("Phone mom please.")

        self.assertEqual(music_theory.intent, "open_domain")
        self.assertEqual(music_theory.route, "cloud_handoff")
        self.assertEqual(music_theory.reason, "understood_open_domain")
        self.assertEqual(bought_phone.intent, "open_domain")
        self.assertEqual(bought_phone.route, "cloud_handoff")
        self.assertEqual(latest_news.intent, "open_domain")
        self.assertEqual(latest_news.route, "cloud_handoff")
        self.assertEqual(latest_news.reason, "understood_open_domain")
        self.assertEqual(fun_volcanoes.intent, "open_domain")
        self.assertEqual(fun_volcanoes.route, "cloud_handoff")
        self.assertEqual(valid_phone_action.intent, "social_contact")
        self.assertEqual(valid_phone_action.route, "device_action")

    def test_bare_domain_words_do_not_route_without_chatframe_relation(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        structured_cases = (
            "What is a story?",
            "What is bedtime?",
            "What is health?",
            "What is Doctor Strange?",
            "What is medicine?",
            "What is dinner?",
            "What is a naked mole rat?",
            "What is a routine?",
            "What is a household?",
            "What is family?",
            "Doctor Strange is a movie.",
            "I took medicine yesterday.",
            "I eat dinner every day.",
            "Call of Duty is a game.",
            "My family likes chess.",
        )

        for utterance in structured_cases:
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(decision.intent, "open_domain")
                self.assertEqual(decision.route, "cloud_handoff")
                self.assertEqual(parsed["chat_frame"]["intent"], "open_domain")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "weighted_functional_relation")

        fragment = router.handle("A naked mole rat fact.")
        self.assertEqual(fragment.intent, "unknown")
        self.assertEqual(fragment.route, "cloud_handoff")

    def test_weather_concepts_do_not_route_as_cache_shortcuts(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        concept_cases = (
            "What is weather?",
            "What is temperature?",
            "How does weather work?",
        )

        for utterance in concept_cases:
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(decision.intent, "open_domain")
                self.assertEqual(decision.route, "cloud_handoff")
                self.assertEqual(parsed["chat_frame"]["intent"], "open_domain")
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "weighted_functional_relation")
                self.assertFalse(parsed["chat_frame"]["can_answer_locally"])

        live_weather = router.handle("What is the weather?")
        parsed_live_weather = parse_assistant_debug_frame("What is the weather?").to_dict()

        self.assertEqual(live_weather.intent, "weather")
        self.assertEqual(live_weather.route, "cached_tool")
        self.assertEqual(parsed_live_weather["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")

    def test_owned_memory_relations_still_route_without_exact_phrase_shortcuts(self) -> None:
        cases = {
            "What is my morning routine?": ("personal_memory", "routine_memory"),
            "What do you know about our household?": ("personal_memory", "household_memory"),
            "Who uses this device?": ("personal_memory", "household_memory"),
            "Tell me about myself.": ("personal_memory", "user_profile"),
        }
        for utterance, (intent, object_value) in cases.items():
            with self.subTest(utterance=utterance):
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(parsed["chat_frame"]["intent"], intent)
                self.assertEqual(parsed["uol"]["object"], object_value)
                self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")
                self.assertFalse(
                    any(
                        item.startswith("secondary_meaning_hints:")
                        or item.startswith("vocabulary_hits:")
                        for item in parsed["chat_frame"]["primary_routing_basis"]
                    )
                )

    def test_noncanonical_requests_still_route_through_slot_relations(self) -> None:
        cases = {
            "Read me a story.": ("story", "request_story_inventory"),
            "Tell me a tale.": ("story", "request_story_inventory"),
            "What healthy thing can I do tonight?": ("health_advice", "request_bounded_health_advice"),
            "I cannot breathe.": ("health_advice", "request_bounded_health_advice"),
            "Should I go outside naked?": ("common_sense_safety", "judgement_safety_policy"),
            "What should I cook for dinner?": ("meal_suggestion", "request_meal_suggestion"),
            "Call mom please.": ("social_contact", "command_trusted_contact"),
            "Please call my daughter.": ("social_contact", "command_trusted_contact"),
        }

        for utterance, (intent, pattern) in cases.items():
            with self.subTest(utterance=utterance):
                parsed = parse_assistant_debug_frame(utterance).to_dict()

                self.assertEqual(parsed["chat_frame"]["intent"], intent)
                self.assertEqual(parsed["nlp"]["compositional_parse"]["source"], "slot_role_relation")
                self.assertEqual(parsed["nlp"]["compositional_parse"]["pattern"], pattern)
                self.assertFalse(
                    any(
                        item.startswith("secondary_meaning_hints:")
                        or item.startswith("vocabulary_hits:")
                        for item in parsed["chat_frame"]["primary_routing_basis"]
                    )
                )

    def test_about_me_memory_uses_about_object_not_response_target(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile(facts={"favorite_color": "green"}))

        about_me = router.handle("Tell me about me.")
        volcanoes = router.handle("Tell me something fun about volcanoes.")

        self.assertEqual(about_me.intent, "personal_memory")
        self.assertEqual(about_me.route, "local_answer")
        self.assertEqual(about_me.reason, "personal_memory_summary")
        self.assertEqual(volcanoes.intent, "open_domain")
        self.assertEqual(volcanoes.route, "cloud_handoff")

    def test_advice_question_with_you_think_is_not_self_status_shortcut(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        health = router.handle("What do you think I should do to improve my health?")
        meal = router.handle("What do you think I should eat today?")
        status = router.handle("What did you do?")

        self.assertEqual(health.intent, "health_advice")
        self.assertEqual(health.route, "local_answer")
        self.assertEqual(meal.intent, "meal_suggestion")
        self.assertEqual(meal.route, "local_answer")
        self.assertEqual(status.intent, "assistant_status")

    def test_task_questions_do_not_collapse_to_self_identity_or_status_shortcuts(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        health = router.handle("What can you do to improve my health?")
        dinner = router.handle("What can you do about dinner?")
        cloud_concept = router.handle("Can you explain cloud computing?")
        story_concept = router.handle("Can you explain story structure?")
        weather_concept = router.handle("Can you explain weather systems?")

        self.assertEqual(health.intent, "health_advice")
        self.assertEqual(health.route, "local_answer")
        self.assertEqual(dinner.intent, "open_domain")
        self.assertEqual(dinner.route, "cloud_handoff")
        self.assertEqual(cloud_concept.intent, "open_domain")
        self.assertEqual(cloud_concept.route, "cloud_handoff")
        self.assertEqual(story_concept.intent, "open_domain")
        self.assertEqual(story_concept.route, "cloud_handoff")
        self.assertEqual(weather_concept.intent, "open_domain")
        self.assertEqual(weather_concept.route, "cloud_handoff")

    def test_private_cloud_export_maps_to_memory_frame_not_hidden_preparse_shortcut(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile(facts={"favorite_color": "green"}))

        decision = router.handle("Send my favorite color to the cloud.")
        parsed = parse_assistant_debug_frame("Send my favorite color to the cloud.").to_dict()
        location = router.handle("Upload where I live to the cloud.")
        location_parsed = parse_assistant_debug_frame("Upload where I live to the cloud.").to_dict()

        self.assertEqual(decision.intent, "personal_memory")
        self.assertEqual(decision.route, "cloud_handoff")
        self.assertEqual(decision.reason, "private_memory_cloud_request")
        self.assertEqual(decision.evidence_keys, ("facts.favorite_color",))
        self.assertEqual(parsed["chat_frame"]["intent"], "personal_memory")
        self.assertEqual(parsed["chat_frame"]["route"], "cloud_handoff")
        self.assertEqual(parsed["chat_frame"]["reason"], "private_memory_cloud_request")
        self.assertTrue(parsed["chat_frame"]["needs_cloud"])
        self.assertFalse(parsed["chat_frame"]["can_answer_locally"])
        self.assertEqual(parsed["uol"]["action"], "send")
        self.assertEqual(parsed["uol"]["object"], "facts.favorite_color")
        self.assertEqual(parsed["uol"]["target"], "external_cloud_model")
        self.assertEqual(parsed["uol"]["slot_sources"]["object"]["source"], "owned_fact_memory_slots")
        self.assertEqual(parsed["uol"]["slot_sources"]["target"]["source"], "policy_boundary_target")
        self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")
        self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["pattern"], "request_private_memory_cloud_boundary")
        self.assertEqual(parsed["nlp"]["secondary_hint_policy"], "debug_only_never_primary_route")
        self.assertIn("token_role:target:external_cloud_model", parsed["chat_frame"]["primary_routing_basis"])
        self.assertFalse(
            any(
                item.startswith("secondary_meaning_hints:")
                or item.startswith("vocabulary_hits:")
                for item in parsed["chat_frame"]["primary_routing_basis"]
            )
        )
        self.assertEqual(location.intent, "personal_memory")
        self.assertEqual(location.route, "cloud_handoff")
        self.assertEqual(location.reason, "private_memory_cloud_request")
        self.assertEqual(location.evidence_keys, ("profile.location",))
        self.assertEqual(location_parsed["chat_frame"]["intent"], "personal_memory")
        self.assertEqual(location_parsed["uol"]["object"], "profile.location")
        self.assertEqual(location_parsed["uol"]["target"], "external_cloud_model")
        self.assertEqual(location_parsed["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")

    def test_secondary_domain_hints_do_not_mask_unknown_complexity(self) -> None:
        story_concept = parse_assistant_debug_frame("What is a story?").to_dict()
        doctor_concept = parse_assistant_debug_frame("What is Doctor Strange?").to_dict()
        local_concept = parse_assistant_debug_frame("Can you explain local cooking?").to_dict()

        self.assertEqual(story_concept["chat_frame"]["intent"], "open_domain")
        self.assertEqual(story_concept["nlp"]["secondary_domain_hints"], {"story": ["story"]})
        self.assertIn("story", story_concept["nlp"]["semantic_unknown_tokens"])
        self.assertEqual(story_concept["chat_frame"]["secondary_debug_hints"], [])
        self.assertNotIn("secondary_routing_hints", story_concept["chat_frame"])
        self.assertEqual(story_concept["chat_frame"]["secondary_hint_policy"], "debug_only_never_primary_route")
        self.assertIn("doctor", doctor_concept["nlp"]["semantic_unknown_tokens"])
        self.assertIn("strange", doctor_concept["nlp"]["semantic_unknown_tokens"])
        self.assertNotIn("assistant_status", local_concept["nlp"]["secondary_domain_hints"])
        self.assertIn("local", local_concept["nlp"]["semantic_unknown_tokens"])

    def test_unknown_token_debug_does_not_borrow_secondary_hint_lexicon(self) -> None:
        parsed = parse_assistant_debug_frame("Tell me a bedtime story about quasar algebra.").to_dict()
        source = inspect.getsource(router_module._known_debug_tokens)

        self.assertEqual(parsed["chat_frame"]["intent"], "story")
        self.assertIn("story", parsed["secondary_meaning_hints"])
        self.assertIn("bedtime", parsed["secondary_meaning_hints"])
        self.assertEqual(parsed["nlp"]["primary_domain_evidence"]["source"], "slot_role_relation")
        self.assertIn("quasar", parsed["nlp"]["unknown_tokens"])
        self.assertIn("algebra", parsed["nlp"]["unknown_tokens"])
        self.assertNotIn("story", parsed["nlp"]["unknown_tokens"])
        self.assertNotIn("_secondary_meaning_hint_groups", source)

    def test_slot_sources_do_not_depend_on_secondary_hint_context(self) -> None:
        parsed = parse_assistant_debug_frame("Tell me a bedtime story.").to_dict()
        source = "\n".join(
            (
                inspect.getsource(router_module._slot_sources),
                inspect.getsource(router_module._object_source),
            )
        )

        self.assertEqual(parsed["uol"]["slot_sources"]["object"]["source"], "story_request_slots")
        self.assertNotIn("secondary_meaning_hints", source)
        self.assertNotIn("secondary_meaning", source)
        self.assertNotIn("object_slot_with_secondary_context", source)

    def test_child_memory_debug_maps_to_owned_fact_object_not_generic_profile(self) -> None:
        parsed = parse_assistant_debug_frame("What is my child's school?").to_dict()

        self.assertEqual(parsed["chat_frame"]["intent"], "personal_memory")
        self.assertEqual(parsed["uol"]["object"], "facts.child_school")
        self.assertEqual(parsed["uol"]["slot_sources"]["object"]["source"], "child_owned_memory_slots")
        self.assertEqual(parsed["nlp"]["compositional_parse"]["pattern"], "request_child_owned_memory")
        self.assertIn("uol_object:facts.child_school", parsed["chat_frame"]["primary_routing_basis"])
        self.assertIn("token_role:object:facts.child_school", parsed["chat_frame"]["primary_routing_basis"])

    def test_bare_play_verb_is_not_a_media_action_shortcut(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        chess = router.handle("Play chess with me.")
        song = router.handle("Play a song for me.")

        self.assertEqual(chess.intent, "open_domain")
        self.assertEqual(chess.route, "cloud_handoff")
        self.assertEqual(chess.reason, "understood_open_domain")
        self.assertEqual(song.intent, "media_playback")
        self.assertEqual(song.route, "device_action")

    def test_media_debug_object_stays_category_not_seed_title_shortcut(self) -> None:
        parsed = parse_assistant_debug_frame("Play calm piano.").to_dict()

        self.assertEqual(parsed["chat_frame"]["intent"], "media_playback")
        self.assertEqual(parsed["uol"]["object"], "music")
        self.assertNotEqual(parsed["uol"]["object"], "calm piano")
        self.assertIn("calm", parsed["nlp"]["unknown_tokens"])
        self.assertIn("uol_object:music", parsed["chat_frame"]["primary_routing_basis"])
        self.assertIn("token_role:object:music", parsed["chat_frame"]["primary_routing_basis"])

    def test_meal_request_requires_user_choice_frame_not_you_cook_shortcut(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        you_cook = router.handle("Can you cook dinner?")
        you_have = router.handle("Can you have lunch?")
        user_choice = router.handle("What can I cook for dinner?")

        self.assertEqual(you_cook.intent, "open_domain")
        self.assertEqual(you_cook.route, "cloud_handoff")
        self.assertEqual(you_have.intent, "open_domain")
        self.assertEqual(you_have.route, "cloud_handoff")
        self.assertEqual(user_choice.intent, "meal_suggestion")
        self.assertEqual(user_choice.route, "local_answer")

    def test_phone_noun_is_not_a_contact_action_shortcut(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile())

        physical_phone = router.handle("Bring my phone please.")
        phone_contact = router.handle("Phone mom please.")

        self.assertEqual(physical_phone.intent, "open_domain")
        self.assertEqual(physical_phone.route, "cloud_handoff")
        self.assertEqual(phone_contact.intent, "social_contact")
        self.assertEqual(phone_contact.route, "device_action")

    def test_generic_contact_target_uses_trusted_contact_order_not_named_shortcut(self) -> None:
        router = OnDeviceAssistantRouter(
            LocalAssistantProfile(contacts={"leo": "+234-000-LEO", "mom": "+234-000-MOM"})
        )

        decision = router.handle("I need to talk to someone.")

        self.assertEqual(decision.intent, "social_contact")
        self.assertEqual(decision.route, "device_action")
        self.assertEqual(decision.evidence_keys, ("contacts.leo",))

    def test_named_contact_routes_from_profile_memory_not_static_name_shortcut(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile(contacts={"sam": "+1-000-SAM"}))
        unknown_router = OnDeviceAssistantRouter(LocalAssistantProfile(contacts={"mom": "+234-000-MOM"}))

        decision = router.handle("Call Sam please.")
        unknown = unknown_router.handle("Call Sam please.")
        contact_helper_source = inspect.getsource(router_module._has_contact_target) + inspect.getsource(
            router_module._contact_object_from_tokens
        )

        self.assertEqual(decision.intent, "social_contact")
        self.assertEqual(decision.route, "device_action")
        self.assertEqual(decision.evidence_keys, ("contacts.sam",))
        self.assertEqual(unknown.intent, "open_domain")
        self.assertNotIn("sam", contact_helper_source.lower())
        self.assertNotIn("leo", contact_helper_source.lower())

    def test_relationship_contact_debug_object_stays_generic_without_profile_memory(self) -> None:
        parsed = parse_assistant_debug_frame("Call mom please.").to_dict()

        self.assertEqual(parsed["chat_frame"]["intent"], "social_contact")
        self.assertEqual(parsed["uol"]["object"], "relationship_contact")
        self.assertNotEqual(parsed["uol"]["object"], "mom")
        self.assertIn(
            {"index": 1, "token": "mom", "role": "uol_object", "meaning": "relationship_contact"},
            parsed["nlp"]["token_roles"],
        )
        self.assertIn("uol_object:relationship_contact", parsed["chat_frame"]["primary_routing_basis"])

    def test_private_cloud_contact_evidence_uses_profile_memory_not_static_name(self) -> None:
        router = OnDeviceAssistantRouter(LocalAssistantProfile(contacts={"sam": "+1-000-SAM"}))
        unknown_router = OnDeviceAssistantRouter(LocalAssistantProfile(contacts={"mom": "+234-000-MOM"}))

        decision = router.handle("Send Sam contact to the cloud.")
        unknown = unknown_router.handle("Send Sam contact to the cloud.")
        evidence_source = inspect.getsource(router_module._private_cloud_evidence_keys)

        self.assertEqual(decision.intent, "personal_memory")
        self.assertEqual(decision.route, "cloud_handoff")
        self.assertEqual(decision.reason, "private_memory_cloud_request")
        self.assertEqual(decision.evidence_keys, ("contacts.sam",))
        self.assertEqual(unknown.intent, "personal_memory")
        self.assertEqual(unknown.route, "cloud_handoff")
        self.assertEqual(unknown.evidence_keys, ("contacts.local",))
        self.assertNotIn("contacts.sam", evidence_source)
        self.assertNotIn("contacts.mom", evidence_source)

    def test_autobiographical_debug_uses_frame_scope_not_exact_recall_phrases(self) -> None:
        statement = parse_assistant_debug_frame("I dropped the last thing yesterday.").to_dict()
        latest_question = parse_assistant_debug_frame("What was my last question?").to_dict()
        paraphrase = parse_assistant_debug_frame("What was the last thing I asked you?").to_dict()
        source = "\n".join(
            (
                inspect.getsource(router_module.compose_autobiographical_memory_frame),
                inspect.getsource(router_module.classify_autobiographical_memory_scope),
                inspect.getsource(router_module._is_autobiographical_debug_request),
            )
        )

        self.assertEqual(statement["chat_frame"]["intent"], "unknown")
        self.assertEqual(statement["nlp"]["primary_domain_evidence"]["source"], "no_local_composition")
        self.assertEqual(latest_question["chat_frame"]["intent"], "autobiographical_memory")
        self.assertEqual(paraphrase["chat_frame"]["intent"], "autobiographical_memory")
        self.assertEqual(
            router_module.classify_autobiographical_memory_scope("What was the last thing I asked you?"),
            "latest_event",
        )
        self.assertNotIn("what was my last question", source)
        self.assertNotIn("last question", source)
        self.assertNotIn("recall_markers", source)

    def test_meal_choice_scores_inventory_weather_and_scope_not_static_combos(self) -> None:
        router = OnDeviceAssistantRouter(
            LocalAssistantProfile(
                preferences={"breakfast": "oatmeal"},
                food_inventory=("soup", "salad", "chicken"),
                weekly_weather={"today": "cool rain"},
                story_models={},
                contacts={},
            )
        )

        decision = router.handle("What should I eat for dinner?")

        self.assertEqual(decision.intent, "meal_suggestion")
        self.assertEqual(decision.route, "local_answer")
        self.assertIn("soup", decision.answer)
        self.assertIn("chicken", decision.answer)
        self.assertIn("rain", decision.answer)
        self.assertNotIn("rice and beans with plantain", decision.answer)
        self.assertNotIn("oatmeal with eggs", decision.answer)
        parsed = parse_assistant_debug_frame("What should I eat for dinner?").to_dict()
        self.assertEqual(parsed["uol"]["slot_sources"]["object"]["source"], "meal_request_slots")
        self.assertNotIn("secondary_meaning_hints:eat", parsed["chat_frame"]["primary_routing_basis"])

    def test_food_inventory_tags_use_token_sequences_not_substrings(self) -> None:
        self.assertEqual(router_module._food_tags("weatherfish"), {"food"})
        self.assertEqual(router_module._food_tags("breaded rice"), {"grain", "staple", "warm"})
        self.assertFalse(router_module._weather_suggests_warm_food("brainstorming about cooking"))
        self.assertTrue(router_module._weather_suggests_warm_food("cool rain"))

    def test_child_memory_routes_through_owned_fact_keys_not_generic_age_or_school(self) -> None:
        router = OnDeviceAssistantRouter(
            LocalAssistantProfile(
                facts={"child_age": "8", "child_school": "Bright School"},
                contacts={},
                weekly_weather={},
                story_models={},
            )
        )

        recall = router.handle("What is my child's school?")
        cloud = router.handle("Send my child's age and school to the cloud.")
        child_location = router.handle("Send my child's location to the cloud.")

        self.assertEqual(recall.intent, "personal_memory")
        self.assertEqual(recall.route, "local_answer")
        self.assertEqual(recall.reason, "personal_memory_recall")
        self.assertEqual(recall.evidence_keys, ("facts.child_school",))
        self.assertEqual(cloud.route, "cloud_handoff")
        self.assertEqual(cloud.evidence_keys, ("facts.child_age", "facts.child_school"))
        self.assertEqual(child_location.route, "cloud_handoff")
        self.assertEqual(child_location.evidence_keys, ("facts.child_location",))
        self.assertNotIn("profile.age", cloud.evidence_keys)
        self.assertNotIn("facts.school", cloud.evidence_keys)

    def test_custom_utterance_comparison_scores_same_surface_across_strategies(self) -> None:
        profile = LocalAssistantProfile(story_models={}, weekly_weather={}, facts={}, contacts={}, media_library=())
        reports = {
            report.strategy: report
            for report in compare_assistant_strategy_reports_for_utterances(
                ("Who are you?", "Tell me a story.", "What is the weather today?"),
                profile=profile,
            )
        }

        local_state = reports["local_state_router_no_lifecycle"]
        thin_tools = reports["thin_tools_plus_cloud"]
        secondary_lexical = reports["secondary_lexical_baseline"]

        self.assertEqual(local_state.cases, 3)
        self.assertEqual(local_state.local_or_device_resolved, 1)
        self.assertEqual(local_state.cloud_handoffs, 1)
        self.assertEqual(local_state.external_fetches, 1)
        self.assertEqual(thin_tools.local_or_device_resolved, 0)
        self.assertEqual(thin_tools.cloud_handoffs, 2)
        self.assertEqual(thin_tools.external_fetches, 1)
        self.assertEqual(secondary_lexical.local_or_device_resolved, 0)
        self.assertEqual(secondary_lexical.cloud_handoffs, 3)
        self.assertEqual(secondary_lexical.external_fetches, 0)
        self.assertEqual(secondary_lexical.decisions[0].intent, "unknown")
        self.assertEqual(secondary_lexical.decisions[0].reason, "intent_without_grounded_runtime")


class CapabilityManifestEnforcementMvpTests(unittest.TestCase):
    """M3 exit gate: zero capability grants via manifest enforcement."""

    def test_uninstalled_family_routes_to_open_domain(self) -> None:
        from melm.appliance.local_assistant_router import (
            replace_installed_families,
            _get_capability_manifest,
        )
        # Remove "story" from installed families
        all_installed, managed = _get_capability_manifest()
        reduced = frozenset(f for f in all_installed if f != "story")
        replace_installed_families(reduced, managed)
        try:
            router = OnDeviceAssistantRouter(LocalAssistantProfile(
                story_models={"test": "A test story."},
            ))
            # "story" is NOT installed → should route to open_domain
            decision = router.handle("Tell me a story about a dragon.")
            self.assertEqual(decision.intent, "story")
            self.assertEqual(decision.route, "open_domain")
            self.assertIn("family_not_installed", decision.reason)
        finally:
            replace_installed_families(all_installed, managed)

    def test_unmanaged_family_passes_through(self) -> None:
        from melm.appliance.local_assistant_router import (
            replace_installed_families,
            _get_capability_manifest,
        )
        all_installed, managed = _get_capability_manifest()
        replace_installed_families(all_installed, managed)
        try:
            router = OnDeviceAssistantRouter(LocalAssistantProfile())
            decision = router.handle("Who are you")
            # assistant_identity is in the default manifest → should handle normally
            self.assertEqual(decision.intent, "assistant_identity")
            self.assertEqual(decision.route, "local_answer")
        finally:
            replace_installed_families(all_installed, managed)

    def test_uninstalled_family_blocks_acquired_vocabulary(self) -> None:
        """Teaching a word that maps to an uninstalled family does not enable it."""
        from melm.appliance.local_assistant_router import (
            replace_installed_families,
            _get_capability_manifest,
        )
        all_installed, managed = _get_capability_manifest()
        reduced = frozenset(f for f in all_installed if f != "meal_suggestion")
        replace_installed_families(reduced, managed)
        try:
            router = OnDeviceAssistantRouter(LocalAssistantProfile())
            decision = router.handle("suggest a pasta recipe")
            self.assertEqual(decision.intent, "meal_suggestion")
            self.assertEqual(decision.route, "open_domain")
            self.assertIn("family_not_installed", decision.reason)
        finally:
            replace_installed_families(all_installed, managed)


if __name__ == "__main__":
    unittest.main()
