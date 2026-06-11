import unittest

from melm.appliance import (
    ChildMemoryOS,
    ChildWorldAtlas,
    FiveYearOldGroundedChat,
    MicroHybridSlm,
    StatePatch,
    apply_state_patch,
    generated_child_capability_cases,
    parse_uol,
    parse_uol_candidates,
    run_child_capability_probe,
    run_child_context_budget_probe,
)
from melm.grounding import StateSet


class GroundedChildChatMvpTests(unittest.TestCase):
    def test_state_patch_replaces_old_state_before_checking_conflicts(self) -> None:
        current = StateSet.from_mapping({"physical": ["closed"]})
        valid_patch = StatePatch(
            object_name="blue box",
            required=StateSet.from_mapping({"physical": ["closed"]}),
            remove=StateSet.from_mapping({"physical": ["closed"]}),
            add=StateSet.from_mapping({"physical": ["open"]}),
        )
        invalid_patch = StatePatch(
            object_name="blue box",
            required=StateSet.from_mapping({"physical": ["closed"]}),
            add=StateSet.from_mapping({"physical": ["open"]}),
        )

        valid = apply_state_patch(current, valid_patch)
        invalid = apply_state_patch(current, invalid_patch)

        self.assertTrue(valid.valid)
        self.assertIsNotNone(valid.next_state)
        self.assertIn("open", valid.next_state.physical)
        self.assertNotIn("closed", valid.next_state.physical)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.conflicts, [("physical", "open", "closed")])

    def test_parse_uol_keeps_subject_action_object_target_slots(self) -> None:
        parsed = parse_uol("Maya put the red block in the blue box.")

        self.assertFalse(hasattr(parsed, "rejection_code"))
        self.assertEqual(parsed.subject, "maya")
        self.assertEqual(parsed.action, "put")
        self.assertEqual(parsed.object, "red block")
        self.assertEqual(parsed.target, "blue box")

    def test_parse_candidates_score_known_slots_above_unknown_slots(self) -> None:
        known = parse_uol_candidates("Maya put the red block in the blue box.")
        harder_action = parse_uol_candidates("Maya moved the red block to the blue box.")
        unknown = parse_uol_candidates("Maya put the silver train in the blue box.")
        missing_slot = parse_uol("Open.")
        unsupported = parse_uol("Where did Maya hide the red block?")

        self.assertGreater(known[0].score, unknown[0].score)
        self.assertGreater(known[0].complexity, 0)
        self.assertGreater(harder_action[0].complexity, known[0].complexity)
        self.assertEqual(known[0].uol.parse_score, known[0].score)
        self.assertEqual(unknown[0].uol.object, "silver train")
        self.assertIn("unknown_object", unknown[0].notes)
        self.assertEqual(missing_slot.route, "reject")
        self.assertIn("below_parse_threshold", missing_slot.uol.parse_notes)
        self.assertEqual(unsupported.route, "reject")
        self.assertEqual(unsupported.uol.parse_score, 0.0)
        self.assertIn("no_parse_candidate", unsupported.uol.parse_notes)

    def test_complete_child_room_path_answers_from_validated_memory(self) -> None:
        chat = FiveYearOldGroundedChat()

        opened = chat.handle("Maya opened the blue box.")
        put = chat.handle("Maya put the red block in the blue box.")
        moved = chat.handle("Leo moved the red block to the green basket.")
        answer = chat.handle("Where is the red block?")

        self.assertEqual(opened.status, "accepted")
        self.assertEqual(put.status, "accepted")
        self.assertEqual(moved.status, "accepted")
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.answer, "The red block is in the green basket.")
        self.assertEqual(answer.evidence_event_ids, ("child_e3",))
        self.assertEqual(answer.frame.route, "memory_read")
        self.assertEqual(answer.bound_plan.schema, "melm.bound_child_plan.v1")
        self.assertIsNotNone(answer.slm_input)
        self.assertEqual(answer.slm_input.model_family, "micro_hybrid_ssm_attention")
        self.assertEqual(answer.slm_input.bound_plan_schema, "melm.bound_child_plan.v1")
        self.assertEqual(answer.slm_input.attended_evidence_event_ids, ("child_e3",))
        self.assertEqual(answer.slm_input.attention.event_ids, ("child_e3",))
        self.assertIn("Leo moved the red block", answer.slm_input.attention.context)
        self.assertIn("red block:location=green basket", answer.slm_input.compact_state)
        self.assertIn("red block:location=green basket", answer.slm_input.ssm_state.lines)
        self.assertEqual(
            [stage.stage for stage in answer.trace],
            [
                "uol_parse",
                "semantic_atlas",
                "frame_build",
                "plan_bind",
                "evidence_admission",
                "memory_os_projection",
                "hybrid_slm",
            ],
        )
        self.assertEqual(opened.trace[-1].stage, "hybrid_slm")
        self.assertEqual(put.trace[-2].stage, "memory_os_projection")
        self.assertEqual(moved.trace[-3].stage, "memory_os")

    def test_micro_components_are_swappable_and_not_hidden_globals(self) -> None:
        atlas = ChildWorldAtlas(aliases={"cube": "red block", "box": "blue box"})
        memory_os = ChildMemoryOS(
            initial_states={"blue box": StateSet.from_mapping({"physical": ["closed"]})}
        )
        renderer = MicroHybridSlm()
        chat = FiveYearOldGroundedChat(
            atlas=atlas,
            memory_os=memory_os,
            renderer=renderer,
        )

        chat.handle("Maya opened the box.")
        chat.handle("Maya put the cube in the box.")
        answer = chat.handle("Where is the cube?")

        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.answer, "The red block is in the blue box.")
        self.assertEqual(renderer.calls, 3)
        self.assertIn("blue box:physical=open", answer.slm_input.ssm_state.lines)

    def test_put_into_closed_box_fails_before_memory_write(self) -> None:
        chat = FiveYearOldGroundedChat()
        renderer = chat.renderer

        response = chat.handle("Maya put the red block in the blue box.")

        self.assertEqual(response.status, "rejected")
        self.assertEqual(response.rejection_packet.code, "target_closed")
        self.assertEqual(response.rejection_packet.stage, "memory_os")
        self.assertEqual(chat.events(), ())
        self.assertIsNone(response.slm_input)
        self.assertEqual(renderer.calls, 0)

    def test_reopening_already_open_box_fails_closed(self) -> None:
        chat = FiveYearOldGroundedChat()
        chat.handle("Maya opened the blue box.")

        response = chat.handle("Open the blue box.")

        self.assertEqual(response.status, "rejected")
        self.assertEqual(response.rejection_packet.code, "state_transition_invalid")
        self.assertEqual(response.rejection_packet.stage, "state_algebra")
        self.assertIn("missing physical: closed", response.rejection_packet.detail)
        self.assertIsNone(response.slm_input)

    def test_unknown_noun_rejects_without_memory_or_model(self) -> None:
        chat = FiveYearOldGroundedChat()
        renderer = chat.renderer

        response = chat.handle("Maya put the silver train in the blue box.")

        self.assertEqual(response.status, "rejected")
        self.assertEqual(response.rejection_packet.code, "unknown_object")
        self.assertEqual(response.rejection_packet.stage, "semantic_atlas")
        self.assertEqual(chat.events(), ())
        self.assertIsNone(response.slm_input)
        self.assertEqual(renderer.calls, 0)

    def test_false_presupposed_move_abstains_with_empty_attention_slice(self) -> None:
        chat = FiveYearOldGroundedChat()
        chat.handle("Maya opened the blue box.")
        chat.handle("Maya put the red block in the blue box.")
        chat.handle("Leo moved the red block to the green basket.")

        response = chat.handle("Did Maya move the red block to the green basket?")

        self.assertEqual(response.status, "abstained")
        self.assertEqual(response.evidence_event_ids, ())
        self.assertIsNotNone(response.slm_input)
        self.assertEqual(response.slm_input.attended_evidence_event_ids, ())
        self.assertEqual(response.slm_input.response_intent, "not_enough_evidence")
        self.assertEqual(response.trace[-2].stage, "memory_os_projection")
        self.assertEqual(response.trace[-1].stage, "hybrid_slm")
        self.assertIn("red block:location=green basket", response.slm_input.ssm_state.lines)

    def test_generated_capability_probe_measures_the_mvp_envelope(self) -> None:
        cases = generated_child_capability_cases()
        report = run_child_capability_probe(cases=cases)

        self.assertEqual(report.cases, len(cases))
        self.assertEqual(report.cases, 50)
        self.assertEqual(report.accepted, 22)
        self.assertEqual(report.answered, 4)
        self.assertEqual(report.abstained, 11)
        self.assertEqual(report.rejected, 13)
        self.assertEqual(report.average_parse_score, 0.863)
        self.assertEqual(report.average_complexity, 1.234)
        self.assertEqual(report.max_complexity, 1.93)
        self.assertEqual(
            report.by_reason,
            {
                "evidence_check": 3,
                "memory_read": 1,
                "memory_write": 16,
                "no_location_observation": 2,
                "no_matching_action_evidence": 9,
                "state_transition": 6,
                "state_transition_invalid": 2,
                "unsupported_object_action": 4,
                "unsupported_target": 4,
                "unsupported_uol_shape": 3,
            },
        )
        self.assertTrue(
            any(
                result.status == "abstained"
                and result.reason == "no_matching_action_evidence"
                for result in report.results
            )
        )
        self.assertTrue(
            any(
                result.status == "rejected"
                and result.reason == "unsupported_uol_shape"
                and result.parse_score == 0.0
                for result in report.results
            )
        )

    def test_context_budget_probe_identifies_best_next_mvp_direction(self) -> None:
        report = run_child_context_budget_probe(move_counts=(32, 128), evidence_top_k=1)
        by_kind_and_count = {
            (result.query_kind, result.move_count): result
            for result in report.results
        }

        location_32 = by_kind_and_count[("location", 32)]
        positive_32 = by_kind_and_count[("positive_evidence", 32)]
        positive_128 = by_kind_and_count[("positive_evidence", 128)]
        negative_128 = by_kind_and_count[("negative_evidence", 128)]

        self.assertEqual(location_32.status, "answered")
        self.assertGreater(location_32.budgeted_compression_ratio, 4.0)
        self.assertEqual(location_32.budgeted_evidence_count, 1)
        self.assertEqual(positive_32.status, "answered")
        self.assertEqual(positive_32.unbudgeted_evidence_count, 16)
        self.assertEqual(positive_32.budgeted_evidence_count, 1)
        self.assertEqual(positive_32.matching_evidence_count, 16)
        self.assertLess(positive_32.unbudgeted_compression_ratio, 2.0)
        self.assertGreater(positive_32.budgeted_compression_ratio, 4.0)
        self.assertGreater(
            positive_128.budgeted_compression_ratio,
            positive_32.budgeted_compression_ratio,
        )
        self.assertEqual(positive_128.matching_evidence_count, 64)
        self.assertEqual(negative_128.status, "abstained")
        self.assertEqual(negative_128.budgeted_evidence_count, 0)
        self.assertGreater(negative_128.budgeted_compression_ratio, 20.0)


if __name__ == "__main__":
    unittest.main()
