import json
from pathlib import Path
import tempfile
import unittest

from melm.appliance import (
    SAFE_TRANSCRIPT_TURN_CONTROL_KEYS,
    STATIC_TRANSCRIPT_EXPECTATION_KEYS,
    import_transcript_replay_fixture,
    load_transcript_replay_scenarios,
    run_transcript_replay_suite,
)


class AssistantTranscriptImportMvpTests(unittest.TestCase):
    def test_import_redacts_raw_chat_and_strips_static_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            out = Path(tmp) / "imported_replay.jsonl"
            _write_jsonl(
                raw,
                [
                    {
                        "role": "user",
                        "session_id": "s1",
                        "content": "My email is maya@example.com and call +234-555-0101.",
                        "expected_route": "local_answer",
                    },
                    {
                        "role": "assistant",
                        "content": "A static assistant answer that must not become evidence.",
                        "expected_answer": "bad",
                    },
                    {
                        "role": "user",
                        "session_id": "s2",
                        "message": "Visit https://example.com and remember code 123456.",
                        "assistant_response": "bad",
                    },
                    {
                        "role": "user",
                        "text": "Alice likes calm piano.",
                    },
                ],
            )

            report = import_transcript_replay_fixture(
                input_path=raw,
                output_path=out,
                replacements=(("Alice", "<person_1>"),),
            )
            payload = report.to_dict()
            output_text = out.read_text(encoding="utf-8")
            output_records = [json.loads(line) for line in output_text.splitlines()]
            turn_records = [item for item in output_records if item.get("type") == "turn"]

            self.assertTrue(report.passed)
            self.assertEqual(payload["schema"], "melm.local_assistant_transcript_import_report.v1")
            self.assertEqual(payload["turns_written"], 3)
            self.assertEqual(payload["assistant_rows_skipped"], 1)
            self.assertEqual(payload["redaction_counts"]["email"], 1)
            self.assertEqual(payload["redaction_counts"]["phone"], 1)
            self.assertEqual(payload["redaction_counts"]["url"], 1)
            self.assertEqual(payload["redaction_counts"]["long_number"], 1)
            self.assertEqual(payload["redaction_counts"]["manual_rule_1"], 1)
            self.assertEqual(payload["static_expectation_fields_dropped"]["expected_route"], 1)
            self.assertEqual(payload["static_expectation_fields_dropped"]["expected_answer"], 1)
            self.assertEqual(output_records[0]["source_type"], "redacted_user_transcript_import")
            self.assertTrue(
                all(record["capture_surface"] == "imported_redacted_transcript" for record in turn_records)
            )
            self.assertTrue(
                all(record["capture_source"] == "redacted_user_transcript_import" for record in turn_records)
            )
            self.assertFalse(any(key in record for record in turn_records for key in STATIC_TRANSCRIPT_EXPECTATION_KEYS))
            self.assertNotIn("maya@example.com", output_text)
            self.assertNotIn("+234-555-0101", output_text)
            self.assertNotIn("https://example.com", output_text)
            self.assertNotIn("Alice", output_text)
            self.assertIn("<email>", output_text)
            self.assertIn("<phone>", output_text)
            self.assertIn("<url>", output_text)
            self.assertIn("<person_1>", output_text)

            scenarios = load_transcript_replay_scenarios(out)
            self.assertEqual(len(scenarios), 1)
            self.assertEqual(len(scenarios[0].turns), 3)
            self.assertFalse(scenarios[0].expectations["required_baseline_win"])

            replay = run_transcript_replay_suite(transcript_path=out, db_dir=Path(tmp) / "db", reset=True).to_dict()
            self.assertTrue(replay["passed"])
            self.assertEqual(replay["source_type"], "redacted_user_transcript_import")
            self.assertEqual(
                replay["capture_provenance"]["capture_surface_counts"],
                {"imported_redacted_transcript": 3},
            )
            self.assertEqual(
                replay["capture_provenance"]["capture_source_counts"],
                {"redacted_user_transcript_import": 3},
            )
            self.assertTrue(replay["capture_provenance"]["has_capture_provenance"])
            self.assertTrue(replay["fixture_checks"]["source_type_supported"])
            self.assertTrue(replay["fixture_checks"]["no_static_answer_or_route_expectations"])
            self.assertFalse(replay["baseline_comparison"]["required"])

    def test_import_applies_safe_lifecycle_controls_without_static_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            out = Path(tmp) / "controlled_replay.jsonl"
            _write_jsonl(
                raw,
                [
                    {
                        "role": "user",
                        "label": "story_gap",
                        "content": "Tell me a story.",
                        "expected_route": "local_answer",
                    },
                    {
                        "role": "user",
                        "content": "What is the weather today?",
                        "run_reflection": False,
                    },
                ],
            )
            controls = {
                "expectations": {
                    "min_turns": 2,
                    "min_route_kinds": 1,
                    "required_priority_signals": True,
                },
                "defaults": {"run_reflection": True},
                "turns": {
                    "story_gap": {"schedule_refreshes": True, "execute_jobs": True},
                    "2": {"network_available": False, "min_story_models": 1},
                },
            }

            report = import_transcript_replay_fixture(input_path=raw, output_path=out, controls=controls)
            output_records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            meta = output_records[0]
            turns = [item for item in output_records if item.get("type") == "turn"]

            self.assertTrue(report.passed)
            self.assertEqual(report.control_fields_applied["run_reflection"], 2)
            self.assertEqual(report.control_fields_applied["schedule_refreshes"], 1)
            self.assertEqual(report.control_fields_applied["execute_jobs"], 1)
            self.assertEqual(report.control_fields_applied["network_available"], 1)
            self.assertEqual(meta["expectations"]["required_priority_signals"], True)
            self.assertEqual(meta["import_report"]["control_fields_applied"]["schedule_refreshes"], 1)
            self.assertTrue(set(turns[0]) >= {"run_reflection", "schedule_refreshes", "execute_jobs"})
            self.assertEqual(turns[1]["network_available"], False)
            self.assertEqual(turns[1]["min_story_models"], 1)
            self.assertFalse(any(key in record for record in turns for key in STATIC_TRANSCRIPT_EXPECTATION_KEYS))
            self.assertTrue(SAFE_TRANSCRIPT_TURN_CONTROL_KEYS >= {"run_reflection", "schedule_refreshes"})

            scenarios = load_transcript_replay_scenarios(out)
            self.assertTrue(scenarios[0].turns[0].schedule_refreshes)
            self.assertTrue(scenarios[0].turns[0].execute_jobs)
            self.assertFalse(scenarios[0].turns[1].network_available)
            self.assertTrue(scenarios[0].expectations["required_priority_signals"])

    def test_import_rejects_static_expectations_inside_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            out = Path(tmp) / "bad_controls.jsonl"
            _write_jsonl(raw, [{"role": "user", "content": "Tell me a story."}])

            with self.assertRaises(ValueError):
                import_transcript_replay_fixture(
                    input_path=raw,
                    output_path=out,
                    controls={"turns": {"1": {"expected_route": "local_answer"}}},
                )

            with self.assertRaisesRegex(ValueError, "controls.turns.1 must be an object"):
                import_transcript_replay_fixture(
                    input_path=raw,
                    output_path=out,
                    controls={"turns": {"1": ["run_reflection"]}},
                )

    def test_shipped_safe_lifecycle_controls_template_is_importable(self) -> None:
        controls_path = Path("config/safe_lifecycle_controls.example.json")
        controls = json.loads(controls_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_chat.jsonl"
            out = Path(tmp) / "controlled_replay.jsonl"
            _write_jsonl(
                raw,
                [
                    {
                        "role": "user",
                        "label": "story_refresh",
                        "content": "Tell me a story.",
                        "expected_route": "local_answer",
                    },
                    {
                        "role": "user",
                        "label": "weather_miss",
                        "content": "What is the weather today?",
                    },
                    {
                        "role": "user",
                        "label": "long_horizon_digest",
                        "content": "What happened over the last few days?",
                    },
                    {
                        "role": "user",
                        "content": "Who are you?",
                    },
                ],
            )

            report = import_transcript_replay_fixture(
                input_path=raw,
                output_path=out,
                controls=controls,
            )
            output_records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            meta = output_records[0]
            turns = [item for item in output_records if item.get("type") == "turn"]

            self.assertTrue(report.passed)
            self.assertEqual(report.static_expectation_fields_dropped["expected_route"], 1)
            self.assertEqual(meta["expectations"]["min_turns"], 4)
            self.assertEqual(meta["expectations"]["min_route_kinds"], 2)
            self.assertEqual(meta["expectations"]["required_priority_signals"], False)
            self.assertTrue(all(turn["run_reflection"] for turn in turns))
            self.assertTrue(turns[0]["schedule_refreshes"])
            self.assertTrue(turns[0]["execute_jobs"])
            self.assertEqual(turns[0]["min_story_models"], 3)
            self.assertTrue(turns[1]["schedule_refreshes"])
            self.assertTrue(turns[1]["execute_jobs"])
            self.assertFalse(any(key in record for record in turns for key in STATIC_TRANSCRIPT_EXPECTATION_KEYS))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
