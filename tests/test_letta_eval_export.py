import json
from pathlib import Path
import tempfile
import unittest

from melm.benchmarks import load_locomo_public_memory_benchmark
from melm.integrations import (
    export_locomo_letta_eval_pack,
    validate_letta_dataset_records,
)


class LettaEvalExportTests(unittest.TestCase):
    def test_exports_letta_eval_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "mini_locomo.json"
            out_dir = Path(tmp) / "letta_pack"
            source.write_text(json.dumps([_mini_sample()]), encoding="utf-8")
            benchmark = load_locomo_public_memory_benchmark(source)

            pack = export_locomo_letta_eval_pack(benchmark, out_dir)

            dataset = _read_jsonl(Path(pack.dataset_path))
            memories = _read_jsonl(Path(pack.memory_path))
            suite = Path(pack.suite_path).read_text(encoding="utf-8")

        self.assertEqual(pack.samples, 1)
        self.assertEqual(pack.memories, 1)
        self.assertEqual(validate_letta_dataset_records(dataset), [])
        self.assertIn("input", dataset[0])
        self.assertEqual(dataset[0]["ground_truth"], "Orbit")
        self.assertIn("agent_args", dataset[0])
        self.assertEqual(memories[0]["schema"], "melm.appliance.memory.v1")
        self.assertIn("target:", suite)
        self.assertIn("kind: agent", suite)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mini_sample() -> dict:
    return {
        "sample_id": "conv-mini",
        "conversation": {
            "speaker_a": "Ava",
            "speaker_b": "Ben",
            "session_1_date_time": "2026/05/01 10:00",
            "session_1": [
                {
                    "speaker": "Ava",
                    "dia_id": "D1:1",
                    "text": "I adopted a rescue dog named Orbit yesterday.",
                }
            ],
        },
        "observation": {"session_1_observation": "Ava adopted a rescue dog named Orbit."},
        "session_summary": {"session_1_summary": "Ava talks about adopting Orbit."},
        "event_summary": {"events_session_1": "Ava adopted Orbit."},
        "qa": [
            {
                "question": "What is the name of Ava's rescue dog?",
                "answer": "Orbit",
                "category": 2,
                "evidence": ["D1:1"],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
