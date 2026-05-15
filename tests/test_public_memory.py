import json
from pathlib import Path
import tempfile
import unittest

from melm.benchmarks import (
    evaluate_public_context_budget,
    evaluate_public_memory_architectures,
    load_locomo_public_memory_benchmark,
)


class PublicMemoryBenchmarkTests(unittest.TestCase):
    def test_loads_locomo_style_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini_locomo.json"
            path.write_text(json.dumps([_mini_sample()]), encoding="utf-8")

            benchmark = load_locomo_public_memory_benchmark(path)

        self.assertEqual(benchmark.name, "locomo_session_evidence_retrieval")
        self.assertEqual(len(benchmark.documents), 2)
        self.assertEqual(len(benchmark.questions), 2)
        self.assertEqual(benchmark.questions[0].evidence_session_ids, ("conv-mini::S1",))

    def test_public_architecture_report_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini_locomo.json"
            path.write_text(json.dumps([_mini_sample()]), encoding="utf-8")
            benchmark = load_locomo_public_memory_benchmark(path)

            report = evaluate_public_memory_architectures(benchmark, k=1)
            context_report = evaluate_public_context_budget(benchmark, report, token_budget=120)

        self.assertEqual(report.questions, 2)
        self.assertIn("melm_memory_os", report.architectures)
        self.assertGreaterEqual(report.architectures["melm_memory_os"].mean_recall, 0.5)
        self.assertIn("melm_memory_os", context_report)
        self.assertGreaterEqual(context_report["melm_memory_os"].answer_support_rate, 0.5)


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
            "session_2_date_time": "2026/05/02 10:00",
            "session_2": [
                {
                    "speaker": "Ben",
                    "dia_id": "D2:1",
                    "text": "I booked a ceramics class for Saturday.",
                }
            ],
        },
        "observation": {
            "session_1_observation": "Ava adopted a rescue dog named Orbit.",
            "session_2_observation": "Ben booked a ceramics class.",
        },
        "session_summary": {
            "session_1_summary": "Ava talks about adopting Orbit.",
            "session_2_summary": "Ben talks about ceramics.",
        },
        "event_summary": {
            "events_session_1": "Ava adopted Orbit.",
            "events_session_2": "Ben booked a class.",
        },
        "qa": [
            {
                "question": "What is the name of Ava's rescue dog?",
                "answer": "Orbit",
                "category": 2,
                "evidence": ["D1:1"],
            },
            {
                "question": "What class did Ben book?",
                "answer": "ceramics",
                "category": 2,
                "evidence": ["D2:1"],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
