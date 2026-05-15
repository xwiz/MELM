import json
import tempfile
from pathlib import Path
import unittest

from melm.benchmarks import (
    load_blimp_fast_cases,
    load_entity_tracking_fast_cases,
    profile_blimp_fast,
)


class BabyLMEvalAdapterTests(unittest.TestCase):
    def test_load_blimp_fast_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agreement.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sentence_good": "The dog is running.",
                        "sentence_bad": "The dog are running.",
                        "UID": "agreement",
                        "pair_id": 7,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cases = load_blimp_fast_cases(tmp)
            profile = profile_blimp_fast(tmp)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "agreement:7")
        self.assertEqual(cases[0].category, "agreement")
        self.assertEqual(profile["files"], 1)
        self.assertEqual(profile["cases"], 1)

    def test_load_entity_tracking_fast_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regular.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": 3,
                        "numops": 1,
                        "input_prefix": "Box 1 contains ",
                        "options": ["the hat.", "the map."],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cases = load_entity_tracking_fast_cases(tmp)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "regular:3")
        self.assertEqual(cases[0].category, "regular_1_ops")
        self.assertEqual(cases[0].option_texts()[0], "Box 1 contains the hat.")


if __name__ == "__main__":
    unittest.main()
