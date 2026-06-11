import json
from pathlib import Path
import tempfile
import unittest

from melm.benchmarks import (
    build_support_refunds_freeze_manifest,
    load_authored_support_refund_dataset,
    load_support_refunds_preregistration,
    summarize_support_refund_dataset,
    validate_support_refunds_preregistration,
    verify_support_refunds_freeze,
)


ROOT = Path(__file__).resolve().parents[1]
SEED_DATASET = ROOT / "benchmarks" / "support_refunds_authored.jsonl"


class SupportRefundFreezeTests(unittest.TestCase):
    def test_preregistration_file_sets_external_blind_requirements(self) -> None:
        preregistration = load_support_refunds_preregistration()

        self.assertEqual(
            preregistration["dataset_id"],
            "melm_support_refunds_external_blind_v0_1",
        )
        self.assertEqual(preregistration["authoring_mode"], "external_blind_batch")
        self.assertGreaterEqual(preregistration["minimums"]["annotator_count"], 2)
        self.assertGreaterEqual(preregistration["minimums"]["overlap_labeled_percent"], 20)
        self.assertIn("valid_low_value", preregistration["required_guard_category_counts"])
        self.assertIn("unknown_order", preregistration["required_memory_category_counts"])

    def test_seed_dataset_is_not_external_blind_publishable(self) -> None:
        dataset = load_authored_support_refund_dataset(SEED_DATASET)
        preregistration = load_support_refunds_preregistration()

        errors = validate_support_refunds_preregistration(dataset, preregistration)

        self.assertGreater(len(errors), 0)
        self.assertTrue(any("external_blind_batch" in error for error in errors))
        self.assertTrue(any("dataset_id" in error for error in errors))

    def test_dataset_summary_counts_categories(self) -> None:
        dataset = load_authored_support_refund_dataset(SEED_DATASET)

        summary = summarize_support_refund_dataset(dataset)

        self.assertEqual(summary.turns, 22)
        self.assertEqual(summary.fact_events, 56)
        self.assertEqual(summary.guard_cases, 13)
        self.assertEqual(summary.memory_cases, 40)
        self.assertGreaterEqual(summary.guard_category_counts["valid_low_value"], 2)
        self.assertGreaterEqual(summary.memory_category_counts["unknown_order"], 8)

    def test_freeze_manifest_verifies_matching_hash(self) -> None:
        dataset = load_authored_support_refund_dataset(SEED_DATASET)
        preregistration = {
            "schema": "melm.support_refunds.preregistration.v1",
            "preregistration_id": "seed_batch_freeze_test",
            "dataset_id": dataset.metadata["dataset_id"],
            "minimums": {
                "turns": 1,
                "fact_events": 1,
                "guard_cases": 1,
                "memory_cases": 1,
            },
            "required_guard_category_counts": {"valid_low_value": 1},
            "required_memory_category_counts": {"unknown_order": 1},
            "adjudication_required": False,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            preregistration_path = Path(temp_dir) / "prereg.json"
            manifest_path = Path(temp_dir) / "manifest.json"
            preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")

            manifest = build_support_refunds_freeze_manifest(SEED_DATASET, preregistration_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            self.assertTrue(manifest["schema_validation_passed"])
            self.assertTrue(manifest["preregistration_passed"])
            self.assertEqual(verify_support_refunds_freeze(SEED_DATASET, manifest_path), [])

            manifest["dataset_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(verify_support_refunds_freeze(SEED_DATASET, manifest_path))


if __name__ == "__main__":
    unittest.main()
