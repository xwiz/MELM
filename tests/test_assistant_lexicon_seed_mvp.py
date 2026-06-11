import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from melm.appliance import LexiconCandidateSource, build_lexicon_seed
from melm.contracts import ContractValidationError


def _candidate(lemma: str, class_id: str, source_ref: str) -> dict:
    return {
        "schema_id": "melm.sense_candidate.v1",
        "batch_id": "seed_test",
        "lemma": lemma,
        "language": "en",
        "pos": "noun",
        "source": {
            "provenance": "wordnet",
            "source_ref": source_ref,
            "license": "test-license",
        },
        "definition": f"a definition for {lemma}",
        "semantic_class_candidates": [
            {
                "class_id": class_id,
                "method": "supersense_map",
                "confidence": 0.85,
            }
        ],
        "forms": [],
        "relations": [],
        "safety": {"reserved_conflict": False, "policy_term_overlap": False},
        "suggested_status": "active",
        "confidence_prior": 0.85,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AssistantLexiconSeedMvpTests(unittest.TestCase):
    def test_builder_records_sources_counts_collisions_and_rebuilds_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "candidates.jsonl"
            candidates = [
                _candidate("bass", "living_thing.animal", "wn:bass:animal"),
                _candidate("zither", "physical_object.instrument", "wn:zither"),
                _candidate("bass", "physical_object.instrument", "wn:bass:instrument"),
            ]
            source.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in reversed(candidates)),
                encoding="utf-8",
            )
            first_db = root / "first.sqlite"
            second_db = root / "second.sqlite"
            first_manifest = root / "first.json"
            second_manifest = root / "second.json"

            first = build_lexicon_seed(
                [LexiconCandidateSource(source, "wordnet")],
                output_db=first_db,
                manifest_path=first_manifest,
                reset=True,
            )
            second = build_lexicon_seed(
                [LexiconCandidateSource(source, "wordnet")],
                output_db=second_db,
                manifest_path=second_manifest,
                reset=True,
            )

            self.assertTrue(first.passed)
            self.assertEqual(first.candidates_read, 3)
            self.assertEqual(first.table_counts["lexemes"], 2)
            self.assertEqual(first.table_counts["lexical_senses"], 3)
            self.assertEqual(first.source_counts, {"wordnet": 3})
            self.assertEqual(first.collisions[0]["lemma"], "bass")
            self.assertEqual(first.collisions[0]["sense_count"], 2)
            self.assertEqual(first.output_db_sha256, _sha256(first_db))
            self.assertEqual(first.output_db_sha256, second.output_db_sha256)
            self.assertEqual(first.router_families, ())
            self.assertEqual(first_db.read_bytes(), second_db.read_bytes())
            self.assertEqual(
                json.loads(first_manifest.read_text(encoding="utf-8"))["source_files"][0]["sha256"],
                _sha256(source),
            )

    def test_builder_reports_rejections_without_partial_lexicon_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.jsonl"
            bad = _candidate("weather", "communication_content", "wn:bad-weather")
            bad["source"]["provenance"] = "user_taught"
            bad["suggested_status"] = "quarantined"
            bad["confidence_prior"] = 0.6
            bad["semantic_class_candidates"][0]["method"] = "genus_walk"
            bad["semantic_class_candidates"][0]["confidence"] = 0.6
            source.write_text(json.dumps(bad) + "\n", encoding="utf-8")

            report = build_lexicon_seed(
                [LexiconCandidateSource(source, "user_taught")],
                output_db=root / "seed.sqlite",
                manifest_path=root / "manifest.json",
                reset=True,
            )

            self.assertFalse(report.passed)
            self.assertEqual(report.candidates_rejected, 1)
            self.assertEqual(report.table_counts["lexemes"], 0)
            self.assertEqual(report.table_counts["lexical_senses"], 0)
            self.assertEqual(report.table_counts["lexicon_ingestions"], 1)
            self.assertIn("reserved", report.rejections[0]["error"])
            self.assertEqual(report.output_db_sha256, "")
            self.assertFalse((root / "seed.sqlite").exists())
            self.assertFalse((root / "manifest.json").exists())

    def test_builder_retries_genus_dependencies_and_rejects_source_spoofing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "candidates.jsonl"
            genus = _candidate("zither", "physical_object.instrument", "wn:zither")
            child = _candidate("autoharp", "physical_object.instrument", "wn:autoharp")
            child["genus_lemma"] = "zither"
            child["semantic_class_candidates"][0]["confidence"] = 0.79
            child["confidence_prior"] = 0.79
            source.write_text(
                json.dumps(child) + "\n" + json.dumps(genus) + "\n",
                encoding="utf-8",
            )

            report = build_lexicon_seed(
                [LexiconCandidateSource(source, "wordnet")],
                output_db=root / "seed.sqlite",
                manifest_path=root / "manifest.json",
                reset=True,
            )
            self.assertTrue(report.passed)
            self.assertEqual(report.candidates_applied, 2)

            spoof = root / "spoof.jsonl"
            spoof.write_text(json.dumps(genus) + "\n", encoding="utf-8")
            rejected = build_lexicon_seed(
                [LexiconCandidateSource(spoof, "wiktextract")],
                output_db=root / "spoof.sqlite",
                manifest_path=root / "spoof.json",
                reset=True,
            )
            self.assertFalse(rejected.passed)
            self.assertIn("adapter is bound", rejected.rejections[0]["error"])

    def test_builder_rejects_unknown_router_family_before_creating_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "candidates.jsonl"
            source.write_text(
                json.dumps(_candidate("zither", "physical_object.instrument", "wn:zither"))
                + "\n",
                encoding="utf-8",
            )
            output_db = root / "seed.sqlite"
            manifest = root / "manifest.json"

            with self.assertRaisesRegex(
                ContractValidationError,
                "unknown lexicon router families",
            ):
                build_lexicon_seed(
                    [LexiconCandidateSource(source, "wordnet")],
                    output_db=output_db,
                    manifest_path=manifest,
                    reset=True,
                    router_families=("made_up",),
                )

            self.assertFalse(output_db.exists())
            self.assertFalse(manifest.exists())

    def test_failed_reset_preserves_previous_passing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good_source = root / "good.jsonl"
            good_source.write_text(
                json.dumps(_candidate("zither", "physical_object.instrument", "wn:zither"))
                + "\n",
                encoding="utf-8",
            )
            output_db = root / "seed.sqlite"
            manifest = root / "manifest.json"
            good = build_lexicon_seed(
                [LexiconCandidateSource(good_source, "wordnet")],
                output_db=output_db,
                manifest_path=manifest,
                reset=True,
            )
            original_db = output_db.read_bytes()
            original_manifest = manifest.read_bytes()

            bad_source = root / "bad.jsonl"
            bad = _candidate("weather", "communication_content", "wn:bad")
            bad["source"]["provenance"] = "user_taught"
            bad["suggested_status"] = "quarantined"
            bad["confidence_prior"] = 0.6
            bad["semantic_class_candidates"][0]["method"] = "genus_walk"
            bad["semantic_class_candidates"][0]["confidence"] = 0.6
            bad_source.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            failed = build_lexicon_seed(
                [LexiconCandidateSource(bad_source, "user_taught")],
                output_db=output_db,
                manifest_path=manifest,
                reset=True,
            )

            self.assertTrue(good.passed)
            self.assertFalse(failed.passed)
            self.assertEqual(output_db.read_bytes(), original_db)
            self.assertEqual(manifest.read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
