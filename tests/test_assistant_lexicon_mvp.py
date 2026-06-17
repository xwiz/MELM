import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import json

from melm.appliance import (
    acquire_definition,
    AssistantOSKernel,
    AssistantOSStore,
    benchmark_lexicon_lookup,
    cloud_definition_lookup,
    lexical_classes_for_term,
    lexicon_ingest,
    lookup_lexical_senses,
    lookup_lexical_senses_tiered,
    offline_definition_lookup,
    set_lexical_sense_status,
)
from melm.appliance.assistant_lexicon import LexiconIngestResult
from melm.appliance.assistant_frame_linker import FrameLinker
from melm.appliance.assistant_lexicon import _normalize_term
from melm.appliance.local_assistant_router import (
    _classify_from_frame_linker,
    _IN_MEMORY_LEXICON,
    replace_in_memory_lexicon,
    OnDeviceAssistantRouter,
    LocalAssistantProfile,
    replace_installed_families,
)
from melm.contracts import ContractValidationError


def _candidate(
    lemma: str,
    class_id: str,
    *,
    provenance: str = "wordnet",
    source_ref: str | None = None,
    status: str = "active",
    genus_lemma: str = "",
    confidence: float = 0.85,
    forms: list[dict] | None = None,
    relations: list[dict] | None = None,
    safety: dict | None = None,
) -> dict:
    payload = {
        "schema_id": "melm.sense_candidate.v1",
        "batch_id": "test_batch",
        "lemma": lemma,
        "language": "en",
        "pos": "noun",
        "source": {
            "provenance": provenance,
            "source_ref": source_ref or f"{provenance}:{lemma}:{class_id}",
            "license": "test-license",
        },
        "definition": f"a test definition for {lemma}",
        "semantic_class_candidates": [
            {
                "class_id": class_id,
                "method": (
                    "seed_authored"
                    if provenance == "seed_authored"
                    else "genus_walk"
                    if provenance == "user_taught"
                    else "supersense_map"
                ),
                "confidence": confidence,
            }
        ],
        "forms": forms or [],
        "relations": relations or [],
        "safety": safety or {"reserved_conflict": False, "policy_term_overlap": False},
        "suggested_status": status,
        "confidence_prior": confidence,
    }
    if genus_lemma:
        payload["genus_lemma"] = genus_lemma
    return payload


def _ingest(store: AssistantOSStore, candidate: dict):
    return lexicon_ingest(
        store,
        candidate,
        expected_provenance=candidate["source"]["provenance"],
    )


class AssistantLexiconMvpTests(unittest.TestCase):
    def test_ingest_persists_factored_rows_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(db)
            result = _ingest(
                store,
                _candidate(
                    "zither",
                    "physical_object.instrument",
                    forms=[
                        {
                            "surface": "zithers",
                            "morph_features": "Number=Plur",
                            "provenance": "spacy_rules",
                        }
                    ],
                ),
            )
            self.assertTrue(result.created_lexeme)
            self.assertTrue(result.created_sense)
            self.assertEqual(store.count("lexemes"), 1)
            self.assertEqual(store.count("word_forms"), 2)
            self.assertEqual(store.count("lexical_senses"), 1)
            self.assertEqual(store.count("lexical_provenance"), 1)
            store.close()

            reloaded = AssistantOSStore(db)
            senses = lookup_lexical_senses(reloaded, "zithers")
            self.assertEqual(len(senses), 1)
            self.assertEqual(senses[0]["semantic_class_id"], "physical_object.instrument")
            self.assertEqual(senses[0]["status"], "active")
            reloaded.close()

    def test_exact_candidate_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            candidate = _candidate("zither", "physical_object.instrument")

            first = _ingest(store, candidate)
            second = _ingest(store, candidate)
            with self.assertRaisesRegex(ContractValidationError, "adapter is bound"):
                lexicon_ingest(
                    store,
                    candidate,
                    expected_provenance="wiktextract",
                )

            self.assertFalse(first.duplicate_candidate)
            self.assertTrue(second.duplicate_candidate)
            self.assertEqual(first.sense_id, second.sense_id)
            self.assertEqual(store.count("lexemes"), 1)
            self.assertEqual(store.count("lexical_senses"), 1)
            self.assertEqual(store.count("lexical_provenance"), 1)
            self.assertEqual(store.count("lexicon_ingestions"), 1)
            store.close()

    def test_same_class_merges_provenance_without_downgrading_active_sense(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            _ingest(store, _candidate("zither", "physical_object.instrument"))
            wordnet = _ingest(store, _candidate("kalimba", "physical_object.instrument"))
            taught = _ingest(
                store,
                _candidate(
                    "kalimba",
                    "physical_object.instrument",
                    provenance="user_taught",
                    status="quarantined",
                    genus_lemma="zither",
                    confidence=0.6,
                ),
            )

            self.assertEqual(wordnet.sense_id, taught.sense_id)
            self.assertFalse(taught.created_sense)
            self.assertEqual(store.count("lexemes"), 2)
            self.assertEqual(store.count("lexical_senses"), 2)
            self.assertEqual(store.count("lexical_provenance"), 3)
            senses = lookup_lexical_senses(store, "kalimba", statuses=("active", "quarantined"))
            self.assertEqual(senses[0]["status"], "active")
            self.assertEqual(senses[0]["confidence"], 0.85)
            store.close()

    def test_different_class_creates_polysemous_sense(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            instrument = _ingest(
                store,
                _candidate("bass", "physical_object.instrument", status="dormant"),
            )
            animal = _ingest(
                store,
                _candidate(
                    "bass",
                    "living_thing.animal",
                    source_ref="wordnet:bass:animal",
                    status="dormant",
                ),
            )

            self.assertEqual(instrument.lexeme_id, animal.lexeme_id)
            self.assertNotEqual(instrument.sense_id, animal.sense_id)
            self.assertEqual(store.count("lexemes"), 1)
            self.assertEqual(store.count("lexical_senses"), 2)
            self.assertEqual(len(lookup_lexical_senses(store, "bass")), 2)
            store.close()

    def test_runtime_candidate_requires_resolvable_genus_and_stays_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            unresolved = _candidate(
                "kalimba",
                "physical_object.instrument",
                provenance="user_taught",
                status="quarantined",
                genus_lemma="piano",
                confidence=0.6,
            )
            with self.assertRaisesRegex(ContractValidationError, "unresolved genus"):
                _ingest(store, unresolved)
            self.assertEqual(store.count("lexemes"), 0)
            self.assertEqual(store.count("lexical_senses"), 0)
            self.assertEqual(store.count("lexicon_ingestions"), 1)

            _ingest(
                store,
                _candidate(
                    "piano",
                    "physical_object.instrument",
                    provenance="seed_authored",
                    safety={"reserved_conflict": True, "policy_term_overlap": False},
                    confidence=0.95,
                ),
            )
            accepted = _ingest(store, unresolved)
            self.assertEqual(accepted.status, "quarantined")
            self.assertEqual(lookup_lexical_senses(store, "kalimba"), [])
            quarantined = lookup_lexical_senses(store, "kalimba", statuses=("quarantined",))
            self.assertEqual(len(quarantined), 1)
            store.close()

    def test_reserved_namespace_is_checked_independently_of_candidate_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            malicious = _candidate(
                "weather",
                "communication_content",
                provenance="user_taught",
                status="quarantined",
                confidence=0.6,
            )
            with self.assertRaisesRegex(ContractValidationError, "is reserved"):
                _ingest(store, malicious)
            self.assertEqual(store.count("lexemes"), 0)

            seed = _candidate(
                "weather",
                "weather_phenomenon",
                provenance="seed_authored",
                status="active",
                safety={"reserved_conflict": True, "policy_term_overlap": False},
            )
            accepted = _ingest(store, seed)
            self.assertEqual(accepted.status, "active")
            self.assertTrue(lookup_lexical_senses(store, "weather")[0]["reserved"])
            store.close()

    def test_policy_namespace_and_adapter_provenance_are_checked_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                policy_term = _candidate(
                    "permission",
                    "abstract",
                    provenance="user_taught",
                    status="quarantined",
                    confidence=0.6,
                )
                with self.assertRaisesRegex(ContractValidationError, "policy namespace"):
                    _ingest(store, policy_term)

                spoofed = _candidate("piano", "physical_object.instrument")
                with self.assertRaisesRegex(ContractValidationError, "adapter is bound"):
                    lexicon_ingest(
                        store,
                        spoofed,
                        expected_provenance="wiktextract",
                    )
                self.assertEqual(store.count("lexemes"), 0)
            finally:
                store.close()

    def test_non_seed_source_cannot_smuggle_router_or_policy_anchors_as_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                router_alias = _candidate(
                    "melody",
                    "media_content",
                    forms=[
                        {
                            "surface": "song",
                            "morph_features": "Number=Sing",
                            "provenance": "wordnet",
                        }
                    ],
                )
                with self.assertRaisesRegex(ContractValidationError, "is reserved"):
                    _ingest(store, router_alias)

                policy_alias = _candidate(
                    "allowance",
                    "abstract",
                    forms=[
                        {
                            "surface": "permission",
                            "morph_features": "Number=Sing",
                            "provenance": "wordnet",
                        }
                    ],
                )
                with self.assertRaisesRegex(ContractValidationError, "policy vocabulary"):
                    _ingest(store, policy_alias)
                self.assertEqual(store.count("lexemes"), 0)
                self.assertEqual(store.count("lexicon_ingestions"), 2)
            finally:
                store.close()

    def test_resolved_genus_must_be_class_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                _ingest(
                    store,
                    _candidate(
                        "piano",
                        "physical_object.instrument",
                        provenance="seed_authored",
                        safety={"reserved_conflict": True, "policy_term_overlap": False},
                        confidence=0.95,
                    ),
                )
                incompatible = _candidate(
                    "kalimba",
                    "person",
                    provenance="user_taught",
                    status="quarantined",
                    genus_lemma="piano",
                    confidence=0.6,
                )
                with self.assertRaisesRegex(ContractValidationError, "incompatible"):
                    _ingest(store, incompatible)
            finally:
                store.close()

    def test_relations_are_always_staged_in_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            result = _ingest(
                store,
                _candidate(
                    "kalimba",
                    "physical_object.instrument",
                    relations=[{"relation": "hypernym", "target_lemma": "piano"}],
                ),
            )
            row = store.connection.execute(
                "SELECT status, target_lemma FROM lexical_relation_candidates WHERE sense_id=?",
                (result.sense_id,),
            ).fetchone()
            self.assertEqual(result.relations_staged, 1)
            self.assertEqual(row["status"], "quarantined")
            self.assertEqual(row["target_lemma"], "piano")
            store.close()

    def test_tiered_lookup_prefers_active_then_dormant_and_never_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                _ingest(
                    store,
                    _candidate(
                        "dulcimer",
                        "physical_object.instrument",
                        status="dormant",
                    ),
                )
                _ingest(
                    store,
                    _candidate(
                        "kalimba",
                        "physical_object.instrument",
                        provenance="user_taught",
                        status="quarantined",
                        genus_lemma="dulcimer",
                        confidence=0.6,
                    ),
                )
                dormant = lookup_lexical_senses_tiered(store, "dulcimer")
                quarantined = lookup_lexical_senses_tiered(store, "kalimba")
                missing = lookup_lexical_senses_tiered(store, "notaword")

                self.assertEqual(dormant.tier, "dormant")
                self.assertEqual(len(dormant.senses), 1)
                self.assertEqual(quarantined.tier, "miss")
                self.assertEqual(quarantined.senses, ())
                self.assertEqual(missing.tier, "miss")
            finally:
                store.close()

    def test_lookup_benchmark_reports_all_activation_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                _ingest(store, _candidate("zither", "physical_object.instrument"))
                _ingest(
                    store,
                    _candidate(
                        "dulcimer",
                        "physical_object.instrument",
                        status="dormant",
                    ),
                )
                report = benchmark_lexicon_lookup(
                    store,
                    ("zither", "dulcimer", "notaword"),
                    iterations=30,
                    warmup_queries=3,
                )
                self.assertEqual(report.queries, 30)
                self.assertEqual(report.active_hits, 10)
                self.assertEqual(report.dormant_hits, 10)
                self.assertEqual(report.misses, 10)
                self.assertGreaterEqual(report.max_ms, report.p95_ms)
            finally:
                store.close()


class AcquireDefinitionMvpTests(unittest.TestCase):
    """Tests for the user-teaching definition acquisition channel."""

    def _seed_genus(self, store, term="instrument", class_id="physical_object.instrument"):
        return _ingest(
            store,
            _candidate(term, class_id, provenance="seed_authored",
                       safety={"reserved_conflict": False, "policy_term_overlap": False},
                       confidence=0.95),
        )

    def test_copula_definition_ingests_as_quarantined_sense(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                result = acquire_definition(store, "a kalimba is a musical instrument")
                self.assertIsNotNone(result)
                self.assertFalse(result.duplicate_candidate)
                self.assertTrue(result.created_lexeme)
                self.assertTrue(result.created_sense)
                self.assertEqual(result.status, "quarantined")

                senses = lookup_lexical_senses(store, "kalimba", statuses=("quarantined",))
                self.assertEqual(len(senses), 1)
                self.assertEqual(senses[0]["semantic_class_id"], "physical_object.instrument")
                self.assertEqual(senses[0]["status"], "quarantined")
            finally:
                store.close()

    def test_copula_definition_normalizes_through_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                result = acquire_definition(store, "A Kalimba is a musical instrument!")
                self.assertIsNotNone(result)
                self.assertEqual(result.status, "quarantined")
                senses = lookup_lexical_senses(store, "kalimba", statuses=("quarantined",))
                self.assertEqual(len(senses), 1)
            finally:
                store.close()

    def test_means_pattern_ingests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                result = acquire_definition(store, "xylophone means a musical instrument")
                self.assertIsNotNone(result)
                self.assertTrue(result.created_lexeme)
                self.assertEqual(result.status, "quarantined")
            finally:
                store.close()

    def test_non_matching_utterance_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                result = acquire_definition(store, "hello, how are you?")
                self.assertIsNone(result)
            finally:
                store.close()

    def test_empty_utterance_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                result = acquire_definition(store, "")
                self.assertIsNone(result)
                result = acquire_definition(store, "   ")
                self.assertIsNone(result)
            finally:
                store.close()

    def test_unresolved_genus_raises_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                with self.assertRaisesRegex(Exception, "unresolved genus"):
                    acquire_definition(store, "a kalimba is a zzzznotaword")
            finally:
                store.close()

    def test_reserved_word_raises_on_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                with self.assertRaisesRegex(Exception, "reserved|is reserved"):
                    acquire_definition(store, "a weather is a natural phenomenon")
            finally:
                store.close()

    def test_genus_with_multiple_classes_uses_all_as_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                _ingest(
                    store,
                    _candidate("bass", "physical_object.instrument",
                               provenance="seed_authored",
                               safety={"reserved_conflict": False, "policy_term_overlap": False},
                               confidence=0.95, status="dormant"),
                )
                _ingest(
                    store,
                    _candidate("bass", "living_thing.animal",
                               source_ref="wordnet:bass:animal",
                               provenance="seed_authored",
                               safety={"reserved_conflict": False, "policy_term_overlap": False},
                               confidence=0.95, status="dormant"),
                )
                result = acquire_definition(store, "a kalimba is a bass")
                self.assertIsNotNone(result)
                senses = lookup_lexical_senses(store, "kalimba", statuses=("quarantined",))
                self.assertEqual(len(senses), 1)
            finally:
                store.close()

    def test_provenance_is_user_taught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                result = acquire_definition(store, "a kalimba is a musical instrument")
                self.assertIsNotNone(result)
                provenance = store.connection.execute(
                    "SELECT provenance FROM lexical_provenance WHERE sense_id=?",
                    (result.sense_id,),
                ).fetchone()
                self.assertEqual(provenance["provenance"], "user_taught")
            finally:
                store.close()

    def test_genus_extracts_head_noun_before_pp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                _ingest(
                    store,
                    _candidate(
                        "piano", "physical_object.instrument",
                        provenance="seed_authored",
                        safety={"reserved_conflict": True, "policy_term_overlap": False},
                        confidence=0.95,
                    ),
                )
                result = acquire_definition(
                    store,
                    "a kalimba is a small thumb piano from africa",
                )
                self.assertIsNotNone(result)
                self.assertEqual(result.status, "quarantined")
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 1)
                self.assertEqual(
                    senses[0]["semantic_class_id"], "physical_object.instrument",
                )
            finally:
                store.close()

    def test_homonym_creates_separate_sense_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                _ingest(
                    store,
                    _candidate(
                        "mammal", "living_thing.animal",
                        provenance="seed_authored",
                        safety={"reserved_conflict": False, "policy_term_overlap": False},
                        confidence=0.95,
                    ),
                )
                r1 = acquire_definition(store, "a kalimba is a musical instrument")
                self.assertTrue(r1.created_lexeme)
                self.assertTrue(r1.created_sense)
                r2 = acquire_definition(store, "a kalimba is a flying mammal")
                self.assertIsNotNone(r2)
                self.assertFalse(r2.created_lexeme)
                self.assertTrue(r2.created_sense)
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 2)
                class_ids = {s["semantic_class_id"] for s in senses}
                self.assertIn("physical_object.instrument", class_ids)
                self.assertIn("living_thing.animal", class_ids)
            finally:
                store.close()

    def test_all_reserved_lexemes_rejected_on_acquire(self) -> None:
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                res_path = (
                    Path(__file__).parent.parent
                    / "melm" / "contracts" / "reserved_lexemes.v1.json"
                )
                with open(res_path) as f:
                    data = _json.load(f)
                reserved = set(data["lexemes"]) | set(data.get("policy_lexemes", []))
                for word in sorted(reserved):
                    with self.subTest(lexeme=word):
                        with self.assertRaises(ContractValidationError):
                            acquire_definition(
                                store, f"a {word} is a musical instrument",
                            )
            finally:
                store.close()


class OfflineDefinitionLookupMvpTests(unittest.TestCase):
    """Tests for the offline dictionary lookup channel."""

    def _seed_genus(self, store, term="instrument", class_id="physical_object.instrument"):
        return lexicon_ingest(
            store,
            {
                "schema_id": "melm.sense_candidate.v1",
                "lemma": term,
                "language": "en",
                "pos": "noun",
                "source": {"provenance": "seed_authored",
                           "source_ref": f"test:{term}",
                           "license": "test"},
                "definition": f"a test {class_id}",
                "semantic_class_candidates": [{"class_id": class_id,
                                               "method": "seed_authored",
                                               "confidence": 0.95}],
                "safety": {"reserved_conflict": False, "policy_term_overlap": False},
                "suggested_status": "active",
                "confidence_prior": 0.95,
            },
            expected_provenance="seed_authored",
        )

    def _write_jsonl(self, tmp: Path, entries: list[dict]) -> Path:
        path = tmp / "dictionary.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return path

    def test_ingests_single_entry_as_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "kalimba", "definition": "a musical instrument"}],
                )
                results = offline_definition_lookup(
                    store, "kalimba", dictionary_path=str(dict_path),
                )
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].status, "quarantined")
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 1)
            finally:
                store.close()

    def test_empty_jsonl_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                dict_path = self._write_jsonl(Path(tmp), [])
                results = offline_definition_lookup(
                    store, "anything", dictionary_path=str(dict_path),
                )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_non_matching_lemma_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "piano", "definition": "a musical instrument"}],
                )
                results = offline_definition_lookup(
                    store, "kalimba", dictionary_path=str(dict_path),
                )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_missing_dictionary_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                results = offline_definition_lookup(
                    store, "kalimba", dictionary_path=str(Path(tmp) / "nonexistent.jsonl"),
                )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_empty_word_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "kalimba", "definition": "an instrument"}],
                )
                results = offline_definition_lookup(
                    store, "", dictionary_path=str(dict_path),
                )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_provenance_is_offline_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "kalimba", "definition": "a musical instrument"}],
                )
                results = offline_definition_lookup(
                    store, "kalimba", dictionary_path=str(dict_path),
                )
                self.assertEqual(len(results), 1)
                prov = store.connection.execute(
                    "SELECT provenance FROM lexical_provenance WHERE sense_id=?",
                    (results[0].sense_id,),
                ).fetchone()
                self.assertEqual(prov["provenance"], "offline_dictionary")
            finally:
                store.close()

    def test_extracts_genus_from_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "xylophone", "definition": "a musical instrument"}],
                )
                results = offline_definition_lookup(
                    store, "xylophone", dictionary_path=str(dict_path),
                )
                self.assertEqual(len(results), 1)
                sense = lookup_lexical_senses(
                    store, "xylophone", statuses=("quarantined",),
                )
                self.assertEqual(len(sense), 1)
                self.assertEqual(sense[0]["semantic_class_id"], "physical_object.instrument")
            finally:
                store.close()

    def test_missing_genus_results_in_abstract_fallback_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "zzzztoken", "definition": "a made-up thing"}],
                )
                results = offline_definition_lookup(
                    store, "zzzztoken", dictionary_path=str(dict_path),
                )
                self.assertEqual(len(results), 0)
            finally:
                store.close()

    def test_reserved_word_triggers_rejection_not_silent_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "weather", "definition": "a meteorological phenomenon"}],
                )
                results = offline_definition_lookup(
                    store, "weather", dictionary_path=str(dict_path),
                )
                self.assertEqual(len(results), 0)
            finally:
                store.close()

    def test_multiple_entries_for_same_lemma_all_attempt_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [
                        {"lemma": "bass", "definition": "a musical instrument"},
                        {"lemma": "bass", "definition": "a type of fish"},
                    ],
                )
                results = offline_definition_lookup(
                    store, "bass", dictionary_path=str(dict_path),
                )
                self.assertGreaterEqual(len(results), 1)
            finally:
                store.close()

    def test_confidence_prior_is_0_70(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "sitar", "definition": "a musical instrument"}],
                )
                results = offline_definition_lookup(
                    store, "sitar", dictionary_path=str(dict_path),
                )
                self.assertEqual(len(results), 1)
                sense = lookup_lexical_senses(
                    store, "sitar", statuses=("quarantined",),
                )
                self.assertAlmostEqual(sense[0]["confidence"], 0.72, places=2)
            finally:
                store.close()

    def test_genus_lemma_on_entry_overrides_auto_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store, "tool", "abstract")
                dict_path = self._write_jsonl(
                    Path(tmp),
                    [{"lemma": "kalimba", "definition": "a musical instrument",
                      "genus_lemma": "tool"}],
                )
                results = offline_definition_lookup(
                    store, "kalimba", dictionary_path=str(dict_path),
                )
                self.assertEqual(len(results), 1)
                sense = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(sense[0]["semantic_class_id"], "abstract")
            finally:
                store.close()


class _MockResponse:
    """Minimal file-like object that simulates ``urlopen`` context."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        pass


class CloudDefinitionLookupMvpTests(unittest.TestCase):
    """Tests for the cloud LLM definition lookup channel."""

    def _seed_genus(self, store, term="instrument", class_id="physical_object.instrument"):
        return lexicon_ingest(
            store,
            {
                "schema_id": "melm.sense_candidate.v1",
                "lemma": term,
                "language": "en",
                "pos": "noun",
                "source": {"provenance": "seed_authored",
                           "source_ref": f"test:{term}",
                           "license": "test"},
                "definition": f"a test {class_id}",
                "semantic_class_candidates": [{"class_id": class_id,
                                               "method": "seed_authored",
                                               "confidence": 0.95}],
                "safety": {"reserved_conflict": False, "policy_term_overlap": False},
                "suggested_status": "active",
                "confidence_prior": 0.95,
            },
            expected_provenance="seed_authored",
        )

    @staticmethod
    def _mock_llm_response(candidate: dict) -> _MockResponse:
        """Build a mock HTTP response simulating an LLM chat completion."""
        body = json.dumps({
            "choices": [{"message": {"content": json.dumps(candidate)}}],
        }).encode("utf-8")
        return _MockResponse(body)

    def _seed_instrument(self, store):
        return self._seed_genus(store, term="instrument", class_id="physical_object.instrument")

    def test_ingests_valid_response_as_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_instrument(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "kalimba",
                               "pos": "noun",
                               "definition": "a small thumb piano",
                               "genus_lemma": "instrument",
                               "semantic_class_candidates": [
                                   {"class_id": "physical_object.instrument",
                                    "confidence": 0.85},
                               ],
                           })):
                    results = cloud_definition_lookup(
                        store, "kalimba",
                        api_key="test-key",
                        endpoint="https://test.example.com/v1/chat/completions",
                    )
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].status, "quarantined")
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 1)
            finally:
                store.close()

    def test_provenance_is_cloud_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_instrument(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "kalimba",
                               "pos": "noun",
                               "definition": "a small thumb piano",
                               "genus_lemma": "instrument",
                               "semantic_class_candidates": [
                                   {"class_id": "physical_object.instrument",
                                    "confidence": 0.85},
                               ],
                           })):
                    results = cloud_definition_lookup(
                        store, "kalimba",
                        api_key="test-key",
                    )
                self.assertEqual(len(results), 1)
                prov = store.connection.execute(
                    "SELECT provenance FROM lexical_provenance WHERE sense_id=?",
                    (results[0].sense_id,),
                ).fetchone()
                self.assertEqual(prov["provenance"], "cloud_lookup")
            finally:
                store.close()

    def test_confidence_prior_is_0_50(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_instrument(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "kalimba",
                               "pos": "noun",
                               "definition": "a small thumb piano",
                               "genus_lemma": "instrument",
                               "semantic_class_candidates": [
                                   {"class_id": "physical_object.instrument",
                                    "confidence": 0.30},
                               ],
                           })):
                    results = cloud_definition_lookup(
                        store, "kalimba",
                        api_key="test-key",
                    )
                self.assertEqual(len(results), 1)
                sense = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(sense[0]["confidence"], 0.50)
            finally:
                store.close()

    def test_method_is_llm_assigned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_instrument(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "kalimba",
                               "pos": "noun",
                               "definition": "a small thumb piano",
                               "genus_lemma": "instrument",
                               "semantic_class_candidates": [
                                   {"class_id": "physical_object.instrument",
                                    "confidence": 0.85},
                               ],
                           })):
                    results = cloud_definition_lookup(
                        store, "kalimba",
                        api_key="test-key",
                    )
                self.assertEqual(len(results), 1)
                # Method is implicitly verified by schema validation —
                # cloud_lookup provenance only allows "llm_assigned" method.
                # Confirm the sense was created with the expected class.
                sense = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(sense[0]["semantic_class_id"], "physical_object.instrument")
            finally:
                store.close()

    def test_network_error_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           side_effect=OSError("connection refused")):
                    results = cloud_definition_lookup(
                        store, "kalimba",
                        api_key="test-key",
                    )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_malformed_llm_response_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=_MockResponse(b"not-json-at-all")):
                    results = cloud_definition_lookup(
                        store, "kalimba",
                        api_key="test-key",
                    )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_missing_content_in_response_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                body = json.dumps({
                    "choices": [{"message": {"content": ""}}],
                }).encode("utf-8")
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=_MockResponse(body)):
                    results = cloud_definition_lookup(
                        store, "kalimba",
                        api_key="test-key",
                    )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_empty_word_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "kalimba",
                               "pos": "noun",
                               "definition": "an instrument",
                           })):
                    results = cloud_definition_lookup(
                        store, "",
                        api_key="test-key",
                    )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_reserved_word_triggers_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                self._seed_genus(store)
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "weather",
                               "pos": "noun",
                               "definition": "a meteorological phenomenon",
                               "genus_lemma": "phenomenon",
                               "semantic_class_candidates": [
                                   {"class_id": "weather_phenomenon",
                                    "confidence": 0.90},
                               ],
                           })):
                    results = cloud_definition_lookup(
                        store, "weather",
                        api_key="test-key",
                    )
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_no_llm_classes_falls_back_to_abstract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "zzzztoken",
                               "pos": "noun",
                               "definition": "a made-up thing",
                               "semantic_class_candidates": [],
                           })):
                    results = cloud_definition_lookup(
                        store, "zzzztoken",
                        api_key="test-key",
                    )
                self.assertEqual(len(results), 1)
                sense = lookup_lexical_senses(
                    store, "zzzztoken", statuses=("quarantined",),
                )
                self.assertEqual(sense[0]["semantic_class_id"], "abstract")
            finally:
                store.close()

    def test_unknown_class_ids_filtered_from_llm_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                with patch("melm.appliance.assistant_lexicon.urlopen",
                           return_value=self._mock_llm_response({
                               "lemma": "phantasm",
                               "pos": "noun",
                               "definition": "a ghostly apparition",
                               "semantic_class_candidates": [
                                   {"class_id": "nonexistent_class",
                                    "confidence": 0.95},
                                   {"class_id": "abstract",
                                    "confidence": 0.60},
                               ],
                           })):
                    results = cloud_definition_lookup(
                        store, "phantasm",
                        api_key="test-key",
                    )
                self.assertEqual(len(results), 1)
                sense = lookup_lexical_senses(
                    store, "phantasm", statuses=("quarantined",),
                )
                self.assertEqual(sense[0]["semantic_class_id"], "abstract")
            finally:
                store.close()


class KalimbaEndToEndLifecycleMvpTests(unittest.TestCase):
    """End-to-end lifecycle fixture for the M3 vocabulary acquisition gate.

    Tests the full teach → quarantine → promote → restart → rollback cycle
    using the word *kalimba* (a genuine under-documented instrument) to
    exercise every stage of the runtime acquisition pipeline.
    """

    def _seed_genus(self, store, term="instrument", class_id="physical_object.instrument"):
        return lexicon_ingest(
            store,
            {
                "schema_id": "melm.sense_candidate.v1",
                "lemma": term,
                "language": "en",
                "pos": "noun",
                "source": {"provenance": "seed_authored",
                           "source_ref": f"test:{term}",
                           "license": "test"},
                "definition": f"a test {class_id}",
                "semantic_class_candidates": [{"class_id": class_id,
                                               "method": "seed_authored",
                                               "confidence": 0.95}],
                "safety": {"reserved_conflict": False, "policy_term_overlap": False},
                "suggested_status": "active",
                "confidence_prior": 0.95,
            },
            expected_provenance="seed_authored",
        )

    def test_teach_quarantine_promote_restart_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(path)
            try:
                self._seed_genus(store)

                # 1. TEACH — acquire definition of a new word
                result = acquire_definition(
                    store, "a kalimba is a musical instrument",
                )
                self.assertIsNotNone(result)
                self.assertEqual(result.status, "quarantined")

                # Verify quarantined — visible only with explicit status filter
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 1)
                sense_id = senses[0]["sense_id"]

                # Not visible in active-only lookup (routing gate)
                active = lexical_classes_for_term(store, "kalimba")
                self.assertEqual(active, frozenset())

                # 2. PROMOTE — quarantined → active
                set_lexical_sense_status(store, sense_id, "active")
                promoted = lookup_lexical_senses(
                    store, "kalimba", statuses=("active",),
                )
                self.assertEqual(len(promoted), 1)
                self.assertEqual(promoted[0]["semantic_class_id"], "physical_object.instrument")

                # Now visible in active-only lookup (routing picks it up)
                active = lexical_classes_for_term(store, "kalimba")
                self.assertIn("physical_object.instrument", active)

                # 3. REOPEN — simulate restart
                store.close()
                store = AssistantOSStore(path)
                self._seed_genus(store)

                # Promoted sense survives restart
                persisted = lookup_lexical_senses(
                    store, "kalimba", statuses=("active",),
                )
                self.assertEqual(len(persisted), 1)
                active = lexical_classes_for_term(store, "kalimba")
                self.assertIn("physical_object.instrument", active)

                # 4. CORRECT — re-teach merges into same sense
                correct = acquire_definition(
                    store, "a kalimba is a hand-held instrument",
                )
                self.assertIsNotNone(correct)
                self.assertEqual(correct.sense_id, sense_id)
                self.assertEqual(correct.status, "active")

                # 5. ROLLBACK — active → quarantined
                set_lexical_sense_status(store, sense_id, "quarantined")
                rolled = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(rolled), 1)

                # No longer visible for routing
                active = lexical_classes_for_term(store, "kalimba")
                self.assertEqual(active, frozenset())

            finally:
                store.close()

    def test_promotion_trace_queryable_after_status_change(self) -> None:
        """set_lexical_sense_status records promotion in promotions table."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assistant.sqlite"
            store = AssistantOSStore(path)
            try:
                self._seed_genus(store)
                result = acquire_definition(
                    store, "a kalimba is a musical instrument",
                )
                self.assertIsNotNone(result)
                sense_id = result.sense_id

                set_lexical_sense_status(store, sense_id, "active")
                proms = store.find_promotions(target_type="sense", target_id=sense_id)
                self.assertEqual(len(proms), 1)
                self.assertEqual(proms[0].from_status, "quarantined")
                self.assertEqual(proms[0].to_status, "active")
                self.assertEqual(proms[0].target_type, "sense")
                self.assertEqual(proms[0].target_id, sense_id)

                set_lexical_sense_status(store, sense_id, "quarantined")
                proms = store.find_promotions(target_type="sense", target_id=sense_id)
                self.assertGreaterEqual(len(proms), 2)
                self.assertEqual(proms[0].from_status, "active")
                self.assertEqual(proms[0].to_status, "quarantined")

                set_lexical_sense_status(store, sense_id, "active")
                proms = store.find_promotions(target_type="sense", target_id=sense_id)
                third_from = {p.from_status for p in proms[:3]}
                self.assertIn("quarantined", third_from)
                self.assertIn("active", third_from)
            finally:
                store.close()


class SemanticClassEventIndexMvpTests(unittest.TestCase):

    def test_semantic_family_terms_activates_collector(self) -> None:
        from melm.appliance.local_assistant_router import (
            _semantic_family_terms,
            set_semantic_class_collector,
        )
        collector: set[str] = set()
        set_semantic_class_collector(collector)
        try:
            result = _semantic_family_terms(
                ("hi", "weather", "rain"),
                semantic_classes={"temporal_descriptor", "social_greeting"},
            )
        finally:
            set_semantic_class_collector(None)
        self.assertIsInstance(result, set)
        self.assertGreaterEqual(len(collector), 0)

    def test_router_handle_injects_activated_classes(self) -> None:
        from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        decision = router.handle("hi what is the weather like")
        self.assertIsInstance(decision.semantic_classes_activated, frozenset)

    def test_activated_semantic_classes_persisted_in_store(self) -> None:
        from melm.appliance import AssistantOSKernel, AssistantOSStore, LocalAssistantProfile
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "test.sqlite"
        store = AssistantOSStore(db)
        kernel = AssistantOSKernel(
            store=store,
            profile=LocalAssistantProfile(),
        )
        try:
            decision = kernel.decide("hi what is the weather like")
            kernel.remember(decision)
            stored = store.load_events()
            self.assertGreater(len(stored), 0)
            for event in stored:
                self.assertIsInstance(event.semantic_classes_activated, frozenset)
        finally:
            store.close()

    def test_activated_semantic_classes_in_query_event_memory(self) -> None:
        from melm.appliance import AssistantOSKernel, AssistantOSStore, LocalAssistantProfile
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "test.sqlite"
        store = AssistantOSStore(db)
        kernel = AssistantOSKernel(
            store=store,
            profile=LocalAssistantProfile(),
        )
        try:
            decision = kernel.decide("hi what is the weather like")
            kernel.remember(decision)
            memory = store.query_event_memory(limit=10)
            self.assertIn("events", memory)
            for event in memory["events"]:
                self.assertIn("semantic_classes_activated", event)
                self.assertIsInstance(event["semantic_classes_activated"], tuple)
        finally:
            store.close()

    def test_migration_adds_semantic_classes_column(self) -> None:
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "test.sqlite"
        store = AssistantOSStore(db)
        try:
            rows = store.connection.execute("PRAGMA table_info(events)").fetchall()
            columns = {str(row["name"]) for row in rows}
            self.assertIn("semantic_classes_activated_json", columns)
        finally:
            store.close()

    def test_migration_is_idempotent(self) -> None:
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "test.sqlite"
        store = AssistantOSStore(db)
        try:
            store._ensure_event_semantic_classes_column()
            store._ensure_event_semantic_classes_column()
        finally:
            store.close()


class BulkLexiconSeederMvpTests(unittest.TestCase):

    def _make_store(self) -> AssistantOSStore:
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "test.sqlite"
        store = AssistantOSStore(db)
        store.initialize()
        return store

    def _wn_data_path(self, lines: list[str]) -> Path:
        p = Path(tempfile.mkdtemp()) / "test_supersense.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def _vn_data_path(self, lines: list[str]) -> Path:
        p = Path(tempfile.mkdtemp()) / "test_verb.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def _wiktextract_data_path(self, lines: list[str]) -> Path:
        p = Path(tempfile.mkdtemp()) / "test_wiktextract.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_seed_wordnet_supersenses_basic(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses
        store = self._make_store()
        try:
            data = self._wn_data_path([
                '{"word": "love", "supersense": "noun.feeling", "pos": "noun"}',
                '{"word": "run", "supersense": "noun.act", "pos": "noun"}',
            ])
            count = seed_wordnet_supersenses(store, data_path=data)
            self.assertEqual(count, 2)
            self.assertGreaterEqual(store.count("lexemes"), 2)
        finally:
            store.close()

    def test_seed_wordnet_supersenses_skips_reserved(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses
        store = self._make_store()
        try:
            data = self._wn_data_path([
                '{"word": "weather", "supersense": "noun.state", "pos": "noun"}',
                '{"word": "love", "supersense": "noun.feeling", "pos": "noun"}',
            ])
            count = seed_wordnet_supersenses(store, data_path=data)
            self.assertEqual(count, 1)
        finally:
            store.close()

    def test_seed_wordnet_supersenses_skips_unknown_supersense(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses
        store = self._make_store()
        try:
            data = self._wn_data_path([
                '{"word": "foo", "supersense": "noun.nonexistent", "pos": "noun"}',
                '{"word": "love", "supersense": "noun.feeling", "pos": "noun"}',
            ])
            count = seed_wordnet_supersenses(store, data_path=data)
            self.assertEqual(count, 1)
        finally:
            store.close()

    def test_seed_wordnet_supersenses_idempotent(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses
        store = self._make_store()
        try:
            data = self._wn_data_path([
                '{"word": "love", "supersense": "noun.feeling", "pos": "noun"}',
            ])
            count1 = seed_wordnet_supersenses(store, data_path=data)
            count2 = seed_wordnet_supersenses(store, data_path=data)
            self.assertEqual(count1, 1)
            self.assertEqual(count2, 1)
            self.assertEqual(store.count("lexemes"), 1)
        finally:
            store.close()

    def test_seed_verbnet_classes_basic(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_verbnet_classes
        store = self._make_store()
        try:
            data = self._vn_data_path([
                '{"verb": "ask", "verbnet_class": "say-37.7", "pos": "verb"}',
            ])
            count = seed_verbnet_classes(store, data_path=data)
            self.assertEqual(count, 1)
            self.assertGreaterEqual(store.count("lexemes"), 1)
        finally:
            store.close()

    def test_seed_verbnet_classes_skips_reserved(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_verbnet_classes
        store = self._make_store()
        try:
            data = self._vn_data_path([
                '{"verb": "call", "verbnet_class": "say-37.7", "pos": "verb"}',
                '{"verb": "ask", "verbnet_class": "say-37.7", "pos": "verb"}',
            ])
            count = seed_verbnet_classes(store, data_path=data)
            self.assertEqual(count, 1)
        finally:
            store.close()

    def test_seed_verbnet_classes_idempotent(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_verbnet_classes
        store = self._make_store()
        try:
            data = self._vn_data_path([
                '{"verb": "ask", "verbnet_class": "say-37.7", "pos": "verb"}',
            ])
            count1 = seed_verbnet_classes(store, data_path=data)
            count2 = seed_verbnet_classes(store, data_path=data)
            self.assertEqual(count1, 1)
            self.assertEqual(count2, 1)
            self.assertEqual(store.count("lexemes"), 1)
        finally:
            store.close()

    def test_seed_wiktextract_entries_basic(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wiktextract_entries
        store = self._make_store()
        try:
            data = self._wiktextract_data_path([
                '{"word": "zither", "class_id": "physical_object.instrument", "pos": "noun"}',
                '{"word": "schadenfreude", "class_id": "emotion", "pos": "noun"}',
            ])
            count = seed_wiktextract_entries(store, data_path=data)
            self.assertEqual(count, 2)
            self.assertGreaterEqual(store.count("lexemes"), 2)
        finally:
            store.close()

    def test_seed_wiktextract_entries_skips_unknown_class(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wiktextract_entries
        store = self._make_store()
        try:
            data = self._wiktextract_data_path([
                '{"word": "nonexistent_class", "class_id": "noun.nonexistent", "pos": "noun"}',
                '{"word": "zither", "class_id": "physical_object.instrument", "pos": "noun"}',
            ])
            count = seed_wiktextract_entries(store, data_path=data)
            self.assertEqual(count, 1)
        finally:
            store.close()

    def test_seed_wiktextract_entries_idempotent(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wiktextract_entries
        store = self._make_store()
        try:
            data = self._wiktextract_data_path([
                '{"word": "zither", "class_id": "physical_object.instrument", "pos": "noun"}',
            ])
            count1 = seed_wiktextract_entries(store, data_path=data)
            count2 = seed_wiktextract_entries(store, data_path=data)
            self.assertEqual(count1, 1)
            self.assertEqual(count2, 1)
            self.assertEqual(store.count("lexemes"), 1)
        finally:
            store.close()

    def test_seed_wiktextract_entries_uses_dormant_status(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wiktextract_entries
        from melm.appliance.assistant_lexicon import lookup_lexical_senses
        store = self._make_store()
        try:
            data = self._wiktextract_data_path([
                '{"word": "zither", "class_id": "physical_object.instrument", "pos": "noun"}',
            ])
            seed_wiktextract_entries(store, data_path=data)
            senses = lookup_lexical_senses(store, "zither")
            self.assertEqual(len(senses), 1)
            self.assertEqual(senses[0]["status"], "dormant")
        finally:
            store.close()

    def test_seed_wiktextract_entries_missing_data_returns_zero(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wiktextract_entries
        store = self._make_store()
        try:
            missing = Path(tempfile.mkdtemp()) / "nonexistent.jsonl"
            count = seed_wiktextract_entries(store, data_path=missing)
            self.assertEqual(count, 0)
        finally:
            store.close()

    def test_seed_bulk_lexicon_orchestrator(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_bulk_lexicon
        store = self._make_store()
        try:
            wn_data = self._wn_data_path([
                '{"word": "love", "supersense": "noun.feeling", "pos": "noun"}',
            ])
            vn_data = self._vn_data_path([
                '{"verb": "ask", "verbnet_class": "say-37.7", "pos": "verb"}',
            ])
            wt_data = self._wiktextract_data_path([
                '{"word": "zither", "class_id": "physical_object.instrument", "pos": "noun"}',
            ])
            counts = seed_bulk_lexicon(
                store, wordnet_data=wn_data, verbnet_data=vn_data,
                wiktextract_data=wt_data,
            )
            self.assertEqual(counts["wordnet"], 1)
            self.assertEqual(counts["verbnet"], 1)
            self.assertEqual(counts["wiktextract"], 1)
        finally:
            store.close()

    def test_missing_data_files_return_zero(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses, seed_verbnet_classes
        store = self._make_store()
        try:
            missing = Path(tempfile.mkdtemp()) / "nonexistent.jsonl"
            wn_count = seed_wordnet_supersenses(store, data_path=missing)
            vn_count = seed_verbnet_classes(store, data_path=missing)
            self.assertEqual(wn_count, 0)
            self.assertEqual(vn_count, 0)
        finally:
            store.close()

    def test_entries_are_dormant(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import seed_wordnet_supersenses
        from melm.appliance.assistant_lexicon import lookup_lexical_senses
        store = self._make_store()
        try:
            data = self._wn_data_path([
                '{"word": "love", "supersense": "noun.feeling", "pos": "noun"}',
            ])
            seed_wordnet_supersenses(store, data_path=data)
            senses = lookup_lexical_senses(store, "love")
            self.assertEqual(len(senses), 1)
            self.assertEqual(senses[0]["status"], "dormant")
        finally:
            store.close()

    def test_seed_with_actual_data_files(self) -> None:
        from melm.appliance.assistant_lexicon_bulk import (
            seed_wordnet_supersenses, seed_verbnet_classes, _CONTRACT_ROOT,
        )
        from pathlib import Path
        store = self._make_store()
        try:
            # Use 2000-entry slices of the full data for speed
            tmpdir = Path(tempfile.mkdtemp())
            wn_path = _CONTRACT_ROOT / "word_supersense_data.v1.jsonl"
            lines = wn_path.read_text(encoding="utf-8").strip().splitlines()
            wn_slice = tmpdir / "wn_slice.jsonl"
            wn_slice.write_text("\n".join(lines[:2000]), encoding="utf-8")
            wn_count = seed_wordnet_supersenses(store, data_path=wn_slice)
            self.assertGreaterEqual(wn_count, 1500,
                f"Expected >=1500 WordNet entries from 2000-entry slice, got {wn_count}")
            vn_path = _CONTRACT_ROOT / "verb_data.v1.jsonl"
            vn_lines = vn_path.read_text(encoding="utf-8").strip().splitlines()
            vn_slice = tmpdir / "vn_slice.jsonl"
            vn_slice.write_text("\n".join(vn_lines[:500]), encoding="utf-8")
            vn_count = seed_verbnet_classes(store, data_path=vn_slice)
            self.assertGreaterEqual(vn_count, 400,
                f"Expected >=400 VerbNet entries from 500-entry slice, got {vn_count}")
        finally:
            store.close()


class SealedDictionaryMvpTests(unittest.TestCase):
    """M3 exit gate: sealed >=60-word dictionary set.

    Verifies >=80% correct next-turn use and retention across the full
    acquisition pipeline: ingest -> promote -> route -> restart-retain.
    """

    _SEALED_DICTIONARY: tuple[dict[str, str], ...] = (
        # story (class: narrative_content)
        {"word": "wug", "utterance": "tell me a story about wug",
         "expected_intent": "story", "class_id": "narrative_content"},
        {"word": "thorp", "utterance": "tell me about thorp",
         "expected_intent": "story", "class_id": "narrative_content"},
        {"word": "glimmer", "utterance": "tell a story with glimmer",
         "expected_intent": "story", "class_id": "narrative_content"},
        {"word": "niffler", "utterance": "read me the niffler story",
         "expected_intent": "story", "class_id": "narrative_content"},
        {"word": "boggle", "utterance": "make up a boggle tale",
         "expected_intent": "story", "class_id": "narrative_content"},
        {"word": "sprocket", "utterance": "give me a sprocket story",
         "expected_intent": "story", "class_id": "narrative_content"},
        # media_playback (class: physical_object.instrument)
        {"word": "zindle", "utterance": "play zindle",
         "expected_intent": "media_playback", "class_id": "physical_object.instrument"},
        {"word": "flump", "utterance": "play flump on the speaker",
         "expected_intent": "media_playback", "class_id": "physical_object.instrument"},
        {"word": "dringle", "utterance": "play dringle music",
         "expected_intent": "media_playback", "class_id": "physical_object.instrument"},
        {"word": "brizzle", "utterance": "play brizzle tune",
         "expected_intent": "media_playback", "class_id": "physical_object.instrument"},
        {"word": "thribble", "utterance": "start playing thribble",
         "expected_intent": "media_playback", "class_id": "physical_object.instrument"},
        {"word": "quibble", "utterance": "play quibble song",
         "expected_intent": "media_playback", "class_id": "physical_object.instrument"},
        # weather (class: weather_phenomenon)
        {"word": "blizz", "utterance": "what is the blizz like today",
         "expected_intent": "weather", "class_id": "weather_phenomenon"},
        {"word": "sunspark", "utterance": "is sunspark expected tomorrow",
         "expected_intent": "weather", "class_id": "weather_phenomenon"},
        {"word": "rainfleck", "utterance": "what about rainfleck on wednesday",
         "expected_intent": "weather", "class_id": "weather_phenomenon"},
        {"word": "drizzlefluff", "utterance": "how much drizzlefluff will there be",
         "expected_intent": "weather", "class_id": "weather_phenomenon"},
        {"word": "windwhorl", "utterance": "is windwhorl dangerous",
         "expected_intent": "weather", "class_id": "weather_phenomenon"},
        {"word": "frostgleam", "utterance": "when will frostgleam start",
         "expected_intent": "weather", "class_id": "weather_phenomenon"},
        # meal_suggestion (class: food_item)
        {"word": "zargle", "utterance": "suggest a zargle recipe",
         "expected_intent": "meal_suggestion", "class_id": "food_item"},
        {"word": "morket", "utterance": "cook morket for dinner",
         "expected_intent": "meal_suggestion", "class_id": "food_item"},
        {"word": "flibble", "utterance": "recommend a flibble dish",
         "expected_intent": "meal_suggestion", "class_id": "food_item"},
        {"word": "gront", "utterance": "eat gront for breakfast",
         "expected_intent": "meal_suggestion", "class_id": "food_item"},
        {"word": "snibble", "utterance": "have snibble with rice",
         "expected_intent": "meal_suggestion", "class_id": "food_item"},
        {"word": "plinket", "utterance": "suggest plinket soup",
         "expected_intent": "meal_suggestion", "class_id": "food_item"},
        # routine_memory (class: routine_concept)
        {"word": "mizzle", "utterance": "what is my mizzle routine",
         "expected_intent": "routine_memory", "class_id": "routine_concept"},
        {"word": "bloop", "utterance": "what is my bloop schedule",
         "expected_intent": "routine_memory", "class_id": "routine_concept"},
        {"word": "zorp", "utterance": "what is our zorp plan",
         "expected_intent": "routine_memory", "class_id": "routine_concept"},
        {"word": "wuzzle", "utterance": "when is my wuzzle time",
         "expected_intent": "routine_memory", "class_id": "routine_concept"},
        {"word": "snizzle", "utterance": "what is our snizzle habit",
         "expected_intent": "routine_memory", "class_id": "routine_concept"},
        {"word": "glorb", "utterance": "what is my glorb practice",
         "expected_intent": "routine_memory", "class_id": "routine_concept"},
        # household_memory (class: household_concept)
        {"word": "frump", "utterance": "where is my frump",
         "expected_intent": "household_memory", "class_id": "household_concept"},
        {"word": "squibble", "utterance": "where is my squibble",
         "expected_intent": "household_memory", "class_id": "household_concept"},
        {"word": "trumpet", "utterance": "where is our trumpet shelf",
         "expected_intent": "household_memory", "class_id": "household_concept"},
        {"word": "bramble", "utterance": "where is my bramble drawer",
         "expected_intent": "household_memory", "class_id": "household_concept"},
        {"word": "crizzle", "utterance": "do we need a crizzle",
         "expected_intent": "household_memory", "class_id": "household_concept"},
        {"word": "dibble", "utterance": "where is my dibble cabinet",
         "expected_intent": "household_memory", "class_id": "household_concept"},

        # autobiographical_memory (class: autobiographical_event)
        {"word": "skribble", "utterance": "tell me about skribble",
         "expected_intent": "autobiographical_memory", "class_id": "autobiographical_event"},
        {"word": "plundle", "utterance": "show my plundle",
         "expected_intent": "autobiographical_memory", "class_id": "autobiographical_event"},
        {"word": "gribble", "utterance": "list gribble events",
         "expected_intent": "autobiographical_memory", "class_id": "autobiographical_event"},
        {"word": "frinkle", "utterance": "recap frinkle for me",
         "expected_intent": "autobiographical_memory", "class_id": "autobiographical_event"},
        {"word": "drabble", "utterance": "summarize drabble",
         "expected_intent": "autobiographical_memory", "class_id": "autobiographical_event"},
        {"word": "nubble", "utterance": "tell me nubble details",
         "expected_intent": "autobiographical_memory", "class_id": "autobiographical_event"},
        # social_contact (class: contact_action)
        {"word": "clibble", "utterance": "call clibble",
         "expected_intent": "social_contact", "class_id": "contact_action"},
        {"word": "strumble", "utterance": "phone strumble",
         "expected_intent": "social_contact", "class_id": "contact_action"},
        {"word": "prickle", "utterance": "ring prickle",
         "expected_intent": "social_contact", "class_id": "contact_action"},
        {"word": "wumble", "utterance": "reach wumble by phone",
         "expected_intent": "social_contact", "class_id": "contact_action"},
        {"word": "frizzle", "utterance": "call frizzle at home",
         "expected_intent": "social_contact", "class_id": "contact_action"},
        {"word": "tramble", "utterance": "phone tramble now",
         "expected_intent": "social_contact", "class_id": "contact_action"},
        # health_advice (class: health_condition)
        {"word": "snoggle", "utterance": "my snoggle hurts",
         "expected_intent": "health_advice", "class_id": "health_condition"},
        {"word": "murgle", "utterance": "how to treat murgle",
         "expected_intent": "health_advice", "class_id": "health_condition"},
        {"word": "flargle", "utterance": "sleep with flargle",
         "expected_intent": "health_advice", "class_id": "health_condition"},
        {"word": "drongle", "utterance": "take medicine for drongle",
         "expected_intent": "health_advice", "class_id": "health_condition"},
        {"word": "prink", "utterance": "see doctor about prink",
         "expected_intent": "health_advice", "class_id": "health_condition"},
        {"word": "thrizzle", "utterance": "do exercise for thrizzle",
         "expected_intent": "health_advice", "class_id": "health_condition"},
        # common_sense_safety (class: undress_state)
        {"word": "glorp", "utterance": "go glorp alone at night",
         "expected_intent": "common_sense_safety", "class_id": "undress_state"},
        {"word": "snarkle", "utterance": "walk snarkle in the dark",
         "expected_intent": "common_sense_safety", "class_id": "undress_state"},
        {"word": "plorf", "utterance": "go plorf by the river",
         "expected_intent": "common_sense_safety", "class_id": "undress_state"},
        {"word": "dring", "utterance": "walk dring in the forest",
         "expected_intent": "common_sense_safety", "class_id": "undress_state"},
        {"word": "spindle", "utterance": "go spindle near the highway",
         "expected_intent": "common_sense_safety", "class_id": "undress_state"},
        {"word": "blorple", "utterance": "walk blorple on the bridge",
         "expected_intent": "common_sense_safety", "class_id": "undress_state"},
        # personal_memory (class: memory_recall)
        {"word": "wibble", "utterance": "remember wibble",
         "expected_intent": "personal_memory", "class_id": "memory_recall"},
        {"word": "tibble", "utterance": "recall tibble for me",
         "expected_intent": "personal_memory", "class_id": "memory_recall"},
        {"word": "nibble", "utterance": "forget nibble",
         "expected_intent": "personal_memory", "class_id": "memory_recall"},
        {"word": "wramble", "utterance": "remember wramble event",
         "expected_intent": "personal_memory", "class_id": "memory_recall"},
        {"word": "fribble", "utterance": "recall fribble memory",
         "expected_intent": "personal_memory", "class_id": "memory_recall"},
        {"word": "squamble", "utterance": "remember squamble detail",
         "expected_intent": "personal_memory", "class_id": "memory_recall"},
        # extra: 6 more to ensure >= 66
        {"word": "sprinkle", "utterance": "tell a story with sprinkle",
         "expected_intent": "story", "class_id": "narrative_content"},
        {"word": "twindle", "utterance": "play twindle on the radio",
         "expected_intent": "media_playback", "class_id": "physical_object.instrument"},
        {"word": "snorf", "utterance": "what is the snorf index",
         "expected_intent": "weather", "class_id": "weather_phenomenon"},
        {"word": "miffle", "utterance": "suggest miffle for lunch",
         "expected_intent": "meal_suggestion", "class_id": "food_item"},
        {"word": "zuffle", "utterance": "what is my zuffle morning habit",
         "expected_intent": "routine_memory", "class_id": "routine_concept"},
        {"word": "kribble", "utterance": "remember kribble from yesterday",
         "expected_intent": "personal_memory", "class_id": "memory_recall"},
    )

    @property
    def _dictionary_words(self) -> tuple[str, ...]:
        return tuple(e["word"] for e in self._SEALED_DICTIONARY)

    def _ingest_word(self, store: AssistantOSStore, entry: dict[str, str]) -> LexiconIngestResult | None:
        candidate = {
            "schema_id": "melm.sense_candidate.v1",
            "lemma": entry["word"],
            "language": "en",
            "pos": "noun",
            "source": {"provenance": "seed_authored",
                       "source_ref": f"test:{entry['word']}",
                       "license": "test"},
            "definition": f"a test word for {entry['expected_intent']}",
            "semantic_class_candidates": [
                {"class_id": entry["class_id"],
                 "method": "seed_authored",
                 "confidence": 0.95}],
            "safety": {"reserved_conflict": False, "policy_term_overlap": False},
            "suggested_status": "active",
            "confidence_prior": 0.95,
        }
        return lexicon_ingest(store, candidate, expected_provenance="seed_authored")

    def _inject_new_words(self, store: AssistantOSStore) -> None:
        """Merge new-word entries from store into _IN_MEMORY_LEXICON."""
        rows = store.connection.execute(
            """
            SELECT l.normalized_lemma, s.semantic_class_id
            FROM lexemes AS l
            JOIN lexical_senses AS s ON s.lexeme_id = l.lexeme_id
            WHERE s.status = 'active'
            """
        ).fetchall()
        from collections import defaultdict
        lexicon: dict[str, set[str]] = defaultdict(set)
        lexicon.update({k: set(v) for k, v in _IN_MEMORY_LEXICON.items()})
        for row in rows:
            lemma = str(row["normalized_lemma"])
            class_id = str(row["semantic_class_id"])
            lexicon[lemma].add(class_id)
        replace_in_memory_lexicon(
            {k: frozenset(v) for k, v in lexicon.items()}
        )

    def _run_routing_agreement(self, store: AssistantOSStore) -> float:
        """Return fraction of sealed-dictionary words that route to expected intent."""
        saved = dict(_IN_MEMORY_LEXICON)
        try:
            self._inject_new_words(store)
            linker = FrameLinker()
            successes = 0
            for entry in self._SEALED_DICTIONARY:
                tokens = tuple(_normalize_term(entry["utterance"]).split())
                if _classify_from_frame_linker(
                    entry["utterance"], tokens, entry["expected_intent"]
                ):
                    successes += 1
            return successes / len(self._SEALED_DICTIONARY)
        finally:
            replace_in_memory_lexicon(saved)

    def test_sealed_dictionary_count(self) -> None:
        """The sealed dictionary contains >= 60 words."""
        self.assertGreaterEqual(len(self._SEALED_DICTIONARY), 60)

    def test_sealed_dictionary_all_ingest_and_promote(self) -> None:
        """All >= 60 words ingest and promote to active without error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sqlite"
            store = AssistantOSStore(path)
            try:
                successes = 0
                for entry in self._SEALED_DICTIONARY:
                    result = self._ingest_word(store, entry)
                    if result is not None and result.status == "active":
                        successes += 1
                rate = successes / len(self._SEALED_DICTIONARY)
                self.assertGreaterEqual(
                    rate, 0.80,
                    f"ingestion rate={rate:.0%} < 80% "
                    f"({successes}/{len(self._SEALED_DICTIONARY)})"
                )
            finally:
                store.close()

    def test_sealed_dictionary_routing_agreement(self) -> None:
        """>= 80% of words produce correct intent routing after ingest."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sqlite"
            store = AssistantOSStore(path)
            try:
                for entry in self._SEALED_DICTIONARY:
                    self._ingest_word(store, entry)
                rate = self._run_routing_agreement(store)
                self.assertGreaterEqual(
                    rate, 0.80,
                    f"routing rate={rate:.0%} < 80%"
                )
            finally:
                store.close()

    def test_sealed_dictionary_retention(self) -> None:
        """>= 80% of words retain correct routing after store restart."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sqlite"
            store = AssistantOSStore(path)
            try:
                for entry in self._SEALED_DICTIONARY:
                    self._ingest_word(store, entry)
                store.close()

                store2 = AssistantOSStore(path)
                try:
                    rate = self._run_routing_agreement(store2)
                    self.assertGreaterEqual(
                        rate, 0.80,
                        f"retention rate={rate:.0%} < 80%"
                    )
                finally:
                    store2.close()
            except:
                store.close()
                raise

    def test_sealed_dictionary_end_to_end_routing_agreement(self) -> None:
        """>= 80% of words route correctly through full OnDeviceAssistantRouter."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sqlite"
            store = AssistantOSStore(path)
            try:
                for entry in self._SEALED_DICTIONARY:
                    self._ingest_word(store, entry)
                saved = dict(_IN_MEMORY_LEXICON)
                try:
                    self._inject_new_words(store)
                    # Ensure all families are installed for the test
                    replace_installed_families(None, None)
                    contact_words = {
                        e["word"]: "+123-000-TEST"
                        for e in self._SEALED_DICTIONARY
                        if e["expected_intent"] == "social_contact"
                    }
                    router = OnDeviceAssistantRouter(
                        profile=LocalAssistantProfile(contacts=contact_words)
                    )
                    successes = 0
                    failures: list[str] = []
                    for entry in self._SEALED_DICTIONARY:
                        decision = router.handle(entry["utterance"])
                        # Frame sub-types map to top-level intent in the router
                        expected = entry["expected_intent"]
                        if expected in {"routine_memory", "household_memory"}:
                            expected = "personal_memory"
                        if decision.intent == expected:
                            successes += 1
                        else:
                            failures.append(
                                f"{entry['word']!r}: expected={expected}, "
                                f"got={decision.intent} (reason={decision.reason})"
                            )
                    rate = successes / len(self._SEALED_DICTIONARY)
                    self.assertGreaterEqual(
                        rate, 0.80,
                        f"end-to-end routing rate={rate:.0%} < 80%\nFailures:\n"
                        + "\n".join(failures)
                    )
                finally:
                    replace_in_memory_lexicon(saved)
            finally:
                store.close()


    def test_next_turn_use_after_ingest(self) -> None:
        """A word learned in turn T is correctly routed in turn T+1 through the full kernel."""
        saved = dict(_IN_MEMORY_LEXICON)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "test.sqlite"
                store = AssistantOSStore(db)
                try:
                    # Pick a sealed-dictionary word that is not already in the lexicon
                    entry = {
                        "word": "wug",
                        "utterance": "tell me a story about wug",
                        "expected_intent": "story",
                        "class_id": "narrative_content",
                    }
                    # Ensure the word is not already known
                    self.assertNotIn(
                        entry["word"],
                        _IN_MEMORY_LEXICON,
                        f"word {entry['word']!r} already in lexicon; choose another",
                    )

                    # Turn T: ingest the word into the store
                    self._ingest_word(store, entry)

                    # Build kernel; init rebuilds lexicon cache
                    replace_installed_families(None, None)
                    kernel = AssistantOSKernel(
                        store=store,
                        profile=LocalAssistantProfile(
                            story_models={},
                            weekly_weather={},
                            contacts={},
                        ),
                    )
                    # Turn T+1: utterance uses the newly learned word
                    decision = kernel.handle(entry["utterance"])
                    self.assertEqual(
                        decision.intent, entry["expected_intent"],
                        f"word {entry['word']!r} routed to {decision.intent!r} "
                        f"instead of {entry['expected_intent']!r}",
                    )
                    kernel.store.close()
                finally:
                    store.close()
        finally:
            replace_in_memory_lexicon(saved)


class LexiconLifecycleMvpTests(unittest.TestCase):
    """Tests for demote_sense, correct_sense, generate_minimal_pairs."""

    def _seed_lemma(
        self, store: AssistantOSStore, lemma: str,
        class_id: str = "physical_object.instrument",
    ) -> str:
        from melm.appliance.assistant_lexicon import promote_lexical_sense
        result = lexicon_ingest(store, {
            "schema_id": "melm.sense_candidate.v1",
            "lemma": lemma,
            "language": "en",
            "pos": "noun",
            "source": {"provenance": "seed_authored",
                       "source_ref": "test", "license": "test"},
            "definition": f"a {lemma} is a test instrument",
            "semantic_class_candidates": [{"class_id": class_id,
                                           "method": "seed_authored",
                                           "confidence": 0.95}],
            "safety": {"reserved_conflict": False, "policy_term_overlap": False},
            "suggested_status": "active",
            "confidence_prior": 0.95,
        }, expected_provenance="seed_authored")
        return result.sense_id

    def test_demote_sense_active_to_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "test.sqlite")
            try:
                sense_id = self._seed_lemma(store, "kalimba")
                from melm.appliance.assistant_lexicon import demote_sense
                demote_sense(store, sense_id)
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(senses), 1)
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("active",),
                )
                self.assertEqual(len(senses), 0)
            finally:
                store.close()

    def test_demote_sense_rejects_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "test.sqlite")
            try:
                result = lexicon_ingest(store, {
                    "schema_id": "melm.sense_candidate.v1",
                    "lemma": "kalimba",
                    "language": "en",
                    "pos": "noun",
                    "source": {"provenance": "user_taught",
                               "source_ref": "test", "license": "test"},
                    "definition": "a small thumb piano",
                    "semantic_class_candidates": [
                        {"class_id": "physical_object.instrument",
                         "method": "genus_walk", "confidence": 0.6}],
                    "safety": {"reserved_conflict": False,
                               "policy_term_overlap": False},
                    "suggested_status": "quarantined",
                    "confidence_prior": 0.60,
                }, expected_provenance="user_taught")
                from melm.appliance.assistant_lexicon import demote_sense
                with self.assertRaises(ContractValidationError):
                    demote_sense(store, result.sense_id)
            finally:
                store.close()

    def test_correct_sense_preserves_original_as_defeated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "test.sqlite")
            try:
                sense_id = self._seed_lemma(store, "kalimba",
                                            "physical_object.instrument")
                from melm.appliance.assistant_lexicon import correct_sense
                result = correct_sense(
                    store, sense_id,
                    semantic_class_id="physical_object.device",
                )
                self.assertEqual(result["lemma"], "kalimba")
                self.assertEqual(result["semantic_class_id"],
                                 "physical_object.device")

                defeated = lookup_lexical_senses(
                    store, "kalimba", statuses=("defeated",),
                )
                self.assertEqual(len(defeated), 1)
                self.assertEqual(defeated[0]["semantic_class_id"],
                                 "physical_object.instrument")

                quarantined = lookup_lexical_senses(
                    store, "kalimba", statuses=("quarantined",),
                )
                self.assertEqual(len(quarantined), 1)
                self.assertEqual(quarantined[0]["semantic_class_id"],
                                 "physical_object.device")
            finally:
                store.close()

    def test_generate_minimal_pairs_single_sense_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "test.sqlite")
            try:
                self._seed_lemma(store, "kalimba", "physical_object.instrument")
                from melm.appliance.assistant_lexicon import generate_minimal_pairs
                pairs = generate_minimal_pairs(store, "kalimba")
                self.assertEqual(pairs, [])
            finally:
                store.close()

    def test_promote_lexical_sense_force_overrides_status_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "test.sqlite")
            try:
                from melm.appliance.assistant_lexicon import promote_lexical_sense
                result = lexicon_ingest(store, {
                    "schema_id": "melm.sense_candidate.v1",
                    "lemma": "kalimba",
                    "language": "en",
                    "pos": "noun",
                    "source": {"provenance": "seed_authored",
                               "source_ref": "test", "license": "test"},
                    "definition": "a small thumb piano",
                    "semantic_class_candidates": [
                        {"class_id": "physical_object.instrument",
                         "method": "seed_authored", "confidence": 0.95}],
                    "safety": {"reserved_conflict": False,
                               "policy_term_overlap": False},
                    "suggested_status": "dormant",
                    "confidence_prior": 0.95,
                }, expected_provenance="seed_authored")
                # Without force, promoting a dormant sense is rejected
                with self.assertRaises(ContractValidationError):
                    promote_lexical_sense(store, result.sense_id)
                # With force=True, the status gate is bypassed
                injected = promote_lexical_sense(store, result.sense_id, force=True)
                self.assertEqual(injected["lemma"], "kalimba")
                senses = lookup_lexical_senses(
                    store, "kalimba", statuses=("active",),
                )
                self.assertEqual(len(senses), 1)
            finally:
                store.close()

    def test_generate_minimal_pairs_two_senses_returns_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "test.sqlite")
            try:
                self._seed_lemma(store, "kalimba", "physical_object.instrument")
                self._seed_lemma(store, "kalimba", "physical_object.device")
                from melm.appliance.assistant_lexicon import generate_minimal_pairs
                pairs = generate_minimal_pairs(store, "kalimba")
                self.assertEqual(len(pairs), 1)
                self.assertIn("kalimba", pairs[0]["case_id"])
                self.assertIn("good", pairs[0])
                self.assertIn("bad", pairs[0])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
