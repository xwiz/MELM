import tempfile
from pathlib import Path
import unittest

from melm.appliance import (
    acquire_definition,
    AssistantOSStore,
    benchmark_lexicon_lookup,
    lexicon_ingest,
    lookup_lexical_senses,
    lookup_lexical_senses_tiered,
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


if __name__ == "__main__":
    unittest.main()
