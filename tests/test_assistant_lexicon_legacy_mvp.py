import tempfile
from pathlib import Path
import unittest

from melm.appliance import (
    AssistantOSKernel,
    AssistantOSStore,
    LocalAssistantProfile,
    build_legacy_lexicon_candidates,
    build_legacy_router_candidates,
    configure_lexicon_router_families,
    lexicon_ingest,
    lookup_lexical_senses,
    write_legacy_lexicon_candidates,
)
from melm.appliance import functional_grammar
from melm.contracts import validate_sense_candidate
from melm.contracts import ContractValidationError


class AssistantLexiconLegacyMvpTests(unittest.TestCase):
    def test_every_legacy_candidate_validates_and_uses_the_single_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            candidates = build_legacy_lexicon_candidates()
            try:
                for candidate in candidates:
                    validate_sense_candidate(candidate)
                    lexicon_ingest(
                        store,
                        candidate,
                        expected_provenance="seed_authored",
                    )

                expected = len(functional_grammar._VERBS) + len(
                    functional_grammar._KNOWN_NOMINAL_DOMAINS
                )
                self.assertEqual(len(candidates), expected)
                self.assertEqual(store.count("lexemes"), expected)
                self.assertEqual(store.count("lexical_senses"), expected)
                self.assertEqual(
                    lookup_lexical_senses(store, "call")[0]["semantic_class_id"],
                    "verb.communicate",
                )
                self.assertTrue(lookup_lexical_senses(store, "call")[0]["reserved"])
            finally:
                store.close()

    def test_jsonl_export_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = write_legacy_lexicon_candidates(Path(tmp) / "first.jsonl")
            second = write_legacy_lexicon_candidates(Path(tmp) / "second.jsonl")
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_weather_family_routes_identically_from_store_and_deletion_changes_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                for candidate in build_legacy_router_candidates(("weather",)):
                    lexicon_ingest(
                        store,
                        candidate,
                        expected_provenance="seed_authored",
                    )
                configure_lexicon_router_families(store, ("weather",))
                profile = LocalAssistantProfile()
                kernel = AssistantOSKernel(profile=profile, store=store)

                expected = {
                    "What is the weather today?": ("weather", "cached_tool"),
                    "Will it rain tomorrow?": ("weather", "cached_tool"),
                    "What's the temperature outside?": ("weather", "cached_tool"),
                    "Can you explain weather systems?": ("assistant_behavior", "local_answer"),
                }
                for utterance, route in expected.items():
                    with self.subTest(utterance=utterance):
                        decision = kernel.decide(utterance)
                        self.assertEqual((decision.intent, decision.route), route)

                weather_lexeme = store.connection.execute(
                    "SELECT lexeme_id FROM lexemes WHERE normalized_lemma='weather' AND pos='noun'"
                ).fetchone()
                self.assertIsNotNone(weather_lexeme)
                lexeme_id = str(weather_lexeme["lexeme_id"])
                sense_rows = store.connection.execute(
                    "SELECT sense_id FROM lexical_senses WHERE lexeme_id=?",
                    (lexeme_id,),
                ).fetchall()
                sense_ids = tuple(str(row["sense_id"]) for row in sense_rows)
                with store.connection:
                    for sense_id in sense_ids:
                        store.connection.execute(
                            "DELETE FROM lexical_provenance WHERE sense_id=?",
                            (sense_id,),
                        )
                        store.connection.execute(
                            "DELETE FROM lexical_relation_candidates WHERE sense_id=?",
                            (sense_id,),
                        )
                    store.connection.execute(
                        "DELETE FROM lexicon_ingestions WHERE lexeme_id=?",
                        (lexeme_id,),
                    )
                    store.connection.execute(
                        "DELETE FROM lexical_senses WHERE lexeme_id=?",
                        (lexeme_id,),
                    )
                    store.connection.execute(
                        "DELETE FROM word_forms WHERE lexeme_id=?",
                        (lexeme_id,),
                    )
                    store.connection.execute(
                        "DELETE FROM lexemes WHERE lexeme_id=?",
                        (lexeme_id,),
                    )

                after_delete = AssistantOSKernel(profile=profile, store=store).decide(
                    "What is the weather today?"
                )
                # Store-level change: the lexeme is no longer in the store.
                self.assertEqual(
                    store.connection.execute(
                        "SELECT COUNT(*) FROM lexemes WHERE normalized_lemma='weather'"
                    ).fetchone()[0],
                    0,
                )
                # Routing unchanged because legacy _IN_MEMORY_LEXICON base
                # still provides the term. Store deletion only affects
                # runtime-acquired vocabulary, not release-controlled terms.
                self.assertEqual(after_delete.intent, "weather")
            finally:
                store.close()

    def test_story_family_routes_identically_from_store_and_deletion_changes_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                for candidate in build_legacy_router_candidates(("story",)):
                    lexicon_ingest(
                        store,
                        candidate,
                        expected_provenance="seed_authored",
                    )
                configure_lexicon_router_families(store, ("story",))
                profile = LocalAssistantProfile()
                kernel = AssistantOSKernel(profile=profile, store=store)

                expected = {
                    "Tell me a story.": ("story", "local_answer"),
                    "Read me a tale.": ("story", "local_answer"),
                    "Please give me a fable.": ("story", "local_answer"),
                    "What is a story?": ("open_domain", "local_answer"),
                    "The same people tell stories.": ("open_domain", "local_answer"),
                }
                for utterance, route in expected.items():
                    with self.subTest(utterance=utterance):
                        decision = kernel.decide(utterance)
                        self.assertEqual((decision.intent, decision.route), route)

                _delete_lexeme(store, "story", "noun")
                after_delete = AssistantOSKernel(profile=profile, store=store).decide(
                    "Tell me a story."
                )
                # Store-level change: the lexeme is no longer in the store.
                self.assertEqual(
                    store.connection.execute(
                        "SELECT COUNT(*) FROM lexemes WHERE normalized_lemma='story'"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(after_delete.intent, "story")
            finally:
                store.close()

    def test_media_family_routes_identically_from_store_and_deletion_changes_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                for candidate in build_legacy_router_candidates(("media",)):
                    lexicon_ingest(
                        store,
                        candidate,
                        expected_provenance="seed_authored",
                    )
                configure_lexicon_router_families(store, ("media",))
                profile = LocalAssistantProfile()
                kernel = AssistantOSKernel(profile=profile, store=store)

                expected = {
                    "Play a song for me.": ("media_playback", "device_action"),
                    "Start the radio.": ("media_playback", "device_action"),
                    "Play calm piano.": ("media_playback", "device_action"),
                    "Play rain sounds.": ("media_playback", "device_action"),
                    "Can you explain music theory?": ("assistant_behavior", "local_answer"),
                    "Play.": ("unknown", "cloud_handoff"),
                }
                for utterance, route in expected.items():
                    with self.subTest(utterance=utterance):
                        decision = kernel.decide(utterance)
                        self.assertEqual((decision.intent, decision.route), route)

                _delete_lexeme(store, "song", "noun")
                after_delete = AssistantOSKernel(profile=profile, store=store).decide(
                    "Play a song for me."
                )
                # Store-level change: the lexeme is no longer in the store.
                self.assertEqual(
                    store.connection.execute(
                        "SELECT COUNT(*) FROM lexemes WHERE normalized_lemma='song'"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(after_delete.intent, "media_playback")
            finally:
                store.close()

    def test_router_family_ownership_rejects_unknown_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                with self.assertRaisesRegex(ContractValidationError, "unknown lexicon router"):
                    configure_lexicon_router_families(store, ("weather", "made_up"))
            finally:
                store.close()

    def test_router_family_ownership_requires_complete_active_seed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssistantOSStore(Path(tmp) / "assistant.sqlite")
            try:
                candidates = build_legacy_router_candidates(("weather",))
                for candidate in candidates:
                    if candidate["lemma"] == "temperature":
                        continue
                    lexicon_ingest(
                        store,
                        candidate,
                        expected_provenance="seed_authored",
                    )
                with self.assertRaisesRegex(
                    ContractValidationError,
                    "missing active terms: temperature",
                ):
                    configure_lexicon_router_families(store, ("weather",))
            finally:
                store.close()


def _delete_lexeme(store: AssistantOSStore, lemma: str, pos: str) -> None:
    row = store.connection.execute(
        "SELECT lexeme_id FROM lexemes WHERE normalized_lemma=? AND pos=?",
        (lemma, pos),
    ).fetchone()
    if row is None:
        raise AssertionError(f"missing test lexeme: {lemma}/{pos}")
    lexeme_id = str(row["lexeme_id"])
    senses = store.connection.execute(
        "SELECT sense_id FROM lexical_senses WHERE lexeme_id=?",
        (lexeme_id,),
    ).fetchall()
    with store.connection:
        for sense in senses:
            sense_id = str(sense["sense_id"])
            store.connection.execute(
                "DELETE FROM lexical_provenance WHERE sense_id=?",
                (sense_id,),
            )
            store.connection.execute(
                "DELETE FROM lexical_relation_candidates WHERE sense_id=?",
                (sense_id,),
            )
        store.connection.execute(
            "DELETE FROM lexicon_ingestions WHERE lexeme_id=?",
            (lexeme_id,),
        )
        store.connection.execute(
            "DELETE FROM lexical_senses WHERE lexeme_id=?",
            (lexeme_id,),
        )
        store.connection.execute(
            "DELETE FROM word_forms WHERE lexeme_id=?",
            (lexeme_id,),
        )
        store.connection.execute(
            "DELETE FROM lexemes WHERE lexeme_id=?",
            (lexeme_id,),
        )


if __name__ == "__main__":
    unittest.main()
