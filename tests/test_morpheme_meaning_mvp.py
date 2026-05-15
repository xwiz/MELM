import unittest
from pathlib import Path

from melm.semantics import evaluate_meaning_mvp, load_meaning_corpus
from melm.semantics.morpheme_meaning import MeaningInferencer


CORPUS = Path("benchmarks/morpheme_meaning_mvp.jsonl")


class MorphemeMeaningMvpTests(unittest.TestCase):
    def test_corpus_loads_and_passes_mvp_cases(self) -> None:
        corpus = load_meaning_corpus(CORPUS)

        report = evaluate_meaning_mvp(corpus)

        self.assertEqual(len(corpus.components), 34)
        self.assertEqual(len(corpus.lexemes), 3)
        self.assertEqual(report.word_cases, 22)
        self.assertEqual(report.utterance_cases, 6)
        self.assertEqual(report.word_accuracy, 1.0)
        self.assertEqual(report.utterance_accuracy, 1.0)
        self.assertEqual(report.overall_accuracy, 1.0)

    def test_infers_rewelcome_from_prefix_and_welcome_roots(self) -> None:
        inferencer = MeaningInferencer(load_meaning_corpus(CORPUS))

        inference = inferencer.infer_word("rewelcome")

        self.assertIn("prefix:re", inference.component_ids)
        self.assertIn("root:wil", inference.component_ids)
        self.assertIn("root:cuma", inference.component_ids)
        self.assertGreaterEqual(inference.features["repeat"], 0.8)
        self.assertGreaterEqual(inference.features["arrival"], 0.45)

    def test_routes_novel_word_utterances_to_response_modes(self) -> None:
        inferencer = MeaningInferencer(load_meaning_corpus(CORPUS))

        command = inferencer.infer_utterance("Please rewelcome Mira when she returns.")
        question = inferencer.infer_utterance("What does unwelcome mean?")
        transfer = inferencer.infer_utterance(
            "A giftable means something that can be given."
        )
        clarification = inferencer.infer_utterance(
            "By wilgift I mean a gift offered with happy intent."
        )

        self.assertEqual(command.intent, "command")
        self.assertEqual(command.response_kind, "action_frame")
        self.assertEqual(question.response_kind, "meaning_answer")
        self.assertEqual(transfer.response_kind, "store_candidate")
        self.assertEqual(clarification.response_kind, "clarification_update")

    def test_malformed_command_asks_for_clarification(self) -> None:
        inferencer = MeaningInferencer(load_meaning_corpus(CORPUS))

        inference = inferencer.infer_utterance("Could you helpfulness the puzzle?")

        self.assertEqual(inference.intent, "clarification_needed")
        self.assertEqual(inference.response_kind, "ask_clarification")

    def test_active_mvp_has_no_sound_symbolism_components(self) -> None:
        corpus = load_meaning_corpus(CORPUS)

        active_sound_components = [
            component
            for component in corpus.components.values()
            if component.kind in {"sound_cue", "phonestheme"}
        ]

        self.assertEqual(active_sound_components, [])

    def test_infers_added_high_confidence_morpheme_examples(self) -> None:
        inferencer = MeaningInferencer(load_meaning_corpus(CORPUS))

        unreadable = inferencer.infer_word("unreadable")
        teacher = inferencer.infer_word("teacher")
        prewrite = inferencer.infer_word("prewrite")
        clearwater = inferencer.infer_word("clearwater")

        self.assertIn("prefix:un", unreadable.component_ids)
        self.assertIn("root:read", unreadable.component_ids)
        self.assertIn("suffix:able", unreadable.component_ids)
        self.assertGreaterEqual(unreadable.features["negation"], 0.8)
        self.assertIn("root:teach", teacher.component_ids)
        self.assertIn("suffix:er", teacher.component_ids)
        self.assertGreaterEqual(teacher.features["person"], 0.3)
        self.assertIn("prefix:pre", prewrite.component_ids)
        self.assertIn("root:write", prewrite.component_ids)
        self.assertGreaterEqual(prewrite.features["before"], 0.8)
        self.assertNotIn("suffix:er", clearwater.component_ids)


if __name__ == "__main__":
    unittest.main()
