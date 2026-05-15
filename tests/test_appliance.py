import tempfile
from pathlib import Path
import unittest

from melm.appliance import MelmAppliance, MemoryRecord


class MelmApplianceTests(unittest.TestCase):
    def test_roundtrip_retrieves_and_answers_with_citation(self) -> None:
        records = [
            MemoryRecord(
                memory_id="m1",
                text="Ava adopted a rescue dog named Orbit yesterday.",
                created_at="2026-05-01",
                metadata={
                    "observation": "Ava adopted a rescue dog named Orbit.",
                    "summary": "Ava talks about adopting Orbit.",
                },
            ),
            MemoryRecord(
                memory_id="m2",
                text="Ben booked a ceramics class for Saturday.",
                created_at="2026-05-02",
                metadata={"summary": "Ben talks about a ceramics class."},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            MelmAppliance(records).save_jsonl(path)
            appliance = MelmAppliance.load_jsonl(path)

        answer = appliance.answer("What is the name of Ava's rescue dog?", k=2)

        self.assertIn("Orbit", answer.answer)
        self.assertIn("m1", answer.citations)
        self.assertGreater(answer.confidence, 0.0)

    def test_unknown_question_returns_supported_best_effort(self) -> None:
        appliance = MelmAppliance(
            [MemoryRecord(memory_id="m1", text="The local note says the office closes at five.")]
        )

        answer = appliance.answer("What time does the office close?", k=1)

        self.assertIn("five", answer.answer)
        self.assertEqual(answer.citations, ("m1",))

    def test_answer_uses_neighboring_reply_in_dialogue_memory(self) -> None:
        appliance = MelmAppliance(
            [
                MemoryRecord(
                    memory_id="m1",
                    text=(
                        "Mira: What did you research yesterday? "
                        "Noah: I researched adoption agencies near the city."
                    ),
                    metadata={"observation": "{'Noah': [['Noah researched adoption agencies.']]}"},
                )
            ]
        )

        answer = appliance.answer("What did Noah research?", k=1)

        self.assertIn("adoption agencies", answer.answer)
        self.assertNotIn("{", answer.answer)

    def test_structured_observation_extracts_direct_object(self) -> None:
        appliance = MelmAppliance(
            [
                MemoryRecord(
                    memory_id="m1",
                    text="Noah talked about plans.",
                    metadata={
                        "observation": "{'Noah': [['Noah is researching adoption agencies with his mentor.', 'D1:1']]}"
                    },
                )
            ]
        )

        answer = appliance.answer("What did Noah research?", k=1)

        self.assertEqual(answer.answer, "adoption agencies")

    def test_structured_observation_normalizes_duration(self) -> None:
        appliance = MelmAppliance(
            [
                MemoryRecord(
                    memory_id="m1",
                    text="A friend made it for Mira's 18th birthday ten years ago.",
                    metadata={
                        "observation": "{'Mira': [[\"Mira received a bowl for her 18th birthday ten years ago.\", 'D1:1']]}"
                    },
                )
            ]
        )

        answer = appliance.answer("How long ago was Mira's 18th birthday?", k=1)

        self.assertEqual(answer.answer, "10 years ago")

    def test_structured_answer_aggregates_quoted_book_titles(self) -> None:
        appliance = MelmAppliance(
            [
                MemoryRecord(
                    memory_id="m1",
                    text=(
                        "Mira: I read 'Nothing is Impossible' last month.\n"
                        "Mira: I also reread \"Charlotte's Web\" with my niece."
                    ),
                )
            ]
        )

        answer = appliance.answer("What books has Mira read?", k=1)

        self.assertEqual(answer.answer, "\"Nothing is Impossible\", \"Charlotte's Web\"")

    def test_when_question_resolves_relative_date(self) -> None:
        appliance = MelmAppliance(
            [
                MemoryRecord(
                    memory_id="m1",
                    text="Mira joined the support group yesterday.",
                    created_at="2026-05-08",
                    metadata={"summary": "Mira joined the support group."},
                )
            ]
        )

        answer = appliance.answer("When did Mira join the support group?", k=1)

        self.assertIn("7 May 2026", answer.answer)

    def test_when_question_keeps_relative_weekday_phrase(self) -> None:
        appliance = MelmAppliance(
            [
                MemoryRecord(
                    memory_id="m1",
                    text="Mira went to the workshop last Friday.",
                    created_at="15 July 2026",
                )
            ]
        )

        answer = appliance.answer("When did Mira go to the workshop?", k=1)

        self.assertEqual(answer.answer, "The Friday before 15 July 2026")


if __name__ == "__main__":
    unittest.main()
