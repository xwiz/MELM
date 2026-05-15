import unittest

from melm.training import summarize_progression


class TinyLMProgressionTests(unittest.TestCase):
    def test_summarize_progression_orders_by_last_bits_per_byte(self) -> None:
        progressions = summarize_progression(
            [
                {
                    "config": {"steps": 10},
                    "summaries": [
                        {"tokenizer": "a", "mean_bits_per_byte": 4.0, "std_bits_per_byte": 0.1},
                        {"tokenizer": "b", "mean_bits_per_byte": 3.0, "std_bits_per_byte": 0.1},
                    ],
                },
                {
                    "config": {"steps": 20},
                    "summaries": [
                        {"tokenizer": "a", "mean_bits_per_byte": 2.0, "std_bits_per_byte": 0.1},
                        {"tokenizer": "b", "mean_bits_per_byte": 2.5, "std_bits_per_byte": 0.1},
                    ],
                },
            ]
        )

        self.assertEqual([progression.tokenizer for progression in progressions], ["a", "b"])
        self.assertEqual([point.steps for point in progressions[0].points], [10, 20])
        self.assertAlmostEqual(progressions[0].relative_improvement, 0.5)


if __name__ == "__main__":
    unittest.main()
