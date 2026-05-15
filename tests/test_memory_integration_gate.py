import unittest

from melm.evaluation import decide_memory_integration_gate


class MemoryIntegrationGateTests(unittest.TestCase):
    def test_advances_when_memory_and_state_checks_pass(self) -> None:
        validation_suite = {
            "overall_passed": True,
            "checks": [
                {"name": "synthetic_event_memory_gain", "metric": 0.47},
                {"name": "authored_dialogue_memory_gain", "metric": 0.20},
                {"name": "sample_transcript_memory_gain", "metric": 0.33},
                {"name": "authored_dialogue_abstention", "metric": 1.06},
                {"name": "sample_transcript_abstention", "metric": 1.25},
            ],
        }
        state_assisted = {
            "runs": [
                {
                    "tokenizer": "tiered_morph_unigram",
                    "report": {
                        "accuracy": 1.0,
                        "accuracy_lift": 0.5,
                        "state_answer_rate": 1.0,
                    },
                }
            ]
        }

        decision = decide_memory_integration_gate(validation_suite, state_assisted)

        self.assertEqual(decision.decision, "advance_to_persistent_dialogue_demo")

    def test_holds_when_state_answer_rate_is_low(self) -> None:
        validation_suite = {
            "overall_passed": True,
            "checks": [
                {"name": "synthetic_event_memory_gain", "metric": 0.47},
                {"name": "authored_dialogue_memory_gain", "metric": 0.20},
                {"name": "sample_transcript_memory_gain", "metric": 0.33},
                {"name": "authored_dialogue_abstention", "metric": 1.06},
                {"name": "sample_transcript_abstention", "metric": 1.25},
            ],
        }
        state_assisted = {
            "runs": [
                {
                    "tokenizer": "tiered_morph_unigram",
                    "report": {
                        "accuracy": 1.0,
                        "accuracy_lift": 0.5,
                        "state_answer_rate": 0.5,
                    },
                }
            ]
        }

        decision = decide_memory_integration_gate(validation_suite, state_assisted)

        self.assertEqual(decision.decision, "hold_for_memory_reliability")


if __name__ == "__main__":
    unittest.main()
