import unittest

from melm.evaluation import interpret_phase1


class InterpretationTests(unittest.TestCase):
    def test_interpret_phase1_marks_memory_pass_and_tokenizer_gap(self) -> None:
        payload = {
            "memory_report": {
                "absolute_gain": 0.25,
                "by_category": {
                    "temporal_after": {"absolute_gain": 1.0},
                },
            },
            "memory_strict_report": {
                "absolute_gain": 0.25,
            },
            "memory_ablation_report": {
                "entity_action_temporal": {"absolute_gain": 0.25},
                "entity_action_causal": {"absolute_gain": 0.22},
                "event_memory": {"absolute_gain": 0.47},
            },
            "tokenizer_lm_reports": [
                {"tokenizer": "unigram_like", "nll_per_token": 5.0},
                {"tokenizer": "heuristic_morpheme", "nll_per_token": 6.0},
            ],
            "morphology_boundary_reports": [
                {"tokenizer": "heuristic_morpheme", "f1": 1.0},
            ],
            "tokenizer_decision": {
                "decision": "auxiliary_only",
                "lm_nll_gain": -1.0,
                "boundary_f1_gain": 0.5,
                "recommendation": "Use morphology for auxiliary supervision only.",
            },
            "grounding": {
                "accuracy": 1.0,
                "cases": 4,
            },
            "authored_dialogue": {
                "memory_report": {
                    "event_memory_recall_at_k": 1.0,
                    "rag_recall_at_k": 0.8,
                },
                "abstention_report": {
                    "positive_recall": 0.8,
                    "negative_abstention": 1.0,
                },
                "memory_gate": {"passed": True},
                "abstention_gate": {"passed": True},
            },
            "sample_transcript": {
                "turns": 8,
                "events": 7,
                "memory_report": {
                    "event_memory_recall_at_k": 1.0,
                    "rag_recall_at_k": 0.67,
                },
                "abstention_report": {
                    "positive_recall": 0.83,
                    "negative_abstention": 1.0,
                },
                "memory_gate": {"passed": True},
                "abstention_gate": {"passed": True},
            },
        }
        findings = {finding.area: finding for finding in interpret_phase1(payload)}
        self.assertEqual(findings["event_memory"].status, "pass")
        self.assertEqual(findings["authored_dialogue"].status, "probe_pass")
        self.assertEqual(findings["sample_transcript"].status, "smoke_pass")
        self.assertEqual(findings["tokenizer_lm"].status, "auxiliary_only")
        self.assertEqual(findings["morphology_boundary"].status, "probe_pass")
        self.assertEqual(findings["state_grounding"].status, "probe_pass")

    def test_interpret_phase1_marks_calibration_risk(self) -> None:
        payload = {
            "memory_report": {
                "absolute_gain": 0.25,
                "by_category": {},
            },
            "memory_abstention": {
                "event_memory_best": {
                    "accuracy": 0.75,
                    "threshold": 1.0,
                    "negative_abstention": 0.50,
                },
            },
            "tokenizer_lm_reports": [
                {"tokenizer": "unigram_like", "nll_per_token": 5.0},
                {"tokenizer": "heuristic_morpheme", "nll_per_token": 6.0},
            ],
            "morphology_boundary_reports": [
                {"tokenizer": "heuristic_morpheme", "f1": 1.0},
            ],
            "grounding": {
                "accuracy": 1.0,
                "cases": 4,
            },
        }

        findings = {finding.area: finding for finding in interpret_phase1(payload)}

        self.assertEqual(findings["event_memory"].status, "pass_with_calibration_risk")
        self.assertIn("abstention", findings["event_memory"].finding)


if __name__ == "__main__":
    unittest.main()
