import unittest

from melm.evaluation import evaluate_validation_suite


class ValidationSuiteTests(unittest.TestCase):
    def test_advisory_tokenizer_stability_failure_does_not_fail_suite(self) -> None:
        payload = _base_payload()

        report = evaluate_validation_suite(payload)

        self.assertTrue(report.overall_passed)
        self.assertEqual(report.hard_failures, 0)
        stability = next(
            check for check in report.checks if check.name == "tokenizer_stability"
        )
        self.assertFalse(stability.passed)
        self.assertEqual(stability.severity, "advisory")

    def test_hard_gate_failure_fails_suite(self) -> None:
        payload = _base_payload()
        payload["memory_gate"]["passed"] = False
        payload["memory_gate"]["metric"] = 0.02

        report = evaluate_validation_suite(payload)

        self.assertFalse(report.overall_passed)
        self.assertEqual(report.hard_failures, 1)


def _base_payload() -> dict:
    return {
        "memory_gate": {
            "passed": True,
            "metric": 0.47,
            "threshold": 0.15,
            "recommendation": "Proceed with structured event memory",
        },
        "memory_abstention": {
            "gate": {
                "passed": True,
                "metric": 1.0,
                "threshold": 1.0,
                "recommendation": "Use calibrated confidence threshold",
            },
            "calibrated": {
                "selected": {
                    "confidence_method": "score_with_evidence_veto",
                    "threshold": 1.25,
                    "evaluation_report": {
                        "positive_recall": 0.75,
                        "negative_abstention": 0.85,
                    },
                }
            },
        },
        "tokenizer_decision": {
            "gate_passed": False,
            "decision": "auxiliary_only",
            "lm_nll_gain": -0.5,
            "boundary_f1_gain": 0.4,
        },
        "tokenizer_stability": {
            "stable_primary_candidate": False,
            "average_lm_nll_gain": -0.4,
            "morph_win_rate": 0.0,
            "best_baseline_tokenizer": "unigram_like",
        },
        "grounding": {
            "accuracy": 1.0,
            "cases": 4,
        },
        "state_resolution": {
            "cases": 12,
            "accuracy": 1.0,
            "false_positive_rate": 0.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
