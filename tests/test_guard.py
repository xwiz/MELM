import unittest

from melm.benchmarks import generate_support_refund_benchmark, support_refund_fixture
from melm.guard import (
    Condition,
    Fact,
    RuleEngine,
    WorkingMemory,
    evaluate_condition,
    evaluate_guard_benchmark,
    schema_only_status,
)


class GuardEngineTests(unittest.TestCase):
    def test_condition_matching_supports_comparison_and_missing(self) -> None:
        memory = WorkingMemory(
            [
                Fact(
                    fact_id="f1",
                    subject="order:o1",
                    predicate="amount",
                    value=150,
                    time_index=1,
                    source_event_id="e1",
                )
            ]
        )
        proposal = support_refund_fixture().guard_cases[0].proposal

        self.assertTrue(
            evaluate_condition(
                memory,
                proposal,
                Condition("order:o1", "amount", "gt", 100),
                current_time=2,
            ).matched
        )
        missing = evaluate_condition(
            memory,
            proposal,
            Condition("order:o1", "manager_approval", "missing"),
            current_time=2,
        )
        self.assertTrue(missing.matched)
        self.assertEqual(missing.missing_fact, "order:o1:manager_approval")

    def test_working_memory_caches_derived_facts(self) -> None:
        memory = WorkingMemory()
        memory.derive(
            fact_id="d1",
            subject="order:o1",
            predicate="refund_over_limit",
            value=True,
            time_index=3,
            source_event_id="e1",
        )
        fact = memory.latest_fact("order:o1", "refund_over_limit")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.value, True)
        self.assertEqual(fact.metadata["derived"], "true")

    def test_guard_fixture_passes_core_gate(self) -> None:
        fixture = support_refund_fixture()
        report = evaluate_guard_benchmark(
            fixture.facts,
            fixture.rules,
            fixture.guard_cases,
            current_time=fixture.current_time,
        )

        self.assertTrue(report.gate_passed)
        self.assertEqual(report.melm_false_allow_rate, 0.0)
        self.assertGreaterEqual(report.valid_action_allow_rate, 0.90)
        self.assertEqual(report.traceability, 1.0)

    def test_schema_only_allows_high_value_refund_without_approval(self) -> None:
        fixture = support_refund_fixture()
        high_value_case = next(case for case in fixture.guard_cases if case.category == "approval_required")

        self.assertEqual(schema_only_status(high_value_case.proposal), "allow")

        report = evaluate_guard_benchmark(
            fixture.facts,
            fixture.rules,
            [high_value_case],
            current_time=fixture.current_time,
        )
        self.assertEqual(report.predictions[0].melm_status, "abstain")
        self.assertIn("manager_approval_required", report.predictions[0].melm_decision.triggered_rule_ids)
        self.assertIn("order:o1003:manager_approval", report.predictions[0].melm_decision.missing_facts)

    def test_guard_denies_unverified_and_warns_on_stale_approval(self) -> None:
        fixture = support_refund_fixture()
        report = evaluate_guard_benchmark(
            fixture.facts,
            fixture.rules,
            fixture.guard_cases,
            current_time=fixture.current_time,
        )
        by_action = {prediction.action_id: prediction for prediction in report.predictions}

        self.assertEqual(by_action["g2"].melm_status, "deny")
        self.assertIn("identity_must_be_true", by_action["g2"].melm_decision.triggered_rule_ids)
        self.assertEqual(by_action["g4"].melm_status, "warn")
        self.assertIn("manager_approval_stale", by_action["g4"].melm_decision.triggered_rule_ids)

    def test_rule_evidence_is_propagated(self) -> None:
        fixture = support_refund_fixture()
        engine = RuleEngine(fixture.rules)
        memory = WorkingMemory(fixture.facts)
        fraud_case = next(case for case in fixture.guard_cases if case.category == "fraud_flag")

        decision = engine.decide(memory, fraud_case.proposal, current_time=fixture.current_time)

        self.assertEqual(decision.status, "deny")
        self.assertIn("fraud_blocks_refund", decision.triggered_rule_ids)
        self.assertIn("o1005_e_fraud", decision.evidence_event_ids)

    def test_generated_support_refund_benchmark_preserves_guard_signal(self) -> None:
        fixture = generate_support_refund_benchmark(scenario_repeats=2, seed=23)
        report = evaluate_guard_benchmark(
            fixture.facts,
            fixture.rules,
            fixture.guard_cases,
            current_time=fixture.current_time,
        )

        self.assertTrue(report.gate_passed)
        self.assertEqual(report.melm_false_allow_rate, 0.0)
        self.assertEqual(report.traceability, 1.0)


if __name__ == "__main__":
    unittest.main()
