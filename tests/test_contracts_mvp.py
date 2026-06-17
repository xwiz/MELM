import unittest

from melm.contracts import (
    ContractRegistry,
    ContractValidationError,
    load_contract_json,
    load_pi_benchmark,
    load_uol_normative_cases,
    validate_class_maps,
    validate_contract_registry,
    validate_router_lexicon_families,
    validate_semantic_class_registry,
    validate_sense_candidate,
)
from melm.contracts.validation import _is_compatible_version
from melm.appliance.functional_grammar import parse_functional_relations
from melm.appliance.assistant_lexicon import _normalize_term


def _candidate(**overrides):
    payload = {
        "schema_id": "melm.sense_candidate.v1",
        "batch_id": "event:e_123",
        "lemma": "kalimba",
        "language": "en",
        "pos": "noun",
        "source": {
            "provenance": "user_taught",
            "source_ref": "event:e_123",
            "license": "user-provided",
        },
        "definition": "a small thumb piano",
        "genus_lemma": "piano",
        "semantic_class_candidates": [
            {
                "class_id": "physical_object.instrument",
                "method": "genus_walk",
                "confidence": 0.72,
            }
        ],
        "forms": [],
        "relations": [{"relation": "hypernym", "target_lemma": "piano"}],
        "safety": {"reserved_conflict": False, "policy_term_overlap": False},
        "suggested_status": "quarantined",
        "confidence_prior": 0.6,
    }
    payload.update(overrides)
    return payload


class ContractMvpTests(unittest.TestCase):
    def test_registry_and_class_maps_are_internally_consistent(self) -> None:
        registry = ContractRegistry.load()
        self.assertEqual(registry.require("melm.sense_candidate.v1")["owner"], "meaning-core")
        validate_semantic_class_registry(load_contract_json("semantic_classes.v1.json"))
        validate_class_maps()
        validate_router_lexicon_families(
            load_contract_json("router_lexicon_families.v1.json")
        )

    def test_valid_user_taught_candidate_passes(self) -> None:
        validate_sense_candidate(_candidate())

    def test_unknown_class_is_rejected(self) -> None:
        candidate = _candidate()
        candidate["semantic_class_candidates"][0]["class_id"] = "capability.install_anything"
        with self.assertRaisesRegex(ContractValidationError, "unknown semantic classes"):
            validate_sense_candidate(candidate)

    def test_reserved_or_policy_terms_cannot_activate(self) -> None:
        reserved = _candidate(
            safety={"reserved_conflict": True, "policy_term_overlap": False},
            suggested_status="active",
        )
        with self.assertRaisesRegex(ContractValidationError, "reserved conflicts"):
            validate_sense_candidate(reserved)

        policy = _candidate(
            safety={"reserved_conflict": False, "policy_term_overlap": True},
            suggested_status="dormant",
        )
        with self.assertRaisesRegex(ContractValidationError, "policy-term overlaps"):
            validate_sense_candidate(policy)

    def test_runtime_sources_are_always_quarantined(self) -> None:
        candidate = _candidate(suggested_status="active")
        with self.assertRaisesRegex(ContractValidationError, "user_taught candidates"):
            validate_sense_candidate(candidate)

    def test_unknown_fields_fail_closed(self) -> None:
        candidate = _candidate(debug_override=True)
        with self.assertRaisesRegex(ContractValidationError, "unknown properties"):
            validate_sense_candidate(candidate)

    def test_check_compatibility_passes_for_current_registry(self) -> None:
        """The live registry has no hash or version mismatches."""
        registry = ContractRegistry.load()
        errors = registry.check_compatibility()
        self.assertEqual(errors, [], f"compatibility errors: {errors}")

    def test_check_compatibility_catches_hash_mismatch(self) -> None:
        """A tampered schema_hash is detected."""
        registry = ContractRegistry.load()
        # Mutate the stored hash for one entry.
        contracts = dict(registry.contracts)
        first_id = next(iter(contracts))
        first = dict(contracts[first_id])
        first["schema_hash"] = "0" * 16
        contracts[first_id] = first
        mutated = ContractRegistry(schema_id=registry.schema_id, contracts=contracts)
        errors = mutated.check_compatibility()
        self.assertTrue(any("schema_hash mismatch" in e for e in errors), f"expected hash mismatch in {errors}")

    def test_check_compatibility_catches_version_incompatibility(self) -> None:
        """A v1 contract claiming a v2 predecessor fails."""
        registry = ContractRegistry.load()
        contracts = dict(registry.contracts)
        first_id = next(iter(contracts))
        first = dict(contracts[first_id])
        first["version"] = "1.0.0"
        first["compatible_predecessors"] = ["melm.semantic_classes.v1"]
        contracts[first_id] = first
        # Bump semantic_classes to v2.0.0
        second_id = "melm.semantic_classes.v1"
        if second_id in contracts:
            second = dict(contracts[second_id])
            second["version"] = "2.0.0"
            contracts[second_id] = second
        mutated = ContractRegistry(schema_id=registry.schema_id, contracts=contracts)
        errors = mutated.check_compatibility()
        self.assertTrue(
            any("not compatible" in e for e in errors),
            f"expected version incompatibility in {errors}",
        )

    def test_is_compatible_version_semantics(self) -> None:
        """Semver compatibility helper behaves correctly."""
        self.assertTrue(_is_compatible_version("2.0.0", "1.0.0"))
        self.assertTrue(_is_compatible_version("1.1.0", "1.0.0"))
        self.assertTrue(_is_compatible_version("1.0.0", "1.0.0"))
        self.assertFalse(_is_compatible_version("1.0.0", "2.0.0"))
        self.assertFalse(_is_compatible_version("1.0.0", "1.1.0"))
        self.assertFalse(_is_compatible_version("1.0", "not_a_version"))

    def test_registry_validation_requires_version_and_schema_hash(self) -> None:
        """Registry entries missing version or schema_hash are rejected."""
        payload = {
            "schema_id": "melm.contract_registry.v1",
            "contracts": [
                {
                    "schema_id": "melm.test.v1",
                    "path": "semantic_classes.v1.json",
                    "owner": "test",
                    "producers": ["test"],
                    "consumers": ["test"],
                    "validator": "melm.contracts.validate_semantic_class_registry",
                    "compatible_predecessors": [],
                    "failure_behavior": "test",
                    "safety_critical": False,
                }
            ]
        }
        with self.assertRaisesRegex(ContractValidationError, "missing required property"):
            validate_contract_registry(payload)

    def test_uol_normative_cases_count(self) -> None:
        """The normative UOL set contains >= 60 cases."""
        cases = load_uol_normative_cases()
        self.assertGreaterEqual(len(cases), 60)

    def test_uol_normative_cases_parse_accuracy(self) -> None:
        """>= 80% of normative cases parse with expected speech_act, subject, action, object, target."""
        cases = load_uol_normative_cases()
        successes = 0
        failures: list[str] = []
        for case in cases:
            tokens = tuple(_normalize_term(case["utterance"]).split())
            parsed = parse_functional_relations(tokens, question_mark="?" in case["utterance"])
            if parsed is None:
                failures.append(f"{case['utterance']!r}: parse returned None")
                continue
            checks = [
                ("speech_act", parsed.speech_act, case["speech_act"]),
                ("subject", parsed.subject, case["subject"]),
                ("action", parsed.action, case["action"]),
                ("object", parsed.object, case["object"]),
                ("target", parsed.target, case["target"]),
            ]
            mismatches = [
                f"{field}={actual!r} (expected {expected!r})"
                for field, actual, expected in checks
                if actual != expected
            ]
            if mismatches:
                failures.append(
                    f"{case['utterance']!r}: " + ", ".join(mismatches)
                )
            else:
                successes += 1
        rate = successes / len(cases)
        self.assertGreaterEqual(
            rate, 0.80,
            f"UOL parse accuracy={rate:.0%} < 80% ({successes}/{len(cases)})\n"
            + "\n".join(failures[:10]),
        )

    def test_pi_benchmark_loads_and_records_go_no_go(self) -> None:
        """Pi benchmark artifact loads and contains go/no-go assessment."""
        bench = load_pi_benchmark()
        self.assertIn("measurements", bench)
        self.assertIn("go_no_go", bench)
        go_no_go = bench["go_no_go"]
        self.assertIsInstance(go_no_go["template_fallback_ready"], bool)
        self.assertIsInstance(go_no_go["model_loaded"], bool)
        self.assertIsInstance(go_no_go["pi_target_met"], bool)


if __name__ == "__main__":
    unittest.main()
