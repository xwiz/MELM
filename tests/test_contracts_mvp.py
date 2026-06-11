import unittest

from melm.contracts import (
    ContractRegistry,
    ContractValidationError,
    load_contract_json,
    validate_class_maps,
    validate_router_lexicon_families,
    validate_semantic_class_registry,
    validate_sense_candidate,
)


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


if __name__ == "__main__":
    unittest.main()
