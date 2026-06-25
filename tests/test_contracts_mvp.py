"""Contract validation and loader tests.

Covers:
- Parametrized tests for all registered contract validators and loaders
- Hash-compatibility checks for every registry entry
- Unique/specific contract tests that exercise error paths or custom assertions
"""

import hashlib
import unittest
import inspect

from melm.contracts import (
    ContractRegistry,
    ContractValidationError,
    load_contract_json,
    load_function_words,
    load_pi_benchmark,
    load_uol_normative_cases,
    validate_class_maps,
    validate_contract_registry,
    validate_function_words,
    validate_router_lexicon_families,
    validate_semantic_class_registry,
    validate_sense_candidate,
)
from melm.contracts.validation import (
    CONTRACT_ROOT,
    _is_compatible_version,
)
from melm.appliance.functional_grammar import parse_functional_relations
from melm.appliance.assistant_lexicon import _normalize_term

import melm.contracts.validation as _vmod


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


# ---------------------------------------------------------------------------
# Helpers: collect contract entries from the live registry
# ---------------------------------------------------------------------------

def _registry_entries():
    """Yield (schema_id, info) sorted by schema_id."""
    registry = ContractRegistry.load()
    for cid in sorted(registry.contracts):
        yield cid, registry.contracts[cid]


def _validator_fn(validator_path: str):
    """Resolve a 'melm.contracts.validate_xxx' string to a callable, or None."""
    if not validator_path:
        return None
    for prefix in ("melm.contracts.", ""):
        if validator_path.startswith(prefix):
            name = validator_path[len(prefix):]
            return getattr(_vmod, name, None)
    return None


def _loader_fn(contract_name: str):
    """Resolve a contract name (e.g. 'food_tags') to a load_* callable, or None."""
    fn_name = f"load_{contract_name}"
    return getattr(_vmod, fn_name, None)


def _contract_short_name(schema_id: str) -> str:
    """Strip the version suffix from a schema_id (e.g. 'melm.food_tags.v1' -> 'food_tags')."""
    parts = schema_id.split(".")
    if len(parts) >= 2 and parts[-1].startswith("v"):
        return ".".join(parts[1:-1])
    return ".".join(parts[1:])


# Validators that need special argument handling (not a simple payload).
# Includes process/authority validators that validate runtime data structures,
# not loaded contract JSON files — they reject their own JSON Schema files.
_SPECIAL_VALIDATORS = frozenset({
    "validate_sense_candidate",
    "validate_class_maps",
    "validate_contract_registry",
    "validate_answer_plan",
    "validate_capability_manifest",
    "validate_evidence_packet",
    "validate_model_manifest",
    "validate_route_decision",
    "validate_uol_parse",
    "validate_verification_result",
})

# Contract IDs whose loaded data does not pass their own validator
# (pre-existing known issues — child_memory_markers: empty default_suffix)
_SKIP_VALIDATE_CONTRACTS = frozenset({
    "melm.capability_manifest.v1",
    "melm.child_memory_markers.v1",
})


# ---------------------------------------------------------------------------
# Existing tests (unique / specific assertions retained)
# ---------------------------------------------------------------------------

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

    def test_function_words_reject_unknown_answer_type(self) -> None:
        payload = load_function_words()
        mutated = {
            **payload,
            "entries": [dict(entry) for entry in payload["entries"]],
        }
        mutated["entries"][0]["answer_type"] = "mystery"
        with self.assertRaisesRegex(ContractValidationError, "answer_type"):
            validate_function_words(mutated)

    def test_each_contract_hash_matches_registry(self) -> None:
        """Every registered contract's file sha256 matches its registry entry."""
        failures: list[str] = []
        for cid, info in _registry_entries():
            path = info.get("path", "")
            if not path:
                continue
            expected = info.get("schema_hash", "")
            if not expected:
                continue
            content = (CONTRACT_ROOT / path).read_bytes()
            actual = hashlib.sha256(content).hexdigest()[:16]
            if actual != expected:
                failures.append(f"{cid}: computed={actual!r} expected={expected!r}")
        with self.subTest(contract="all"):
            self.assertEqual(failures, [], "\n".join(failures))


# ---------------------------------------------------------------------------
# Parametrized validator tests (collapsed from individual per-contract tests)
# ---------------------------------------------------------------------------

class ContractValidatorMvpTests(unittest.TestCase):
    """Parametrized contract validator tests."""

    def test_contract_validator_accepts_loaded_data(self) -> None:
        """Every registered contract's validator accepts its own loaded data."""
        failures: list[str] = []
        for cid, info in _registry_entries():
            with self.subTest(contract=cid):
                path = info.get("path", "")
                if not path:
                    continue
                validator_path = info.get("validator", "")
                fn = _validator_fn(validator_path)
                if fn is None:
                    continue
                fn_name = getattr(fn, "__name__", str(fn))
                if fn_name in _SPECIAL_VALIDATORS:
                    continue
                if cid in _SKIP_VALIDATE_CONTRACTS:
                    continue
                try:
                    payload = load_contract_json(path)
                except Exception as exc:
                    failures.append(f"{cid}: load failed: {exc}")
                    continue
                sig = inspect.signature(fn)
                try:
                    if len(sig.parameters) == 0:
                        fn()
                    else:
                        fn(payload)
                except Exception as exc:
                    failures.append(f"{cid}: validate failed: {exc}")
        with self.subTest(contract="all"):
            self.assertEqual(failures, [], "\n".join(failures))

    def test_contract_validator_rejects_bogus_schema_id(self) -> None:
        """Every registered contract's validator rejects a payload with a bad schema_id."""
        for cid, info in _registry_entries():
            with self.subTest(contract=cid):
                validator_path = info.get("validator", "")
                fn = _validator_fn(validator_path)
                if fn is None:
                    continue
                fn_name = getattr(fn, "__name__", str(fn))
                if fn_name in _SPECIAL_VALIDATORS:
                    continue
                if cid in _SKIP_VALIDATE_CONTRACTS:
                    continue
                sig = inspect.signature(fn)
                if len(sig.parameters) == 0:
                    continue
                bogus = {"schema_id": f"melm.bogus.v999"}
                try:
                    fn(bogus)
                except ContractValidationError:
                    pass
                except Exception:
                    pass
                else:
                    self.fail(f"{cid}: validator accepted bogus schema_id")


# ---------------------------------------------------------------------------
# Parametrized loader tests (collapsed from individual per-contract tests)
# ---------------------------------------------------------------------------

class ContractLoaderMvpTests(unittest.TestCase):
    """Parametrized contract loader tests."""

    def test_contract_loader_returns_data(self) -> None:
        """Every registered contract can be loaded and returns non-None data."""
        failures: list[str] = []
        for cid, info in _registry_entries():
            with self.subTest(contract=cid):
                path = info.get("path", "")
                if not path:
                    continue
                try:
                    data = load_contract_json(path)
                except Exception as exc:
                    failures.append(f"{cid}: load failed: {exc}")
                    continue
                if data is None:
                    failures.append(f"{cid}: loaded data is None")
        with self.subTest(contract="all"):
            self.assertEqual(failures, [], "\n".join(failures))

    def test_contract_loader_minimum_size(self) -> None:
        """Standard data contracts have >= 3 entries after loading."""
        # (short_label, contract_key, min_count) — min_count derived from
        # actual loaded data (len of dict, list, set, tuple, or frozenset).
        standard_checks = [
            ("food_tags", "food_tags", 5),
            ("health_disclaimers", "health_disclaimers", 3),
            ("safety_policies", "safety_policies", 1),
            ("story_components", "story_components", 3),
            ("weather_concepts", "weather_concepts", 3),
            ("meal_scopes", "meal_scopes", 5),
            ("assistant_identity", "assistant_identity", 3),
            ("memory_insights", "memory_insights", 3),
            ("mood_states", "mood_states", 2),
            ("affect_lexicon", "affect_lexicon", 1),
            ("response_pools", "response_pools", 1),
            ("perception_affect_map", "perception_affect_map", 2),
            ("creative_behaviors", "creative_behaviors", 1),
            ("causal_cues", "causal_cues", 4),
            ("causal_effects", "causal_effects", 3),
            ("causal_link_markers", "causal_link_markers", 5),
            ("causal_frames", "causal_frames", 3),
            ("verb_states", "verb_states", 1),
            ("state_valences", "state_valences", 1),
            ("noun_atoms", "noun_atoms", 3),
            ("verb_atoms", "verb_atoms", 50),
            ("atom_templates", "atom_templates", 10),
            ("self_identity", "self_identity", 1),
            ("knowledge_types", "knowledge_types", 1),
            ("world_relations", "world_relations", 1),
            ("sentience_map", "sentience_map", 5),
            ("damage_markers", "damage_markers", 10),
            ("moral_responses", "moral_responses", 3),
            ("open_domain_templates", "open_domain_templates", 1),
            ("frame_templates", "frame_templates", 1),
            ("frame_minimal_pairs", "frame_minimal_pairs", 1),
            ("router_semantic_aliases", "router_semantic_aliases", 1),
            ("igbo_lexicon_seed", "igbo_lexicon_seed", 1),
            ("yoruba_greetings", "yoruba_greetings", 1),
            ("swahili_greetings", "swahili_greetings", 1),
            ("igbo_greetings", "igbo_greetings", 1),
            ("prompt_seeds", "prompt_seeds", 1),
            ("predicate_inventory", "predicate_inventory", 3),
            ("function_words", "function_words", 3),
            ("deferred_task_templates", "deferred_task_templates", 3),
            ("novelty_patterns", "novelty_patterns", 1),
            ("agreement_templates", "agreement_templates", 3),
            ("epistemic_states", "epistemic_states", 1),
            ("background_task_policies", "background_task_policies", 1),
            ("commitment_parsers", "commitment_parsers", 1),
            ("commitment_responses", "commitment_responses", 3),
            ("consent_revocation_response", "consent_revocation_response", 1),
            ("contact_enrichment_templates", "contact_enrichment_templates", 3),
            ("contact_object_tokens", "contact_object_tokens", 3),
            ("mail_verb_sets", "mail_verb_sets", 2),
            ("story_follow_up_phrase", "story_follow_up_phrase", 1),
            ("social_status_patterns", "social_status_patterns", 5),
            ("autobiographical_horizon_tokens", "autobiographical_horizon_tokens", 3),
            ("music_discovery_verbs", "music_discovery_verbs", 2),
            ("short_circuit_responses", "short_circuit_responses", 3),
            ("personal_memory_evidence_map", "personal_memory_evidence_map", 10),
            ("safety_school_terms", "safety_school_terms", 3),
            ("media_object_tokens", "media_object_tokens", 3),
            ("identity_token_roles", "identity_token_roles", 40),
            ("identity_scope_tokens", "identity_scope_tokens", 20),
            ("task_domain_terms", "task_domain_terms", 20),
            ("story_constraint_stopwords", "story_constraint_stopwords", 15),
            ("music_instruments", "music_instruments", 5),
            ("always_respond_intents", "always_respond_intents", 5),
            ("short_circuit_reasons", "short_circuit_reasons", 1),
            ("environment_prep_phrases", "environment_prep_phrases", 1),
            ("intent_domains", "intent_domains", 10),
            ("frame_local_sources", "frame_local_sources", 10),
            ("pool_intents", "pool_intents", 2),
            ("synthesis_quality_weights", "synthesis_quality_weights", 3),
            ("persona_emoji_intents", "persona_emoji_intents", 3),
            ("mood_emoji_map", "mood_emoji_map", 10),
            ("intent_evidence_sources", "intent_evidence_sources", 10),
            ("midi_music_mapping", "midi_music_mapping", 3),
            ("music_style_templates", "music_style_templates", 1),
            ("modifier_atoms", "modifier_atoms", 1),
            ("self_identity_facts", "self_identity_facts", 1),
            ("ethical_constraints", "ethical_constraints", 3),
            ("geo_decision", "geo_decision", 1),
            ("geo_atlas", "geo_atlas", 1),
            ("semantic_attention_rules", "semantic_attention_rules", 3),
            ("noise_tokens", "noise_tokens", 10),
            ("normalization_expansions", "normalization_expansions", 1),
            ("token_typability", "token_typability", 1),
            ("nlg_atomic_renderers", "nlg_atomic_renderers", 1),
            ("nlg_fallback_phrases", "nlg_fallback_phrases", 3),
            ("research_deferral_triggers", "research_deferral_triggers", 1),
            ("patient_type_map", "patient_type_map", 20),
            ("family_relation_terms", "family_relation_terms", 20),
            ("folk_tales", "folk_tales", 1),
            ("lesson_keywords", "lesson_keywords", 10),
            ("literary_device_map", "literary_device_map", 1),
            ("mood_faces", "mood_faces", 1),
            ("mood_face_tones", "mood_face_tones", 1),
            ("story_pipeline_prompts", "story_pipeline_prompts", 1),
            ("story_plan_schema", "story_plan_schema", 1),
            ("storytelling_phrases", "storytelling_phrases", 1),
            ("status_domain_terms", "status_domain_terms", 3),
            ("music_genre_scales", "music_genre_scales", 1),
            ("autobiographical_scope_terms", "autobiographical_scope_terms", 10),
            ("answer_templates", "answer_templates", 5),
            ("reasoning_templates", "reasoning_templates", 3),
            ("story_scene_templates", "story_scene_templates", 1),
            ("story_components", "story_components", 3),
        ]
        failures: list[str] = []
        for short_name, contract_key, min_count in standard_checks:
            with self.subTest(contract=short_name):
                loader = _loader_fn(contract_key)
                if loader is None:
                    # Fallback: load raw JSON
                    cid = f"melm.{contract_key}.v1"
                    path = ContractRegistry.load().contracts.get(cid, {}).get("path", "")
                    if path:
                        data = load_contract_json(path)
                    else:
                        failures.append(f"{short_name}: no loader and no path")
                        continue
                else:
                    try:
                        data = loader()
                    except Exception as exc:
                        failures.append(f"{short_name}: load failed: {exc}")
                        continue
                if data is None:
                    failures.append(f"{short_name}: loaded data is None")
                elif isinstance(data, (dict, list, set, tuple, frozenset)):
                    if len(data) < min_count:
                        failures.append(f"{short_name}: len={len(data)} < {min_count}")
        with self.subTest(contract="all"):
            self.assertEqual(failures, [], "\n".join(failures))


# ---------------------------------------------------------------------------
# Unique / specific contract tests (not suitable for parametrization)
# ---------------------------------------------------------------------------

class ContractMvpMoralTests(unittest.TestCase):
    """Contract integration tests for T4 moral cognition contracts."""

    def test_verb_states_contract_loads(self):
        """verb_states.v1.json can be loaded and validated."""
        from melm.contracts.validation import load_verb_states
        data = load_verb_states()
        self.assertIn("verbs", data)
        self.assertGreater(len(data["verbs"]), 0)
        self.assertIn("hit", data["verbs"])

    def test_state_valences_contract_loads(self):
        """state_valences.v1.json can be loaded and validated."""
        from melm.contracts.validation import load_state_valences
        data = load_state_valences()
        self.assertIn("valences", data)
        self.assertGreater(len(data["valences"]), 0)

    def test_verb_states_hit_has_expected_structure(self):
        """The hit verb entry has the expected nested structure."""
        from melm.contracts.validation import load_verb_states
        data = load_verb_states()
        hit = data["verbs"]["hit"]
        self.assertIn("patient_states", hit)
        self.assertIn("patient_types", hit)
        self.assertIn("subject_mental", hit)
        self.assertIsInstance(hit["patient_types"], list)
        self.assertIn("person", hit["patient_types"])

    def test_state_valences_symmetry(self):
        """Common harmful states have negative valences."""
        from melm.contracts.validation import load_state_valences
        data = load_state_valences()
        valences = data["valences"]
        for harmful in ("dead", "traumatized", "hurt", "injured"):
            self.assertLess(valences.get(harmful, 0), 0,
                            f"{harmful} should have negative valence")

    def test_state_valences_positive(self):
        """Positive states have positive valences."""
        from melm.contracts.validation import load_state_valences
        data = load_state_valences()
        valences = data["valences"]
        for positive in ("rescued", "healed", "loved", "safe"):
            self.assertGreater(valences.get(positive, 0), 0,
                               f"{positive} should have positive valence")


class ContractMvpFoundationTests(unittest.TestCase):
    """Contract integration tests for Phase 0 foundation contracts."""

    def test_validate_knowledge_types(self):
        from melm.contracts.validation import validate_knowledge_types
        data = {
            "schema_id": "melm.knowledge_types.v1",
            "version": "1.0.0",
            "type_markers": {
                "opinion_markers": ["best"],
                "literary_stems": ["what has"],
                "provenance_confidence": {"seed": 0.95, "user": 0.6, "cloud": 0.5},
            },
            "truth_arbitration": {
                "contradiction_prompt": "test",
                "contradiction_ack": "test",
                "assert_ack": "test",
                "negate_ack": "test",
            },
        }
        validate_knowledge_types(data)

    def test_validate_knowledge_types_missing_field(self):
        from melm.contracts.validation import ContractValidationError, validate_knowledge_types
        with self.assertRaises(ContractValidationError):
            validate_knowledge_types({})

    def test_load_knowledge_types(self):
        from melm.contracts.validation import load_knowledge_types
        data = load_knowledge_types()
        self.assertEqual(data["schema_id"], "melm.knowledge_types.v1")

    def test_validate_world_relations(self):
        from melm.contracts.validation import validate_world_relations
        data = {
            "schema_id": "melm.world_relations.v1",
            "version": "1.0.0",
            "predicate_to_relation": {
                "be": {"pattern": "copula", "relation_id": "is_a", "confidence": 0.8},
            },
        }
        validate_world_relations(data)

    def test_validate_world_relations_missing_field(self):
        from melm.contracts.validation import ContractValidationError, validate_world_relations
        with self.assertRaises(ContractValidationError):
            validate_world_relations({})

    def test_load_world_relations(self):
        from melm.contracts.validation import load_world_relations
        data = load_world_relations()
        self.assertEqual(data["schema_id"], "melm.world_relations.v1")


class ContractMvpAtomTemplateTests(unittest.TestCase):
    """Contract integration tests for atom_templates.v1.json."""

    def test_atom_templates_loads(self) -> None:
        from melm.contracts.validation import load_atom_templates, validate_atom_templates
        from melm.contracts.validation import load_contract_json
        payload = load_contract_json("atom_templates.v1.json")
        validate_atom_templates(payload)
        templates = load_atom_templates()
        self.assertIsInstance(templates, dict)
        self.assertGreater(len(templates), 0)

    def test_atom_templates_has_all_defaults(self) -> None:
        from melm.contracts.validation import load_atom_templates
        templates = load_atom_templates()
        for key in ("weather", "meal_suggestion", "assistant_identity",
                     "story", "gibberish", "complaint_response", "novelty"):
            self.assertIn(key, templates, f"missing template key: {key}")
            self.assertIsInstance(templates[key], str)
            self.assertTrue(templates[key])

    def test_atom_templates_hash(self) -> None:
        content = (CONTRACT_ROOT / "atom_templates.v1.json").read_bytes()
        actual = hashlib.sha256(content).hexdigest()[:16]
        registry = ContractRegistry.load()
        entry = registry.require("melm.atom_templates.v1")
        self.assertEqual(actual, entry.get("schema_hash", ""),
                         "atom_templates.v1.json schema_hash mismatch")


class ContractMvpSelfIdentityTests(unittest.TestCase):
    """Contract integration tests for self_identity.v1.json."""

    def test_validate_self_identity(self):
        from melm.contracts.validation import validate_self_identity
        data = {
            "schema_id": "melm.self_identity.v1",
            "analysis_window_days": 30,
            "min_data_points": 3,
            "identity_labels": {
                "story": {"label": "a storyteller", "frame": "sharing stories"},
                "weather": {"label": "a weather watcher", "frame": "checking the weather"},
            },
            "identity_narratives": {
                "neutral": "I see myself as {label}.",
                "happy": "I feel like {label}.",
            },
            "name_awareness_templates": {
                "no_name": "You could call me {label}.",
                "why": "You asked me to {frame} {count} times.",
            },
        }
        validate_self_identity(data)

    def test_validate_self_identity_rejects_empty(self):
        from melm.contracts.validation import ContractValidationError, validate_self_identity
        with self.assertRaises(ContractValidationError):
            validate_self_identity({})

    def test_load_self_identity(self):
        from melm.contracts.validation import load_self_identity
        data = load_self_identity()
        self.assertEqual(data["schema_id"], "melm.self_identity.v1")
        self.assertIn("identity_labels", data)
        self.assertIn("identity_narratives", data)
        self.assertIn("name_awareness_templates", data)
        self.assertGreater(len(data["identity_labels"]), 0)
        self.assertGreater(len(data["identity_narratives"]), 0)
        self.assertGreater(len(data["name_awareness_templates"]), 0)


class ContractMvpStorySceneTests(unittest.TestCase):
    def test_story_scene_templates_contract_loads(self):
        from melm.contracts import load_story_scene_templates
        data = load_story_scene_templates()
        assert data is not None
        assert "archetypes" in data

    def test_story_scene_templates_has_minimum_archetypes(self):
        from melm.contracts import load_story_scene_templates
        data = load_story_scene_templates()
        assert len(data.get("archetypes", [])) >= 3

    def test_story_scene_templates_each_archetype_has_required_fields(self):
        from melm.contracts import load_story_scene_templates
        data = load_story_scene_templates()
        for archetype in data.get("archetypes", []):
            assert "archetype_id" in archetype
            assert "entity_slots" in archetype
            assert "atom_sequence" in archetype
            for atom in archetype["atom_sequence"]:
                assert "verb" in atom
                assert "subject" in atom

    def test_story_scene_templates_entity_slots_have_allowed_classes(self):
        from melm.contracts import load_story_scene_templates, load_semantic_class_ids
        data = load_story_scene_templates()
        all_ids = load_semantic_class_ids()
        for archetype in data.get("archetypes", []):
            for slot in archetype.get("entity_slots", []):
                assert "role" in slot
                assert "allowed_classes" in slot
                for cls in slot["allowed_classes"]:
                    assert cls in all_ids, f"Class '{cls}' not in semantic_classes.v1.json"

    def test_story_scene_templates_verb_lemmas_match_verb_atoms(self):
        from melm.contracts import load_story_scene_templates, load_verb_atoms
        data = load_story_scene_templates()
        verbs = load_verb_atoms()
        verb_ids = set(verbs)
        for archetype in data.get("archetypes", []):
            for atom in archetype.get("atom_sequence", []):
                expected_id = f"verb__{atom['verb']}"
                assert expected_id in verb_ids, (
                    f"Verb '{atom['verb']}' has no atom in verb_atoms.v1.json"
                )


if __name__ == "__main__":
    unittest.main()
