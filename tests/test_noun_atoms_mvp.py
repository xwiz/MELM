"""Tests for noun atoms: contract validation, enrichment, seeding."""

import unittest
from dataclasses import replace

from melm.contracts import (
    ContractRegistry,
    ContractValidationError,
    load_noun_atoms,
    validate_noun_atoms,
)
from melm.appliance.uol_types import RoleAssignment
from melm.appliance.uol_atomizer import enrich_role_entities


class TestNounContractValidation(unittest.TestCase):
    def test_contract_loads_and_validates(self) -> None:
        data = load_noun_atoms()
        self.assertEqual(data["schema_id"], "melm.noun_atoms.v1")
        self.assertGreater(len(data["entities"]), 200)

    def test_all_entity_ids_unique(self) -> None:
        data = load_noun_atoms()
        ids = [e["entity_id"] for e in data["entities"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_kinds_valid(self) -> None:
        data = load_noun_atoms()
        valid = {"object", "place", "person", "animal", "plant", "concept"}
        for e in data["entities"]:
            self.assertIn(e.get("kind", ""), valid, f"{e['entity_id']}: bad kind")

    def test_missing_schema_id_fails(self) -> None:
        with self.assertRaises(ContractValidationError):
            validate_noun_atoms({"entities": [{"entity_id": "x", "label": "x"}]})

    def test_duplicate_entity_id_fails(self) -> None:
        payload = {
            "schema_id": "melm.noun_atoms.v1",
            "entities": [
                {"entity_id": "noun__x", "label": "x", "kind": "object"},
                {"entity_id": "noun__x", "label": "x", "kind": "object"},
            ],
        }
        with self.assertRaisesRegex(ContractValidationError, "duplicate"):
            validate_noun_atoms(payload)

    def test_empty_entities_fails(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "non-empty"):
            validate_noun_atoms({"schema_id": "melm.noun_atoms.v1", "entities": []})

    def test_invalid_kind_fails(self) -> None:
        payload = {
            "schema_id": "melm.noun_atoms.v1",
            "entities": [{"entity_id": "noun__x", "label": "x", "kind": "invalid_kind"}],
        }
        with self.assertRaisesRegex(ContractValidationError, "kind"):
            validate_noun_atoms(payload)


class TestNounEnrichment(unittest.TestCase):
    def test_known_entity_enriched(self) -> None:
        roles = [RoleAssignment(role="theme", value="vase", confidence=1.0)]
        enriched = enrich_role_entities(roles)
        self.assertEqual(enriched[0].entity_id, "noun__vase")
        self.assertEqual(enriched[0].semantic_class, "physical_object")

    def test_known_entity_case_insensitive(self) -> None:
        roles = [RoleAssignment(role="theme", value="VASE", confidence=1.0)]
        enriched = enrich_role_entities(roles)
        self.assertEqual(enriched[0].entity_id, "noun__vase")

    def test_unknown_entity_passthrough(self) -> None:
        roles = [RoleAssignment(role="theme", value="gobbledygook")]
        enriched = enrich_role_entities(roles)
        self.assertEqual(enriched[0].entity_id, "")
        self.assertEqual(enriched[0].semantic_class, "")
        self.assertEqual(enriched[0].value, "gobbledygook")

    def test_multiple_roles_mixed(self) -> None:
        roles = [
            RoleAssignment(role="agent", value="dog"),
            RoleAssignment(role="theme", value="unknown_thing"),
            RoleAssignment(role="location", value="kitchen"),
        ]
        enriched = enrich_role_entities(roles)
        self.assertEqual(enriched[0].entity_id, "noun__dog")
        self.assertEqual(enriched[1].entity_id, "")
        self.assertEqual(enriched[2].entity_id, "noun__kitchen")

    def test_enrichment_preserves_other_fields(self) -> None:
        role = RoleAssignment(
            role="theme", value="vase", status="asserted", confidence=0.75
        )
        [enriched] = enrich_role_entities([role])
        self.assertEqual(enriched.role, "theme")
        self.assertEqual(enriched.status, "asserted")
        self.assertEqual(enriched.confidence, 0.75)

    def test_empty_roles_list(self) -> None:
        self.assertEqual(enrich_role_entities([]), [])

    def test_entity_from_agent_semantic_class(self) -> None:
        roles = [RoleAssignment(role="agent", value="dog")]
        enriched = enrich_role_entities(roles)
        self.assertEqual(enriched[0].semantic_class, "living_thing.animal")

    def test_entity_from_location_semantic_class(self) -> None:
        roles = [RoleAssignment(role="location", value="kitchen")]
        enriched = enrich_role_entities(roles)
        self.assertEqual(enriched[0].semantic_class, "location")


class TestNounStoreSeeding(unittest.TestCase):
    def test_seed_noun_atoms_writes_to_store(self) -> None:
        from melm.appliance.assistant_os_store import AssistantOSStore
        import tempfile, os
        db_path = os.path.join(tempfile.gettempdir(), "test_noun_seed.db")
        try:
            store = AssistantOSStore(path=db_path)
            from melm.contracts import seed_noun_atoms
            seed_noun_atoms(store)
            # Verify a few entities
            vase = store.get_entity("noun__vase")
            self.assertIsNotNone(vase)
            self.assertEqual(vase.label, "vase")
            self.assertEqual(vase.semantic_class_id, "physical_object")
            # Verify slots
            dims = store.get_entity_slot("noun__vase", "dimensions")
            self.assertIsNotNone(dims)
            self.assertIn("shape", dims.value_json)
            # Verify place entity
            kitchen = store.get_entity("noun__kitchen")
            self.assertIsNotNone(kitchen)
            self.assertEqual(kitchen.kind, "place")
            self.assertEqual(kitchen.semantic_class_id, "location")
        finally:
            store.close()
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_seed_idempotent(self) -> None:
        from melm.appliance.assistant_os_store import AssistantOSStore
        import tempfile, os
        db_path = os.path.join(tempfile.gettempdir(), "test_noun_seed_idempotent.db")
        try:
            store = AssistantOSStore(path=db_path)
            from melm.contracts import seed_noun_atoms
            seed_noun_atoms(store)
            seed_noun_atoms(store)
            entities = store.find_entities()
            vase_count = sum(1 for e in entities if e.entity_id == "noun__vase")
            self.assertEqual(vase_count, 1)
        finally:
            store.close()
            if os.path.exists(db_path):
                os.remove(db_path)


class TestNounRegistryIntegration(unittest.TestCase):
    def test_registry_contains_noun_atoms(self) -> None:
        registry = ContractRegistry.load()
        entry = registry.require("melm.noun_atoms.v1")
        self.assertEqual(entry["path"], "noun_atoms.v1.json")
        self.assertEqual(entry["owner"], "meaning-core")

    def test_contract_hash_matches(self) -> None:
        registry = ContractRegistry.load()
        errors = registry.check_compatibility()
        noun_errors = [e for e in errors if "noun_atoms" in e]
        self.assertEqual(noun_errors, [], f"Hash mismatch: {noun_errors}")


class TestRoleAssignmentBackwardCompat(unittest.TestCase):
    def test_default_semantic_class_is_empty(self) -> None:
        r = RoleAssignment(role="agent", value="user")
        self.assertEqual(r.semantic_class, "")

    def test_default_entity_id_is_empty(self) -> None:
        r = RoleAssignment(role="agent", value="user")
        self.assertEqual(r.entity_id, "")

    def test_explicit_semantic_class_works(self) -> None:
        r = RoleAssignment(role="agent", value="dog", semantic_class="living_thing.animal")
        self.assertEqual(r.semantic_class, "living_thing.animal")

    def test_explicit_entity_id_works(self) -> None:
        r = RoleAssignment(role="theme", value="vase", entity_id="noun__vase")
        self.assertEqual(r.entity_id, "noun__vase")

    def test_replace_preserves_frozen(self) -> None:
        r = RoleAssignment(role="theme", value="vase")
        r2 = replace(r, entity_id="noun__vase")
        self.assertEqual(r.entity_id, "")
        self.assertEqual(r2.entity_id, "noun__vase")

    def test_pipeline_roundtrip_enrichment(self) -> None:
        roles = [RoleAssignment(role="theme", value="vase")]
        enriched = enrich_role_entities(roles)
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].entity_id, "noun__vase")
        # Verify original is unchanged (frozen)
        self.assertEqual(roles[0].entity_id, "")

    def test_uol_serialization_includes_entity_fields(self) -> None:
        from melm.appliance.uol_types import UolAtom, UolAct, PredicateRef, AtomContext
        roles = (
            RoleAssignment(role="theme", value="vase", semantic_class="physical_object", entity_id="noun__vase"),
        )
        pred = PredicateRef(id="buy", semantic_class="verb.posses", language="en")
        atom = UolAtom(id="test", kind="event", predicate=pred, roles=roles)
        act = UolAct(id="uol_test", act="command", content=(atom,))
        d = act.to_dict()
        atom_dict = d["content"][0]
        self.assertEqual(len(atom_dict["roles"]), 1)
        role_dict = atom_dict["roles"][0]
        self.assertEqual(role_dict["entity_id"], "noun__vase")
        self.assertEqual(role_dict["semantic_class"], "physical_object")
        self.assertEqual(role_dict["value"], "vase")


class TestSynthesisNlgEnrichment(unittest.TestCase):
    def setUp(self):
        from melm.appliance.assistant_os_store import AssistantOSStore
        from melm.appliance.local_assistant_router import AssistantDecision, LocalAssistantProfile
        from melm.appliance.assistant_synthesis import BoundedLocalSynthesizer
        self.store = AssistantOSStore(path=":memory:")
        from melm.contracts import seed_noun_atoms, seed_verb_atoms
        seed_noun_atoms(self.store)
        seed_verb_atoms(self.store)
        profile = LocalAssistantProfile()
        self.synth = BoundedLocalSynthesizer(profile=profile, store=self.store)

    def tearDown(self):
        self.store.close()

    def _make_decision(self, uol_act: dict | None = None) -> "AssistantDecision":
        from melm.appliance.local_assistant_router import AssistantDecision
        return AssistantDecision(
            utterance="",
            intent="open_domain",
            route="local_answer",
            answer="That is a good question.",
            evidence_keys=(),
            reason="fallback",
            cloud_needed=False,
            confidence=1.0,
            uol_act=uol_act,
        )

    def test_append_nlg_fragile_entity(self) -> None:
        roles = [
            {"role": "theme", "value": "vase", "semantic_class": "physical_object",
             "entity_id": "noun__vase", "status": "asserted", "confidence": 1.0},
        ]
        uol_act = {
            "id": "uol_test", "act": "command", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "event",
                "predicate": {"id": "buy", "semantic_class": "verb.consume", "lemma": "buy", "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []},
            }],
            "expected_answer_type": None,
        }
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("Buy a vase.", decision)
        self.assertIn("fragile", result)
        self.assertIn("handle with care", result)

    def test_append_nlg_place_context(self) -> None:
        roles = [
            {"role": "location", "value": "kitchen", "semantic_class": "location",
             "entity_id": "noun__kitchen", "status": "asserted", "confidence": 1.0},
        ]
        uol_act = {
            "id": "uol_test", "act": "question", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "state",
                "predicate": {"id": "be", "semantic_class": "verb.stative", "lemma": "be", "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []},
            }],
            "expected_answer_type": None,
        }
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("Where is the kitchen?", decision)
        self.assertIn("kitchen is located", result.lower())
        self.assertIn("inside a building", result)

    def test_no_store_no_enrichment(self) -> None:
        from melm.appliance.local_assistant_router import LocalAssistantProfile
        from melm.appliance.assistant_synthesis import BoundedLocalSynthesizer
        profile = LocalAssistantProfile()
        synth_no_store = BoundedLocalSynthesizer(profile=profile, store=None)
        decision = self._make_decision(uol_act=None)
        result = synth_no_store._append_entity_nlg("Buy a vase.", decision)
        self.assertEqual(result, "Buy a vase.")

    def test_unknown_entity_no_enrichment(self) -> None:
        roles = [
            {"role": "theme", "value": "gobbledygook", "semantic_class": "",
             "entity_id": "", "status": "asserted", "confidence": 1.0},
        ]
        uol_act = {"id": "uol_test", "act": "command", "speaker": "user", "addressee": "assistant",
                   "content": [{"id": "uol_atom", "kind": "event",
                                "predicate": {"id": "buy", "semantic_class": "verb.consume", "lemma": "buy", "language": "en"},
                                "roles": roles,
                                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                                            "tense": "", "aspect": "", "certainty": 1.0,
                                            "time": {"time_kind": "", "ref": ""}},
                                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                                          "subordinate_atoms": []}}],
                   "expected_answer_type": None}
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("Buy something.", decision)
        self.assertEqual(result, "Buy something.")

    def test_append_nlg_verb_harm_property(self) -> None:
        uol_act = {
            "id": "uol_test", "act": "command", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "event",
                "predicate": {"id": "hit", "semantic_class": "verb.contact", "lemma": "hit",
                              "language": "en", "entity_id": "verb__hit"},
                "roles": [],
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []}}],
            "expected_answer_type": None}
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("Don't hit your sister.", decision)
        self.assertIn("harm", result.lower())

    def test_append_nlg_verb_no_harm(self) -> None:
        uol_act = {
            "id": "uol_test", "act": "command", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "event",
                "predicate": {"id": "hug", "semantic_class": "verb.contact", "lemma": "hug",
                              "language": "en", "entity_id": "verb__hug"},
                "roles": [],
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []}}],
            "expected_answer_type": None}
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("Give your sister a hug.", decision)
        self.assertEqual(result, "Give your sister a hug.")

    def test_append_nlg_materials(self) -> None:
        roles = [
            {"role": "theme", "value": "bed", "semantic_class": "physical_object",
             "entity_id": "noun__bed", "status": "asserted", "confidence": 1.0},
        ]
        uol_act = {
            "id": "uol_test", "act": "command", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "event",
                "predicate": {"id": "buy", "semantic_class": "verb.consume", "lemma": "buy", "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []},
            }],
            "expected_answer_type": None,
        }
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("Buy a bed.", decision)
        self.assertIn("made of", result)
        self.assertIn("wood", result)
        self.assertIn("metal", result)
        self.assertIn("fabric", result)

    def test_append_nlg_functional_uses(self) -> None:
        roles = [
            {"role": "theme", "value": "bed", "semantic_class": "physical_object",
             "entity_id": "noun__bed", "status": "asserted", "confidence": 1.0},
        ]
        uol_act = {
            "id": "uol_test", "act": "question", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "state",
                "predicate": {"id": "need", "semantic_class": "verb.stative", "lemma": "need", "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []},
            }],
            "expected_answer_type": None,
        }
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("I need a bed.", decision)
        self.assertIn("used for", result.lower())
        self.assertIn("sleeping", result.lower())

    def test_append_nlg_color(self) -> None:
        roles = [
            {"role": "theme", "value": "rose", "semantic_class": "physical_object",
             "entity_id": "noun__rose", "status": "asserted", "confidence": 1.0},
        ]
        uol_act = {
            "id": "uol_test", "act": "statement", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "state",
                "predicate": {"id": "see", "semantic_class": "verb.perceive", "lemma": "see", "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []},
            }],
            "expected_answer_type": None,
        }
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("I see a rose.", decision)
        self.assertIn("roses are typically", result.lower())
        self.assertIn("pink", result.lower())
        self.assertIn("white", result.lower())

    def test_append_nlg_slot_cap_enforced(self) -> None:
        """An entity with many populated slots should emit at most 2 appendices."""
        roles = [
            {"role": "theme", "value": "vase", "semantic_class": "physical_object",
             "entity_id": "noun__vase", "status": "asserted", "confidence": 1.0},
        ]
        uol_act = {
            "id": "uol_test", "act": "command", "speaker": "user", "addressee": "assistant",
            "content": [{
                "id": "uol_atom", "kind": "event",
                "predicate": {"id": "buy", "semantic_class": "verb.consume", "lemma": "buy", "language": "en"},
                "roles": roles,
                "context": {"polarity": "positive", "modality": "", "negation_scope": "",
                            "tense": "", "aspect": "", "certainty": 1.0,
                            "time": {"time_kind": "", "ref": ""}},
                "links": {"causes": [], "caused_by": [], "enables": [], "prevents": [],
                          "subordinate_atoms": []},
            }],
            "expected_answer_type": None,
        }
        decision = self._make_decision(uol_act=uol_act)
        result = self.synth._append_entity_nlg("Buy a vase.", decision)
        # Vase has fragile + materials + uses — cap should keep it to 2
        period_count = result.count(".")
        self.assertLessEqual(period_count, 3, msg=f"Too many appendices: {result}")


class TestVerbAtomContractValidation(unittest.TestCase):
    def test_verb_contract_loads_and_validates(self) -> None:
        from melm.contracts import validate_verb_atoms, load_contract_json
        payload = load_contract_json("verb_atoms.v1.json")
        validate_verb_atoms(payload)
        self.assertEqual(payload["schema_id"], "melm.verb_atoms.v1")
        self.assertGreaterEqual(len(payload["entities"]), 59)

    def test_verb_entity_kind_is_action(self) -> None:
        from melm.contracts import load_verb_atoms
        atoms = load_verb_atoms()
        for eid, entry in atoms.items():
            self.assertEqual(entry["kind"], "action", f"{eid} kind must be action")

    def test_verb_semantic_class_ids_valid(self) -> None:
        from melm.contracts import validate_semantic_class_registry, load_verb_atoms
        atoms = load_verb_atoms()
        for eid, entry in atoms.items():
            sc = str(entry.get("semantic_class_id", ""))
            self.assertTrue(sc.startswith("verb."), f"{eid} semantic_class_id={sc}")

    def test_verb_harm_severity_range(self) -> None:
        from melm.contracts import load_verb_atoms
        atoms = load_verb_atoms()
        for eid, entry in atoms.items():
            slots = entry.get("slots", {})
            harm = slots.get("harm_severity", 0)
            if harm:
                self.assertGreaterEqual(harm, 0.0)
                self.assertLessEqual(harm, 2.0)

    def test_verb_seed_writes_to_store(self) -> None:
        from melm.appliance.assistant_os_store import AssistantOSStore
        from melm.contracts import seed_verb_atoms
        store = AssistantOSStore(path=":memory:")
        seed_verb_atoms(store)
        entity = store.get_entity("verb__hit")
        self.assertIsNotNone(entity)
        self.assertEqual(entity.kind, "action")
        slot = store.get_entity_slot("verb__hit", "patient_states")
        self.assertIsNotNone(slot)
        store.close()

    def test_enrich_verb_predicate_matches(self) -> None:
        from melm.appliance.uol_types import PredicateRef
        from melm.appliance.uol_atomizer import enrich_verb_predicate
        pred = PredicateRef(id="hit", semantic_class="verb.contact", lemma="hit")
        enriched = enrich_verb_predicate(pred)
        self.assertEqual(enriched.entity_id, "verb__hit")

    def test_enrich_verb_predicate_unknown(self) -> None:
        from melm.appliance.uol_types import PredicateRef
        from melm.appliance.uol_atomizer import enrich_verb_predicate
        pred = PredicateRef(id="xyzzy", semantic_class="verb.contact", lemma="xyzzy")
        enriched = enrich_verb_predicate(pred)
        self.assertEqual(enriched.entity_id, "")

    def test_predicate_ref_backward_compat(self) -> None:
        from melm.appliance.uol_types import PredicateRef
        pred = PredicateRef(id="test", semantic_class="verb.test")
        self.assertEqual(pred.entity_id, "")

    def test_predicate_ref_serialization(self) -> None:
        from melm.appliance.uol_types import PredicateRef, UolAtom, AtomKind, AtomContext, AtomLinks
        pred = PredicateRef(id="hit", semantic_class="verb.contact", lemma="hit",
                            entity_id="verb__hit")
        atom = UolAtom(
            id="t1", kind="event", predicate=pred, context=AtomContext(),
            links=AtomLinks(),
        )
        d = atom.to_dict()
        self.assertEqual(d["predicate"]["entity_id"], "verb__hit")
