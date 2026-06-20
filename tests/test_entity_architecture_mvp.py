"""Tests for the unified entity architecture (class_schemas, entities, entity_slots).

Covers: schema creation, migration, seeding, CRUD, slot states, class hierarchy,
idempotency, and restart persistence.
"""

import json
import tempfile
from pathlib import Path
import unittest

from melm.appliance import AssistantOSStore
from melm.appliance.assistant_os_store import (
    seed_class_schemas,
    StoredEntity,
    StoredEntitySlot,
    StoredEntityRelation,
    migrate_contacts_to_entities,
    migrate_self_facts_to_entities,
    _now,
)


class EntitySchemaMvpTests(unittest.TestCase):
    """Entity tables exist after initialize, migration creates missing tables."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_entity_tables_exist_after_initialize(self) -> None:
        store = self.make_store()
        try:
            for table in ("class_schemas", "class_schema_slots", "entities", "entity_slots", "entity_relations"):
                row = store.connection.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                ).fetchone()
                self.assertIsNotNone(row, f"Table {table} not found in schema")
        finally:
            store.close()

    def test_entity_indexes_exist(self) -> None:
        store = self.make_store()
        try:
            indexes = {
                str(r["name"])
                for r in store.connection.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            }
            for idx in ("idx_class_schemas_parent", "idx_entities_kind", "idx_entities_semantic_class",
                        "idx_entity_slots_entity", "idx_entity_slots_state",
                        "idx_entity_relations_entity", "idx_entity_relations_relation"):
                self.assertIn(idx, indexes, f"Missing index: {idx}")
        finally:
            store.close()

    def test_migration_creates_entity_tables_on_existing_store(self) -> None:
        store = self.make_store()
        try:
            store.connection.execute("DROP TABLE entity_relations")
            store.connection.execute("DROP TABLE entity_slots")
            store.connection.execute("DROP TABLE entities")
            store.connection.execute("DROP TABLE class_schema_slots")
            store.connection.execute("DROP TABLE class_schemas")
            store.connection.commit()
            store._ensure_entity_tables()
            for table in ("class_schemas", "class_schema_slots", "entities", "entity_slots", "entity_relations"):
                row = store.connection.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                ).fetchone()
                self.assertIsNotNone(row, f"Table {table} not found after migration")
        finally:
            store.close()

    def test_migration_is_idempotent_when_tables_exist(self) -> None:
        store = self.make_store()
        try:
            store._ensure_entity_tables()
            store._ensure_entity_tables()
            for table in ("class_schemas", "class_schema_slots", "entities", "entity_slots", "entity_relations"):
                count = store.connection.execute(
                    f"SELECT COUNT(*) AS c FROM {table}"
                ).fetchone()
                self.assertEqual(0, count["c"])
        finally:
            store.close()

    def test_entity_tables_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "entity.sqlite"
            store = AssistantOSStore(path)
            try:
                store.add_entity("p1", "person", "Alice", "person")
                store.connection.commit()
            finally:
                store.close()
            store = AssistantOSStore(path)
            try:
                entity = store.get_entity("p1")
                self.assertIsNotNone(entity)
                self.assertEqual("Alice", entity.label)
            finally:
                store.close()


class SeedClassSchemasMvpTests(unittest.TestCase):
    """seed_class_schemas populates class hierarchy and slot definitions."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_seeds_twenty_classes(self) -> None:
        store = self.make_store()
        try:
            seed_class_schemas(store)
            rows = store.connection.execute(
                "SELECT semantic_class_id, label FROM class_schemas ORDER BY semantic_class_id"
            ).fetchall()
            ids = [str(r["semantic_class_id"]) for r in rows]
            self.assertEqual(
                ["abstract", "anonymous_fact", "cognition", "competition", "deferred_task", "entity", "epistemic_state", "event", "learned_fact", "mood_ambient", "mood_session_summary", "mood_state", "novelty_candidate", "object", "person", "personal_experience", "place", "uol_parse", "user_commitment", "world_fact"],
                ids,
            )
        finally:
            store.close()

    def test_seeds_slot_definitions(self) -> None:
        store = self.make_store()
        try:
            seed_class_schemas(store)
            rows = store.connection.execute(
                "SELECT semantic_class_id, slot_name, required FROM class_schema_slots ORDER BY semantic_class_id, slot_name"
            ).fetchall()
            self.assertGreater(len(rows), 0)
            slot_keys = {(str(r["semantic_class_id"]), str(r["slot_name"])) for r in rows}
            self.assertIn(("person", "name"), slot_keys)
            self.assertIn(("person", "phone"), slot_keys)
            self.assertIn(("event", "start_time"), slot_keys)
            self.assertIn(("competition", "winner"), slot_keys)
        finally:
            store.close()

    def test_idempotent(self) -> None:
        store = self.make_store()
        try:
            seed_class_schemas(store)
            seed_class_schemas(store)
            count = store.connection.execute("SELECT COUNT(*) AS c FROM class_schemas").fetchone()
            self.assertEqual(20, count["c"])
        finally:
            store.close()

    def test_class_hierarchy_parent_refs(self) -> None:
        store = self.make_store()
        try:
            seed_class_schemas(store)
            event_parent = store.connection.execute(
                "SELECT parent_class_id FROM class_schemas WHERE semantic_class_id='event'"
            ).fetchone()
            self.assertEqual("entity", event_parent["parent_class_id"])
            comp_parent = store.connection.execute(
                "SELECT parent_class_id FROM class_schemas WHERE semantic_class_id='competition'"
            ).fetchone()
            self.assertEqual("event", comp_parent["parent_class_id"])
            root_parent = store.connection.execute(
                "SELECT parent_class_id FROM class_schemas WHERE semantic_class_id='entity'"
            ).fetchone()
            self.assertIsNone(root_parent["parent_class_id"])
        finally:
            store.close()


class EntityCrudMvpTests(unittest.TestCase):
    """add_entity, get_entity, find_entities, set_entity_slot, delete_entity."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_add_and_get_entity(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            entity = store.get_entity("alice")
            self.assertIsNotNone(entity)
            self.assertEqual("alice", entity.entity_id)
            self.assertEqual("person", entity.kind)
            self.assertEqual("Alice", entity.label)
            self.assertEqual("person", entity.semantic_class_id)
            self.assertIsInstance(entity.created_at, str)
            self.assertIsInstance(entity.updated_at, str)
        finally:
            store.close()

    def test_get_nonexistent_entity(self) -> None:
        store = self.make_store()
        try:
            entity = store.get_entity("nonexistent")
            self.assertIsNone(entity)
        finally:
            store.close()

    def test_add_duplicate_is_idempotent(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("alice", "person", "Alice", "person")
            entity = store.get_entity("alice")
            self.assertIsNotNone(entity)
            self.assertEqual("Alice", entity.label)
        finally:
            store.close()

    def test_add_entity_with_defaults(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("e1", "event_type", "Test Event")
            entity = store.get_entity("e1")
            self.assertIsNotNone(entity)
            self.assertEqual("", entity.semantic_class_id)
            self.assertEqual("", entity.canonical_lemma)
        finally:
            store.close()

    def test_find_entities_by_kind(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.add_entity("cup", "object", "Cup", "object")
            people = store.find_entities(kind="person")
            self.assertEqual(2, len(people))
            ids = {p.entity_id for p in people}
            self.assertEqual({"alice", "bob"}, ids)
        finally:
            store.close()

    def test_find_entities_by_semantic_class(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.add_entity("cup", "object", "Cup", "object")
            objects = store.find_entities(semantic_class_id="object")
            self.assertEqual(1, len(objects))
            self.assertEqual("cup", objects[0].entity_id)
        finally:
            store.close()

    def test_find_entities_empty_filters(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("cup", "object", "Cup", "object")
            all_entities = store.find_entities()
            self.assertEqual(2, len(all_entities))
        finally:
            store.close()

    def test_find_entities_no_matches(self) -> None:
        store = self.make_store()
        try:
            results = store.find_entities(kind="person")
            self.assertEqual([], results)
        finally:
            store.close()

    def test_stored_entity_is_frozen_dataclass(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            entity = store.get_entity("alice")
            self.assertIsInstance(entity, StoredEntity)
            with self.assertRaises(AttributeError):
                entity.label = "Bob"
        finally:
            store.close()


class EntitySlotCrudMvpTests(unittest.TestCase):
    """set_entity_slot, get_entity_slots, get_entity_slot, slot_state."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_set_and_get_slot(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.set_entity_slot("alice", "phone", "555-0100")
            slots = store.get_entity_slots("alice")
            self.assertEqual(1, len(slots))
            self.assertEqual("phone", slots[0].slot_name)
            self.assertEqual("555-0100", json.loads(slots[0].value_json))
            self.assertEqual("filled", slots[0].slot_state)
        finally:
            store.close()

    def test_set_slot_updates_existing(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.set_entity_slot("alice", "phone", "555-0100")
            store.set_entity_slot("alice", "phone", "555-0199")
            slot = store.get_entity_slot("alice", "phone")
            self.assertIsNotNone(slot)
            self.assertEqual("555-0199", json.loads(slot.value_json))
        finally:
            store.close()

    def test_missing_slot_returns_none(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            slot = store.get_entity_slot("alice", "nonexistent")
            self.assertIsNone(slot)
        finally:
            store.close()

    def test_multiple_slots_per_entity(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.set_entity_slot("alice", "phone", "555-0100")
            store.set_entity_slot("alice", "email", "alice@example.com")
            store.set_entity_slot("alice", "name", "Alice Smith")
            slots = store.get_entity_slots("alice")
            self.assertEqual(3, len(slots))
            names = {s.slot_name for s in slots}
            self.assertEqual({"phone", "email", "name"}, names)
        finally:
            store.close()

    def test_slot_state(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.set_entity_slot("alice", "phone", "555-0100", slot_state="filled")
            store.set_entity_slot("alice", "email", "", slot_state="asked_but_empty")
            store.set_entity_slot("alice", "address", "", slot_state="unknown")
            slots = store.get_entity_slots("alice")
            states = {s.slot_name: s.slot_state for s in slots}
            self.assertEqual("filled", states["phone"])
            self.assertEqual("asked_but_empty", states["email"])
            self.assertEqual("unknown", states["address"])
        finally:
            store.close()

    def test_slot_provenance(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.set_entity_slot("alice", "phone", "555-0100", provenance="user_taught")
            slot = store.get_entity_slot("alice", "phone")
            self.assertEqual("user_taught", slot.provenance)
        finally:
            store.close()

    def test_stored_entity_slot_is_frozen_dataclass(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.set_entity_slot("alice", "phone", "555-0100")
            slot = store.get_entity_slot("alice", "phone")
            self.assertIsInstance(slot, StoredEntitySlot)
            with self.assertRaises(AttributeError):
                slot.slot_name = "email"
        finally:
            store.close()


class DeleteEntityMvpTests(unittest.TestCase):
    """delete_entity removes entity, slots, and relations."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_delete_entity_removes_entity_and_slots(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.set_entity_slot("alice", "phone", "555-0100")
            store.delete_entity("alice")
            self.assertIsNone(store.get_entity("alice"))
            self.assertEqual([], store.get_entity_slots("alice"))
        finally:
            store.close()

    def test_delete_nonexistent_entity_does_not_raise(self) -> None:
        store = self.make_store()
        try:
            store.delete_entity("nonexistent")
        finally:
            store.close()

    def test_delete_entity_does_not_affect_others(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.set_entity_slot("alice", "phone", "555-0100")
            store.set_entity_slot("bob", "phone", "555-0199")
            store.delete_entity("alice")
            self.assertIsNotNone(store.get_entity("bob"))
            bob_slots = store.get_entity_slots("bob")
            self.assertEqual(1, len(bob_slots))
        finally:
            store.close()

    def test_delete_entity_removes_entity_relations(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            now = _now()
            store.connection.execute(
                "INSERT INTO entity_relations(relation_id, entity_id, relation, target_entity_id, provenance, strength, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("r1", "alice", "knows", "bob", "user_taught", 1.0, now),
            )
            store.delete_entity("alice")
            remaining = store.connection.execute(
                "SELECT COUNT(*) AS c FROM entity_relations"
            ).fetchone()
            self.assertEqual(0, remaining["c"])
        finally:
            store.close()

    def test_delete_entity_removes_target_relations(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            now = _now()
            store.connection.execute(
                "INSERT INTO entity_relations(relation_id, entity_id, relation, target_entity_id, provenance, strength, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("r1", "bob", "knows", "alice", "user_taught", 1.0, now),
            )
            store.delete_entity("alice")
            remaining = store.connection.execute(
                "SELECT COUNT(*) AS c FROM entity_relations"
            ).fetchone()
            self.assertEqual(0, remaining["c"])
        finally:
            store.close()


class EntityRelationMvpTests(unittest.TestCase):
    """entity_relations CRUD (direct SQL integration)."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_insert_and_query_relation(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            now = _now()
            store.connection.execute(
                "INSERT INTO entity_relations(relation_id, entity_id, relation, target_entity_id, provenance, strength, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("r1", "alice", "knows", "bob", "user_taught", 1.0, now),
            )
            row = store.connection.execute(
                "SELECT * FROM entity_relations WHERE relation_id='r1'"
            ).fetchone()
            self.assertEqual("alice", row["entity_id"])
            self.assertEqual("knows", row["relation"])
            self.assertEqual("bob", row["target_entity_id"])
        finally:
            store.close()

    def test_duplicate_relation_is_rejected(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            now = _now()
            store.connection.execute(
                "INSERT INTO entity_relations(relation_id, entity_id, relation, target_entity_id, provenance, strength, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("r1", "alice", "knows", "bob", "user_taught", 1.0, now),
            )
            with self.assertRaises(Exception):
                store.connection.execute(
                    "INSERT INTO entity_relations(relation_id, entity_id, relation, target_entity_id, provenance, strength, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("r2", "alice", "knows", "bob", "user_taught", 1.0, now),
                )
        finally:
            store.close()

    def test_relation_entity_fk_enforced(self) -> None:
        store = self.make_store()
        try:
            now = _now()
            with self.assertRaises(Exception):
                store.connection.execute(
                    "INSERT INTO entity_relations(relation_id, entity_id, relation, target_entity_id, provenance, strength, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("r1", "nonexistent", "knows", "also_nonexistent", "user_taught", 1.0, now),
                )
        finally:
            store.close()

    def test_add_relation_returns_id(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            rid = store.add_relation("alice", "knows", "bob")
            self.assertIsInstance(rid, str)
            self.assertTrue(len(rid) > 0)
        finally:
            store.close()

    def test_add_relation_with_provenance_and_strength(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.add_relation("alice", "knows", "bob", provenance="user_taught", strength=0.95)
            relations = store.get_entity_relations("alice")
            self.assertEqual(1, len(relations))
            self.assertEqual("knows", relations[0].relation)
            self.assertEqual("bob", relations[0].target_entity_id)
            self.assertEqual("user_taught", relations[0].provenance)
            self.assertAlmostEqual(0.95, relations[0].strength)
        finally:
            store.close()

    def test_get_entity_relations_empty(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            self.assertEqual([], store.get_entity_relations("alice"))
        finally:
            store.close()

    def test_get_entity_relations_multiple(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.add_entity("carol", "person", "Carol", "person")
            store.add_relation("alice", "knows", "bob")
            store.add_relation("alice", "knows", "carol")
            relations = store.get_entity_relations("alice")
            self.assertEqual(2, len(relations))
            targets = {r.target_entity_id for r in relations}
            self.assertIn("bob", targets)
            self.assertIn("carol", targets)
        finally:
            store.close()

    def test_find_relations_by_type(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.add_entity("carol", "person", "Carol", "person")
            store.add_entity("event1", "event_instance", "Party", "event")
            store.add_relation("alice", "knows", "bob")
            store.add_relation("alice", "knows", "carol")
            store.add_relation("alice", "attended", "event1")
            knows = store.find_relations_by_type("knows")
            self.assertEqual(2, len(knows))
            attended = store.find_relations_by_type("attended")
            self.assertEqual(1, len(attended))
        finally:
            store.close()

    def test_find_relations_by_type_empty(self) -> None:
        store = self.make_store()
        try:
            self.assertEqual([], store.find_relations_by_type("nonexistent"))
        finally:
            store.close()

    def test_find_relations_by_target(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.add_entity("carol", "person", "Carol", "person")
            store.add_relation("alice", "knows", "bob")
            store.add_relation("carol", "knows", "bob")
            result = store.find_relations_by_target("bob")
            self.assertEqual(2, len(result))
        finally:
            store.close()

    def test_delete_relation(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            rid = store.add_relation("alice", "knows", "bob")
            self.assertEqual(1, len(store.get_entity_relations("alice")))
            store.delete_relation(rid)
            self.assertEqual(0, len(store.get_entity_relations("alice")))
        finally:
            store.close()

    def test_delete_nonexistent_relation_does_not_raise(self) -> None:
        store = self.make_store()
        try:
            store.delete_relation("nonexistent")
        finally:
            store.close()

    def test_add_duplicate_via_method_is_idempotent(self) -> None:
        store = self.make_store()
        try:
            store.add_entity("alice", "person", "Alice", "person")
            store.add_entity("bob", "person", "Bob", "person")
            store.add_relation("alice", "knows", "bob")
            store.add_relation("alice", "knows", "bob")
            self.assertEqual(1, len(store.get_entity_relations("alice")))
        finally:
            store.close()


class EntityCountWhitelistMvpTests(unittest.TestCase):
    """Entity tables are counted by store.count() and table_counts()."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_count_entity_tables(self) -> None:
        store = self.make_store()
        try:
            for table in ("class_schemas", "class_schema_slots", "entities", "entity_slots", "entity_relations"):
                self.assertEqual(0, store.count(table))
        finally:
            store.close()

    def test_table_counts_includes_entity_tables(self) -> None:
        store = self.make_store()
        try:
            counts = store.table_counts()
            for table in ("class_schemas", "class_schema_slots", "entities", "entity_slots", "entity_relations"):
                self.assertIn(table, counts)
                self.assertEqual(0, counts[table])
        finally:
            store.close()

    def test_count_unknown_table_raises(self) -> None:
        store = self.make_store()
        try:
            with self.assertRaises(ValueError):
                store.count("nonexistent_table")
        finally:
            store.close()


class ContactMigrationMvpTests(unittest.TestCase):
    """migrate_contacts_to_entities ports inventory contacts to person entities."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_migrates_contacts(self) -> None:
        store = self.make_store()
        try:
            store.upsert_inventory("contact", "mom", {"number": "+234-000-MOM"}, source="test", license="private", tags=("contact",))
            store.upsert_inventory("contact", "dad", {"number": "+234-000-DAD"}, source="test", license="private", tags=("contact",))
            count = migrate_contacts_to_entities(store)
            self.assertEqual(2, count)
            mom = store.get_entity("contact:mom")
            self.assertIsNotNone(mom)
            self.assertEqual("person", mom.kind)
            self.assertEqual("mom", mom.label)
            mom_phone = store.get_entity_slot("contact:mom", "phone")
            self.assertEqual("+234-000-MOM", json.loads(mom_phone.value_json))
            mom_name = store.get_entity_slot("contact:mom", "name")
            self.assertEqual("mom", json.loads(mom_name.value_json))
        finally:
            store.close()

    def test_idempotent(self) -> None:
        store = self.make_store()
        try:
            store.upsert_inventory("contact", "mom", {"number": "+234-000-MOM"}, source="test", license="private", tags=("contact",))
            migrate_contacts_to_entities(store)
            count = migrate_contacts_to_entities(store)
            self.assertEqual(0, count)
            self.assertIsNotNone(store.get_entity("contact:mom"))
        finally:
            store.close()

    def test_empty_inventory(self) -> None:
        store = self.make_store()
        try:
            count = migrate_contacts_to_entities(store)
            self.assertEqual(0, count)
        finally:
            store.close()

    def test_contact_without_number(self) -> None:
        store = self.make_store()
        try:
            store.upsert_inventory("contact", "unknown", {}, source="test", license="private", tags=("contact",))
            count = migrate_contacts_to_entities(store)
            self.assertEqual(1, count)
            phone = store.get_entity_slot("contact:unknown", "phone")
            self.assertIsNone(phone)
        finally:
            store.close()


class SelfFactsMigrationMvpTests(unittest.TestCase):
    """migrate_self_facts_to_entities ports user_facts to self entity slots."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_migrates_self_facts(self) -> None:
        store = self.make_store()
        try:
            store.upsert_user_fact("profile.user_name", "Alice", source="profile", confidence=1.0)
            store.upsert_user_fact("facts.favorite_color", "blue", source="profile.facts", confidence=0.9)
            count = migrate_self_facts_to_entities(store)
            self.assertEqual(1, count)
            self_entity = store.get_entity("self")
            self.assertIsNotNone(self_entity)
            self.assertEqual("self", self_entity.kind)
            color_slot = store.get_entity_slot("self", "favorite_color")
            self.assertEqual("blue", json.loads(color_slot.value_json))
            # Profile fields ("profile.*") are NOT migrated — only "facts.*" keys.
            name_slot = store.get_entity_slot("self", "profile.user_name")
            self.assertIsNone(name_slot)
        finally:
            store.close()

    def test_skips_revoked_facts(self) -> None:
        store = self.make_store()
        try:
            store.upsert_user_fact("facts.public", "visible", source="profile.facts", confidence=1.0)
            store.upsert_user_fact("facts.secret", "hidden", source="profile.facts", confidence=1.0, consent=False)
            count = migrate_self_facts_to_entities(store)
            self.assertEqual(1, count)
            public_slot = store.get_entity_slot("self", "public")
            self.assertIsNotNone(public_slot)
            secret = store.get_entity_slot("self", "secret")
            self.assertIsNone(secret)
        finally:
            store.close()

    def test_idempotent(self) -> None:
        store = self.make_store()
        try:
            store.upsert_user_fact("facts.color", "red", source="profile.facts", confidence=1.0)
            migrate_self_facts_to_entities(store)
            count = migrate_self_facts_to_entities(store)
            self.assertEqual(0, count)
        finally:
            store.close()

    def test_empty_facts(self) -> None:
        store = self.make_store()
        try:
            count = migrate_self_facts_to_entities(store)
            self.assertEqual(0, count)
            self_entity = store.get_entity("self")
            self.assertIsNotNone(self_entity)
        finally:
            store.close()


class EntityLexiconIndexMvpTests(unittest.TestCase):
    """rebuild_entity_lexicon_index injects entity labels into _IN_MEMORY_LEXICON."""

    def setUp(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            build_legacy_in_memory_lexicon,
        )
        self._saved = dict(_IN_MEMORY_LEXICON)
        _IN_MEMORY_LEXICON.clear()
        _IN_MEMORY_LEXICON.update(build_legacy_in_memory_lexicon())

    def tearDown(self) -> None:
        from melm.appliance.local_assistant_router import _IN_MEMORY_LEXICON
        _IN_MEMORY_LEXICON.clear()
        _IN_MEMORY_LEXICON.update(self._saved)

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    def test_injects_entity_label(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            rebuild_entity_lexicon_index,
        )
        store = self.make_store()
        try:
            store.add_entity("contact:mom", "person", "mom", "person")
            rebuild_entity_lexicon_index(store)
            classes = _IN_MEMORY_LEXICON.get("mom", frozenset())
            self.assertIn("person", classes)
        finally:
            store.close()

    def test_injects_multi_word_entity_as_compound(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            rebuild_entity_lexicon_index,
        )
        store = self.make_store()
        try:
            store.add_entity("wc2026", "event_instance", "World Cup", "competition")
            rebuild_entity_lexicon_index(store)
            classes = _IN_MEMORY_LEXICON.get("world_cup", frozenset())
            self.assertIn("competition", classes)
        finally:
            store.close()

    def test_injects_canonical_lemma(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            rebuild_entity_lexicon_index,
        )
        store = self.make_store()
        try:
            store.add_entity("contact:mom", "person", "mother", "person", canonical_lemma="mom")
            rebuild_entity_lexicon_index(store)
            self.assertIn("mom", _IN_MEMORY_LEXICON)
            self.assertIn("mother", _IN_MEMORY_LEXICON)
        finally:
            store.close()

    def test_skip_entity_without_semantic_class(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            rebuild_entity_lexicon_index,
            build_legacy_in_memory_lexicon,
        )
        baseline = set(build_legacy_in_memory_lexicon().keys())
        store = self.make_store()
        try:
            store.add_entity("e1", "person", "unknown_person")
            rebuild_entity_lexicon_index(store)
            self.assertNotIn("unknown_person", _IN_MEMORY_LEXICON)
        finally:
            store.close()

    def test_entity_unknown_kind_not_injected(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            rebuild_entity_lexicon_index,
        )
        store = self.make_store()
        try:
            store.add_entity("self", "self", "Self", "person")
            rebuild_entity_lexicon_index(store)
            self.assertNotIn("self", _IN_MEMORY_LEXICON)
        finally:
            store.close()

    def test_compound_token_matches_via_semantic_family_terms(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            _semantic_family_terms,
            rebuild_entity_lexicon_index,
        )
        store = self.make_store()
        try:
            store.add_entity("wc2026", "event_instance", "World Cup", "competition")
            rebuild_entity_lexicon_index(store)
            result = _semantic_family_terms(("world", "cup", "final"), {"competition"})
            self.assertIn("world_cup", result)
        finally:
            store.close()

    def test_single_token_still_matches(self) -> None:
        from melm.appliance.local_assistant_router import (
            _IN_MEMORY_LEXICON,
            _semantic_family_terms,
            rebuild_entity_lexicon_index,
        )
        store = self.make_store()
        try:
            store.add_entity("contact:mom", "person", "mom", "person")
            rebuild_entity_lexicon_index(store)
            result = _semantic_family_terms(("call", "mom", "now"), {"person"})
            self.assertIn("mom", result)
        finally:
            store.close()


class LearningLedgerCrudMvpTests(unittest.TestCase):
    """CRUD for atlas_edges, learning_candidates, corrections, promotions."""

    def make_store(self) -> AssistantOSStore:
        return AssistantOSStore(":memory:")

    # ── atlas_edges ────────────────────────────────────────────────────────

    def test_add_and_get_atlas_edge(self) -> None:
        store = self.make_store()
        try:
            edge_id = store.add_atlas_edge("dog", "used_with", "bone", provenance="uol_parse")
            edge = store.get_atlas_edge(edge_id)
            self.assertIsNotNone(edge)
            self.assertEqual("dog", edge.subject_concept_id)
            self.assertEqual("used_with", edge.relation_id)
            self.assertEqual("bone", edge.object_concept_id)
            self.assertEqual("quarantined", edge.status)
        finally:
            store.close()

    def test_add_atlas_edge_idempotent(self) -> None:
        store = self.make_store()
        try:
            e1 = store.add_atlas_edge("dog", "used_with", "bone")
            e2 = store.add_atlas_edge("dog", "used_with", "bone")
            self.assertNotEqual(e1, e2)
            edges = store.find_atlas_edges(subject_concept_id="dog", relation_id="used_with")
            self.assertEqual(1, len(edges))
        finally:
            store.close()

    def test_find_atlas_edges_by_status(self) -> None:
        store = self.make_store()
        try:
            store.add_atlas_edge("dog", "used_with", "bone", provenance="uol")
            e2 = store.add_atlas_edge("cat", "used_with", "yarn", provenance="uol")
            store.set_atlas_edge_status(e2, "promoted")
            edges = store.find_atlas_edges(status="quarantined")
            self.assertEqual(1, len(edges))
            self.assertEqual("dog", edges[0].subject_concept_id)
        finally:
            store.close()

    def test_touch_atlas_edge_does_not_fail(self) -> None:
        store = self.make_store()
        try:
            edge_id = store.add_atlas_edge("dog", "used_with", "bone")
            store.touch_atlas_edge(edge_id)
            edge = store.get_atlas_edge(edge_id)
            self.assertIsNotNone(edge)
            self.assertEqual("quarantined", edge.status)
        finally:
            store.close()

    def test_set_atlas_edge_status(self) -> None:
        store = self.make_store()
        try:
            edge_id = store.add_atlas_edge("dog", "used_with", "bone")
            store.set_atlas_edge_status(edge_id, "promoted")
            edge = store.get_atlas_edge(edge_id)
            self.assertEqual("promoted", edge.status)
        finally:
            store.close()

    # ── learning_candidates ────────────────────────────────────────────────

    def test_add_and_get_learning_candidate(self) -> None:
        store = self.make_store()
        try:
            cid = store.add_learning_candidate("offline_dict", "xylophone", context="a musical instrument")
            cand = store.get_learning_candidate(cid)
            self.assertIsNotNone(cand)
            self.assertEqual("xylophone", cand.surface_form)
            self.assertEqual("offline_dict", cand.source)
            self.assertEqual("quarantined", cand.status)
        finally:
            store.close()

    def test_add_learning_candidate_idempotent(self) -> None:
        store = self.make_store()
        try:
            store.add_learning_candidate("offline_dict", "xylophone")
            store.add_learning_candidate("offline_dict", "xylophone")
            candidates = store.find_learning_candidates()
            self.assertEqual(1, len(candidates))
        finally:
            store.close()

    def test_find_learning_candidates_by_status(self) -> None:
        store = self.make_store()
        try:
            store.add_learning_candidate("offline_dict", "dog")
            c2 = store.add_learning_candidate("offline_dict", "cat")
            store.set_learning_candidate_status(c2, "promoted")
            quarantined = store.find_learning_candidates(status="quarantined")
            self.assertEqual(1, len(quarantined))
            self.assertEqual("dog", quarantined[0].surface_form)
        finally:
            store.close()

    def test_set_learning_candidate_status_with_error(self) -> None:
        store = self.make_store()
        try:
            cid = store.add_learning_candidate("offline_dict", "xyzzy")
            store.set_learning_candidate_status(cid, "failed", error="not_found")
            cand = store.get_learning_candidate(cid)
            self.assertEqual("failed", cand.status)
            self.assertEqual("not_found", cand.error)
        finally:
            store.close()

    # ── corrections ────────────────────────────────────────────────────────

    def test_add_and_find_corrections(self) -> None:
        store = self.make_store()
        try:
            corr_id = store.add_correction(
                "edge", "edge123", "strength_down",
                user_utterance="that is not right",
            )
            corr = store.find_corrections(target_type="edge", target_id="edge123")
            self.assertEqual(1, len(corr))
            self.assertEqual(corr_id, corr[0].correction_id)
            self.assertEqual("strength_down", corr[0].correction_type)
        finally:
            store.close()

    def test_find_corrections_all(self) -> None:
        store = self.make_store()
        try:
            store.add_correction("edge", "e1", "strength_down")
            store.add_correction("sense", "s1", "delete")
            all_corr = store.find_corrections()
            self.assertEqual(2, len(all_corr))
        finally:
            store.close()

    # ── promotions ─────────────────────────────────────────────────────────

    def test_add_and_find_promotions(self) -> None:
        store = self.make_store()
        try:
            prom_id = store.add_promotion(
                "candidate", "cand123", "quarantined", "promoted",
                reason="verified_by_user",
                provenance="user_teach",
            )
            proms = store.find_promotions(target_type="candidate", target_id="cand123")
            self.assertEqual(1, len(proms))
            self.assertEqual(prom_id, proms[0].promotion_id)
            self.assertEqual("promoted", proms[0].to_status)
            self.assertEqual("quarantined", proms[0].from_status)
        finally:
            store.close()

    def test_find_promotions_orders_by_created_desc(self) -> None:
        store = self.make_store()
        try:
            p1 = store.add_promotion("sense", "s1", "quarantined", "promoted")
            p2 = store.add_promotion("sense", "s1", "promoted", "rolled_back")
            proms = store.find_promotions(target_type="sense", target_id="s1")
            self.assertEqual(2, len(proms))
            self.assertEqual(p2, proms[0].promotion_id)
            self.assertEqual(p1, proms[1].promotion_id)
        finally:
            store.close()

    # ── count whitelist ────────────────────────────────────────────────────

    def test_count_new_tables(self) -> None:
        store = self.make_store()
        try:
            self.assertEqual(0, store.count("atlas_edges"))
            self.assertEqual(0, store.count("learning_candidates"))
            self.assertEqual(0, store.count("corrections"))
            self.assertEqual(0, store.count("promotions"))
        finally:
            store.close()
