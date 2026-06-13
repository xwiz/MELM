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

    def test_seeds_seven_classes(self) -> None:
        store = self.make_store()
        try:
            seed_class_schemas(store)
            rows = store.connection.execute(
                "SELECT semantic_class_id, label FROM class_schemas ORDER BY semantic_class_id"
            ).fetchall()
            ids = [str(r["semantic_class_id"]) for r in rows]
            self.assertEqual(["competition", "entity", "event", "object", "person", "personal_experience", "place"], ids)
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
            self.assertEqual(7, count["c"])
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
            self.assertEqual(2, count)
            self_entity = store.get_entity("self")
            self.assertIsNotNone(self_entity)
            self.assertEqual("self", self_entity.kind)
            name_slot = store.get_entity_slot("self", "profile.user_name")
            self.assertEqual("Alice", json.loads(name_slot.value_json))
            color_slot = store.get_entity_slot("self", "facts.favorite_color")
            self.assertEqual("blue", json.loads(color_slot.value_json))
        finally:
            store.close()

    def test_skips_revoked_facts(self) -> None:
        store = self.make_store()
        try:
            store.upsert_user_fact("profile.user_name", "Alice", source="profile", confidence=1.0)
            store.upsert_user_fact("facts.secret", "hidden", source="profile.facts", confidence=1.0, consent=False)
            count = migrate_self_facts_to_entities(store)
            self.assertEqual(1, count)
            secret = store.get_entity_slot("self", "facts.secret")
            self.assertIsNone(secret)
        finally:
            store.close()

    def test_idempotent(self) -> None:
        store = self.make_store()
        try:
            store.upsert_user_fact("profile.user_name", "Alice", source="profile", confidence=1.0)
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
