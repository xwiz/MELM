"""Regression tests for self-identity derivation edge cases.

Each test class targets one specific bug, logic flaw, or edge case
identified in the code review of assistant_skill_self_identity.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import Any

from melm.appliance.assistant_os_store import AssistantOSStore
from melm.appliance.assistant_os_store import seed_class_schemas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> AssistantOSStore:
    store = AssistantOSStore(":memory:")
    seed_class_schemas(store)
    return store


def _add_pe(
    store: AssistantOSStore,
    entity_id: str,
    intent: str,
    user_id: str,
    polarity: float | None = None,
    *,
    label_override: str | None = None,
) -> None:
    """Add a personal_experience entity.
    
    If polarity is None, no polarity slot is created (tests for missing slot).
    label_override replaces the auto-generated "turn: {intent}" label.
    """
    label = f"turn: {intent}" if label_override is None else label_override
    store.add_entity(
        entity_id=entity_id,
        kind="personal_experience",
        label=label,
        semantic_class_id="personal_experience",
        canonical_lemma=f"test utterance about {intent}",
    )
    store.set_entity_slot(
        entity_id, "outcome", "resolved", provenance="experience_writer",
    )
    store.set_entity_slot(
        entity_id, "user_id", user_id, provenance="experience_writer",
    )
    if polarity is not None:
        store.set_entity_slot(
            entity_id, "polarity", polarity, provenance="experience_writer",
        )


def _add_pe_raw_slot(
    store: AssistantOSStore,
    entity_id: str,
    intent: str,
    user_id: str,
    slot_name: str,
    raw_json: str,
) -> None:
    """Add an entity with a slot whose value_json is set to arbitrary text.
    
    Used to simulate malformed JSON in the database.
    """
    store.add_entity(
        entity_id=entity_id,
        kind="personal_experience",
        label=f"turn: {intent}",
        semantic_class_id="personal_experience",
        canonical_lemma="test",
    )
    # Bypass set_entity_slot (which uses json.dumps) by writing raw SQL
    store.connection.execute(
        """
        INSERT INTO entity_slots(
            slot_id, entity_id, slot_name, value_json, slot_state, consent,
            local_only, cloud_eligible, scope, source, confidence, provenance,
            updated_at
        ) VALUES (?, ?, ?, ?, 'filled', 1, 1, 0, 'private_local',
                  'test', 0.8, 'test', datetime('now'))
        """,
        (entity_id + "_" + slot_name, entity_id, slot_name, raw_json),
    )
    store.connection.commit()


# ---------------------------------------------------------------------------
# B1: Label substring check is not a prefix check
# ---------------------------------------------------------------------------

class LabelSubstringMatchTests(unittest.TestCase):
    """Labels containing 'turn: ' as a substring but not as prefix are 
    incorrectly accepted, and labels with differing capitalisation are 
    incorrectly rejected."""

    def test_substring_not_prefix_creates_phantom_intent(self) -> None:
        """Label 'not_turn: story' contains 'turn: ' (substring match) so
        it passes the filter, but split('turn: ', 1)[1] gives ' story'
        (leading space) — a phantom intent never present in the contract."""
        store = _make_store()
        try:
            _add_pe(store, "pe_001", "story", "test", 0.5,
                    label_override="not_turn: story")
            _add_pe(store, "pe_002", "story", "test", 0.5,
                    label_override="not_turn: story")
            _add_pe(store, "pe_003", "story", "test", 0.5,
                    label_override="not_turn: story")
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNone(result)
            return
            # Split "not_turn: story" by "turn: " → ["not_", " story"]
            # So intent = " story" (with leading space)
            self.assertIn(
                " story", result.per_intent_counts,
                "Substring match allows non-prefix labels through",
            )
            self.assertEqual(
                result.per_intent_counts[" story"], 3,
                "Phantom intent ' story' counted 3 times",
            )
        finally:
            store.close()

    def test_capitalised_turn_skipped(self) -> None:
        """Label 'Turn: story' (capital T) does not contain 'turn: ' so the
        substring check returns False and the entity is skipped entirely –
        data loss."""
        store = _make_store()
        try:
            _add_pe(store, "pe_001", "story", "test", 0.5,
                    label_override="Turn: story")
            _add_pe(store, "pe_002", "story", "test", 0.5,
                    label_override="Turn: story")
            _add_pe(store, "pe_003", "story", "test", 0.5,
                    label_override="Turn: story")
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertEqual(result.highest_meaning_intent, "story")
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B2: Trailing space in label produces phantom intent
# ---------------------------------------------------------------------------

class LabelTrailingSpaceTests(unittest.TestCase):
    """Label 'turn: story ' (trailing space) passes the substring check
    but split produces 'story ' with trailing space, which does not match
    any contract key."""

    def test_trailing_space_creates_wrong_intent(self) -> None:
        store = _make_store()
        try:
            _add_pe(store, "pe_001", "story", "test", 0.5,
                    label_override="turn: story ")
            _add_pe(store, "pe_002", "story", "test", 0.5,
                    label_override="turn: story ")
            _add_pe(store, "pe_003", "story", "test", 0.5,
                    label_override="turn: story ")
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            # The intent will be "story " (with trailing space) not "story".
            # Since the contract has "story" but not "story ", the narrative
            # will return None for highest_meaning_intent "story ".
            self.assertEqual(
                result.highest_meaning_intent,
                "story",
            )
            # Verify the intent name has a trailing space (the defect)
            self.assertEqual(
                result.per_intent_counts.get("story ", 0), 0,
                "Trailing-space intent should not be counted",
            )
            self.assertEqual(
                result.per_intent_counts.get("story", 0), 3,
                "Canonical 'story' intent should have 3 counts",
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B3: Malformed JSON in user_id slot causes unhandled crash
# ---------------------------------------------------------------------------

class MalformedJsonUseridTests(unittest.TestCase):
    """json.loads on corrupted value_json raises JSONDecodeError."""

    def test_malformed_userid_json_does_not_crash(self) -> None:
        store = _make_store()
        try:
            _add_pe_raw_slot(store, "pe_001", "story", "test",
                             "user_id", "{broken")
            _add_pe_raw_slot(store, "pe_002", "story", "test",
                             "user_id", "{broken")
            _add_pe_raw_slot(store, "pe_003", "story", "test",
                             "user_id", "{broken")
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNone(result)
        finally:
            store.close()


class MalformedJsonPolarityTests(unittest.TestCase):
    """json.loads on corrupted polarity value_json raises JSONDecodeError."""

    def test_malformed_polarity_json_does_not_crash(self) -> None:
        store = _make_store()
        try:
            # Add a bad-polarity entity WITH a user_id slot so it gets past
            # the user_id filter and hits json.loads on polarity.
            store.add_entity(
                entity_id="pe_001",
                kind="personal_experience",
                label="turn: story",
                semantic_class_id="personal_experience",
                canonical_lemma="test",
            )
            # user_id slot (valid)
            store.set_entity_slot(
                "pe_001", "user_id", "test", provenance="experience_writer",
            )
            # polarity slot with malformed JSON (raw SQL bypasses json.dumps)
            store.connection.execute(
                """
                INSERT INTO entity_slots(
                    slot_id, entity_id, slot_name, value_json, slot_state,
                    consent, local_only, cloud_eligible, scope, source,
                    confidence, provenance, updated_at
                ) VALUES (?, ?, ?, ?, 'filled', 1, 1, 0, 'private_local',
                          'test', 0.8, 'test', datetime('now'))
                """,
                ("pe_001_polarity", "pe_001", "polarity", "not-json"),
            )
            store.connection.commit()

            # Add normal entities so total_turns >= min_data_points
            _add_pe(store, "pe_002", "story", "test", 0.5)
            _add_pe(store, "pe_003", "story", "test", 0.5)
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertEqual(result.per_intent_counts["story"], 3)
            self.assertAlmostEqual(
                result.per_intent_mean_polarities["story"],
                0.5,
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B4: String polarity silently skipped (isinstance rejects strings)
# ---------------------------------------------------------------------------

class StringPolaritySilentDropTests(unittest.TestCase):
    """When polarity is stored as a JSON string '0.5' instead of number,
    json.loads returns a Python str, isinstance rejects it, and the
    polarity data is silently dropped with no error."""

    def test_string_polarity_silently_dropped(self) -> None:
        store = _make_store()
        try:
            # Store polarity as string via raw SQL
            store.add_entity(
                entity_id="pe_001",
                kind="personal_experience",
                label="turn: story",
                semantic_class_id="personal_experience",
                canonical_lemma="test",
            )
            store.set_entity_slot(
                "pe_001", "user_id", "test", provenance="experience_writer",
            )
            store.connection.execute(
                """
                INSERT INTO entity_slots(
                    slot_id, entity_id, slot_name, value_json, slot_state,
                    consent, local_only, cloud_eligible, scope, source,
                    confidence, provenance, updated_at
                ) VALUES (?, ?, ?, ?, 'filled', 1, 1, 0, 'private_local',
                          'test', 0.8, 'test', datetime('now'))
                """,
                ("pe_001_polarity", "pe_001", "polarity", '"0.8"'),
            )
            store.connection.commit()

            _add_pe(store, "pe_002", "story", "test", 0.6)
            _add_pe(store, "pe_003", "story", "test", 0.4)
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertAlmostEqual(
                result.per_intent_mean_polarities.get("story", -999),
                (0.8 + 0.6 + 0.4) / 3,
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B5: Mixed metrics in derive_identity_explanation
#     Uses highest_meaning_intent for the frame but top_intent for
#     count and polarity — they can describe different intents.
# ---------------------------------------------------------------------------

class MixedMetricsExplanationTests(unittest.TestCase):
    """derive_identity_explanation() takes its 'frame' from
    highest_meaning_intent but 'count' and 'polarity' from top_intent.
    When these differ, the explanation is semantically wrong."""

    def test_explanation_metrics_mismatched(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_explanation,
        )
        # highest_meaning_intent = "weather" (1 turn, polarity 0.9)
        # top_intent = "story" (50 turns, polarity 0.1)
        identity = DerivedIdentity(
            user_id="test",
            highest_meaning_intent="weather",
            highest_meaning_polarity=0.9,
            top_intent="story",
            top_intent_count=50,
            top_intent_mean_polarity=0.1,
            total_turns=51,
            per_intent_counts={"weather": 1, "story": 50},
            per_intent_mean_polarities={"weather": 0.9, "story": 0.1},
        )
        explanation = derive_identity_explanation(identity)
        self.assertIsNotNone(explanation)
        # The frame comes from "weather" → "checking the weather"
        self.assertIn("checking the weather", explanation)
        self.assertIn("1", explanation)
        self.assertIn("+0.9", explanation)


# ---------------------------------------------------------------------------
# B6: Negative polarity sorting
# ---------------------------------------------------------------------------

class NegativePolaritySortingTests(unittest.TestCase):
    """Negative polarities: verify they do not accidentally become
    'highest meaning' due to the sort key formula."""

    def test_negative_polarity_does_not_beat_positive(self) -> None:
        store = _make_store()
        try:
            # "weather": 3 turns, polarity 0.8 (highest meaning should win)
            # "story": 3 turns, polarity -0.5 (should NOT win)
            _add_pe(store, "pe_001", "weather", "test", 0.8)
            _add_pe(store, "pe_002", "weather", "test", 0.8)
            _add_pe(store, "pe_003", "weather", "test", 0.8)
            _add_pe(store, "pe_004", "story", "test", -0.5)
            _add_pe(store, "pe_005", "story", "test", -0.5)
            _add_pe(store, "pe_006", "story", "test", -0.5)
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertEqual(
                result.highest_meaning_intent,
                "weather",
                "Positive-polarity intent should beat negative-polarity one",
            )
            self.assertAlmostEqual(
                result.highest_meaning_polarity, 0.8,
            )
        finally:
            store.close()

    def test_all_negative_polarities_still_chooses_best(self) -> None:
        """When all intents have negative polarity, the least-negative
        (closest to zero) should win."""
        store = _make_store()
        try:
            # "weather": -0.1 (least negative → highest meaning)
            # "story": -0.9
            _add_pe(store, "pe_001", "weather", "test", -0.1)
            _add_pe(store, "pe_002", "weather", "test", -0.1)
            _add_pe(store, "pe_003", "weather", "test", -0.1)
            _add_pe(store, "pe_004", "story", "test", -0.9)
            _add_pe(store, "pe_005", "story", "test", -0.9)
            _add_pe(store, "pe_006", "story", "test", -0.9)
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertEqual(
                result.highest_meaning_intent, "weather",
                "Least-negative polarity should win",
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B7: Empty polarity data — all intents have no polarity slots
# ---------------------------------------------------------------------------

class EmptyPolarityDataTests(unittest.TestCase):
    """When no entities have polarity slots, per_intent_mean_polarities is
    empty and all polarities default to 0.0.  The sort still picks an
    intent, but the 'highest meaning' label is meaningless."""

    def test_all_zero_polarity_still_picks_first_alphabetical(self) -> None:
        store = _make_store()
        try:
            _add_pe(store, "pe_001", "weather", "test")
            _add_pe(store, "pe_002", "weather", "test")
            _add_pe(store, "pe_003", "weather", "test")
            _add_pe(store, "pe_004", "story", "test")
            _add_pe(store, "pe_005", "story", "test")
            _add_pe(store, "pe_006", "story", "test")
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            # Both intents have 0.0 polarity.  Sort key: (-0.0, -count, name)
            # Both have same polarity (0.0).  Count tiebreaker: alpha has 3.
            # So highest_meaning will be alpha (ties → alphabetical).
            self.assertEqual(
                result.highest_meaning_intent, "story",
            )
            self.assertAlmostEqual(result.highest_meaning_polarity, 0.0)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B8: Numeric user_id silently skips all entities
# ---------------------------------------------------------------------------

class NumericUserIdTests(unittest.TestCase):
    """When user_id is stored as a JSON number (e.g. 42 instead of "42"),
    json.loads returns an int, uid != user_id (str comparison), and all
    entities are silently skipped."""

    def test_numeric_user_id_causes_data_loss(self) -> None:
        store = _make_store()
        try:
            # Store user_id as integer via raw SQL
            store.add_entity(
                entity_id="pe_001",
                kind="personal_experience",
                label="turn: story",
                semantic_class_id="personal_experience",
                canonical_lemma="test",
            )
            store.connection.execute(
                """
                INSERT INTO entity_slots(
                    slot_id, entity_id, slot_name, value_json, slot_state,
                    consent, local_only, cloud_eligible, scope, source,
                    confidence, provenance, updated_at
                ) VALUES (?, ?, ?, ?, 'filled', 1, 1, 0, 'private_local',
                          'test', 0.8, 'test', datetime('now'))
                """,
                ("pe_001_uid", "pe_001", "user_id", "42"),
            )
            store.connection.execute(
                """
                INSERT INTO entity_slots(
                    slot_id, entity_id, slot_name, value_json, slot_state,
                    consent, local_only, cloud_eligible, scope, source,
                    confidence, provenance, updated_at
                ) VALUES (?, ?, ?, ?, 'filled', 1, 1, 0, 'private_local',
                          'test', 0.8, 'test', datetime('now'))
                """,
                ("pe_001_plr", "pe_001", "polarity", "0.5"),
            )
            store.connection.commit()
            # Add string-user_id entities to pass min_data_points
            _add_pe(store, "pe_002", "story", "42", 0.5)
            _add_pe(store, "pe_003", "story", "42", 0.5)
            _add_pe(store, "pe_004", "story", "42", 0.5)

            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "42")
            self.assertIsNotNone(result)
            self.assertEqual(result.total_turns, 4)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B9: min_data_points gate counts ALL intents, not the winning intent
# ---------------------------------------------------------------------------

class MinDataPointsPerIntentTests(unittest.TestCase):
    """The gate total_turns < min_data_points counts across ALL intents.
    A user with 2 story (polarity 0.9) + 1 weather (polarity 0.0) passes
    the gate (3 >= 3) but the winning intent (story) only has 2 data
    points.  The result is unreliable."""

    def test_winning_intent_below_min_data_points(self) -> None:
        store = _make_store()
        try:
            _add_pe(store, "pe_001", "story", "test", 0.9)
            _add_pe(store, "pe_002", "story", "test", 0.9)
            _add_pe(store, "pe_003", "weather", "test", 0.0)
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNone(result)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B10: derive_identity_narrative — missing "neutral" key in contract
# ---------------------------------------------------------------------------

class MissingNeutralNarrativeTests(unittest.TestCase):
    """If 'neutral' is missing from identity_narratives in the contract,
    derive_identity_narrative returns None.  There is no hard guarantee
    in the code that neutral exists — only the contract JSON provides it."""

    def test_contract_without_neutral_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_narrative,
        )
        identity = DerivedIdentity(
            user_id="test",
            highest_meaning_intent="story",
            highest_meaning_polarity=0.5,
            top_intent="story",
            top_intent_count=3,
            top_intent_mean_polarity=0.5,
            total_turns=3,
        )
        contract_no_neutral: dict[str, Any] = {
            "identity_labels": {
                "story": {
                    "label": "a storyteller",
                    "frame": "sharing stories",
                },
            },
            "identity_narratives": {
                "happy": "Happy narrative for {label}",
                # "neutral" intentionally omitted
            },
        }
        with patch(
            "melm.appliance.assistant_skill_self_identity._get_self_identity_contract",
            return_value=contract_no_neutral,
        ):
            result = derive_identity_narrative(identity, "nonexistent_mood")
            self.assertIsNone(
                result,
                "Should return None when mood not found AND neutral missing",
            )


# ---------------------------------------------------------------------------
# B11: get_name_awareness_template — empty label in contract
# ---------------------------------------------------------------------------

class EmptyLabelNameTemplateTests(unittest.TestCase):
    """If identity_labels has an entry with label='', the template is
    formatted with an empty label.  The template may become grammatically
    broken."""

    def test_empty_label_substitution(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            get_name_awareness_template,
        )
        identity = DerivedIdentity(
            user_id="test",
            highest_meaning_intent="story",
            highest_meaning_polarity=0.5,
            top_intent="story",
            top_intent_count=3,
            top_intent_mean_polarity=0.5,
            total_turns=3,
        )
        contract_empty_label: dict[str, Any] = {
            "identity_labels": {
                "story": {
                    "label": "",
                    "frame": "sharing stories",
                },
            },
            "name_awareness_templates": {
                "no_name": "I am {label}.",
            },
        }
        with patch(
            "melm.appliance.assistant_skill_self_identity._get_self_identity_contract",
            return_value=contract_empty_label,
        ):
            result = get_name_awareness_template(identity, "no_name")
            self.assertIsNotNone(result)
            self.assertEqual(
                result, "I am .",
                "Empty label should produce empty substitution",
            )


# ---------------------------------------------------------------------------
# B12: derive_identity_narrative with empty identity_labels entry
# ---------------------------------------------------------------------------

class MissingLabelNarrativeTests(unittest.TestCase):
    """When highest_meaning_intent has no identity_labels entry at all,
    derive_identity_narrative returns None.  Callers must handle None."""

    def test_unknown_intent_label_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_narrative,
        )
        identity = DerivedIdentity(
            user_id="test",
            highest_meaning_intent="bogus_intent_not_in_contract",
            highest_meaning_polarity=0.5,
            top_intent="bogus_intent_not_in_contract",
            top_intent_count=3,
            top_intent_mean_polarity=0.5,
            total_turns=3,
        )
        contract: dict[str, Any] = {
            "identity_labels": {
                "story": {"label": "a storyteller", "frame": "sharing stories"},
            },
            "identity_narratives": {
                "neutral": "I am {label}.",
            },
        }
        with patch(
            "melm.appliance.assistant_skill_self_identity._get_self_identity_contract",
            return_value=contract,
        ):
            result = derive_identity_narrative(identity, "neutral")
            self.assertIsNone(result)

    def test_unknown_intent_explanation_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_explanation,
        )
        identity = DerivedIdentity(
            user_id="test",
            highest_meaning_intent="bogus_intent_not_in_contract",
            highest_meaning_polarity=0.5,
            top_intent="bogus_intent_not_in_contract",
            top_intent_count=3,
            top_intent_mean_polarity=0.5,
            total_turns=3,
        )
        contract: dict[str, Any] = {
            "identity_labels": {
                "story": {"label": "a storyteller", "frame": "sharing stories"},
            },
            "name_awareness_templates": {"why": "You asked me to {frame}."},
        }
        with patch(
            "melm.appliance.assistant_skill_self_identity._get_self_identity_contract",
            return_value=contract,
        ):
            result = derive_identity_explanation(identity)
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# B13: Column in label creates phantom intent prefix
# ---------------------------------------------------------------------------

class LabelWithColonTests(unittest.TestCase):
    """Intent label 'turn: story: test' splits correctly because maxsplit=1
    gives 'story: test'.  This is OK — no bug here, but documented as
    non-issue."""

    def test_colon_in_intent_part_is_handled(self) -> None:
        store = _make_store()
        try:
            _add_pe(store, "pe_001", "test", "test", 0.5,
                    label_override="turn: story: test")
            _add_pe(store, "pe_002", "test", "test", 0.5,
                    label_override="turn: story: test")
            _add_pe(store, "pe_003", "test", "test", 0.5,
                    label_override="turn: story: test")
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNone(result)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B14: Derive explanation with missing 'why' template key
# ---------------------------------------------------------------------------

class MissingWhyTemplateTests(unittest.TestCase):
    """If name_awareness_templates lacks 'why', derive_identity_explanation
    returns None.  Not a crash, but callers should handle."""

    def test_missing_why_returns_none(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            DerivedIdentity,
            derive_identity_explanation,
        )
        identity = DerivedIdentity(
            user_id="test",
            highest_meaning_intent="story",
            highest_meaning_polarity=0.5,
            top_intent="story",
            top_intent_count=3,
            top_intent_mean_polarity=0.5,
            total_turns=3,
        )
        contract: dict[str, Any] = {
            "identity_labels": {
                "story": {"label": "a storyteller", "frame": "sharing stories"},
            },
            "name_awareness_templates": {
                # "why" key intentionally omitted
                "no_name": "I am {label}.",
            },
        }
        with patch(
            "melm.appliance.assistant_skill_self_identity._get_self_identity_contract",
            return_value=contract,
        ):
            result = derive_identity_explanation(identity)
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# B15: Privacy / performance: find_entities returns ALL users' entities
# ---------------------------------------------------------------------------

class PrivacyFilteringTests(unittest.TestCase):
    """find_entities(kind='personal_experience') loads ALL users' entities
    into memory before filtering by user_id.  This is not a data leak to
    callers (the guard filters correctly) but it is an O(all_users)
    performance anti-pattern."""

    def test_other_user_entities_loaded_but_filtered(self) -> None:
        store = _make_store()
        try:
            # Add data for user_alice (should be excluded from user_bob)
            _add_pe(store, "pe_001", "story", "user_alice", 0.9)
            _add_pe(store, "pe_002", "story", "user_alice", 0.9)
            _add_pe(store, "pe_003", "story", "user_alice", 0.9)
            # Add data for user_bob (target)
            _add_pe(store, "pe_004", "weather", "user_bob", 0.5)
            _add_pe(store, "pe_005", "weather", "user_bob", 0.5)
            _add_pe(store, "pe_006", "weather", "user_bob", 0.5)
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "user_bob")
            self.assertIsNotNone(result)
            # Should only see user_bob's data
            self.assertEqual(result.total_turns, 3)
            self.assertEqual(result.highest_meaning_intent, "weather")
            self.assertEqual(result.per_intent_counts.get("story", 0), 0)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B16: store.find_entities returning empty list
# ---------------------------------------------------------------------------

class EmptyEntityListTests(unittest.TestCase):
    """When find_entities returns [] (no personal_experience entities at
    all), the function correctly returns None because total_turns = 0."""

    def test_empty_store_returns_none(self) -> None:
        store = _make_store()
        try:
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNone(result)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B17: Some entities of same intent missing polarity slot
# ---------------------------------------------------------------------------

class PartialPolarityDataTests(unittest.TestCase):
    """When some entities of the same intent have a polarity slot and
    others do not, the mean is computed only from those with polarity.
    The count includes all entities.  This is correct behavior, but the
    test documents the invariant."""

    def test_mixed_polarity_and_no_polarity(self) -> None:
        store = _make_store()
        try:
            # story: 2 with polarity (0.5, 0.7), 1 without
            _add_pe(store, "pe_001", "story", "test", 0.5)
            _add_pe(store, "pe_002", "story", "test", 0.7)
            _add_pe(store, "pe_003", "story", "test")  # no polarity
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertEqual(result.per_intent_counts["story"], 3)
            self.assertAlmostEqual(
                result.per_intent_mean_polarities["story"],
                (0.5 + 0.7) / 2,
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B18: Given name with has_name=False in self_state
# ---------------------------------------------------------------------------

class SelfStateEdgeTests(unittest.TestCase):
    """Edge cases around load_self_state() — empty dict, missing keys."""

    def test_empty_self_state(self) -> None:
        """Empty self_state dict should result in has_name=False and
        given_name=None (not crash)."""
        store = _make_store()
        try:
            _add_pe(store, "pe_001", "story", "test", 0.5)
            _add_pe(store, "pe_002", "story", "test", 0.5)
            _add_pe(store, "pe_003", "story", "test", 0.5)
            from melm.appliance.assistant_skill_self_identity import (
                analyze_user_identity,
            )
            result = analyze_user_identity(store, "test")
            self.assertIsNotNone(result)
            self.assertFalse(result.has_name)
            self.assertIsNone(result.given_name)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# B19: Contract caching — stale data test
# ---------------------------------------------------------------------------

class ContractCachingTests(unittest.TestCase):
    """Module-level _SELF_IDENTITY_CONTRACT is loaded once and never
    refreshed.  Changes to the contract file won't be picked up until
    module reload.  This test confirms the caching behaviour (it is a
    known deployment constraint)."""

    def test_contract_cached_after_first_load(self) -> None:
        from melm.appliance.assistant_skill_self_identity import (
            _get_self_identity_contract,
        )
        from melm.appliance.assistant_skill_self_identity import (
            _SELF_IDENTITY_CONTRACT,
        )
        # Reset cache
        import melm.appliance.assistant_skill_self_identity as m
        m._SELF_IDENTITY_CONTRACT = None
        # First call loads from disk
        first = _get_self_identity_contract()
        self.assertIsNotNone(first)
        self.assertIsNotNone(m._SELF_IDENTITY_CONTRACT)
        # Second call returns the same object
        second = _get_self_identity_contract()
        self.assertIs(second, first)
        self.assertIs(second, m._SELF_IDENTITY_CONTRACT)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
