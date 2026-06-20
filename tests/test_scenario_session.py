"""Scenario working memory is intentionally session/process-scoped (ADTC Issue 8a).

These tests document the deliberate design: scenario state persists within a
session (across the per-turn router rebuild, since the store is long-lived) but
sessions are isolated from one another and a fresh store starts clean (the
restart-resets-by-design semantics).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas


class ScenarioSessionScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_same_session_persistence(self) -> None:
        scenario = {"places": ["enugu", "lagos"], "stays": {"oniru": 2.0}}
        self.store.set_current_scenario("sess-1", scenario)
        self.assertEqual(self.store.get_current_scenario("sess-1"), scenario)

    def test_sessions_are_isolated(self) -> None:
        self.store.set_current_scenario("sess-1", {"places": ["enugu"]})
        self.assertIsNone(self.store.get_current_scenario("a-different-session-id"))

    def test_fresh_store_starts_clean(self) -> None:
        # Restart semantics: a new store instance has no carried-over scenarios,
        # even pointing at a DB file that previously had scenario writes.
        self.store.set_current_scenario("sess-1", {"places": ["enugu"]})
        self.store.connection.close()

        restarted = AssistantOSStore(self.tmp.name)
        seed_class_schemas(restarted)
        try:
            self.assertIsNone(restarted.get_current_scenario("sess-1"))
        finally:
            restarted.connection.close()


if __name__ == "__main__":
    unittest.main()
