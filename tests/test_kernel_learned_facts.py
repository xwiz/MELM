"""Kernel integration tests for learned-fact open-domain fallback."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.assistant_skill_research import record_learned_fact
from melm.appliance.local_assistant_router import LocalAssistantProfile


class KernelLearnedFactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(),
            store=self.store,
        )

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_open_domain_uses_learned_fact(self):
        record_learned_fact(self.store, "Mars", "Mars is the fourth planet from the Sun.")
        decision = self.kernel.decide("Tell me about Mars")
        self.assertEqual(decision.route, "local_answer")
        self.assertIn("Mars", decision.answer)
        self.assertIn("fourth planet", decision.answer)
        self.assertEqual(decision.reason, "learned_fact_answer")

    def test_open_domain_acknowledges_topic_without_fact(self):
        decision = self.kernel.decide("Tell me about Pluto")
        self.assertIn("pluto", decision.answer.lower())
        # Route stays cloud_handoff because no learned fact exists
        self.assertEqual(decision.route, "cloud_handoff")

    def test_learned_fact_wired_to_experience(self):
        record_learned_fact(self.store, "Mars", "Mars is the fourth planet from the Sun.")
        self.kernel.handle("Tell me about Mars")
        # Check that the last experience has the learned_fact_ids slot populated
        rows = self.store.connection.execute(
            """
            SELECT es.value_json
            FROM entity_slots es
            JOIN entities e ON e.entity_id = es.entity_id
            WHERE e.kind = 'personal_experience' AND es.slot_name = 'learned_fact_ids'
            ORDER BY e.created_at DESC
            LIMIT 1
            """
        ).fetchall()
        self.assertTrue(rows)
        import json
        fact_ids = json.loads(rows[0]["value_json"])
        self.assertTrue(any("learned_fact" in fid for fid in fact_ids))

    def test_auto_research_triggers_when_no_learned_fact(self):
        from melm.appliance.assistant_skill_research import StubResearchProvider
        provider = StubResearchProvider(canned={"pluto": "Pluto is a dwarf planet."})
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(),
            store=self.store,
            research_provider=provider,
        )
        decision = kernel.decide("Tell me about Pluto")
        self.assertEqual(decision.route, "local_answer")
        self.assertIn("Pluto", decision.answer)
        self.assertIn("dwarf planet", decision.answer)
        self.assertEqual(decision.reason, "auto_research_answer")

    def test_auto_research_idempotent(self):
        from melm.appliance.assistant_skill_research import StubResearchProvider
        provider = StubResearchProvider(canned={"pluto": "Pluto is a dwarf planet."})
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(),
            store=self.store,
            research_provider=provider,
        )
        # First call should trigger research and store
        decision1 = kernel.decide("Tell me about Pluto")
        self.assertEqual(decision1.reason, "auto_research_answer")
        # Second call should hit the stored fact, not re-research
        call_count = provider.canned.copy()
        provider.canned = {"pluto": "CHANGED"}
        decision2 = kernel.decide("Tell me about Pluto")
        self.assertEqual(decision2.reason, "learned_fact_answer")
        # Should still have the original text
        self.assertIn("dwarf planet", decision2.answer)

    def test_auto_research_falls_back_when_provider_finds_nothing(self):
        from melm.appliance.assistant_skill_research import StubResearchProvider
        provider = StubResearchProvider(canned={})
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(),
            store=self.store,
            research_provider=provider,
        )
        decision = kernel.decide("Tell me about XYZUnknown")
        self.assertEqual(decision.route, "cloud_handoff")
        self.assertIn("xyzunknown", decision.answer.lower())


if __name__ == "__main__":
    unittest.main()
