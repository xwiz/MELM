"""Issue 5 — open-domain auto-research must have a SINGLE owner (the kernel).

Before the fix, ``learn_topic()`` ran in both the kernel decision-shaping
(the correct owner) and in the synthesis handlers ``_handle_open_domain`` /
``_handle_unknown``. A single open_domain turn therefore invoked the
ResearchProvider at least twice (double-fetch / double network+store side
effect).

These tests pin the invariant: one open_domain turn ⇒ provider invoked AT MOST
ONCE, and the learned-fact answer still renders end-to-end (no regression).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.assistant_skill_research import ResearchProvider, ResearchResult
from melm.appliance.local_assistant_router import LocalAssistantProfile


class CountingResearchProvider(ResearchProvider):
    """ResearchProvider stub that counts how many times ``research`` is called."""

    def __init__(self, canned: dict[str, str] | None = None) -> None:
        self.canned = canned or {}
        self.call_count = 0
        self.topics_seen: list[str] = []

    def research(self, topic: str) -> ResearchResult:
        self.call_count += 1
        self.topics_seen.append(topic)
        summary = self.canned.get(topic.lower(), "")
        return ResearchResult(
            topic=topic,
            summary=summary,
            source="counting-stub",
            found=bool(summary),
        )


class OpenDomainSingleOwnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _kernel(self, provider: ResearchProvider) -> AssistantOSKernel:
        return AssistantOSKernel(
            profile=LocalAssistantProfile(),
            store=self.store,
            research_provider=provider,
        )

    def test_provider_invoked_at_most_once_per_open_domain_turn(self):
        provider = CountingResearchProvider(
            canned={"photosynthesis": "Photosynthesis converts light into energy."}
        )
        kernel = self._kernel(provider)

        # ONE full end-to-end turn (router -> kernel decision-shaping -> synthesis).
        kernel.handle("Tell me about photosynthesis")

        # Was >= 2 before the fix (kernel + synthesis each called learn_topic).
        self.assertLessEqual(
            provider.call_count,
            1,
            f"research() invoked {provider.call_count}x for one open_domain turn "
            f"(topics: {provider.topics_seen}); expected at most once",
        )

    def test_learned_fact_answer_still_renders_end_to_end(self):
        provider = CountingResearchProvider(
            canned={"photosynthesis": "Photosynthesis converts light into energy."}
        )
        kernel = self._kernel(provider)

        decision = kernel.handle("Tell me about photosynthesis")

        # The kernel auto-researched, stored the fact, and re-answered locally.
        self.assertEqual(decision.route, "local_answer")
        self.assertIn("light into energy", decision.answer)
        self.assertNotEqual(decision.intent, "")
        # Provider was consulted exactly once for the fetch.
        self.assertEqual(provider.call_count, 1)

    def test_second_turn_hits_stored_fact_no_refetch(self):
        provider = CountingResearchProvider(
            canned={"photosynthesis": "Photosynthesis converts light into energy."}
        )
        kernel = self._kernel(provider)

        kernel.handle("Tell me about photosynthesis")
        count_after_first = provider.call_count
        # Second identical turn should reuse the stored learned_fact, not re-fetch.
        decision2 = kernel.handle("Tell me about photosynthesis")

        self.assertEqual(provider.call_count, count_after_first)
        self.assertIn("light into energy", decision2.answer)


if __name__ == "__main__":
    unittest.main()
