"""ADTC fix 1: compound profile extraction stores every fact in one utterance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile


def _kernel():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = AssistantOSStore(tmp.name)
    seed_class_schemas(store)
    # Start from blank name/location so we observe what the utterance stores.
    profile = LocalAssistantProfile(user_name="", location="", age=0)
    return AssistantOSKernel(profile=profile, store=store), store, tmp.name


class CompoundProfileTests(unittest.TestCase):
    def setUp(self):
        self.kernel, self.store, self.path = _kernel()

    def tearDown(self):
        self.store.connection.close()
        Path(self.path).unlink(missing_ok=True)

    def test_name_and_location_in_one_sentence(self):
        self.kernel.handle("My name is Ade. I live in Lagos.")
        self.assertEqual(self.kernel.profile.user_name, "Ade")
        self.assertEqual(self.kernel.profile.location, "Lagos")

    def test_order_independent(self):
        self.kernel.handle("I live in Lagos. My name is Ade.")
        self.assertEqual(self.kernel.profile.user_name, "Ade")
        self.assertEqual(self.kernel.profile.location, "Lagos")

    def test_three_facts(self):
        self.kernel.handle("I'm 30. My name is Ada and I live in Abuja.")
        self.assertEqual(self.kernel.profile.user_name, "Ada")
        self.assertEqual(self.kernel.profile.location, "Abuja")
        self.assertEqual(self.kernel.profile.age, 30)

    def test_single_fact_still_works(self):
        d = self.kernel.handle("My name is Bola.")
        self.assertEqual(self.kernel.profile.user_name, "Bola")
        self.assertEqual(d.reason, "profile_update")

    def test_evidence_keys_cover_all_stored(self):
        d = self.kernel.handle("My name is Ade. I live in Lagos.")
        self.assertIn("profile.user_name", d.evidence_keys)
        self.assertIn("profile.location", d.evidence_keys)


if __name__ == "__main__":
    unittest.main()
