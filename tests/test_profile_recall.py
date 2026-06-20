"""ADTC fix 2: first-person attribute questions recall locally, never open_domain/cloud."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import (
    LocalAssistantProfile, _profile_attribute_requested,
)


def _tokens(s):
    return tuple(s.lower().replace("?", "").replace("'", "").split())


class AttributeDetectionTests(unittest.TestCase):
    def test_name(self):
        self.assertEqual(_profile_attribute_requested(_tokens("What is my name?")), "name")

    def test_location(self):
        self.assertEqual(_profile_attribute_requested(_tokens("Where do I live?")), "location")

    def test_age(self):
        self.assertEqual(_profile_attribute_requested(_tokens("How old am I?")), "age")

    def test_child_question_is_not_self(self):
        self.assertIsNone(_profile_attribute_requested(_tokens("How old is my child?")))

    def test_meal_question_is_not_attribute(self):
        self.assertIsNone(_profile_attribute_requested(_tokens("What should I eat today?")))


class RecallKernelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(user_name="", location="", age=0), store=self.store)

    def tearDown(self):
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_name_recall_local_no_cloud(self):
        self.kernel.handle("My name is Ade. I live in Lagos.")
        d = self.kernel.handle("What is my name?")
        self.assertEqual(d.intent, "personal_memory")
        self.assertNotEqual(d.route, "open_domain")
        self.assertFalse(d.cloud_needed)
        self.assertIn("Ade", d.answer)

    def test_location_recall(self):
        self.kernel.handle("I live in Lagos.")
        d = self.kernel.handle("Where do I live?")
        self.assertEqual(d.intent, "personal_memory")
        self.assertIn("Lagos", d.answer)

    def test_unset_attribute_clarifies_no_fabrication(self):
        d = self.kernel.handle("What is my name?")  # name never set (blank profile)
        self.assertEqual(d.intent, "personal_memory")
        self.assertEqual(d.route, "clarify")
        self.assertFalse(d.cloud_needed)


if __name__ == "__main__":
    unittest.main()
