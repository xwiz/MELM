"""Slice 8: geospatial walk/drive decision with entity-purpose override."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile
from melm.appliance.reasoning.solvers import solve
from melm.appliance.reasoning.task_router import detect_reasoning_task
from melm.appliance.reasoning.value_extract import extract_distance


class DistanceExtractionTests(unittest.TestCase):
    def test_meters(self):
        self.assertAlmostEqual(extract_distance("50m away")["value_km"], 0.05)

    def test_km(self):
        self.assertAlmostEqual(extract_distance("3km away")["value_km"], 3.0)

    def test_miles(self):
        self.assertAlmostEqual(extract_distance("5 miles")["value_km"], 8.04672, places=4)


class GeoDetectionTests(unittest.TestCase):
    def test_car_wash(self):
        t = detect_reasoning_task("The car wash is 50m away, should I drive or walk?")
        self.assertEqual(t["task"], "geo_decision")
        self.assertAlmostEqual(t["distance_km"], 0.05)

    def test_shop(self):
        t = detect_reasoning_task("There's a shop 3km away, should I walk or drive there?")
        self.assertEqual(t["task"], "geo_decision")
        self.assertAlmostEqual(t["distance_km"], 3.0)

    def test_no_distance_not_geo(self):
        self.assertIsNone(detect_reasoning_task("Should I walk or drive?"))


class GeoSolverTests(unittest.TestCase):
    def test_car_wash_entity_purpose_overrides_distance(self):
        result, answer, refusal = solve({
            "task": "geo_decision", "distance_km": 0.05, "distance_text": "50m",
            "text": "the car wash is 50m away should i drive or walk",
        })
        self.assertIsNone(refusal)
        self.assertEqual(result["decision"], "drive")
        self.assertEqual(result["purpose"], "car_wash")
        self.assertIn("drive", answer.lower())
        self.assertIn("walk", answer.lower())  # explains the distance tension

    def test_short_generic_place_walk(self):
        result, _, _ = solve({
            "task": "geo_decision", "distance_km": 0.05, "distance_text": "50m",
            "text": "the park is 50m away should i walk or drive",
        })
        self.assertEqual(result["decision"], "walk")

    def test_far_place_drive(self):
        result, _, _ = solve({
            "task": "geo_decision", "distance_km": 3.0, "distance_text": "3km",
            "text": "the shop is 3km away should i walk or drive",
        })
        self.assertEqual(result["decision"], "drive")


class GeoKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_car_wash_drives_despite_short_distance(self):
        d = self.kernel.handle("The car wash is 50m away, should I drive or walk?")
        self.assertEqual(d.intent, "reasoning:geo_decision")
        self.assertEqual(d.route, "local_answer")
        self.assertFalse(d.cloud_needed)
        self.assertIn("drive", d.answer.lower())

    def test_far_shop_drives(self):
        d = self.kernel.handle("There's a shop 3km away, should I walk or drive there?")
        self.assertEqual(d.intent, "reasoning:geo_decision")
        self.assertIn("drive", d.answer.lower())


if __name__ == "__main__":
    unittest.main()
