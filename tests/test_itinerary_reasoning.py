"""Slice 9: multi-turn itinerary scenario reasoning."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile
from melm.appliance.reasoning import geo
from melm.appliance.reasoning.itinerary import (
    detect_itinerary_query, parse_itinerary, solve_itinerary,
)
from melm.contracts import load_geo_atlas

ITINERARY = (
    "You're currently in Enugu, and you will be moving to Lagos on saturday. "
    "When you get to Lagos, you'll stay in Oniru for 2 hours, before moving to "
    "Ikeja, then you'll stop briefly at Nsukka."
)
PLACES = ["enugu", "lagos", "oniru", "ikeja", "nsukka"]


class HaversineTests(unittest.TestCase):
    def test_known_distance_enugu_lagos(self):
        d = geo.haversine_km(6.459964, 7.548949, 6.4541, 3.3947)
        self.assertTrue(450 < d < 475)  # ~460 km


class ParseTests(unittest.TestCase):
    def test_parses_ordered_stops_stays_day(self):
        s = parse_itinerary(ITINERARY, list(load_geo_atlas()["places"].keys()))
        self.assertEqual(s["places"], PLACES)
        self.assertEqual(s["stays"], {"oniru": 2.0})
        self.assertEqual(s["start_day"], "saturday")

    def test_query_detection(self):
        self.assertEqual(detect_itinerary_query("How long will it take you?"), "duration")
        self.assertEqual(detect_itinerary_query("What is the total distance moved?"), "path_distance")
        self.assertEqual(detect_itinerary_query(
            "What is the distance between your final location and your initial location?"), "displacement")
        self.assertEqual(detect_itinerary_query("Where will you be tomorrow at about 4pm?"), "projection")


class SolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.atlas = load_geo_atlas()
        self.scenario = parse_itinerary(ITINERARY, list(self.atlas["places"].keys()))
        self.now = datetime(2026, 6, 18, 10, 0)

    def test_path_distance(self):
        _, ans, refusal = solve_itinerary(self.scenario, self.atlas, "path_distance", self.now)
        self.assertIsNone(refusal)
        self.assertIn("km", ans)

    def test_displacement_less_than_path(self):
        rd, _, _ = solve_itinerary(self.scenario, self.atlas, "displacement", self.now)
        self.assertLess(rd["displacement_km"], rd["path_km"])  # loops back east

    def test_duration_has_hours(self):
        rd, ans, _ = solve_itinerary(self.scenario, self.atlas, "duration", self.now)
        self.assertGreater(rd["total_hours"], 0)
        self.assertIn("hours", ans)

    def test_projection_flags_assumption(self):
        _, ans, refusal = solve_itinerary(self.scenario, self.atlas, "projection", self.now)
        self.assertIsNone(refusal)
        self.assertIn("assuming", ans.lower())

    def test_unknown_place_refuses(self):
        bad = {"places": ["enugu", "atlantis"], "legs": [("enugu", "atlantis")],
               "stays": {}, "start_day": "saturday"}
        _, _, refusal = solve_itinerary(bad, self.atlas, "duration", self.now)
        self.assertTrue(refusal.startswith("unknown_place"))


class ItineraryKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_multi_turn_itinerary(self):
        d1 = self.kernel.handle(ITINERARY + " How long will it take you?")
        self.assertEqual(d1.intent, "reasoning:itinerary")
        self.assertIn("hours", d1.answer)

        d2 = self.kernel.handle("What is the total distance moved?")
        self.assertEqual(d2.intent, "reasoning:itinerary")
        self.assertIn("km", d2.answer)

        d3 = self.kernel.handle(
            "What is the distance between your final location and your initial location?")
        self.assertEqual(d3.intent, "reasoning:itinerary")
        self.assertIn("Nsukka", d3.answer)
        self.assertIn("Enugu", d3.answer)

        with mock.patch("melm.appliance.reasoning.clock.now", return_value=datetime(2026, 6, 18, 10, 0)):
            d4 = self.kernel.handle("Where will you be tomorrow at about 4pm?")
        self.assertEqual(d4.intent, "reasoning:itinerary")
        self.assertIn("assuming", d4.answer.lower())


if __name__ == "__main__":
    unittest.main()
