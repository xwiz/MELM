"""Slice 7: temporal reasoning (time / today / day offsets)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile
from melm.appliance.reasoning.solvers import solve
from melm.appliance.reasoning.task_router import detect_reasoning_task

FIXED = datetime(2026, 6, 19, 14, 30)  # a specific date/time


class TemporalDetectionTests(unittest.TestCase):
    def test_time(self):
        self.assertEqual(detect_reasoning_task("What time is it?"),
                         {"task": "temporal", "op": "time"})

    def test_today(self):
        self.assertEqual(detect_reasoning_task("What day is it today?")["op"], "date_today")

    def test_offset_future_digit(self):
        self.assertEqual(detect_reasoning_task("What day will it be in 3 days?"),
                         {"task": "temporal", "op": "day_offset", "days": 3})

    def test_offset_future_word(self):
        self.assertEqual(detect_reasoning_task("What day is it in three days?")["days"], 3)

    def test_offset_past(self):
        self.assertEqual(detect_reasoning_task("What day was it 2 days ago?")["days"], -2)

    def test_absolute_date_day_lookup(self):
        self.assertEqual(
            detect_reasoning_task("What day was June 11 1991?"),
            {"task": "temporal", "op": "absolute_date", "date": "1991-06-11"},
        )


class TemporalSolverTests(unittest.TestCase):
    def _patch(self):
        return mock.patch("melm.appliance.reasoning.clock.now", return_value=FIXED)

    def test_time(self):
        with self._patch():
            result, answer, refusal = solve({"task": "temporal", "op": "time"})
        self.assertIsNone(refusal)
        self.assertIn("It is", answer)

    def test_date_today(self):
        with self._patch():
            _, answer, _ = solve({"task": "temporal", "op": "date_today"})
        self.assertIn(FIXED.strftime("%A"), answer)

    def test_offset_future(self):
        with self._patch():
            _, answer, _ = solve({"task": "temporal", "op": "day_offset", "days": 3})
        self.assertIn((FIXED + timedelta(days=3)).strftime("%A"), answer)
        self.assertIn("In 3 days", answer)

    def test_offset_past(self):
        with self._patch():
            _, answer, _ = solve({"task": "temporal", "op": "day_offset", "days": -2})
        self.assertIn((FIXED - timedelta(days=2)).strftime("%A"), answer)
        self.assertIn("2 days ago", answer)

    def test_absolute_date(self):
        with self._patch():
            result, answer, refusal = solve(
                {"task": "temporal", "op": "absolute_date", "date": "1991-06-11"}
            )
        self.assertIsNone(refusal)
        self.assertEqual(result["weekday"], "Tuesday")
        self.assertEqual(result["relation"], "past")
        self.assertIn("Tuesday", answer)
        self.assertIn("was", answer)
        self.assertIn("June 11, 1991", answer)

    def test_absolute_future_date_uses_future_tense(self):
        with self._patch():
            result, answer, refusal = solve(
                {"task": "temporal", "op": "absolute_date", "date": "2030-06-11"}
            )
        self.assertIsNone(refusal)
        self.assertEqual(result["relation"], "future")
        self.assertIn("will be", answer)


class TemporalKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = AssistantOSStore(self.tmp.name)
        seed_class_schemas(self.store)
        self.kernel = AssistantOSKernel(profile=LocalAssistantProfile(), store=self.store)

    def tearDown(self) -> None:
        self.store.connection.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_day_in_3_days_local(self):
        with mock.patch("melm.appliance.reasoning.clock.now", return_value=FIXED):
            d = self.kernel.handle("What day will it be in 3 days?")
        self.assertEqual(d.intent, "reasoning:temporal")
        self.assertEqual(d.route, "local_answer")
        self.assertFalse(d.cloud_needed)
        self.assertIn((FIXED + timedelta(days=3)).strftime("%A"), d.answer)

    def test_time_local(self):
        with mock.patch("melm.appliance.reasoning.clock.now", return_value=FIXED):
            d = self.kernel.handle("What time is it?")
        self.assertEqual(d.intent, "reasoning:temporal")
        self.assertIn("It is", d.answer)

    def test_absolute_date_local(self):
        d = self.kernel.handle("What day was June 11 1991?")
        self.assertEqual(d.intent, "reasoning:temporal")
        self.assertIn("Tuesday", d.answer)

    def test_historical_weather_routes_to_dated_fetch(self):
        d = self.kernel.handle("What was the weather on June 11 1991?")
        self.assertEqual(d.intent, "weather")
        self.assertEqual(d.route, "external_fetch")
        self.assertTrue(d.external_fetch_needed)
        self.assertIn("1991-06-11", d.evidence_keys)
        self.assertIn("historical", d.reason)

    def test_future_weather_routes_to_dated_forecast_fetch(self):
        d = self.kernel.handle("What will the weather be on June 11 2030?")
        self.assertEqual(d.intent, "weather")
        self.assertEqual(d.route, "external_fetch")
        self.assertTrue(d.external_fetch_needed)
        self.assertIn("2030-06-11", d.evidence_keys)
        self.assertIn("forecast", d.reason)


if __name__ == "__main__":
    unittest.main()
