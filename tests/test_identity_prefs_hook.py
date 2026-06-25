"""Tests for Hook #2 — identity.json → BoundedLocalSynthesizer persona.

Covers:
  - load_identity_prefs: happy path, missing file, corrupt JSON
  - BoundedLocalSynthesizer: display_name override, persona emoji on/off,
    emoji only on identity/greeting intents
  - Kernel._synthesizer: wires identity_prefs and overrides self_state name
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# load_identity_prefs
# ---------------------------------------------------------------------------

class LoadIdentityPrefsTests(unittest.TestCase):
    def test_returns_dict_from_valid_file(self):
        from melm.appliance.provisioning import load_identity_prefs
        data = {
            "schema": "melm.identity.v1",
            "device_id": "dev-001",
            "display_name": "Aria",
            "emoji": "\U0001F916",
            "prefs": {"use_emoji": True, "greeting_style": "warm", "verbosity": "concise"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = load_identity_prefs(path)
            self.assertEqual(result["display_name"], "Aria")
            self.assertEqual(result["emoji"], "\U0001F916")
            self.assertTrue(result["prefs"]["use_emoji"])
        finally:
            path.unlink(missing_ok=True)

    def test_returns_empty_dict_when_file_missing(self):
        from melm.appliance.provisioning import load_identity_prefs
        result = load_identity_prefs(Path("/nonexistent/path/identity.json"))
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_corrupt_json(self):
        from melm.appliance.provisioning import load_identity_prefs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            path = Path(f.name)
        try:
            result = load_identity_prefs(path)
            self.assertEqual(result, {})
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# BoundedLocalSynthesizer — identity_prefs wiring
# ---------------------------------------------------------------------------

def _make_synthesizer(identity_prefs=None):
    from melm.appliance.assistant_synthesis import BoundedLocalSynthesizer
    from melm.appliance.local_assistant_router import LocalAssistantProfile
    profile = LocalAssistantProfile()
    return BoundedLocalSynthesizer(
        profile,
        self_state={"name": "MELM Local Assistant OS", "purpose": "help"},
        identity_prefs=identity_prefs,
    )


def _make_decision(intent="assistant_identity", answer="Hello, I am MELM."):
    from melm.appliance.local_assistant_router import AssistantDecision
    return AssistantDecision(
        utterance="who are you",
        intent=intent,
        route="local_answer",
        answer=answer,
        reason="test",
    )


class SynthesizerIdentityPrefsTests(unittest.TestCase):
    def test_no_identity_prefs_returns_answer_unchanged(self):
        synth = _make_synthesizer()
        decision = _make_decision()
        result = synth._maybe_emoji(decision, "Hello")
        self.assertEqual(result, "Hello")

    def test_use_emoji_false_no_persona_emoji(self):
        synth = _make_synthesizer({"emoji": "\U0001F916", "prefs": {"use_emoji": False}})
        decision = _make_decision(intent="assistant_identity")
        result = synth._maybe_emoji(decision, "Hello")
        self.assertEqual(result, "Hello")

    def test_use_emoji_true_prepends_on_identity_intent(self):
        synth = _make_synthesizer({"emoji": "\U0001F916", "prefs": {"use_emoji": True}})
        decision = _make_decision(intent="assistant_identity", answer="I am MELM.")
        result = synth._maybe_emoji(decision, "I am MELM.")
        self.assertTrue(result.startswith("\U0001F916"))
        self.assertIn("I am MELM.", result)

    def test_use_emoji_true_prepends_on_greeting_intent(self):
        synth = _make_synthesizer({"emoji": "\U0001F916", "prefs": {"use_emoji": True}})
        decision = _make_decision(intent="social_greeting", answer="Hi there!")
        result = synth._maybe_emoji(decision, "Hi there!")
        self.assertTrue(result.startswith("\U0001F916"))

    def test_use_emoji_true_no_prepend_on_other_intents(self):
        synth = _make_synthesizer({"emoji": "\U0001F916", "prefs": {"use_emoji": True}})
        decision = _make_decision(intent="local_answer", answer="The weather is fine.")
        result = synth._maybe_emoji(decision, "The weather is fine.")
        self.assertEqual(result, "The weather is fine.")

    def test_empty_emoji_string_no_prepend(self):
        synth = _make_synthesizer({"emoji": "", "prefs": {"use_emoji": True}})
        decision = _make_decision(intent="assistant_identity", answer="I am MELM.")
        result = synth._maybe_emoji(decision, "I am MELM.")
        self.assertEqual(result, "I am MELM.")


# ---------------------------------------------------------------------------
# Kernel._synthesizer — display_name override
# ---------------------------------------------------------------------------

class KernelSynthesizerDisplayNameTests(unittest.TestCase):
    def _make_kernel(self, identity_prefs=None):
        from melm.appliance.assistant_os_kernel import AssistantOSKernel
        kernel = AssistantOSKernel.__new__(AssistantOSKernel)
        from melm.appliance.local_assistant_router import LocalAssistantProfile
        from melm.appliance.assistant_os_kernel import SelfModel
        kernel.profile = LocalAssistantProfile()
        kernel.self_model = SelfModel()
        kernel.store = None
        kernel.decoder = None
        kernel.research_provider = None
        kernel._current_self_status = {}
        kernel.identity_prefs = identity_prefs or {}
        return kernel

    def test_display_name_overrides_self_state_name(self):
        kernel = self._make_kernel({"display_name": "Aria"})
        synth = kernel._synthesizer()
        self.assertEqual(synth.self_state["name"], "Aria")

    def test_no_display_name_uses_self_model_name(self):
        kernel = self._make_kernel({})
        synth = kernel._synthesizer()
        self.assertIsNotNone(synth.self_state.get("name"))
        self.assertNotEqual(synth.self_state.get("name"), "Aria")

    def test_identity_prefs_forwarded_to_synthesizer(self):
        prefs = {"display_name": "Aria", "emoji": "\U0001F916", "prefs": {"use_emoji": True}}
        kernel = self._make_kernel(prefs)
        synth = kernel._synthesizer()
        self.assertEqual(synth.identity_prefs["display_name"], "Aria")
        self.assertTrue(synth.identity_prefs["prefs"]["use_emoji"])


if __name__ == "__main__":
    unittest.main()
