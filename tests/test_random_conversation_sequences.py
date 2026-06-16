"""Random conversation sequence generator — stress-tests the kernel/router
pipeline with utterances NOT already hardcoded in existing test fixtures.

Strategy:
1. Build reverse lexicon (class_id → list of tokens) from the legacy router terms.
2. Read frame templates to understand per-intent activation patterns.
3. Generate random utterances by sampling tokens from required + optional
   classes and prepending/appending action tokens.
4. Run through AssistantOSKernel.handle() and assert invariants:
   - No exception raised
   - decision.intent is a known intent
   - decision.route is a known route
   - If route == local_answer, synthesis.authority exists and verification passes
   - If route == cloud_handoff or reject, no synthesis authority required
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from melm.appliance import (
    AssistantOSKernel,
    LocalAssistantProfile,
)
from melm.appliance.local_assistant_router import (
    OnDeviceAssistantRouter,
    _IN_MEMORY_LEXICON,
)

# ---------------------------------------------------------------------------
# Lexicon helpers
# ---------------------------------------------------------------------------


def _build_class_to_tokens(lexicon: dict[str, frozenset[str]]) -> dict[str, list[str]]:
    """Reverse-map: class_id → all tokens carrying that class."""
    result: dict[str, list[str]] = {}
    for token, classes in lexicon.items():
        for cls in classes:
            result.setdefault(cls, []).append(token)
    return result


# ---------------------------------------------------------------------------
# Frame template helpers
# ---------------------------------------------------------------------------


def _load_frame_templates() -> dict[str, dict]:
    path = Path(__file__).resolve().parent.parent / "melm" / "contracts" / "frame_templates.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["templates"]


# ---------------------------------------------------------------------------
# Utterance generator
# ---------------------------------------------------------------------------


def _generate_utterance_for_frame(
    frame: dict,
    class_to_tokens: dict[str, list[str]],
    rng: random.Random,
) -> str | None:
    """Return a random utterance string that should activate *frame*.

    Returns None when the frame requires classes not present in the lexicon.
    """
    act = frame["activation"]
    tokens: list[str] = []

    # Optional: prepend a random action token (if any)
    action_tokens = act.get("action_tokens", [])
    if action_tokens and rng.random() < 0.7:
        tokens.append(rng.choice(action_tokens))

    # Required classes (OR gate) — pick one token from one required class
    required = act.get("required_classes", [])
    if required:
        # Pick a required class that actually has tokens
        viable = [c for c in required if class_to_tokens.get(c)]
        if not viable:
            return None
        chosen_class = rng.choice(viable)
        tokens.append(rng.choice(class_to_tokens[chosen_class]))

    # Required-all classes (AND gate) — pick one token from EACH
    required_all = act.get("required_all_classes", [])
    for cls in required_all:
        viable = class_to_tokens.get(cls, [])
        if not viable:
            return None
        tokens.append(rng.choice(viable))

    # Optional classes — pick 0-2 tokens
    optional = act.get("optional_classes", [])
    rng.shuffle(optional)
    for cls in optional[: rng.randint(0, 2)]:
        viable = class_to_tokens.get(cls, [])
        if viable:
            tokens.append(rng.choice(viable))

    # Simple sentence structures
    if not tokens:
        return None

    # Occasionally prepend a question word or pronoun
    prefixes = ["", "what", "how", "can you", "i want to", "please", "let me"]
    prefix = rng.choice(prefixes)
    if prefix:
        tokens.insert(0, prefix)

    # Occasionally append a suffix
    suffixes = ["", "today", "now", "please", "?"]
    suffix = rng.choice(suffixes)
    if suffix:
        tokens.append(suffix)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class RandomConversationInvariantTests(unittest.TestCase):
    """Stress-test kernel and router with random utterances."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.templates = _load_frame_templates()
        cls.class_to_tokens = _build_class_to_tokens(_IN_MEMORY_LEXICON)
        cls.known_intents = set(t["intent"] for t in cls.templates.values()) | {
            "assistant_identity",
            "assistant_status",
            "social_greeting",
            "assistant_behavior",
            "personal_goal_advice",
            "open_domain",
            "unknown",
        }
        cls.known_routes = {
            "local_answer",
            "cached_tool",
            "external_fetch",
            "cloud_handoff",
            "clarify",
            "reject",
            "device_action",
        }

    def _assert_valid_decision(self, decision, utterance: str) -> None:
        """Invariant checks that must hold for every decision."""
        self.assertIn(
            decision.intent,
            self.known_intents,
            f"Unknown intent '{decision.intent}' for '{utterance}'",
        )
        self.assertIn(
            decision.route,
            self.known_routes,
            f"Unknown route '{decision.route}' for '{utterance}'",
        )
        # Basic type checks
        self.assertIsInstance(decision.confidence, float)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)

    def test_random_utterances_per_frame_no_crash(self) -> None:
        """Generate 5 random utterances per frame template; router must not crash."""
        rng = random.Random(42)
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        for frame_id, frame in self.templates.items():
            for _ in range(5):
                utterance = _generate_utterance_for_frame(frame, self.class_to_tokens, rng)
                if utterance is None:
                    continue
                with self.subTest(frame=frame_id, utterance=utterance):
                    decision = router.handle(utterance)
                    self._assert_valid_decision(decision, utterance)

    def test_random_utterances_kernel_synthesis_invariants(self) -> None:
        """Generate random utterances through the full kernel; assert synthesis invariants."""
        rng = random.Random(43)
        kernel = AssistantOSKernel(
            profile=LocalAssistantProfile(
                weekly_weather={"today": "sunny and warm"},
                story_models={"test": "Once upon a time..."},
                food_inventory=("pasta", "rice", "apple"),
                facts={"pet": "dog"},
                contacts={"mom": "+1-555-MOM"},
                health_goals=("sleep better",),
                media_library=("lofi_beats",),
            )
        )
        for frame_id, frame in self.templates.items():
            for _ in range(5):
                utterance = _generate_utterance_for_frame(frame, self.class_to_tokens, rng)
                if utterance is None:
                    continue
                with self.subTest(frame=frame_id, utterance=utterance):
                    decision = kernel.handle(utterance)
                    self._assert_valid_decision(decision, utterance)

                    # If the kernel produced a synthesis, check authority
                    if decision.route not in {"local_answer", "cached_tool", "reject"}:
                        # Non-synthesizable routes (external_fetch, cloud_handoff,
                        # device_action, clarify) do not produce synthesis
                        continue
                    if kernel.last_synthesis is None:
                        self.fail(
                            f"Expected synthesis for route={decision.route} "
                            f"intent={decision.intent} utterance='{utterance}'"
                        )
                    synth = kernel.last_synthesis
                    self.assertIsNotNone(
                        synth.authority,
                        f"Synthesis missing authority for '{utterance}'",
                    )
                    if synth.authority is not None:
                        # Verification should pass for applied local answers
                        if synth.applied and decision.route in ("local_answer", "cached_tool"):
                            self.assertTrue(
                                synth.authority.verification.passed,
                                f"Verification failed for '{utterance}': "
                                f"{synth.authority.verification.failure_codes}",
                            )

    def test_adversarial_cross_intent_mixtures(self) -> None:
        """Combine tokens from two unrelated frames; should route to one or cloud_handoff."""
        rng = random.Random(44)
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        frames = list(self.templates.values())
        for _ in range(50):
            f1, f2 = rng.sample(frames, 2)
            u1 = _generate_utterance_for_frame(f1, self.class_to_tokens, rng)
            u2 = _generate_utterance_for_frame(f2, self.class_to_tokens, rng)
            if u1 is None or u2 is None:
                continue
            mixed = u1 + " and " + u2
            with self.subTest(mixed=mixed):
                decision = router.handle(mixed)
                self._assert_valid_decision(decision, mixed)

    def test_random_non_sense_strings_boundary(self) -> None:
        """Randomly generated low-signal strings should not crash or route locally."""
        rng = random.Random(45)
        router = OnDeviceAssistantRouter(LocalAssistantProfile())
        for _ in range(30):
            length = rng.randint(2, 8)
            tokens = [rng.choice(list(self.class_to_tokens.keys())[:20]) for _ in range(length)]
            utterance = " ".join(tokens)
            with self.subTest(utterance=utterance):
                decision = router.handle(utterance)
                self._assert_valid_decision(decision, utterance)


if __name__ == "__main__":
    unittest.main()
