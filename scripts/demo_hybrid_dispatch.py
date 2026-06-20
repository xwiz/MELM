"""Demo hybrid dispatch: template for weather, model for story.

Usage::

    python scripts/demo_hybrid_dispatch.py

This demonstrates the Week 2 ADTC deliverable: MELM uses the model
backend only when it adds value (story, open_domain), keeping
deterministic intents (weather, meal, identity) on template for speed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance.assistant_authority import (
    AnswerPlan,
    build_answer_plan,
    build_evidence_packet,
)
from melm.appliance.assistant_decoder import ConstrainedDecoder, DecodingGrammar
from melm.appliance.assistant_synthesis import BoundedLocalSynthesizer
from melm.appliance.local_assistant_router import AssistantDecision, LocalAssistantProfile


def _decision(intent: str, reason: str = "default") -> AssistantDecision:
    return AssistantDecision(
        utterance="demo",
        intent=intent,
        route="local_answer",
        answer="",
        evidence_keys=(),
        confidence=0.9,
        reason=reason,
    )


def main() -> None:
    model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if not Path(model_path).exists():
        print(f"Model not found: {model_path} (run: python scripts/download_model.py)")
        print("Continuing with template-only backend; model turns will fall back to template.")
        model_path = ""

    profile = LocalAssistantProfile(user_name="Ade", location="Lagos", culture="Yoruba")
    decoder = ConstrainedDecoder(preferred="template", model_path=model_path)
    synth = BoundedLocalSynthesizer(profile, decoder=decoder)

    # Helper evidence class
    E = type("E", (), {})

    def make_evidence(key: str, kind: str, value: str):
        e = E()
        e.key = key
        e.kind = kind
        e.value = value
        e.source = "test"
        e.license = "test"
        e.local_only = True
        return e

    print("=" * 60)
    print("MELM Hybrid Dispatch Demo")
    print("=" * 60)

    def _badge(decoder_used: str) -> str:
        return "[model]" if decoder_used == "llamacpp" else "[template]"

    # Turn 1: Weather (template)
    print("\n[Turn 1] Weather -> template backend (fast, deterministic)")
    decision = _decision("weather", "location_available")
    template_answer = "Today in Lagos: sunny 25C."
    evidence = (make_evidence("w1", "weather", "sunny 25C"),)
    packet = build_evidence_packet((), (), "")
    plan = build_answer_plan(decision, packet)
    t0 = time.perf_counter()
    answer, decoder_used = synth._decode_verified(plan, evidence, decision, template_answer, packet)
    t1 = time.perf_counter()
    print(f"  Answer: {answer} {_badge(decoder_used)}")
    print(f"  Time:   {(t1 - t0) * 1000:.2f} ms")
    assert answer == template_answer, "Weather should stay on template"
    assert decoder_used == "template", "Weather should use template decoder"

    # Turn 2: Story (model)
    print("\n[Turn 2] Story -> model backend (rich narrative)")
    decision = _decision("story")
    template_answer = "I picked The Tortoise from the local story inventory. In Lagos, Ade met a patient tortoise."
    evidence = (make_evidence("s1", "story_model", "tortoise"),)
    packet = build_evidence_packet((), (), "")
    plan = build_answer_plan(decision, packet)
    t0 = time.perf_counter()
    answer, decoder_used = synth._decode_verified(plan, evidence, decision, template_answer, packet)
    t1 = time.perf_counter()
    print(f"  Answer: {answer} {_badge(decoder_used)}")
    print(f"  Time:   {(t1 - t0) * 1000:.2f} ms")

    # Turn 3: Meal (template)
    print("\n[Turn 3] Meal -> template backend (contract-based)")
    decision = _decision("meal", "location_available")
    template_answer = "I suggest jollof rice with chicken for lunch in Lagos."
    evidence = (make_evidence("m1", "food_inventory", "rice"),)
    packet = build_evidence_packet((), (), "")
    plan = build_answer_plan(decision, packet)
    t0 = time.perf_counter()
    answer, decoder_used = synth._decode_verified(plan, evidence, decision, template_answer, packet)
    t1 = time.perf_counter()
    print(f"  Answer: {answer} {_badge(decoder_used)}")
    print(f"  Time:   {(t1 - t0) * 1000:.2f} ms")
    assert answer == template_answer, "Meal should stay on template"

    # Turn 4: Open domain (model)
    print("\n[Turn 4] Open domain -> model backend (novelty handling)")
    decision = _decision("open_domain")
    template_answer = "I do not have that information locally."
    evidence = (make_evidence("od1", "user_fact", "likes jazz"),)
    packet = build_evidence_packet((), (), "")
    plan = build_answer_plan(decision, packet)
    t0 = time.perf_counter()
    answer, decoder_used = synth._decode_verified(plan, evidence, decision, template_answer, packet)
    t1 = time.perf_counter()
    print(f"  Answer: {answer} {_badge(decoder_used)}")
    print(f"  Time:   {(t1 - t0) * 1000:.2f} ms")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
