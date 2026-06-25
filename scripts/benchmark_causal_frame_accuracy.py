"""Benchmark causal frame extraction accuracy.

Compares LLM-generated entries against human-curated ground truth
across 5 dimensions: schema compliance, semantic class, cause kind,
effect coverage, and state definition coverage.

Usage:
    python scripts/benchmark_causal_frame_accuracy.py
    python scripts/benchmark_causal_frame_accuracy.py --backend ollama
    python scripts/benchmark_causal_frame_accuracy.py --dry-run   # use embedded gold-only scores
    python scripts/benchmark_causal_frame_accuracy.py --ci        # exit code 1 if overall < 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Ground-truth entries (same as in extract_causal_frames_llm.py)
GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "eat": {
        "predicate": {
            "predicate_id": "eat", "semantic_class": "verb.consume",
            "default_cause_kind": "intentional_action", "roles": ["agent", "patient"],
            "effects": [
                {"target_role": "agent", "state": "satisfied", "domain": "emotional", "relation": "causes", "confidence": 0.88},
                {"target_role": "agent", "state": "nourished", "domain": "physical", "relation": "causes", "confidence": 0.85},
                {"target_role": "agent", "state": "full", "domain": "physical", "relation": "causes", "confidence": 0.80},
            ],
            "surface_aliases": ["eat", "eats", "ate", "eaten", "eating"],
        },
        "states": {
            "satisfied": {"aliases": ["satisfied", "content", "full"], "definition": "has had hunger or desire fulfilled", "opposites": ["hungry"]},
            "nourished": {"aliases": ["nourished", "fed", "sustained"], "definition": "has received nutrition or sustenance", "opposites": ["malnourished"]},
            "full": {"aliases": ["full"], "definition": "has no remaining capacity for food", "opposites": ["hungry"]},
        },
    },
    "sleep": {
        "predicate": {
            "predicate_id": "sleep", "semantic_class": "verb.body",
            "default_cause_kind": "natural_process", "roles": ["agent"],
            "effects": [
                {"target_role": "agent", "state": "rested", "domain": "physical", "relation": "causes", "confidence": 0.92},
                {"target_role": "agent", "state": "energetic", "domain": "physical", "relation": "causes", "confidence": 0.85},
                {"target_role": "agent", "state": "alert", "domain": "mental", "relation": "causes", "confidence": 0.80},
            ],
            "surface_aliases": ["sleep", "sleeps", "slept", "sleeping"],
        },
        "states": {
            "rested": {"aliases": ["rested", "well_rested", "refreshed"], "definition": "has recovered energy through rest or sleep", "opposites": ["tired"]},
            "energetic": {"aliases": ["energetic", "energized", "active"], "definition": "has high physical energy or vitality", "opposites": ["lethargic"]},
            "alert": {"aliases": ["alert", "awake", "attentive", "focused"], "definition": "is mentally awake and attentive", "opposites": ["drowsy"]},
        },
    },
    "break": {
        "predicate": {
            "predicate_id": "break", "semantic_class": "verb.contact",
            "default_cause_kind": "accidental_process", "roles": ["agent", "patient", "instrument"],
            "effects": [
                {"target_role": "patient", "state": "broken", "domain": "physical", "relation": "causes", "confidence": 0.95},
                {"target_role": "patient", "state": "damaged", "domain": "physical", "relation": "causes", "confidence": 0.85},
                {"target_role": "patient", "state": "fragmented", "domain": "physical", "relation": "causes", "confidence": 0.70},
            ],
            "surface_aliases": ["break", "breaks", "broke", "broken", "breaking"],
        },
        "states": {
            "broken": {"aliases": ["broken", "shattered", "cracked", "fractured"], "definition": "is physically separated into pieces or no longer intact", "opposites": ["intact"]},
            "damaged": {"aliases": ["damaged", "harmed", "impaired"], "definition": "has suffered physical harm or impairment to function or integrity", "opposites": ["undamaged"]},
            "fragmented": {"aliases": ["fragmented", "in_pieces", "shattered"], "definition": "is broken into multiple fragments or pieces", "opposites": ["whole"]},
        },
    },
    "teach": {
        "predicate": {
            "predicate_id": "teach", "semantic_class": "verb.communicate",
            "default_cause_kind": "intentional_action", "roles": ["agent", "patient", "theme"],
            "effects": [
                {"target_role": "patient", "state": "knowledgeable", "domain": "mental", "relation": "causes", "confidence": 0.88},
                {"target_role": "patient", "state": "skilled", "domain": "mental", "relation": "causes", "confidence": 0.80},
                {"target_role": "patient", "state": "educated", "domain": "mental", "relation": "causes", "confidence": 0.75},
            ],
            "surface_aliases": ["teach", "teaches", "taught", "teaching"],
        },
        "states": {
            "knowledgeable": {"aliases": ["knowledgeable", "educated", "learned"], "definition": "has acquired knowledge or understanding", "opposites": ["ignorant"]},
            "skilled": {"aliases": ["skilled", "proficient", "adept", "expert"], "definition": "has developed ability or expertise through training or practice", "opposites": ["unskilled"]},
            "educated": {"aliases": ["educated", "trained", "schooled", "instructed"], "definition": "has received formal teaching or instruction", "opposites": ["uneducated"]},
        },
    },
    "rain": {
        "predicate": {
            "predicate_id": "rain", "semantic_class": "verb.weather",
            "default_cause_kind": "natural_process", "roles": ["source", "patient"],
            "effects": [
                {"target_role": "patient", "state": "wet", "domain": "physical", "relation": "causes", "confidence": 0.95},
                {"target_role": "environment", "state": "cooler", "domain": "environmental", "relation": "causes", "confidence": 0.75},
            ],
            "surface_aliases": ["rain", "rains", "rained", "raining"],
        },
        "states": {
            "wet": {"aliases": ["wet", "damp", "moist"], "definition": "has moisture or water on or in the target", "opposites": ["dry"]},
            "cooler": {"aliases": ["cooler", "colder", "cooled"], "definition": "has decreased temperature relative to prior state", "opposites": ["warmer"]},
        },
    },
    "push": {
        "predicate": {
            "predicate_id": "push", "semantic_class": "verb.contact",
            "default_cause_kind": "intentional_action", "roles": ["agent", "patient"],
            "effects": [
                {"target_role": "patient", "state": "moved", "domain": "physical", "relation": "causes", "confidence": 0.90},
                {"target_role": "patient", "state": "displaced", "domain": "physical", "relation": "causes", "confidence": 0.85},
            ],
            "surface_aliases": ["push", "pushes", "pushed", "pushing"],
        },
        "states": {
            "moved": {"aliases": ["moved", "displaced", "relocated", "shifted"], "definition": "has changed physical position or location", "opposites": ["stationary"]},
            "displaced": {"aliases": ["displaced", "shifted"], "definition": "has been moved from its original position", "opposites": ["in_place"]},
        },
    },
}


def _score_effects(
    pred_effects: list[dict[str, Any]],
    exp_effects: list[dict[str, Any]],
) -> float:
    """Jaccard similarity on (state, domain, target_role) triples."""
    pred_set = set()
    for e in pred_effects:
        pred_set.add((e.get("state", ""), e.get("domain", ""), e.get("target_role", "")))
    exp_set = set()
    for e in exp_effects:
        exp_set.add((e.get("state", ""), e.get("domain", ""), e.get("target_role", "")))
    union = pred_set | exp_set
    if not union:
        return 1.0
    return len(pred_set & exp_set) / len(union)


def _score_states(
    pred_states: dict[str, Any],
    exp_states: dict[str, Any],
) -> float:
    """Jaccard on state keys."""
    pred_keys = set(pred_states.keys())
    exp_keys = set(exp_states.keys())
    union = pred_keys | exp_keys
    if not union:
        return 1.0
    return len(pred_keys & exp_keys) / len(union)


def _score_single(
    predicted: dict[str, Any] | None,
    expected: dict[str, Any],
) -> dict[str, float]:
    """Score one verb. Returns {schema, semantic_class, cause_kind, effects, states} ∈ [0,1]."""
    if predicted is None:
        return {"schema": 0.0, "semantic_class": 0.0, "cause_kind": 0.0, "effects": 0.0, "states": 0.0}

    pp = predicted.get("predicate", {})
    ep = expected.get("predicate", {})
    ps = predicted.get("states", {})
    es = expected.get("states", {})

    return {
        "schema": 1.0,  # only filled if it passed validate_causal_frames
        "semantic_class": 1.0 if pp.get("semantic_class") == ep.get("semantic_class") else 0.0,
        "cause_kind": 1.0 if pp.get("default_cause_kind") == ep.get("default_cause_kind") else 0.0,
        "effects": _score_effects(pp.get("effects", []), ep.get("effects", [])),
        "states": _score_states(ps, es),
    }


def _perfect_score(expected: dict[str, Any]) -> dict[str, float]:
    """Return perfect scores for a ground-truth entry (self-consistency check)."""
    return _score_single(expected, expected)


def run_benchmark_dry() -> dict[str, Any]:
    """Dry-run: score ground truth against itself to verify benchmark logic."""
    results: dict[str, dict[str, float]] = {}
    dim_totals: dict[str, float] = {"schema": 0.0, "semantic_class": 0.0, "cause_kind": 0.0, "effects": 0.0, "states": 0.0}
    n = len(GROUND_TRUTH)

    print("--- Dry-Run: Ground Truth Self-Consistency ---")
    for verb, expected in sorted(GROUND_TRUTH.items()):
        scores = _perfect_score(expected)
        results[verb] = scores
        for dim, v in scores.items():
            dim_totals[dim] += v

    agg = {dim: round(total / n, 3) for dim, total in dim_totals.items()}
    agg["overall"] = round(sum(agg.values()) / len(agg), 3)

    for verb, scores in sorted(results.items()):
        parts = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
        print(f"  {verb}: {parts}")
    print(f"\n  Aggregate ({n} verbs):")
    for dim, v in agg.items():
        print(f"    {dim}: {v:.3f}")
    print(f"  ---> OVERALL: {agg['overall']:.3f} (all 1.0 if benchmark logic is correct)")
    return {"verbs": results, "aggregate": agg, "n": n}


def run_benchmark(backend: str = "ollama") -> dict[str, Any]:
    """Run full benchmark with LLM."""
    from scripts.extract_causal_frames_llm import extract_for_verb

    results: dict[str, dict[str, float] | None] = {}
    dim_totals: dict[str, float] = {"schema": 0.0, "semantic_class": 0.0, "cause_kind": 0.0, "effects": 0.0, "states": 0.0}
    n = len(GROUND_TRUTH)

    print(f"--- Causal Frame Accuracy Benchmark (backend={backend}) ---")
    print(f"Verbs: {', '.join(sorted(GROUND_TRUTH.keys()))}\n")

    for verb, expected in sorted(GROUND_TRUTH.items()):
        predicted = extract_for_verb(verb, backend=backend)
        scores = _score_single(predicted, expected)
        results[verb] = scores
        for dim, v in scores.items():
            dim_totals[dim] += v

    agg = {dim: round(total / n, 3) for dim, total in dim_totals.items()}
    agg["overall"] = round(sum(agg.values()) / len(agg), 3)

    print()
    for verb, scores in sorted(results.items()):
        if scores is None:
            print(f"  {verb}: FAILED")
        else:
            parts = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
            print(f"  {verb}: {parts}")

    print()
    print(f"  Aggregate ({n} verbs):")
    for dim, v in agg.items():
        marker = " ✓" if v >= 0.90 else ""
        print(f"    {dim}: {v:.3f}{marker}")
    print(f"  ---> OVERALL: {agg['overall']:.3f} {'✓ PASS' if agg['overall'] >= 0.90 else '✗ NEEDS WORK'}")

    return {"verbs": results, "aggregate": agg, "n": n}


def main():
    parser = argparse.ArgumentParser(description="Benchmark causal frame extraction accuracy")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "ollama", "custom", "transformers", "none", "dry-run"])
    parser.add_argument("--dry-run", action="store_true", help="Run self-consistency check only")
    parser.add_argument("--ci", action="store_true", help="Exit code 1 if overall < 0.90")
    parser.add_argument("--output", type=Path, help="Save results JSON")
    args = parser.parse_args()

    if args.dry_run:
        result = run_benchmark_dry()
    else:
        result = run_benchmark(backend=args.backend)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {args.output}")

    if args.ci:
        overall = result.get("aggregate", {}).get("overall", 0.0)
        if overall < 0.90:
            print(f"\nCI FAIL: overall={overall:.3f} < 0.90")
            sys.exit(1)
        print(f"\nCI PASS: overall={overall:.3f} >= 0.90")


if __name__ == "__main__":
    main()
