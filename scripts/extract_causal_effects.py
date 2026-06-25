"""Extract causal effects from nameless verb data into causal_effects.v1.json.

Usage:
    python scripts/extract_causal_effects.py [--output PATH] [--min-states N]

Output: melm/contracts/causal_effects.v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTRACTS_DIR = Path(__file__).parent.parent / "melm" / "contracts"
NAMELESS_PATH = CONTRACTS_DIR / "nameless_extracted_verbs.json"
OUTPUT_PATH = CONTRACTS_DIR / "causal_effects.v1.json"


def _confidence_from_data(total_states: int) -> float:
    if total_states >= 5:
        return 0.90
    if total_states >= 3:
        return 0.80
    if total_states >= 1:
        return 0.70
    return 0.50


def _convert_verb(verb: str, data: dict) -> dict | None:
    patient_states = data.get("patient_states", {})
    if not patient_states:
        return None
    effects: dict[str, list[str]] = {}
    for domain, states in patient_states.items():
        if not isinstance(states, list):
            continue
        clean = [s for s in states if s and s != "unchanged"]
        if clean:
            effects[domain] = clean
    if not effects:
        return None
    total = sum(len(v) for v in effects.values())
    return {
        "patient_types": data.get("patient_types", []),
        "effects": effects,
        "confidence": _confidence_from_data(total),
        "provenance": "offline_extractor",
        "review_status": "pending",
    }


def extract(
    nameless_path: Path = NAMELESS_PATH,
    output_path: Path = OUTPUT_PATH,
    min_states: int = 1,
) -> int:
    with open(nameless_path, encoding="utf-8") as f:
        nameless = json.load(f)

    rules: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        for verb, rule in existing.get("rules", {}).items():
            if verb not in nameless:
                rules[verb] = rule

    for verb, data in nameless.items():
        rule = _convert_verb(verb, data)
        if rule is None:
            continue
        total = sum(len(v) for v in rule["effects"].values())
        if total < min_states:
            continue
        rules[verb] = rule

    rules = dict(sorted(rules.items()))

    output = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_id": "melm.causal_effects.v1",
        "version": "1.0.0",
        "description": "Maps verbs to probable causal effects for deterministic inference. Auto-enriched from nameless verb data.",
        "rules": rules,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return len(rules)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--min-states", type=int, default=1)
    args = parser.parse_args()
    count = extract(output_path=args.output, min_states=args.min_states)
    print(f"Extracted {count} verbs to {args.output}")


if __name__ == "__main__":
    main()
