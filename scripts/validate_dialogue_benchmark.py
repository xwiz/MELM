"""Validate dialogue benchmark JSONL fixtures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import load_dialogue_benchmark, validate_dialogue_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="JSONL event fixture path.")
    parser.add_argument("--recall-cases", required=True, type=Path, help="JSONL recall-case fixture path.")
    parser.add_argument("--evidence-cases", required=True, type=Path, help="JSONL evidence-case fixture path.")
    args = parser.parse_args()

    events, recall_cases, evidence_cases = load_dialogue_benchmark(
        events_path=args.events,
        recall_cases_path=args.recall_cases,
        evidence_cases_path=args.evidence_cases,
        validate=False,
    )
    errors = validate_dialogue_benchmark(events, recall_cases, evidence_cases)

    print("Dialogue benchmark fixture")
    print(f"- events={len(events)}")
    print(f"- recall_cases={len(recall_cases)}")
    print(f"- evidence_cases={len(evidence_cases)}")
    print(f"- event_path={args.events}")
    print(f"- recall_path={args.recall_cases}")
    print(f"- evidence_path={args.evidence_cases}")

    _print_categories("Recall categories", (case.category for case in recall_cases))
    _print_categories("Evidence categories", (case.category for case in evidence_cases))
    positives = sum(case.expected_event_id is not None for case in evidence_cases)
    negatives = len(evidence_cases) - positives
    print("")
    print("Evidence answerability")
    print(f"- positives={positives}")
    print(f"- negatives={negatives}")

    if errors:
        print("")
        print("Errors")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("")
    print("Validation passed")


def _print_categories(label: str, categories) -> None:
    counts = Counter(categories)
    if not counts:
        return
    print("")
    print(label)
    for category, count in sorted(counts.items()):
        print(f"- {category}: {count}")


if __name__ == "__main__":
    main()
