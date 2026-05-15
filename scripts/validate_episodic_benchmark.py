"""Validate episodic benchmark JSONL fixtures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import load_episodic_benchmark, validate_episodic_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="JSONL event fixture path.")
    parser.add_argument("--cases", required=True, type=Path, help="JSONL recall-case fixture path.")
    args = parser.parse_args()

    events, cases = load_episodic_benchmark(
        events_path=args.events,
        cases_path=args.cases,
        validate=False,
    )
    errors = validate_episodic_benchmark(events, cases)

    print("Episodic benchmark fixture")
    print(f"- events={len(events)}")
    print(f"- cases={len(cases)}")
    print(f"- event_path={args.events}")
    print(f"- case_path={args.cases}")

    categories = Counter(case.category for case in cases)
    if categories:
        print("")
        print("Categories")
        for category, count in sorted(categories.items()):
            print(f"- {category}: {count}")

    if errors:
        print("")
        print("Errors")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("")
    print("Validation passed")


if __name__ == "__main__":
    main()
