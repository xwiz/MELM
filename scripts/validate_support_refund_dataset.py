"""Validate the authored support/refunds JSONL dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import load_authored_support_refund_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="benchmarks/support_refunds_authored.jsonl")
    args = parser.parse_args()

    dataset = load_authored_support_refund_dataset(args.dataset)
    fixture = dataset.fixture
    print("Support/refunds authored dataset validation")
    print(f"- dataset={dataset.path}")
    print(f"- turns={len(dataset.turns)}")
    print(f"- events={len(fixture.events)}")
    print(f"- facts={len(fixture.facts)}")
    print(f"- guard_cases={len(fixture.guard_cases)}")
    print(f"- memory_cases={len(fixture.memory_cases)}")
    if dataset.validation_errors:
        print("- validation_passed=False")
        for error in dataset.validation_errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("- validation_passed=True")


if __name__ == "__main__":
    main()
