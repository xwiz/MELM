"""Export the hand-authored child-dialogue benchmark as JSONL fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import authored_child_dialogue_fixture, save_dialogue_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events-out",
        type=Path,
        default=Path("benchmarks/authored_dialogue_events.jsonl"),
        help="Output JSONL path for event records.",
    )
    parser.add_argument(
        "--recall-out",
        type=Path,
        default=Path("benchmarks/authored_dialogue_recall_cases.jsonl"),
        help="Output JSONL path for positive recall cases.",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=Path("benchmarks/authored_dialogue_evidence_cases.jsonl"),
        help="Output JSONL path for answerability/evidence cases.",
    )
    args = parser.parse_args()

    events, recall_cases, evidence_cases = authored_child_dialogue_fixture()
    save_dialogue_benchmark(
        events,
        recall_cases,
        evidence_cases,
        events_path=args.events_out,
        recall_cases_path=args.recall_out,
        evidence_cases_path=args.evidence_out,
    )

    print("Exported authored child-dialogue benchmark")
    print(f"- events={len(events)} -> {args.events_out}")
    print(f"- recall_cases={len(recall_cases)} -> {args.recall_out}")
    print(f"- evidence_cases={len(evidence_cases)} -> {args.evidence_out}")


if __name__ == "__main__":
    main()
