"""Compile annotated transcript JSONL into dialogue benchmark fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import load_annotated_transcript_benchmark, save_dialogue_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path, help="Annotated transcript JSONL path.")
    parser.add_argument(
        "--events-out",
        type=Path,
        default=Path("benchmarks/sample_transcript_events.jsonl"),
        help="Output JSONL path for event records.",
    )
    parser.add_argument(
        "--recall-out",
        type=Path,
        default=Path("benchmarks/sample_transcript_recall_cases.jsonl"),
        help="Output JSONL path for recall cases.",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=Path("benchmarks/sample_transcript_evidence_cases.jsonl"),
        help="Output JSONL path for evidence cases.",
    )
    args = parser.parse_args()

    benchmark = load_annotated_transcript_benchmark(args.annotations)
    save_dialogue_benchmark(
        benchmark.events,
        benchmark.recall_cases,
        benchmark.evidence_cases,
        events_path=args.events_out,
        recall_cases_path=args.recall_out,
        evidence_cases_path=args.evidence_out,
    )

    print("Compiled annotated transcript benchmark")
    print(f"- annotations={args.annotations}")
    print(f"- turns={len(benchmark.turns)}")
    print(f"- events={len(benchmark.events)} -> {args.events_out}")
    print(f"- recall_cases={len(benchmark.recall_cases)} -> {args.recall_out}")
    print(f"- evidence_cases={len(benchmark.evidence_cases)} -> {args.evidence_out}")
    print(f"- state_cases={len(benchmark.state_cases)} retained in annotation source")


if __name__ == "__main__":
    main()
