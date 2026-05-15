"""Export the deterministic synthetic episodic benchmark as JSONL fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import generate_synthetic_episodic_benchmark, save_episodic_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=25, help="Number of synthetic stories to generate.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic generation seed.")
    parser.add_argument("--distractors", type=int, default=1, help="Same-entity distractor events per story.")
    parser.add_argument(
        "--events-out",
        type=Path,
        default=Path("benchmarks/synthetic_episodic_events.jsonl"),
        help="Output JSONL path for event records.",
    )
    parser.add_argument(
        "--cases-out",
        type=Path,
        default=Path("benchmarks/synthetic_episodic_cases.jsonl"),
        help="Output JSONL path for recall cases.",
    )
    args = parser.parse_args()

    events, cases = generate_synthetic_episodic_benchmark(
        stories=args.stories,
        seed=args.seed,
        distractors_per_story=args.distractors,
    )
    save_episodic_benchmark(
        events,
        cases,
        events_path=args.events_out,
        cases_path=args.cases_out,
    )

    print("Exported synthetic episodic benchmark")
    print(f"- stories={args.stories}")
    print(f"- seed={args.seed}")
    print(f"- distractors_per_story={args.distractors}")
    print(f"- events={len(events)} -> {args.events_out}")
    print(f"- cases={len(cases)} -> {args.cases_out}")


if __name__ == "__main__":
    main()
