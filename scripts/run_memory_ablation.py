"""Run event-memory component ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import generate_synthetic_episodic_benchmark
from melm.memory import EventMemory, evaluate_memory_variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--distractors", type=int, default=1)
    args = parser.parse_args()

    events, cases = generate_synthetic_episodic_benchmark(
        stories=args.stories,
        seed=args.seed,
        distractors_per_story=args.distractors,
    )
    memory = EventMemory(events)
    variants = evaluate_memory_variants(memory, cases, k=args.k)

    print("Memory ablation")
    print(f"- stories={args.stories}")
    print(f"- events={len(events)}")
    print(f"- cases={len(cases)}")
    for name, result in variants.items():
        print(
            f"- {name}: recall@{args.k}={result.event_memory_recall_at_k:.2%}, "
            f"gain={result.absolute_gain:.2%}, "
            f"mrr@{args.k}={result.event_memory_mrr_at_k:.2%}, "
            f"mrr_gain={result.mrr_gain:.2%}"
        )


if __name__ == "__main__":
    main()
