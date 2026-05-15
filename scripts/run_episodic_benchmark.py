"""Run synthetic episodic memory vs RAG benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import generate_synthetic_episodic_benchmark, load_episodic_benchmark
from melm.evaluation import memory_gate
from melm.memory import EventMemory, evaluate_memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=25, help="Number of synthetic stories to generate.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic generation seed.")
    parser.add_argument("--k", type=int, default=2, help="Recall@k.")
    parser.add_argument("--distractors", type=int, default=1, help="Same-entity distractor events per story.")
    parser.add_argument("--events", type=Path, help="Optional JSONL event fixture path.")
    parser.add_argument("--cases", type=Path, help="Optional JSONL recall-case fixture path.")
    args = parser.parse_args()

    if bool(args.events) != bool(args.cases):
        parser.error("--events and --cases must be supplied together")

    if args.events and args.cases:
        events, cases = load_episodic_benchmark(events_path=args.events, cases_path=args.cases)
        source = f"fixtures events={args.events} cases={args.cases}"
    else:
        events, cases = generate_synthetic_episodic_benchmark(
            stories=args.stories,
            seed=args.seed,
            distractors_per_story=args.distractors,
        )
        source = f"generated stories={args.stories} seed={args.seed}"

    memory = EventMemory(events)
    comparison = evaluate_memory(memory, cases, k=args.k)
    gate = memory_gate(comparison.event_memory_recall_at_k, comparison.rag_recall_at_k)

    print("Synthetic episodic benchmark")
    print(f"- source={source}")
    if not args.events:
        print(f"- stories={args.stories}")
        print(f"- distractors_per_story={args.distractors}")
    print(f"- events={len(events)}")
    print(f"- cases={len(cases)}")
    print(f"- recall@{args.k} RAG={comparison.rag_recall_at_k:.2%}")
    print(f"- recall@{args.k} event_memory={comparison.event_memory_recall_at_k:.2%}")
    print(f"- absolute_gain={comparison.absolute_gain:.2%}")
    print(f"- mrr@{args.k} RAG={comparison.rag_mrr_at_k:.2%}")
    print(f"- mrr@{args.k} event_memory={comparison.event_memory_mrr_at_k:.2%}")
    print(f"- mrr_gain={comparison.mrr_gain:.2%}")
    print(f"- gate_passed={gate.passed}")

    if comparison.by_category:
        print("\nBy category")
        for category, result in comparison.by_category.items():
            print(
                f"- {category}: RAG={result.rag_recall_at_k:.2%}, "
                f"event_memory={result.event_memory_recall_at_k:.2%}, "
                f"gain={result.absolute_gain:.2%}, "
                f"mrr_gain={result.mrr_gain:.2%}, cases={result.cases}"
            )


if __name__ == "__main__":
    main()
