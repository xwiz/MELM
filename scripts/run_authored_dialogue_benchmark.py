"""Run the hand-authored child-dialogue memory benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import authored_child_dialogue_fixture, load_dialogue_benchmark
from melm.evaluation import abstention_gate, memory_gate
from melm.memory import EventMemory, evaluate_abstention, evaluate_memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2, help="Recall@k / retrieved evidence count.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.25,
        help="Evidence-admission threshold, defaulting to the synthetic calibrated selector.",
    )
    parser.add_argument("--events", type=Path, help="Optional JSONL event fixture path.")
    parser.add_argument("--recall-cases", type=Path, help="Optional JSONL recall-case fixture path.")
    parser.add_argument("--evidence-cases", type=Path, help="Optional JSONL evidence-case fixture path.")
    args = parser.parse_args()

    fixture_paths = [args.events, args.recall_cases, args.evidence_cases]
    if any(fixture_paths) and not all(fixture_paths):
        parser.error("--events, --recall-cases, and --evidence-cases must be supplied together")

    if args.events and args.recall_cases and args.evidence_cases:
        events, recall_cases, evidence_cases = load_dialogue_benchmark(
            events_path=args.events,
            recall_cases_path=args.recall_cases,
            evidence_cases_path=args.evidence_cases,
        )
        source = (
            f"fixtures events={args.events} recall_cases={args.recall_cases} "
            f"evidence_cases={args.evidence_cases}"
        )
    else:
        events, recall_cases, evidence_cases = authored_child_dialogue_fixture()
        source = "built_in_authored_dialogue"

    memory = EventMemory(events)
    recall = evaluate_memory(memory, recall_cases, k=args.k)
    memory_gate_result = memory_gate(recall.event_memory_recall_at_k, recall.rag_recall_at_k)
    abstention = evaluate_abstention(
        memory,
        evidence_cases,
        k=args.k,
        threshold=args.threshold,
        retriever="event_memory",
        confidence_method="score_with_evidence_veto",
    )
    abstention_gate_result = abstention_gate(
        abstention.negative_abstention,
        positive_recall=abstention.positive_recall,
    )

    print("Authored child-dialogue benchmark")
    print(f"- source={source}")
    print(f"- events={len(events)}")
    print(f"- recall_cases={len(recall_cases)}")
    print(f"- evidence_cases={len(evidence_cases)}")
    print(f"- recall@{args.k} RAG={recall.rag_recall_at_k:.2%}")
    print(f"- recall@{args.k} event_memory={recall.event_memory_recall_at_k:.2%}")
    print(f"- absolute_gain={recall.absolute_gain:.2%}")
    print(f"- mrr@{args.k} event_memory={recall.event_memory_mrr_at_k:.2%}")
    print(f"- memory_gate_passed={memory_gate_result.passed}")
    print("")
    print("Evidence admission")
    print(f"- threshold={args.threshold:.2f}")
    print("- method=score_with_evidence_veto")
    print(f"- accuracy={abstention.accuracy:.2%}")
    print(f"- precision={abstention.precision:.2%}")
    print(f"- positive_recall={abstention.positive_recall:.2%}")
    print(f"- negative_abstention={abstention.negative_abstention:.2%}")
    print(f"- abstention_gate_passed={abstention_gate_result.passed}")

    if recall.by_category:
        print("")
        print("Recall by category")
        for category, report in recall.by_category.items():
            print(
                f"- {category}: RAG={report.rag_recall_at_k:.2%}, "
                f"event_memory={report.event_memory_recall_at_k:.2%}, "
                f"gain={report.absolute_gain:.2%}, cases={report.cases}"
            )


if __name__ == "__main__":
    main()
