"""Print per-case retrieval and evidence-admission diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import load_dialogue_benchmark
from melm.memory import EventMemory, decide_evidence, evaluate_abstention, evaluate_memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="JSONL event fixture path.")
    parser.add_argument("--recall-cases", required=True, type=Path, help="JSONL recall-case fixture path.")
    parser.add_argument("--evidence-cases", required=True, type=Path, help="JSONL evidence-case fixture path.")
    parser.add_argument("--k", type=int, default=2, help="Retrieved evidence count.")
    parser.add_argument("--threshold", type=float, default=1.25, help="Evidence-admission threshold.")
    parser.add_argument("--all", action="store_true", help="Show all cases instead of only misses/errors.")
    args = parser.parse_args()

    events, recall_cases, evidence_cases = load_dialogue_benchmark(
        events_path=args.events,
        recall_cases_path=args.recall_cases,
        evidence_cases_path=args.evidence_cases,
    )
    memory = EventMemory(events)
    recall = evaluate_memory(memory, recall_cases, k=args.k)
    abstention = evaluate_abstention(
        memory,
        evidence_cases,
        k=args.k,
        threshold=args.threshold,
        retriever="event_memory",
        confidence_method="score_with_evidence_veto",
    )

    print("Dialogue benchmark analysis")
    print(f"- events={len(events)}")
    print(f"- recall_cases={len(recall_cases)}")
    print(f"- evidence_cases={len(evidence_cases)}")
    print(f"- recall@{args.k} RAG={recall.rag_recall_at_k:.2%}")
    print(f"- recall@{args.k} event_memory={recall.event_memory_recall_at_k:.2%}")
    print(f"- evidence_accuracy={abstention.accuracy:.2%}")
    print(f"- evidence_positive_recall={abstention.positive_recall:.2%}")
    print(f"- evidence_negative_abstention={abstention.negative_abstention:.2%}")

    print("")
    print("Recall cases")
    recall_rows = 0
    for index, case in enumerate(recall_cases, start=1):
        event_results = memory.retrieve_event_memory(case.query, k=args.k)
        event_ids = [result.event.event_id for result in event_results]
        hit = case.expected_event_id in event_ids
        if not args.all and hit:
            continue
        recall_rows += 1
        print(f"- #{index} {case.category}: hit={hit} expected={case.expected_event_id}")
        print(f"  query={case.query}")
        print(f"  event_memory={_format_results(event_results)}")
        print(f"  rag={_format_results(memory.retrieve_rag(case.query, k=args.k))}")

    if recall_rows == 0:
        print("- no recall misses")

    print("")
    print("Evidence cases")
    evidence_rows = 0
    for index, case in enumerate(evidence_cases, start=1):
        decision = decide_evidence(
            memory,
            case.query,
            k=args.k,
            threshold=args.threshold,
            retriever="event_memory",
            confidence_method="score_with_evidence_veto",
        )
        if case.expected_event_id is None:
            correct = not decision.answered
        else:
            correct = case.expected_event_id in set(decision.candidate_event_ids)
        if not args.all and correct:
            continue
        evidence_rows += 1
        print(
            f"- #{index} {case.category}: correct={correct} "
            f"expected={case.expected_event_id} answered={decision.answered}"
        )
        print(f"  query={case.query}")
        print(
            f"  confidence={decision.confidence:.3f} threshold={decision.threshold:.3f} "
            f"candidates={list(decision.candidate_event_ids)}"
        )
        print(f"  event_memory={_format_results(memory.retrieve_event_memory(case.query, k=args.k))}")

    if evidence_rows == 0:
        print("- no evidence-admission errors")


def _format_results(results) -> str:
    parts = [
        f"{result.event.event_id}:{result.score:.3f}:{result.reason}"
        for result in results
    ]
    return "[" + ", ".join(parts) + "]"


if __name__ == "__main__":
    main()
