"""Run answerability and abstention probes for the event-memory baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import generate_synthetic_evidence_benchmark
from melm.memory import (
    EventMemory,
    best_abstention_report,
    calibrate_abstention_threshold,
    split_evidence_cases,
    sweep_abstention_thresholds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=25, help="Number of synthetic stories to generate.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic generation seed.")
    parser.add_argument("--k", type=int, default=2, help="Retrieved contexts to consider as evidence.")
    parser.add_argument("--distractors", type=int, default=1, help="Same-entity distractor events per story.")
    args = parser.parse_args()

    events, cases = generate_synthetic_evidence_benchmark(
        stories=args.stories,
        seed=args.seed,
        distractors_per_story=args.distractors,
    )
    memory = EventMemory(events)

    rag_reports = sweep_abstention_thresholds(memory, cases, k=args.k, retriever="rag")
    event_reports = sweep_abstention_thresholds(memory, cases, k=args.k, retriever="event_memory")
    hybrid_reports = sweep_abstention_thresholds(
        memory,
        cases,
        k=args.k,
        retriever="event_memory",
        confidence_method="score_with_evidence_veto",
    )
    rag_best = best_abstention_report(rag_reports)
    event_best = best_abstention_report(event_reports)
    hybrid_best = best_abstention_report(hybrid_reports)
    calibration_cases, evaluation_cases = split_evidence_cases(cases)
    calibrated_top_score = calibrate_abstention_threshold(
        memory,
        calibration_cases,
        evaluation_cases,
        k=args.k,
        retriever="event_memory",
        confidence_method="top_score",
    )
    calibrated_hybrid = calibrate_abstention_threshold(
        memory,
        calibration_cases,
        evaluation_cases,
        k=args.k,
        retriever="event_memory",
        confidence_method="score_with_evidence_veto",
    )

    print("Synthetic evidence-abstention benchmark")
    print(f"- stories={args.stories}")
    print(f"- events={len(events)}")
    print(f"- cases={len(cases)}")
    print(f"- positives={event_best.positives}")
    print(f"- negatives={event_best.negatives}")
    print("")
    _print_table("RAG", rag_reports)
    print("")
    _print_table("Event memory, top score", event_reports)
    print("")
    _print_table("Event memory, score + evidence veto", hybrid_reports)
    print("")
    print("Best same-set thresholds")
    print(_summary_line("RAG", rag_best))
    print(_summary_line("Event memory top score", event_best))
    print(_summary_line("Event memory score + evidence veto", hybrid_best))
    print("")
    print("Held-out calibrated thresholds")
    print(_calibrated_line("Top score", calibrated_top_score))
    print(_calibrated_line("Score + evidence veto", calibrated_hybrid))


def _print_table(label: str, reports) -> None:
    print(label)
    print("| threshold | accuracy | precision | positive_recall | negative_abstention | false_positive_rate |")
    print("|---:|---:|---:|---:|---:|---:|")
    for report in reports:
        print(
            f"| {report.threshold:.2f} | {report.accuracy:.2%} | "
            f"{report.precision:.2%} | {report.positive_recall:.2%} | "
            f"{report.negative_abstention:.2%} | {report.false_positive_rate:.2%} |"
        )


def _summary_line(label: str, report) -> str:
    return (
        f"- {label}: threshold={report.threshold:.2f}, accuracy={report.accuracy:.2%}, "
        f"precision={report.precision:.2%}, positive_recall={report.positive_recall:.2%}, "
        f"negative_abstention={report.negative_abstention:.2%}"
    )


def _calibrated_line(label: str, run) -> str:
    report = run.evaluation_report
    return (
        f"- {label}: threshold={run.threshold:.2f}, calibration_cases={run.calibration_cases}, "
        f"evaluation_cases={run.evaluation_cases}, eval_accuracy={report.accuracy:.2%}, "
        f"eval_precision={report.precision:.2%}, eval_positive_recall={report.positive_recall:.2%}, "
        f"eval_negative_abstention={report.negative_abstention:.2%}"
    )


if __name__ == "__main__":
    main()
