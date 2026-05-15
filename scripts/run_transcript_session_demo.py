"""Run a transcript-derived persistent dialogue session demo."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import (
    load_annotated_transcript_benchmark,
    sample_transcript_distractor_events,
    sample_transcript_noisy_evidence_cases,
)
from melm.demo import PersistentDialogueDemo, PersistentDialogueSession, evaluate_dialogue_demo
from melm.memory import EventMemory, evaluate_memory, evaluate_state_resolution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("benchmarks/sample_transcript_annotations.jsonl"),
    )
    parser.add_argument(
        "--session-path",
        type=Path,
        default=Path("reports/sample_transcript_session_events.jsonl"),
    )
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--include-sample-distractors", action="store_true")
    parser.add_argument("--include-sample-noisy-cases", action="store_true")
    parser.add_argument("--out-json", default="reports/transcript_session_demo.json")
    parser.add_argument("--out-md", default="reports/transcript_session_demo.md")
    args = parser.parse_args()

    if args.reset and args.session_path.exists():
        args.session_path.unlink()

    benchmark = load_annotated_transcript_benchmark(args.annotations)
    session = PersistentDialogueSession(
        args.session_path,
        threshold=args.threshold,
        k=args.k,
        confidence_method="score_with_evidence_veto",
    )
    seeded = False
    if not session.events():
        seed_events = list(benchmark.events)
        if args.include_sample_distractors:
            seed_events.extend(sample_transcript_distractor_events())
        session.replace_events(seed_events)
        seeded = True

    reloaded = PersistentDialogueSession(
        args.session_path,
        threshold=args.threshold,
        k=args.k,
        confidence_method="score_with_evidence_veto",
    )
    events = list(reloaded.events())
    demo = PersistentDialogueDemo(
        events,
        threshold=args.threshold,
        k=args.k,
        confidence_method="score_with_evidence_veto",
    )
    dialogue_report = evaluate_dialogue_demo(demo, benchmark.evidence_cases)
    noisy_dialogue_report = None
    if args.include_sample_noisy_cases:
        noisy_dialogue_report = evaluate_dialogue_demo(
            demo,
            sample_transcript_noisy_evidence_cases(),
        )
    memory_report = evaluate_memory(EventMemory(events), benchmark.recall_cases, k=args.k)
    state_report = evaluate_state_resolution(events, benchmark.state_cases)
    payload = {
        "source": "annotated_transcript_persistent_session",
        "annotations": str(args.annotations),
        "session_path": str(args.session_path),
        "seeded": seeded,
        "threshold": args.threshold,
        "k": args.k,
        "turns": len(benchmark.turns),
        "events": len(events),
        "distractor_events": len(sample_transcript_distractor_events())
        if args.include_sample_distractors
        else 0,
        "recall_cases": len(benchmark.recall_cases),
        "evidence_cases": len(benchmark.evidence_cases),
        "state_cases": len(benchmark.state_cases),
        "dialogue_report": _to_jsonable(dialogue_report),
        "noisy_dialogue_report": _to_jsonable(noisy_dialogue_report)
        if noisy_dialogue_report is not None
        else None,
        "memory_report": _to_jsonable(memory_report),
        "state_report": _to_jsonable(state_report),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Transcript session demo")
    print(f"- annotations={args.annotations}")
    print(f"- session_path={args.session_path}")
    print(f"- seeded={seeded}")
    print(f"- turns={len(benchmark.turns)}")
    print(f"- events={len(events)}")
    if args.include_sample_distractors:
        print(f"- distractor_events={len(sample_transcript_distractor_events())}")
    print(f"- dialogue_accuracy={dialogue_report.accuracy:.2%}")
    print(f"- dialogue_positive_recall={dialogue_report.positive_recall:.2%}")
    print(f"- dialogue_negative_abstention={dialogue_report.negative_abstention:.2%}")
    if noisy_dialogue_report is not None:
        print(f"- noisy_dialogue_accuracy={noisy_dialogue_report.accuracy:.2%}")
        print(f"- noisy_positive_recall={noisy_dialogue_report.positive_recall:.2%}")
        print(f"- noisy_negative_abstention={noisy_dialogue_report.negative_abstention:.2%}")
    print(f"- memory_event_recall@{args.k}={memory_report.event_memory_recall_at_k:.2%}")
    print(f"- memory_rag_recall@{args.k}={memory_report.rag_recall_at_k:.2%}")
    print(f"- state_accuracy={state_report.accuracy:.2%}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _markdown(payload: dict[str, Any]) -> str:
    dialogue = payload["dialogue_report"]
    noisy_dialogue = payload.get("noisy_dialogue_report")
    memory = payload["memory_report"]
    state = payload["state_report"]
    lines = [
        "# Transcript Session Demo",
        "",
        f"Source: `{payload['source']}`",
        f"Annotations: `{payload['annotations']}`",
        f"Session path: `{payload['session_path']}`",
        f"Seeded: `{payload['seeded']}`",
        f"Turns: `{payload['turns']}`",
        f"Events after reload: `{payload['events']}`",
        f"Distractor events: `{payload['distractor_events']}`",
        f"Threshold: `{payload['threshold']}`",
        f"k: `{payload['k']}`",
        "",
        "## Dialogue Evidence",
        "",
        f"- Accuracy: `{dialogue['accuracy']:.2%}`",
        f"- Positive recall: `{dialogue['positive_recall']:.2%}`",
        f"- Negative abstention: `{dialogue['negative_abstention']:.2%}`",
        f"- Answer rate: `{dialogue['answer_rate']:.2%}`",
        "",
    ]
    if noisy_dialogue is not None:
        lines.extend(
            [
                "## Noisy Dialogue Evidence",
                "",
                f"- Accuracy: `{noisy_dialogue['accuracy']:.2%}`",
                f"- Positive recall: `{noisy_dialogue['positive_recall']:.2%}`",
                f"- Negative abstention: `{noisy_dialogue['negative_abstention']:.2%}`",
                f"- Answer rate: `{noisy_dialogue['answer_rate']:.2%}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Memory Retrieval",
        "",
        f"- RAG recall@k: `{memory['rag_recall_at_k']:.2%}`",
        f"- Event-memory recall@k: `{memory['event_memory_recall_at_k']:.2%}`",
        f"- Absolute gain: `{memory['absolute_gain']:.2%}`",
        f"- Event-memory MRR@k: `{memory['event_memory_mrr_at_k']:.2%}`",
        "",
        "## State Resolution",
        "",
        f"- Accuracy: `{state['accuracy']:.2%}`",
        f"- Answer rate: `{state['answer_rate']:.2%}`",
        f"- False positive rate: `{state['false_positive_rate']:.2%}`",
        "",
        "| Query | Status | Confidence | Evidence | Answer |",
        "|---|---|---:|---|---|",
        ]
    )
    for response in dialogue["responses"]:
        evidence = ", ".join(response["evidence_event_ids"])
        answer = response["answer"].replace("|", "\\|")
        query = response["query"].replace("|", "\\|")
        lines.append(
            f"| {query} | {response['status']} | {response['confidence']:.3f} | "
            f"{evidence} | {answer} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
