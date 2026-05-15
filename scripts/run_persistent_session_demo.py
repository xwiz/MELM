"""Run the reloadable persistent dialogue session demo."""

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

from melm.benchmarks import authored_child_dialogue_fixture
from melm.demo import PersistentDialogueDemo, PersistentDialogueSession, evaluate_dialogue_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-path",
        default="reports/persistent_dialogue_session_events.jsonl",
    )
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--out-json", default="reports/persistent_dialogue_session_demo.json")
    parser.add_argument("--out-md", default="reports/persistent_dialogue_session_demo.md")
    args = parser.parse_args()

    session_path = Path(args.session_path)
    if args.reset and session_path.exists():
        session_path.unlink()

    authored_events, _recall_cases, evidence_cases = authored_child_dialogue_fixture()
    session = PersistentDialogueSession(
        session_path,
        threshold=args.threshold,
        k=args.k,
        confidence_method="score_with_evidence_veto",
    )
    seeded = False
    if not session.events():
        session.replace_events(authored_events)
        seeded = True

    reloaded = PersistentDialogueSession(
        session_path,
        threshold=args.threshold,
        k=args.k,
        confidence_method="score_with_evidence_veto",
    )
    demo = PersistentDialogueDemo(
        reloaded.events(),
        threshold=args.threshold,
        k=args.k,
        confidence_method="score_with_evidence_veto",
    )
    report = evaluate_dialogue_demo(demo, evidence_cases)
    payload = {
        "source": "jsonl_persistent_session_seeded_from_authored_child_dialogue",
        "session_path": str(session_path),
        "seeded": seeded,
        "threshold": args.threshold,
        "k": args.k,
        "events": len(reloaded.events()),
        "report": _to_jsonable(report),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Persistent session demo")
    print(f"- session_path={session_path}")
    print(f"- seeded={seeded}")
    print(f"- events={len(reloaded.events())}")
    print(f"- cases={report.cases}")
    print(f"- accuracy={report.accuracy:.2%}")
    print(f"- positive_recall={report.positive_recall:.2%}")
    print(f"- negative_abstention={report.negative_abstention:.2%}")
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
    report = payload["report"]
    lines = [
        "# Persistent Session Demo",
        "",
        f"Source: `{payload['source']}`",
        f"Session path: `{payload['session_path']}`",
        f"Seeded: `{payload['seeded']}`",
        f"Events after reload: `{payload['events']}`",
        f"Threshold: `{payload['threshold']}`",
        f"k: `{payload['k']}`",
        "",
        f"- Accuracy: `{report['accuracy']:.2%}`",
        f"- Positive recall: `{report['positive_recall']:.2%}`",
        f"- Negative abstention: `{report['negative_abstention']:.2%}`",
        f"- Answer rate: `{report['answer_rate']:.2%}`",
        "",
        "| Query | Status | Confidence | Evidence | Answer |",
        "|---|---|---:|---|---|",
    ]
    for response in report["responses"]:
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
