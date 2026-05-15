"""Run MELM Memory OS support/refunds benchmark."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import support_refund_fixture
from melm.memory import SupportMemoryOS, evaluate_memory_os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--out-json", default="reports/melm_memory_os_benchmark.json")
    parser.add_argument("--out-md", default="reports/melm_memory_os_benchmark.md")
    args = parser.parse_args()

    fixture = support_refund_fixture()
    os_memory = SupportMemoryOS(fixture.events)
    report = evaluate_memory_os(os_memory, fixture.memory_cases, k=args.k)
    payload = {
        "source": "support_refund_fixture",
        "k": args.k,
        "events": len(fixture.events),
        "report": asdict(report),
    }
    _write_outputs(payload, Path(args.out_json), Path(args.out_md))

    print("MELM Memory OS benchmark")
    print(f"- events={len(fixture.events)}")
    print(f"- cases={report.cases}")
    print(f"- gate_passed={report.gate_passed}")
    print(f"- vector_accuracy={report.vector_accuracy:.2%}")
    print(f"- temporal_entity_accuracy={report.temporal_entity_accuracy:.2%}")
    print(f"- memory_os_accuracy={report.memory_os_accuracy:.2%}")
    print(f"- memory_os_gain_vs_vector={report.memory_os_gain_vs_vector:.2%}")
    print(f"- positive_recall={report.positive_recall:.2%}")
    print(f"- negative_abstention={report.negative_abstention:.2%}")


def _write_outputs(payload: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict) -> str:
    report = payload["report"]
    lines = [
        "# MELM Memory OS Support/Refunds Benchmark",
        "",
        f"Source: `{payload['source']}`",
        f"Events: `{payload['events']}`",
        f"k: `{payload['k']}`",
        "",
        f"- Gate passed: `{report['gate_passed']}`",
        f"- Cases: `{report['cases']}`",
        f"- Vector RAG accuracy: `{report['vector_accuracy']:.2%}`",
        f"- Temporal/entity RAG accuracy: `{report['temporal_entity_accuracy']:.2%}`",
        f"- Memory OS accuracy: `{report['memory_os_accuracy']:.2%}`",
        f"- Memory OS gain vs vector: `{report['memory_os_gain_vs_vector']:.2%}`",
        f"- Positive recall: `{report['positive_recall']:.2%}`",
        f"- Negative abstention: `{report['negative_abstention']:.2%}`",
        "",
        "| Query | Category | Expected | Vector | Temporal/entity | Memory OS | Evidence |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for prediction in report["predictions"]:
        expected = prediction["expected"] if prediction["expected"] is not None else "ABSTAIN"
        evidence = prediction["memory_os_event_id"] or ""
        lines.append(
            f"| {prediction['query']} | {prediction['category']} | {expected} | "
            f"{prediction['vector_correct']} | {prediction['temporal_entity_correct']} | "
            f"{prediction['memory_os_correct']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: Memory OS must improve over vector RAG by resolving "
            "latest state from event history and abstaining on unseen order facts.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
