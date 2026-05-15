"""Run MELM Guard support/refunds benchmark."""

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
from melm.guard import evaluate_guard_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default="reports/melm_guard_benchmark.json")
    parser.add_argument("--out-md", default="reports/melm_guard_benchmark.md")
    args = parser.parse_args()

    fixture = support_refund_fixture()
    report = evaluate_guard_benchmark(
        fixture.facts,
        fixture.rules,
        fixture.guard_cases,
        current_time=fixture.current_time,
    )
    payload = {
        "source": "support_refund_fixture",
        "current_time": fixture.current_time,
        "report": asdict(report),
    }
    _write_outputs(payload, Path(args.out_json), Path(args.out_md))

    print("MELM Guard benchmark")
    print(f"- cases={report.cases}")
    print(f"- gate_passed={report.gate_passed}")
    print(f"- melm_accuracy={report.melm_accuracy:.2%}")
    print(f"- schema_only_accuracy={report.schema_only_accuracy:.2%}")
    print(f"- melm_false_allow_rate={report.melm_false_allow_rate:.2%}")
    print(f"- schema_false_allow_rate={report.schema_only_false_allow_rate:.2%}")
    print(f"- false_allow_reduction_vs_schema={report.false_allow_reduction_vs_schema:.2%}")
    print(f"- valid_action_allow_rate={report.valid_action_allow_rate:.2%}")
    print(f"- traceability={report.traceability:.2%}")


def _write_outputs(payload: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict) -> str:
    report = payload["report"]
    lines = [
        "# MELM Guard Support/Refunds Benchmark",
        "",
        f"Source: `{payload['source']}`",
        f"Current time: `{payload['current_time']}`",
        "",
        f"- Gate passed: `{report['gate_passed']}`",
        f"- Cases: `{report['cases']}`",
        f"- MELM accuracy: `{report['melm_accuracy']:.2%}`",
        f"- Schema-only accuracy: `{report['schema_only_accuracy']:.2%}`",
        f"- Prompt-only accuracy: `{report['prompt_only_accuracy']:.2%}`",
        f"- MELM false-allow rate: `{report['melm_false_allow_rate']:.2%}`",
        f"- Schema-only false-allow rate: `{report['schema_only_false_allow_rate']:.2%}`",
        f"- False-allow reduction vs schema: `{report['false_allow_reduction_vs_schema']:.2%}`",
        f"- Valid-action allow rate: `{report['valid_action_allow_rate']:.2%}`",
        f"- Traceability: `{report['traceability']:.2%}`",
        "",
        "| Action | Category | Expected | Schema | Prompt | MELM | Rules/Missing |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for prediction in report["predictions"]:
        decision = prediction["melm_decision"]
        trace = ", ".join(decision["triggered_rule_ids"] or decision["missing_facts"])
        lines.append(
            f"| {prediction['action_id']} | {prediction['category']} | "
            f"{prediction['expected_status']} | {prediction['schema_only_status']} | "
            f"{prediction['prompt_only_status']} | {prediction['melm_status']} | {trace} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this benchmark tests whether explicit procedural "
            "working memory blocks invalid support actions while preserving valid "
            "refund approvals. Non-allow MELM decisions must cite a rule or "
            "missing fact to count as traceable.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
