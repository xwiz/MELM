"""Summarize a Phase 1 report as pass/fail validation checks."""

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

from melm.evaluation import evaluate_validation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/phase1_report.json", help="Phase 1 JSON report path.")
    parser.add_argument("--out-json", default="reports/validation_suite.json", help="Validation suite JSON output.")
    parser.add_argument("--out-md", default="reports/validation_suite.md", help="Validation suite Markdown output.")
    parser.add_argument("--no-fail", action="store_true", help="Do not exit non-zero when hard gates fail.")
    args = parser.parse_args()

    report_path = Path(args.report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    suite = evaluate_validation_suite(payload)
    suite_payload = _to_jsonable(suite)

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(suite_payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(suite_payload, source=report_path), encoding="utf-8")

    print("MELM validation suite")
    print(f"- source={report_path}")
    print(f"- overall_passed={suite.overall_passed}")
    print(f"- hard_failures={suite.hard_failures}")
    for check in suite.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"- {status} [{check.severity}] {check.name}: {check.detail}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if suite.hard_failures and not args.no_fail:
        raise SystemExit(1)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _markdown(suite: dict[str, Any], *, source: Path) -> str:
    lines = [
        "# MELM Validation Suite",
        "",
        f"Source report: `{source}`",
        "",
        f"Overall passed: `{suite['overall_passed']}`",
        f"Hard failures: `{suite['hard_failures']}`",
        "",
        "| Check | Severity | Status | Metric | Threshold | Detail |",
        "|---|---|---:|---:|---:|---|",
    ]
    for check in suite["checks"]:
        metric = "" if check["metric"] is None else f"{check['metric']:.3f}"
        threshold = "" if check["threshold"] is None else f"{check['threshold']:.3f}"
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"| {check['name']} | {check['severity']} | {status} | "
            f"{metric} | {threshold} | {check['detail']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
