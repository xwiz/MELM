"""Run a symbolic state-tracking baseline on BabyLM entity tracking."""

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
    evaluate_entity_tracking_symbolic,
    load_entity_tracking_fast_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/entity_tracking_fast",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-cases-per-file", type=int)
    parser.add_argument("--out-json", default="reports/entity_tracking_symbolic.json")
    parser.add_argument("--out-md", default="reports/entity_tracking_symbolic.md")
    args = parser.parse_args()

    cases = load_entity_tracking_fast_cases(
        args.data_dir,
        max_files=args.max_files,
        max_cases_per_file=args.max_cases_per_file,
    )
    report = evaluate_entity_tracking_symbolic(cases)
    payload = {
        "data_dir": args.data_dir,
        "max_files": args.max_files,
        "max_cases_per_file": args.max_cases_per_file,
        "report": _to_jsonable(report),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Symbolic BabyLM entity tracking")
    print(f"- data_dir={args.data_dir}")
    print(f"- cases={report.cases}")
    print(f"- accuracy={report.accuracy:.2%}")
    print(f"- abstentions={report.abstentions}")
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
        "# Symbolic BabyLM Entity Tracking",
        "",
        f"Data dir: `{payload['data_dir']}`",
        f"Cases: `{report['cases']}`",
        f"Accuracy: `{report['accuracy']:.2%}`",
        f"Abstentions: `{report['abstentions']}`",
        "",
        "This is not a language-model score. It is a state-tracking oracle-style baseline that tests whether explicit event/state bookkeeping can solve the task.",
        "",
        "| Category | Accuracy | Correct | Cases | Abstentions |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, category_report in report["by_category"].items():
        lines.append(
            f"| {category} | {category_report['accuracy']:.2%} | "
            f"{category_report['correct']} | {category_report['cases']} | "
            f"{category_report['abstentions']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
