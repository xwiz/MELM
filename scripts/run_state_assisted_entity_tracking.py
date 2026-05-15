"""Evaluate LM entity tracking with explicit state-memory assistance."""

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

from melm.benchmarks import load_entity_tracking_fast_cases
from melm.evaluation import evaluate_state_assisted_entity_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-report", default="reports/babylm_2026_small_model_stage_entity_tracking.json")
    parser.add_argument(
        "--data-dir",
        default="local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/entity_tracking_fast",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-cases-per-file", type=int)
    parser.add_argument("--out-json", default="reports/babylm_2026_state_assisted_entity_tracking.json")
    parser.add_argument("--out-md", default="reports/babylm_2026_state_assisted_entity_tracking.md")
    args = parser.parse_args()

    cases = load_entity_tracking_fast_cases(
        args.data_dir,
        max_files=args.max_files,
        max_cases_per_file=args.max_cases_per_file,
    )
    entity_payload = json.loads(Path(args.entity_report).read_text(encoding="utf-8"))
    runs = []
    for run in entity_payload["runs"]:
        report = evaluate_state_assisted_entity_tracking(
            cases,
            run["report"]["predictions"],
        )
        runs.append(
            {
                "name": run["name"],
                "tokenizer": run["tokenizer"],
                "lm_score_field": run.get("score_field"),
                "lm_accuracy": run["report"]["accuracy"],
                "report": _to_jsonable(report),
            }
        )
    runs.sort(key=lambda item: item["report"]["accuracy"], reverse=True)
    payload = {
        "entity_report": args.entity_report,
        "data_dir": args.data_dir,
        "cases": len(cases),
        "runs": runs,
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("State-assisted entity tracking")
    print(f"- entity_report={args.entity_report}")
    print(f"- cases={len(cases)}")
    for rank, run in enumerate(runs, start=1):
        report = run["report"]
        print(
            f"- #{rank} {run['name']}: tokenizer={run['tokenizer']}, "
            f"lm={report['lm_accuracy']:.2%}, assisted={report['accuracy']:.2%}, "
            f"lift={report['accuracy_lift']:.2%}, state_answer_rate={report['state_answer_rate']:.2%}"
        )
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
    lines = [
        "# State-Assisted BabyLM Entity Tracking",
        "",
        f"LM entity report: `{payload['entity_report']}`",
        f"Data dir: `{payload['data_dir']}`",
        f"Cases: `{payload['cases']}`",
        "",
        "State memory gets first refusal. If the state parser cannot resolve a case, the evaluator falls back to the LM prediction.",
        "",
        "| Rank | Run | Tokenizer | LM Accuracy | Assisted Accuracy | Lift | State Answer Rate | LM Fallbacks |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, run in enumerate(payload["runs"], start=1):
        report = run["report"]
        lines.append(
            f"| {rank} | {run['name']} | {run['tokenizer']} | "
            f"{report['lm_accuracy']:.2%} | {report['accuracy']:.2%} | "
            f"{report['accuracy_lift']:.2%} | {report['state_answer_rate']:.2%} | "
            f"{report['lm_fallbacks']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
