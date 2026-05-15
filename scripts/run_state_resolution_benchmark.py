"""Run the synthetic state-resolution benchmark."""

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

from melm.benchmarks import synthetic_state_resolution_fixture
from melm.memory import evaluate_state_resolution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=25, help="Synthetic story count.")
    parser.add_argument("--out-json", help="Optional JSON output path.")
    args = parser.parse_args()

    events, cases = synthetic_state_resolution_fixture(stories=args.stories)
    report = evaluate_state_resolution(events, cases)

    print("State-resolution benchmark")
    print(f"- events={len(events)}")
    print(f"- cases={report.cases}")
    print(f"- accuracy={report.accuracy:.2%}")
    print(f"- answer_rate={report.answer_rate:.2%}")
    print(f"- false_positive_rate={report.false_positive_rate:.2%}")
    print("- by_category:")
    for category, category_report in (report.by_category or {}).items():
        print(
            f"  - {category}: accuracy={category_report.accuracy:.2%}, "
            f"cases={category_report.cases}"
        )

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
        print(f"Wrote {path}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
