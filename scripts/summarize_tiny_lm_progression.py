"""Summarize tiny LM tokenizer ablation trends across step counts."""

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

from melm.training import summarize_progression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        nargs="+",
        default=[
            "reports/babylm_2026_multiseed_tiny_lm_ablation.json",
            "reports/babylm_2026_multiseed_tiny_lm_ablation_50step.json",
            "reports/babylm_2026_multiseed_tiny_lm_ablation_100step.json",
        ],
    )
    parser.add_argument("--out-json", default="reports/babylm_2026_tiny_lm_progression.json")
    parser.add_argument("--out-md", default="reports/babylm_2026_tiny_lm_progression.md")
    args = parser.parse_args()

    payloads = [
        json.loads(Path(report).read_text(encoding="utf-8"))
        for report in args.reports
    ]
    progressions = summarize_progression(payloads)
    output = {
        "source_reports": args.reports,
        "progressions": [_to_jsonable(progression) for progression in progressions],
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(output), encoding="utf-8")

    print("Tiny LM progression")
    for rank, progression in enumerate(progressions, start=1):
        print(
            f"- #{rank} {progression.tokenizer}: "
            f"{progression.first_bits_per_byte:.3f} -> {progression.last_bits_per_byte:.3f}, "
            f"improvement={progression.relative_improvement:.2%}"
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
    return value


def _markdown(payload: dict[str, Any]) -> str:
    all_steps = sorted(
        {
            point["steps"]
            for progression in payload["progressions"]
            for point in progression["points"]
        }
    )
    lines = [
        "# Tiny LM Progression",
        "",
        "Mean bits/byte across multi-seed matched-parameter tiny LM ablations.",
        "",
        "| Rank | Tokenizer | "
        + " | ".join(f"{step} steps" for step in all_steps)
        + " | Relative Improvement |",
        "|---:|---|"
        + "|".join("---:" for _step in all_steps)
        + "|---:|",
    ]
    for rank, progression in enumerate(payload["progressions"], start=1):
        by_step = {
            point["steps"]: point
            for point in progression["points"]
        }
        values = [
            f"{by_step[step]['mean_bits_per_byte']:.3f}"
            if step in by_step
            else ""
            for step in all_steps
        ]
        lines.append(
            f"| {rank} | {progression['tokenizer']} | "
            + " | ".join(values)
            + f" | {progression['relative_improvement']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
