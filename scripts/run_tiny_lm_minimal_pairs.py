"""Evaluate saved tiny LM artifacts on child-level minimal pairs."""

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

from melm.benchmarks import child_language_minimal_pairs_fixture
from melm.evaluation import evaluate_minimal_pair_scores
from melm.tokenization import load_tokenizer_artifact
from melm.training import score_tiny_lm_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/tiny_lm")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--score-field",
        choices=("mean_nll_per_token", "total_nll", "bits_per_byte"),
        default="mean_nll_per_token",
    )
    parser.add_argument("--out-json", default="reports/tiny_lm_minimal_pairs.json")
    parser.add_argument("--out-md", default="reports/tiny_lm_minimal_pairs.md")
    args = parser.parse_args()

    cases = child_language_minimal_pairs_fixture()
    texts = sorted({text for case in cases for text in (case.good, case.bad)})
    runs = []
    root = Path(args.root)
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokenizer = load_tokenizer_artifact(run_manifest["tokenizer_metadata"])
        text_scores = score_tiny_lm_texts(
            texts,
            tokenizer,
            run_manifest["checkpoint_dir"],
            batch_size=args.batch_size,
            device=args.device,
        )
        score_by_text = {
            score.text: float(getattr(score, args.score_field)) for score in text_scores
        }
        report = evaluate_minimal_pair_scores(cases, score_by_text)
        runs.append(
            {
                "name": manifest_path.parent.name,
                "manifest": str(manifest_path),
                "tokenizer": tokenizer.name,
                "checkpoint_dir": run_manifest["checkpoint_dir"],
                "score_field": args.score_field,
                "text_scores": [_to_jsonable(score) for score in text_scores],
                "report": _to_jsonable(report),
            }
        )

    runs.sort(key=lambda item: item["report"]["accuracy"], reverse=True)
    payload = {
        "root": str(root),
        "score_field": args.score_field,
        "cases": [_to_jsonable(case) for case in cases],
        "runs": runs,
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Tiny LM minimal pairs")
    print(f"- root={root}")
    print(f"- cases={len(cases)}")
    print(f"- score_field={args.score_field}")
    for rank, run in enumerate(runs, start=1):
        report = run["report"]
        print(
            f"- #{rank} {run['name']}: tokenizer={run['tokenizer']}, "
            f"accuracy={report['accuracy']:.2%}"
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
    lines = [
        "# Tiny LM Minimal Pairs",
        "",
        f"Artifact root: `{payload['root']}`",
        f"Score field: `{payload['score_field']}`",
        f"Cases: `{len(payload['cases'])}`",
        "",
        "| Rank | Run | Tokenizer | Accuracy | Correct | Cases |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, run in enumerate(payload["runs"], start=1):
        report = run["report"]
        lines.append(
            f"| {rank} | {run['name']} | {run['tokenizer']} | "
            f"{report['accuracy']:.2%} | {report['correct']} | {report['cases']} |"
        )
    lines.extend(["", "## Cases", ""])
    for run in payload["runs"]:
        lines.extend(
            [
                f"### {run['name']}",
                "",
                "| Case | Category | Correct? | Margin |",
                "|---|---|---:|---:|",
            ]
        )
        for prediction in run["report"]["predictions"]:
            lines.append(
                f"| {prediction['case_id']} | {prediction['category']} | "
                f"{str(prediction['chose_good']).lower()} | {prediction['margin']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
