"""Evaluate saved tiny LM artifacts on local BabyLM 2026 fast BLiMP cases."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import load_blimp_fast_cases
from melm.evaluation import evaluate_minimal_pair_scores
from melm.tokenization import load_tokenizer_artifact
from melm.training import score_tiny_lm_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast",
    )
    parser.add_argument("--root", default="artifacts/tiny_lm")
    parser.add_argument(
        "--tokenizers",
        help="Optional comma-separated tokenizer names to include.",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-cases-per-file", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--score-field",
        choices=("mean_nll_per_token", "total_nll", "bits_per_byte"),
        default="mean_nll_per_token",
    )
    parser.add_argument("--out-json", default="reports/tiny_lm_blimp_fast.json")
    parser.add_argument("--out-md", default="reports/tiny_lm_blimp_fast.md")
    args = parser.parse_args()

    cases = load_blimp_fast_cases(
        args.data_dir,
        max_files=args.max_files,
        max_cases_per_file=args.max_cases_per_file,
    )
    texts = sorted({text for case in cases for text in (case.good, case.bad)})
    runs = []
    root = Path(args.root)
    requested = _requested_tokenizers(args.tokenizers)
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokenizer = load_tokenizer_artifact(run_manifest["tokenizer_metadata"])
        if requested is not None and tokenizer.name not in requested:
            continue
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
                "report": _to_jsonable(report),
            }
        )
        _release_torch_memory()

    runs.sort(key=lambda item: item["report"]["accuracy"], reverse=True)
    payload = {
        "data_dir": args.data_dir,
        "root": str(root),
        "score_field": args.score_field,
        "max_files": args.max_files,
        "max_cases_per_file": args.max_cases_per_file,
        "cases": len(cases),
        "unique_texts": len(texts),
        "runs": runs,
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Tiny LM BabyLM fast BLiMP")
    print(f"- data_dir={args.data_dir}")
    print(f"- cases={len(cases)}")
    print(f"- unique_texts={len(texts)}")
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


def _requested_tokenizers(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tiny LM BabyLM Fast BLiMP",
        "",
        f"Data dir: `{payload['data_dir']}`",
        f"Artifact root: `{payload['root']}`",
        f"Score field: `{payload['score_field']}`",
        f"Cases: `{payload['cases']}`",
        f"Unique texts: `{payload['unique_texts']}`",
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
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
