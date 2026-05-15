"""Reload local tiny LM artifacts and evaluate them on a manifest split."""

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

from melm.data import limit_texts_by_bytes, load_train_validation
from melm.tokenization import load_tokenizer_artifact
from melm.training import evaluate_tiny_lm_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="MELM_whitepaper.md")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--root", default="artifacts/tiny_lm")
    parser.add_argument("--max-validation-bytes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", default="reports/tiny_lm_artifact_evaluation.json")
    parser.add_argument("--out-md", default="reports/tiny_lm_artifact_evaluation.md")
    args = parser.parse_args()

    _texts, _train_texts, validation_texts, source = load_train_validation(
        path=args.path,
        manifest_path=args.manifest,
    )
    validation_texts = limit_texts_by_bytes(validation_texts, args.max_validation_bytes)

    root = Path(args.root)
    runs = []
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokenizer = load_tokenizer_artifact(run_manifest["tokenizer_metadata"])
        evaluation = evaluate_tiny_lm_checkpoint(
            validation_texts,
            tokenizer,
            run_manifest["checkpoint_dir"],
            batch_size=args.batch_size,
            device=args.device,
        )
        training_report = run_manifest.get("report", {})
        evaluation_payload = _to_jsonable(evaluation)
        training_bpb = training_report.get("validation_bits_per_byte")
        bpb_delta = (
            evaluation_payload["validation_bits_per_byte"] - training_bpb
            if training_bpb is not None
            else None
        )
        runs.append(
            {
                "name": manifest_path.parent.name,
                "manifest": str(manifest_path),
                "tokenizer_metadata": run_manifest["tokenizer_metadata"],
                "evaluation": evaluation_payload,
                "training_validation_bits_per_byte": training_bpb,
                "bits_per_byte_delta_from_training_report": bpb_delta,
            }
        )

    runs.sort(key=lambda item: item["evaluation"]["validation_bits_per_byte"])
    payload = {
        "source": source,
        "root": str(root),
        "max_validation_bytes": args.max_validation_bytes,
        "runs": runs,
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Tiny LM artifact evaluation")
    print(f"- source={source}")
    print(f"- root={root}")
    print(f"- runs={len(runs)}")
    for rank, run in enumerate(runs, start=1):
        report = run["evaluation"]
        print(
            f"- #{rank} {run['name']}: tokenizer={report['tokenizer']}, "
            f"bpb={report['validation_bits_per_byte']:.3f}"
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
        "# Tiny LM Artifact Evaluation",
        "",
        f"Source: `{payload['source']}`",
        f"Artifact root: `{payload['root']}`",
        f"Max validation bytes: `{payload['max_validation_bytes']}`",
        "",
        "| Rank | Run | Tokenizer | Steps | Params | Eval Bits/Byte | Delta vs Training Report |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, run in enumerate(payload["runs"], start=1):
        report = run["evaluation"]
        delta = run["bits_per_byte_delta_from_training_report"]
        delta_text = "" if delta is None else f"{delta:.6f}"
        lines.append(
            f"| {rank} | {run['name']} | {report['tokenizer']} | "
            f"{report['checkpoint_steps']} | {report['parameters']} | "
            f"{report['validation_bits_per_byte']:.3f} | {delta_text} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
