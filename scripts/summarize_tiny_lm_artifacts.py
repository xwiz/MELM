"""Summarize local tiny LM artifact directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/tiny_lm")
    parser.add_argument("--out-json", default="reports/tiny_lm_artifact_index.json")
    parser.add_argument("--out-md", default="reports/tiny_lm_artifact_index.md")
    args = parser.parse_args()

    root = Path(args.root)
    runs = []
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = manifest["report"]
        runs.append(
            {
                "name": manifest_path.parent.name,
                "manifest": str(manifest_path),
                "tokenizer": report["tokenizer"],
                "parameters": report["parameters"],
                "steps": report["steps"],
                "validation_bits_per_byte": report["validation_bits_per_byte"],
                "checkpoint_dir": manifest["checkpoint_dir"],
                "tokenizer_metadata": manifest["tokenizer_metadata"],
            }
        )
    runs.sort(key=lambda item: item["validation_bits_per_byte"])
    payload = {
        "root": str(root),
        "runs": runs,
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Tiny LM artifact index")
    print(f"- root={root}")
    print(f"- runs={len(runs)}")
    for rank, run in enumerate(runs, start=1):
        print(
            f"- #{rank} {run['name']}: tokenizer={run['tokenizer']}, "
            f"bpb={run['validation_bits_per_byte']:.3f}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _markdown(payload: dict) -> str:
    lines = [
        "# Tiny LM Artifact Index",
        "",
        f"Artifact root: `{payload['root']}`",
        "",
        "`artifacts/` is intentionally gitignored; this report records local outputs.",
        "",
        "| Rank | Run | Tokenizer | Steps | Params | Bits/Byte | Checkpoint |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for rank, run in enumerate(payload["runs"], start=1):
        lines.append(
            f"| {rank} | {run['name']} | {run['tokenizer']} | {run['steps']} | "
            f"{run['parameters']} | {run['validation_bits_per_byte']:.3f} | "
            f"`{run['checkpoint_dir']}` |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
