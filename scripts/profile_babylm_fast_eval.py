"""Profile local BabyLM 2026 fast evaluation assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import profile_blimp_fast, profile_jsonl_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blimp-dir",
        default="local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast",
    )
    parser.add_argument(
        "--entity-dir",
        default="local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/entity_tracking_fast",
    )
    parser.add_argument("--out-json", default="reports/babylm_2026_fast_eval_profile.json")
    parser.add_argument("--out-md", default="reports/babylm_2026_fast_eval_profile.md")
    args = parser.parse_args()

    blimp = profile_blimp_fast(args.blimp_dir)
    entity_tracking = profile_jsonl_directory(args.entity_dir)
    payload = {
        "blimp_dir": args.blimp_dir,
        "entity_dir": args.entity_dir,
        "blimp_fast": blimp,
        "entity_tracking_fast": entity_tracking,
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("BabyLM 2026 fast eval profile")
    print(f"- blimp_dir={args.blimp_dir}")
    print(f"- blimp_files={blimp['files']}")
    print(f"- blimp_cases={blimp['cases']}")
    print(f"- entity_tracking_files={entity_tracking['files']}")
    print(f"- entity_tracking_cases={entity_tracking['cases']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _markdown(payload: dict) -> str:
    blimp = payload["blimp_fast"]
    entity_tracking = payload["entity_tracking_fast"]
    return "\n".join(
        [
            "# BabyLM 2026 Fast Eval Profile",
            "",
            f"BLiMP fast directory: `{payload['blimp_dir']}`",
            f"Entity-tracking fast directory: `{payload['entity_dir']}`",
            "",
            "| Asset | Files | Cases |",
            "|---|---:|---:|",
            f"| BLiMP fast | {blimp['files']} | {blimp['cases']} |",
            f"| Entity tracking fast | {entity_tracking['files']} | {entity_tracking['cases']} |",
            "",
        ]
    )


if __name__ == "__main__":
    main()
