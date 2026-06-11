"""Freeze a support/refunds blind dataset before scoring it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import (
    build_support_refunds_freeze_manifest,
    support_refunds_freeze_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to the support/refunds JSONL dataset to freeze.")
    parser.add_argument(
        "--preregistration",
        default="benchmarks/support_refunds_external_blind_preregistration.json",
        help="Path to the preregistration JSON.",
    )
    parser.add_argument("--out-json", default=None, help="Freeze manifest JSON path.")
    parser.add_argument("--out-md", default=None, help="Freeze manifest Markdown path.")
    parser.add_argument(
        "--allow-validation-errors",
        action="store_true",
        help="Write a manifest even when schema validation fails.",
    )
    parser.add_argument(
        "--allow-preregistration-errors",
        action="store_true",
        help="Write a manifest even when preregistration coverage checks fail.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest even if it freezes a different hash.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    default_stem = dataset_path.stem
    out_json = Path(args.out_json or f"reports/{default_stem}_freeze_manifest.json")
    out_md = Path(args.out_md or f"reports/{default_stem}_freeze_manifest.md")

    manifest = build_support_refunds_freeze_manifest(dataset_path, args.preregistration)
    _guard_existing_manifest(out_json, manifest, force=args.force)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out_md.write_text(support_refunds_freeze_markdown(manifest), encoding="utf-8")

    print("Support/refunds freeze manifest")
    print(f"- dataset={manifest['dataset_path']}")
    print(f"- dataset_sha256={manifest['dataset_sha256']}")
    print(f"- schema_validation_passed={manifest['schema_validation_passed']}")
    print(f"- preregistration_passed={manifest['preregistration_passed']}")
    print(f"- out_json={out_json}")
    print(f"- out_md={out_md}")

    if manifest["validation_errors"] and not args.allow_validation_errors:
        _print_errors("validation", manifest["validation_errors"])
        raise SystemExit(1)
    if manifest["preregistration_errors"] and not args.allow_preregistration_errors:
        _print_errors("preregistration", manifest["preregistration_errors"])
        raise SystemExit(1)


def _guard_existing_manifest(path: Path, manifest: dict, *, force: bool) -> None:
    if force or not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("dataset_sha256") != manifest.get("dataset_sha256"):
        raise SystemExit(
            f"{path} already freezes a different dataset hash; pass --force to replace it"
        )


def _print_errors(label: str, errors: list[str]) -> None:
    print(f"- {label}_errors:")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    main()
