"""Build a deterministic local corpus manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.data import save_manifest, scan_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="File or directory to scan.")
    parser.add_argument("--name", default="local_corpus")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--source", default="local")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--out", default="reports/corpus_manifest.json")
    parser.add_argument(
        "--include-reports",
        action="store_true",
        help="Include generated report files when scanning a directory.",
    )
    args = parser.parse_args()

    manifest = scan_corpus(
        args.path,
        name=args.name,
        version=args.version,
        source=args.source,
        license_name=args.license,
        exclude_dirs=() if args.include_reports else ("reports", ".git", "__pycache__", ".pytest_cache"),
    )
    save_manifest(manifest, args.out)
    print(f"Wrote {args.out}")
    print(f"Documents: {len(manifest.documents)}")
    for split in ("train", "validation", "test"):
        count = sum(1 for document in manifest.documents if document.split == split)
        print(f"- {split}: {count}")


if __name__ == "__main__":
    main()
