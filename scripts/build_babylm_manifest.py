"""Build a manifest from a local BabyLM-style corpus directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.data import save_manifest, scan_babylm_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Local BabyLM-style corpus file or directory.")
    parser.add_argument("--name", default="babylm_local")
    parser.add_argument("--version", default="local")
    parser.add_argument("--track", default="unknown", help="Track label, e.g. 10M or 100M.")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--out", default="reports/babylm_manifest.json")
    args = parser.parse_args()

    summary = scan_babylm_corpus(
        args.path,
        name=args.name,
        version=args.version,
        track=args.track,
        license_name=args.license,
    )
    save_manifest(summary.manifest, args.out)

    print(f"Wrote {args.out}")
    print(f"Documents: {len(summary.manifest.documents)}")
    print(f"Explicit split documents: {summary.explicit_splits}")
    print(f"Fallback split documents: {summary.fallback_splits}")
    for split in ("train", "validation", "test"):
        count = sum(1 for document in summary.manifest.documents if document.split == split)
        bytes_count = sum(document.bytes for document in summary.manifest.documents if document.split == split)
        print(f"- {split}: {count} docs, {bytes_count} bytes")


if __name__ == "__main__":
    main()
