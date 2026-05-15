"""Download the current BabyLM 2026 Strict-Small local corpus bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, snapshot_download

from melm.data import save_manifest, scan_babylm_corpus


TRAIN_REPO = "BabyLM-community/BabyLM-2026-Strict-Small"
DEV_REPO = "BabyLM-community/BabyLM-dev"
TEST_REPO = "BabyLM-community/BabyLM-Test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="local_data/babylm_2026_strict_small")
    parser.add_argument("--manifest-out", default="reports/babylm_2026_strict_small_manifest.json")
    parser.add_argument("--track", default="2026-strict-small")
    parser.add_argument("--license", default="mit for train; dev/test license not declared on HF")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    downloads = [
        ("train", TRAIN_REPO, ["*.txt", "README.md"]),
        ("dev", DEV_REPO, ["*.dev"]),
        ("test", TEST_REPO, ["*.test"]),
    ]
    metadata: dict[str, dict[str, object]] = {}

    for split_dir, repo_id, patterns in downloads:
        info = api.dataset_info(repo_id, files_metadata=True)
        local_dir = out_dir / split_dir
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=patterns,
        )
        files = [
            {
                "path": sibling.rfilename,
                "size": getattr(sibling, "size", None),
            }
            for sibling in info.siblings
            if _matches_any(sibling.rfilename, patterns)
        ]
        metadata[split_dir] = {
            "repo_id": repo_id,
            "sha": info.sha,
            "local_dir": str(local_dir),
            "files": files,
        }

    summary = scan_babylm_corpus(
        out_dir,
        name="babylm_2026_strict_small",
        version=metadata["train"]["sha"],
        track=args.track,
        license_name=args.license,
    )
    save_manifest(summary.manifest, args.manifest_out)
    metadata["manifest"] = {
        "path": args.manifest_out,
        "documents": len(summary.manifest.documents),
        "explicit_splits": summary.explicit_splits,
        "fallback_splits": summary.fallback_splits,
    }

    metadata_path = out_dir / "download_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Downloaded BabyLM 2026 Strict-Small bundle")
    print(f"- target={out_dir}")
    print(f"- metadata={metadata_path}")
    print(f"- manifest={args.manifest_out}")
    print(f"- documents={len(summary.manifest.documents)}")
    print(f"- explicit_splits={summary.explicit_splits}")
    print(f"- fallback_splits={summary.fallback_splits}")
    for split in ("train", "validation", "test"):
        count = sum(1 for document in summary.manifest.documents if document.split == split)
        bytes_count = sum(document.bytes for document in summary.manifest.documents if document.split == split)
        print(f"- {split}: {count} docs, {bytes_count} bytes")


def _matches_any(path: str, patterns: list[str]) -> bool:
    name = Path(path).name
    for pattern in patterns:
        if pattern == "*.txt" and name.endswith(".txt"):
            return True
        if pattern == "*.dev" and name.endswith(".dev"):
            return True
        if pattern == "*.test" and name.endswith(".test"):
            return True
        if pattern == "README.md" and name == "README.md":
            return True
    return False


if __name__ == "__main__":
    main()
