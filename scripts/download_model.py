"""Download a GGUF model from HuggingFace for local inference.

Usage::

    python scripts/download_model.py

This downloads Qwen2.5-0.5B-Instruct-GGUF (Q4_K_M, ~350MB) into
``models/qwen2.5-0.5b-instruct-q4_k_m.gguf``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run(repo_id: str, filename: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}/{filename} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    out = Path(path)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Saved to {out} ({size_mb:.1f} MB)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--filename",
        default="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        help="GGUF filename inside the repo",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the model",
    )
    args = parser.parse_args()
    run(args.repo_id, args.filename, args.local_dir)


if __name__ == "__main__":
    main()
