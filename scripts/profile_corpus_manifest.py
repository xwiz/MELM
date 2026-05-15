"""Profile a corpus manifest without training tokenizers."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.data import load_manifest


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-json", default="reports/corpus_profile.json")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    profile = {
        "manifest": args.manifest,
        "name": manifest.name,
        "version": manifest.version,
        "documents": len(manifest.documents),
        "splits": {},
        "documents_detail": [],
    }
    for document in manifest.documents:
        text = Path(document.path).read_text(encoding="utf-8")
        line_count = text.count("\n") + int(bool(text))
        word_count = len(WORD_RE.findall(text))
        split = profile["splits"].setdefault(
            document.split,
            {"documents": 0, "bytes": 0, "lines": 0, "word_like_tokens": 0},
        )
        split["documents"] += 1
        split["bytes"] += document.bytes
        split["lines"] += line_count
        split["word_like_tokens"] += word_count
        detail = asdict(document)
        detail["lines"] = line_count
        detail["word_like_tokens"] = word_count
        profile["documents_detail"].append(detail)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print("Corpus profile")
    print(f"- manifest={args.manifest}")
    print(f"- documents={profile['documents']}")
    for split, split_profile in sorted(profile["splits"].items()):
        print(
            f"- {split}: docs={split_profile['documents']}, "
            f"bytes={split_profile['bytes']}, "
            f"lines={split_profile['lines']}, "
            f"word_like_tokens={split_profile['word_like_tokens']}"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
