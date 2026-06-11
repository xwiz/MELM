from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance.assistant_lexicon_legacy import (
    build_legacy_lexicon_candidates,
    build_legacy_router_candidates,
    write_legacy_lexicon_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export legacy functional-grammar vocabulary as SenseCandidate JSONL."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output = write_legacy_lexicon_candidates(args.out)
    candidates = build_legacy_lexicon_candidates() + build_legacy_router_candidates()
    print(
        json.dumps(
            {
                "schema": "melm.legacy_lexicon_export.v1",
                "output": str(output),
                "candidate_count": len(candidates),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
