from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance.assistant_lexicon_seed import (
    DEFAULT_BUILD_TIMESTAMP,
    LexiconCandidateSource,
    build_lexicon_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic MELM lexicon seed database from SenseCandidate JSONL."
    )
    parser.add_argument(
        "--candidate-jsonl",
        action="append",
        required=True,
        metavar="PROVENANCE=PATH",
        help="Candidate JSONL bound to an allowed provenance; repeat for multiple sources.",
    )
    parser.add_argument("--out-db", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-timestamp", default=DEFAULT_BUILD_TIMESTAMP)
    parser.add_argument(
        "--router-family",
        action="append",
        default=[],
        help="Mark a migrated router vocabulary family as store-owned.",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    sources = [_parse_source(value) for value in args.candidate_jsonl]
    report = build_lexicon_seed(
        sources,
        output_db=args.out_db,
        manifest_path=args.manifest,
        reset=args.reset,
        build_timestamp=args.build_timestamp,
        router_families=tuple(args.router_family),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def _parse_source(value: str) -> LexiconCandidateSource:
    provenance, separator, raw_path = value.partition("=")
    if not separator or not provenance or not raw_path:
        raise argparse.ArgumentTypeError(
            "--candidate-jsonl must use PROVENANCE=PATH"
        )
    return LexiconCandidateSource(path=Path(raw_path), provenance=provenance)


if __name__ == "__main__":
    raise SystemExit(main())
