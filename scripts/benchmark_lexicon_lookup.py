from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance import AssistantOSStore, benchmark_lexicon_lookup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark active-first, dormant-on-miss MELM lexicon lookup."
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--term", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--p95-budget-ms", type=float, default=5.0)
    args = parser.parse_args()

    store = AssistantOSStore(args.db)
    try:
        report = benchmark_lexicon_lookup(
            store,
            tuple(args.term),
            iterations=args.iterations,
            warmup_queries=args.warmup,
        )
    finally:
        store.close()
    payload = report.to_dict()
    payload["p95_budget_ms"] = args.p95_budget_ms
    payload["passed"] = report.p95_ms < args.p95_budget_ms
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
