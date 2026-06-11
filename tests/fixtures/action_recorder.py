"""Harmless command target for Assistant OS action execution tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("target")
    args = parser.parse_args()

    target = Path(args.target)
    row = {
        "label": args.label,
        "target": args.target,
        "target_exists": target.exists(),
    }
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"recorded {args.label}")


if __name__ == "__main__":
    main()
