"""Export MELM public-memory benchmarks as a Letta Evals pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import LOCOMO_URL, load_locomo_public_memory_benchmark
from melm.integrations import export_locomo_letta_eval_pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="local_data/locomo10.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/letta_eval")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--agent-file", default="agent.af")
    parser.add_argument("--base-url", default="http://localhost:8283")
    parser.add_argument("--gate-value", type=float, default=0.60)
    parser.add_argument("--out-json", default="reports/melm_letta_eval_pack.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        if not args.download:
            raise SystemExit(f"{dataset_path} does not exist. Rerun with --download.")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(LOCOMO_URL, dataset_path)

    benchmark = load_locomo_public_memory_benchmark(dataset_path)
    pack = export_locomo_letta_eval_pack(
        benchmark,
        args.out_dir,
        max_questions=args.max_questions,
        agent_file=args.agent_file,
        base_url=args.base_url,
        gate_value=args.gate_value,
    )
    payload = {
        "source_dataset": str(dataset_path),
        "letta_eval_pack": pack.__dict__,
        "letta_docs": {
            "datasets": "https://docs.letta.com/guides/evals/concepts/datasets/",
            "suite_yaml": "https://docs.letta.com/evals/configuration/suite-yaml-reference",
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("MELM Letta Evals export pack")
    print(f"- samples={pack.samples}")
    print(f"- memories={pack.memories}")
    print(f"- dataset={pack.dataset_path}")
    print(f"- suite={pack.suite_path}")
    print(f"- memory={pack.memory_path}")
    print(f"- readme={pack.readme_path}")


if __name__ == "__main__":
    main()
