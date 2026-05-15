"""Generate a runnable run card for the next BabyLM-style model stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.training import build_small_model_stage_plan, small_model_spec_from_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/babylm/small_model_tokenizer_stage.json",
        help="Small-model stage config.",
    )
    parser.add_argument("--out-json", default="reports/babylm_2026_small_model_stage_plan.json")
    parser.add_argument("--out-md", default="reports/babylm_2026_small_model_stage_plan.md")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = small_model_spec_from_mapping(config)
    gate_payload = _read_optional_json(spec.source_gate)
    proxy_payload = _read_optional_json(spec.source_proxy_decision)
    plan = build_small_model_stage_plan(
        spec,
        gate_payload=gate_payload,
        proxy_payload=proxy_payload,
        config_path=str(config_path),
    )
    plan["config_path"] = str(config_path)

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(plan), encoding="utf-8")

    status = plan["dependency_status"]
    estimates = plan["estimates"]
    print("Small-model stage plan")
    print(f"- config={config_path}")
    print(f"- candidate={plan['candidate']}")
    print(f"- dependency_pass={status['pass']}")
    print(f"- gate_decision={status['gate_decision']}")
    print(f"- proxy_decision={status['proxy_decision']}")
    print(f"- parameters_per_arm={estimates['parameters_per_arm']}")
    print(f"- lower_bound_flops_full_multiseed={estimates['lower_bound_training_flops_full_multiseed']}")
    print(f"- commands={len(plan['commands'])}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _read_optional_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _markdown(plan: dict[str, Any]) -> str:
    status = plan["dependency_status"]
    config = plan["config"]
    estimates = plan["estimates"]
    lines = [
        "# BabyLM 2026 Small-Model Stage Plan",
        "",
        f"Config: `{plan['config_path']}`",
        f"Candidate: `{plan['candidate']}`",
        f"Tokenizers: `{', '.join(plan['tokenizers'])}`",
        "",
        "## Dependency Check",
        "",
        f"- Gate decision: `{status['gate_decision']}`",
        f"- Proxy decision: `{status['proxy_decision']}`",
        f"- Proxy candidate: `{status['proxy_candidate']}`",
        f"- Pass: `{status['pass']}`",
        "",
        "## Training Profile",
        "",
        f"- Manifest: `{config['manifest']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in config['seeds'])}`",
        f"- Train bytes cap: `{config['max_train_bytes']}`",
        f"- Validation bytes cap: `{config['max_validation_bytes']}`",
        f"- Steps: `{config['steps']}`",
        f"- Sequence length: `{config['sequence_length']}`",
        f"- Embedding/layers/heads: `{config['embedding_dim']}/{config['layers']}/{config['heads']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Vocab size: `{config['max_vocab_size']}`",
        "",
        "## Estimates",
        "",
        f"- Parameters per arm: `{estimates['parameters_per_arm']}`",
        f"- Training tokens per arm: `{estimates['training_tokens_per_arm']}`",
        f"- Lower-bound FLOPs per arm: `{estimates['lower_bound_training_flops_per_arm']}`",
        f"- Lower-bound FLOPs, all tokenizers one seed: `{estimates['lower_bound_training_flops_all_tokenizers_one_seed']}`",
        f"- Lower-bound FLOPs, full multi-seed: `{estimates['lower_bound_training_flops_full_multiseed']}`",
        "",
        "## Commands",
        "",
    ]
    for index, command in enumerate(plan["commands"], start=1):
        lines.extend([f"### {index}. Command", "", f"```powershell\n{command}\n```", ""])
    lines.extend(["## Go/No-Go", ""])
    for criterion in plan["go_no_go"]:
        lines.append(f"- {criterion}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
