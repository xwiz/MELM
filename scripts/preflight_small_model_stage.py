"""Check local readiness before launching a small-model stage."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.training import preflight_small_model_stage, small_model_spec_from_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/babylm/small_model_tokenizer_stage.json",
    )
    parser.add_argument("--min-free-disk-gb", type=float, default=2.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--out-json", default="reports/babylm_2026_small_model_stage_preflight.json")
    parser.add_argument("--out-md", default="reports/babylm_2026_small_model_stage_preflight.md")
    args = parser.parse_args()

    config_path = Path(args.config)
    spec = small_model_spec_from_mapping(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    report = preflight_small_model_stage(
        spec,
        root=ROOT,
        min_free_disk_bytes=int(args.min_free_disk_gb * 1_000_000_000),
        require_cuda=args.require_cuda,
    )
    payload = {
        "config": str(config_path),
        "min_free_disk_gb": args.min_free_disk_gb,
        "require_cuda": args.require_cuda,
        "report": _to_jsonable(report),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Small-model stage preflight")
    print(f"- config={config_path}")
    print(f"- status={report.status}")
    print(f"- free_disk_bytes={report.free_disk_bytes}")
    print(f"- cuda_available={report.cuda_available}")
    for check in report.checks:
        print(f"- {check.name}: {check.status} ({check.detail})")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _markdown(payload: dict[str, Any]) -> str:
    report = payload["report"]
    lines = [
        "# Small-Model Stage Preflight",
        "",
        f"Config: `{payload['config']}`",
        f"Status: `{report['status']}`",
        f"Free disk bytes: `{report['free_disk_bytes']}`",
        f"CUDA available: `{report['cuda_available']}`",
        f"Estimated parameters per arm: `{report['estimated_parameters_per_arm']}`",
        f"Estimated checkpoint bytes: `{report['estimated_checkpoint_bytes']}`",
        f"Estimated training memory lower bound bytes: `{report['estimated_training_memory_lower_bound_bytes']}`",
        "",
        "## CUDA Devices",
        "",
    ]
    if report["cuda_devices"]:
        for device in report["cuda_devices"]:
            lines.append(
                f"- `{device.get('index')}` `{device.get('name')}` memory `{device.get('total_memory')}`"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in report["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | {check['status']} | {detail} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
