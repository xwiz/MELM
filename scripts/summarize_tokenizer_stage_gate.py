"""Summarize tokenizer evidence into a stage-gate decision."""

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

from melm.evaluation import decide_tokenizer_stage_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progression", default="reports/babylm_2026_tiered_tiny_lm_progression.json")
    parser.add_argument(
        "--blimp-reports",
        nargs="+",
        default=[
            "reports/tiny_lm_blimp_fast_full.json",
            "reports/tiny_lm_blimp_fast_full_bits_per_byte.json",
            "reports/tiny_lm_blimp_fast_full_total_nll.json",
        ],
    )
    parser.add_argument("--entity-report", default="reports/tiny_lm_entity_tracking_fast.json")
    parser.add_argument(
        "--proxy-decision",
        default="reports/babylm_2026_small_proxy_tokenizer_decision.json",
        help="Optional first larger proxy-model decision report.",
    )
    parser.add_argument("--candidate", default="tiered_morph_unigram")
    parser.add_argument("--out-json", default="reports/tokenizer_stage_gate.json")
    parser.add_argument("--out-md", default="reports/tokenizer_stage_gate.md")
    args = parser.parse_args()

    progression_payload = json.loads(Path(args.progression).read_text(encoding="utf-8"))
    blimp_payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.blimp_reports
    ]
    entity_payload = json.loads(Path(args.entity_report).read_text(encoding="utf-8"))
    proxy_path = Path(args.proxy_decision) if args.proxy_decision else None
    proxy_payload = (
        json.loads(proxy_path.read_text(encoding="utf-8"))
        if proxy_path is not None and proxy_path.exists()
        else None
    )
    decision = decide_tokenizer_stage_gate(
        progression_payload,
        blimp_payloads,
        entity_payload,
        proxy_decision_payload=proxy_payload,
        candidate=args.candidate,
    )
    payload = {
        "progression": args.progression,
        "blimp_reports": args.blimp_reports,
        "entity_report": args.entity_report,
        "proxy_decision": str(proxy_path) if proxy_payload is not None else None,
        "decision": _to_jsonable(decision),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Tokenizer stage gate")
    print(f"- candidate={decision.candidate}")
    print(f"- decision={decision.decision}")
    print(f"- latest_step={decision.latest_step}")
    print(f"- relative_bpb_gain={decision.relative_bits_per_byte_gain:.2%}")
    print(f"- blimp_wins={decision.blimp_wins}/{decision.blimp_reports}")
    print(f"- entity_delta={decision.entity_accuracy_delta:.2%}")
    if decision.proxy_decision is not None:
        print(f"- proxy_decision={decision.proxy_decision}")
        print(f"- proxy_relative_bpb_gain={decision.proxy_relative_bits_per_byte_gain:.2%}")
    print(f"- recommendation={decision.recommendation}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _markdown(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    return "\n".join(
        [
            "# Tokenizer Stage Gate",
            "",
            f"Progression: `{payload['progression']}`",
            f"Entity report: `{payload['entity_report']}`",
            f"Proxy decision: `{payload['proxy_decision']}`",
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Candidate: `{decision['candidate']}`",
            f"- Latest step: `{decision['latest_step']}`",
            f"- Candidate bits/byte: `{decision['candidate_latest_bits_per_byte']:.3f}`",
            f"- Best baseline: `{decision['best_baseline']}`",
            f"- Best baseline bits/byte: `{decision['best_baseline_latest_bits_per_byte']:.3f}`",
            f"- Relative bits/byte gain: `{decision['relative_bits_per_byte_gain']:.2%}`",
            f"- Fast-BLiMP wins: `{decision['blimp_wins']}/{decision['blimp_reports']}`",
            f"- Entity best baseline: `{decision['entity_best_baseline']}`",
            f"- Candidate entity accuracy: `{decision['candidate_entity_accuracy']:.2%}`",
            f"- Best baseline entity accuracy: `{decision['best_baseline_entity_accuracy']:.2%}`",
            f"- Entity accuracy delta: `{decision['entity_accuracy_delta']:.2%}`",
            f"- Proxy decision: `{decision['proxy_decision']}`",
            f"- Proxy best baseline: `{decision['proxy_best_baseline']}`",
            f"- Proxy relative bits/byte gain: `{_format_optional_percent(decision['proxy_relative_bits_per_byte_gain'])}`",
            f"- Proxy supports scale: `{decision['proxy_supports_scale']}`",
            f"- Recommendation: {decision['recommendation']}",
            "",
        ]
    )


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


if __name__ == "__main__":
    main()
