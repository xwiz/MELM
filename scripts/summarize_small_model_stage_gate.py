"""Summarize the checkpointed small-model tokenizer stage."""

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

from melm.evaluation import decide_small_model_stage_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiseed", default="reports/babylm_2026_small_model_stage_multiseed.json")
    parser.add_argument(
        "--blimp-reports",
        nargs="+",
        default=[
            "reports/babylm_2026_small_model_stage_blimp_mean_nll.json",
            "reports/babylm_2026_small_model_stage_blimp_bits_per_byte.json",
            "reports/babylm_2026_small_model_stage_blimp_total_nll.json",
        ],
    )
    parser.add_argument("--entity-report", default="reports/babylm_2026_small_model_stage_entity_tracking.json")
    parser.add_argument(
        "--symbolic-entity-report",
        default="reports/babylm_2026_small_model_stage_entity_tracking_symbolic.json",
    )
    parser.add_argument("--candidate", default="tiered_morph_unigram")
    parser.add_argument("--out-json", default="reports/babylm_2026_small_model_stage_gate.json")
    parser.add_argument("--out-md", default="reports/babylm_2026_small_model_stage_gate.md")
    args = parser.parse_args()

    multiseed = json.loads(Path(args.multiseed).read_text(encoding="utf-8"))
    blimp_payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.blimp_reports
    ]
    entity_payload = json.loads(Path(args.entity_report).read_text(encoding="utf-8"))
    symbolic_path = Path(args.symbolic_entity_report) if args.symbolic_entity_report else None
    symbolic_payload = (
        json.loads(symbolic_path.read_text(encoding="utf-8"))
        if symbolic_path is not None and symbolic_path.exists()
        else None
    )
    decision = decide_small_model_stage_gate(
        multiseed,
        blimp_payloads,
        entity_payload,
        symbolic_entity_payload=symbolic_payload,
        candidate=args.candidate,
    )
    payload = {
        "multiseed": args.multiseed,
        "blimp_reports": args.blimp_reports,
        "entity_report": args.entity_report,
        "symbolic_entity_report": str(symbolic_path) if symbolic_payload is not None else None,
        "decision": _to_jsonable(decision),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Small-model stage gate")
    print(f"- decision={decision.decision}")
    print(f"- candidate={decision.candidate}")
    print(f"- relative_bpb_gain={decision.relative_bits_per_byte_gain:.2%}")
    print(f"- blimp_wins={decision.blimp_wins}/{decision.blimp_reports}")
    print(f"- entity_delta={decision.entity_accuracy_delta:.2%}")
    if decision.symbolic_entity_accuracy is not None:
        print(f"- symbolic_entity_accuracy={decision.symbolic_entity_accuracy:.2%}")
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
            "# BabyLM 2026 Small-Model Stage Gate",
            "",
            f"Multiseed report: `{payload['multiseed']}`",
            f"Entity report: `{payload['entity_report']}`",
            f"Symbolic entity report: `{payload['symbolic_entity_report']}`",
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Candidate: `{decision['candidate']}`",
            f"- Candidate bits/byte: `{decision['candidate_bits_per_byte']:.3f}`",
            f"- Best HF baseline: `{decision['best_hf_baseline']}`",
            f"- Best HF baseline bits/byte: `{decision['best_hf_baseline_bits_per_byte']:.3f}`",
            f"- Relative bits/byte gain: `{decision['relative_bits_per_byte_gain']:.2%}`",
            f"- Compression control: `{decision['compression_control']}`",
            f"- Compression control bits/byte: `{decision['compression_control_bits_per_byte']:.3f}`",
            f"- Fast-BLiMP wins vs HF baselines: `{decision['blimp_wins']}/{decision['blimp_reports']}`",
            f"- Entity best HF baseline: `{decision['entity_best_hf_baseline']}`",
            f"- Candidate entity accuracy: `{decision['candidate_entity_accuracy']:.2%}`",
            f"- Best HF entity accuracy: `{decision['best_hf_entity_accuracy']:.2%}`",
            f"- Entity accuracy delta: `{decision['entity_accuracy_delta']:.2%}`",
            f"- Symbolic entity accuracy: `{_format_optional_percent(decision['symbolic_entity_accuracy'])}`",
            f"- Recommendation: {decision['recommendation']}",
            "",
            "Interpretation: this gate can promote the next integration step, but it does not make the tokenizer final.",
            "",
        ]
    )


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


if __name__ == "__main__":
    main()
