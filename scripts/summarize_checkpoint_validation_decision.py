"""Summarize saved-checkpoint evidence as a tokenizer validation decision."""

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

from melm.evaluation import decide_checkpoint_tokenizer_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-report", default="reports/tiny_lm_artifact_evaluation.json")
    parser.add_argument("--minimal-pair-report", default="reports/tiny_lm_minimal_pairs.json")
    parser.add_argument("--candidate", default="capped_morpheme")
    parser.add_argument("--out-json", default="reports/tiny_lm_checkpoint_validation_decision.json")
    parser.add_argument("--out-md", default="reports/tiny_lm_checkpoint_validation_decision.md")
    args = parser.parse_args()

    artifact_payload = json.loads(Path(args.artifact_report).read_text(encoding="utf-8"))
    minimal_pair_payload = json.loads(Path(args.minimal_pair_report).read_text(encoding="utf-8"))
    decision = decide_checkpoint_tokenizer_validation(
        artifact_payload["runs"],
        minimal_pair_payload["runs"],
        candidate_tokenizer=args.candidate,
    )
    output = {
        "artifact_report": args.artifact_report,
        "minimal_pair_report": args.minimal_pair_report,
        "decision": _to_jsonable(decision),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(output), encoding="utf-8")

    print("Tiny LM checkpoint validation decision")
    print(f"- artifact_report={args.artifact_report}")
    print(f"- minimal_pair_report={args.minimal_pair_report}")
    print(f"- decision={decision.decision}")
    print(f"- candidate={decision.candidate_tokenizer}")
    print(f"- best_baseline={decision.best_baseline_tokenizer}")
    print(f"- relative_bits_per_byte_gain={decision.relative_bits_per_byte_gain:.2%}")
    print(f"- minimal_pair_accuracy_delta={decision.minimal_pair_accuracy_delta:.2%}")
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
            "# Tiny LM Checkpoint Validation Decision",
            "",
            f"Artifact report: `{payload['artifact_report']}`",
            f"Minimal-pair report: `{payload['minimal_pair_report']}`",
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Candidate: `{decision['candidate_tokenizer']}`",
            f"- Best baseline: `{decision['best_baseline_tokenizer']}`",
            f"- Candidate bits/byte: `{decision['candidate_bits_per_byte']:.3f}`",
            f"- Best baseline bits/byte: `{decision['best_baseline_bits_per_byte']:.3f}`",
            f"- Relative bits/byte gain: `{decision['relative_bits_per_byte_gain']:.2%}`",
            f"- Candidate minimal-pair accuracy: `{decision['candidate_minimal_pair_accuracy']:.2%}`",
            f"- Best baseline minimal-pair accuracy: `{decision['best_baseline_minimal_pair_accuracy']:.2%}`",
            f"- Minimal-pair accuracy delta: `{decision['minimal_pair_accuracy_delta']:.2%}`",
            f"- Recommendation: {decision['recommendation']}",
            "",
        ]
    )


if __name__ == "__main__":
    main()
