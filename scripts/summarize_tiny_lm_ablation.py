"""Summarize a tiny LM tokenizer ablation as a scale-up decision."""

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

from melm.training import decide_tiny_lm_tokenizer_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/babylm_2026_matched_tiny_lm_ablation.json")
    parser.add_argument("--candidate", default="capped_morpheme")
    parser.add_argument("--out-json", default="reports/babylm_2026_tiny_lm_ablation_decision.json")
    parser.add_argument("--out-md", default="reports/babylm_2026_tiny_lm_ablation_decision.md")
    args = parser.parse_args()

    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    decision = decide_tiny_lm_tokenizer_ablation(
        payload["reports"],
        candidate_tokenizer=args.candidate,
    )
    output = {
        "source_report": args.report,
        "decision": _to_jsonable(decision),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(output), encoding="utf-8")

    print("Tiny LM ablation decision")
    print(f"- source_report={args.report}")
    print(f"- decision={decision.decision}")
    print(f"- candidate={decision.candidate_tokenizer}")
    print(f"- best_baseline={decision.best_baseline_tokenizer}")
    print(f"- bits_per_byte_gain={decision.bits_per_byte_gain:.3f}")
    print(f"- relative_gain={decision.relative_bits_per_byte_gain:.2%}")
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
            "# Tiny LM Ablation Decision",
            "",
            f"Source report: `{payload['source_report']}`",
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Candidate: `{decision['candidate_tokenizer']}`",
            f"- Best baseline: `{decision['best_baseline_tokenizer']}`",
            f"- Candidate bits/byte: `{decision['candidate_bits_per_byte']:.3f}`",
            f"- Best baseline bits/byte: `{decision['best_baseline_bits_per_byte']:.3f}`",
            f"- Absolute bits/byte gain: `{decision['bits_per_byte_gain']:.3f}`",
            f"- Relative gain: `{decision['relative_bits_per_byte_gain']:.2%}`",
            f"- Recommendation: {decision['recommendation']}",
            "",
            "This supports a longer neural ablation only; it is not a final BabyLM score.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
