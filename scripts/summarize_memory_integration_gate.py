"""Summarize the event/state-memory integration gate."""

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

from melm.evaluation import decide_memory_integration_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-suite", default="reports/validation_suite.json")
    parser.add_argument("--state-assisted", default="reports/babylm_2026_state_assisted_entity_tracking.json")
    parser.add_argument("--candidate", default="tiered_morph_unigram")
    parser.add_argument("--out-json", default="reports/memory_integration_gate.json")
    parser.add_argument("--out-md", default="reports/memory_integration_gate.md")
    args = parser.parse_args()

    validation_suite = json.loads(Path(args.validation_suite).read_text(encoding="utf-8"))
    state_assisted = json.loads(Path(args.state_assisted).read_text(encoding="utf-8"))
    decision = decide_memory_integration_gate(
        validation_suite,
        state_assisted,
        candidate=args.candidate,
    )
    payload = {
        "validation_suite": args.validation_suite,
        "state_assisted": args.state_assisted,
        "decision": _to_jsonable(decision),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Memory integration gate")
    print(f"- candidate={decision.candidate}")
    print(f"- decision={decision.decision}")
    print(f"- state_assisted_accuracy={decision.state_assisted_accuracy:.2%}")
    print(f"- state_assisted_lift={decision.state_assisted_lift:.2%}")
    print(f"- authored_dialogue_memory_gain={decision.authored_dialogue_memory_gain:.2%}")
    print(f"- sample_transcript_memory_gain={decision.sample_transcript_memory_gain:.2%}")
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
            "# Memory Integration Gate",
            "",
            f"Validation suite: `{payload['validation_suite']}`",
            f"State-assisted entity report: `{payload['state_assisted']}`",
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Candidate: `{decision['candidate']}`",
            f"- State-assisted entity accuracy: `{decision['state_assisted_accuracy']:.2%}`",
            f"- State-assisted entity lift: `{decision['state_assisted_lift']:.2%}`",
            f"- State answer rate: `{decision['state_answer_rate']:.2%}`",
            f"- Synthetic memory gain: `{decision['synthetic_memory_gain']:.2%}`",
            f"- Authored dialogue memory gain: `{decision['authored_dialogue_memory_gain']:.2%}`",
            f"- Sample transcript memory gain: `{decision['sample_transcript_memory_gain']:.2%}`",
            f"- Authored dialogue abstention metric: `{decision['authored_dialogue_abstention_metric']:.2f}`",
            f"- Sample transcript abstention metric: `{decision['sample_transcript_abstention_metric']:.2f}`",
            f"- Recommendation: {decision['recommendation']}",
            "",
        ]
    )


if __name__ == "__main__":
    main()
