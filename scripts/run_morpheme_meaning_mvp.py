"""Run the morpheme/root/meaning MVP validation corpus."""

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

from melm.semantics import evaluate_meaning_mvp, load_meaning_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/morpheme_meaning_mvp.jsonl"),
    )
    parser.add_argument("--out-json", default="reports/morpheme_meaning_mvp.json")
    parser.add_argument("--out-md", default="reports/morpheme_meaning_mvp.md")
    args = parser.parse_args()

    corpus = load_meaning_corpus(args.corpus)
    report = evaluate_meaning_mvp(corpus)
    payload = {
        "source": str(args.corpus),
        "components": len(corpus.components),
        "lexemes": len(corpus.lexemes),
        "word_cases": report.word_cases,
        "utterance_cases": report.utterance_cases,
        "report": _to_jsonable(report),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Morpheme meaning MVP")
    print(f"- source={args.corpus}")
    print(f"- components={len(corpus.components)}")
    print(f"- lexemes={len(corpus.lexemes)}")
    print(f"- word_accuracy={report.word_accuracy:.2%}")
    print(f"- utterance_accuracy={report.utterance_accuracy:.2%}")
    print(f"- overall_accuracy={report.overall_accuracy:.2%}")
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
        "# Morpheme Meaning MVP",
        "",
        f"Source: `{payload['source']}`",
        f"Components: `{payload['components']}`",
        f"Known lexemes: `{payload['lexemes']}`",
        f"Word cases: `{payload['word_cases']}`",
        f"Utterance cases: `{payload['utterance_cases']}`",
        "",
        f"- Word inference accuracy: `{report['word_accuracy']:.2%}`",
        f"- Utterance routing accuracy: `{report['utterance_accuracy']:.2%}`",
        f"- Overall accuracy: `{report['overall_accuracy']:.2%}`",
        "",
        "## Word Cases",
        "",
        "| Item | Category | Passed | Predicted Components | Gloss |",
        "|---|---|---:|---|---|",
    ]
    for result in report["word_results"]:
        gloss = str(result["details"].get("gloss", "")).replace("|", "\\|")
        lines.append(
            f"| {result['item']} | {result['category']} | {result['passed']} | "
            f"{result['predicted']} | {gloss} |"
        )

    lines.extend(
        [
            "",
            "## Utterance Cases",
            "",
            "| Utterance | Category | Passed | Prediction | Features |",
            "|---|---|---:|---|---|",
        ]
    )
    for result in report["utterance_results"]:
        features = json.dumps(result["details"].get("features", {}), sort_keys=True)
        utterance = result["item"].replace("|", "\\|")
        lines.append(
            f"| {utterance} | {result['category']} | {result['passed']} | "
            f"{result['predicted']} | `{features}` |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
