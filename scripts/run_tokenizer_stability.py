"""Run cross-fold tokenizer LM stability checks."""

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

from melm.data import load_train_validation
from melm.tokenization import evaluate_tokenizer_lm_stability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="MELM_whitepaper.md", help="Text file or directory for tokenizer metrics.")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--folds", type=int, default=5, help="Maximum number of document folds.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic fold shuffle seed.")
    parser.add_argument("--json", help="Optional JSON output path.")
    args = parser.parse_args()

    texts, _train_texts, _validation_texts, text_source = load_train_validation(
        path=args.text,
        manifest_path=args.manifest,
    )
    report = evaluate_tokenizer_lm_stability(
        texts,
        folds=args.folds,
        seed=args.seed,
    )

    print("Tokenizer LM stability")
    print(f"- text_source={text_source}")
    print(f"- documents={report.documents}")
    print(f"- folds={report.folds}")
    print(f"- morph_win_rate={report.morph_win_rate:.2%}")
    print(f"- stable_primary_candidate={report.stable_primary_candidate}")
    print(f"- best_baseline={report.best_baseline_tokenizer}")
    print(f"- best_baseline_average_nll={report.best_baseline_average_nll_per_token:.3f}")
    print(f"- morph_average_nll={report.morph_average_nll_per_token:.3f}")
    print(f"- average_lm_nll_gain={report.average_lm_nll_gain:.3f}")
    print("- winners:")
    for tokenizer, count in report.winner_counts.items():
        print(f"  - {tokenizer}: {count}")
    print("- average NLL/token:")
    for tokenizer, nll in report.average_nll_per_token.items():
        print(f"  - {tokenizer}: {nll:.3f}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
        print(f"Wrote {path}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
