"""Run the tokenizer decision gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import morphology_fixture
from melm.data import limit_texts_by_bytes, load_train_validation
from melm.tokenization import (
    build_default_tokenizers,
    compare_unigram_lms,
    decide_tokenizer_strategy,
    evaluate_boundary_f1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="MELM_whitepaper.md", help="Text file or directory for tokenizer metrics.")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--max-train-bytes", type=int, help="Optional deterministic train byte cap for large corpora.")
    parser.add_argument("--max-validation-bytes", type=int, help="Optional deterministic validation byte cap.")
    args = parser.parse_args()

    texts, train_texts, validation_texts, text_source = load_train_validation(
        path=args.text,
        manifest_path=args.manifest,
    )
    train_texts = limit_texts_by_bytes(train_texts, args.max_train_bytes)
    validation_texts = limit_texts_by_bytes(validation_texts, args.max_validation_bytes)
    texts = train_texts + validation_texts
    tokenizers = build_default_tokenizers(texts, train_texts=train_texts)
    lm_reports = compare_unigram_lms(tokenizers, train_texts, validation_texts)
    boundary_reports = [
        evaluate_boundary_f1(tokenizer, morphology_fixture())
        for tokenizer in tokenizers
    ]
    decision = decide_tokenizer_strategy(lm_reports, boundary_reports)

    print("Tokenizer decision gate")
    print(f"- text_source={text_source}")
    print(f"- train_bytes={sum(len(text.encode('utf-8')) for text in train_texts)}")
    print(f"- validation_bytes={sum(len(text.encode('utf-8')) for text in validation_texts)}")
    print(f"- decision={decision.decision}")
    print(f"- gate_passed={decision.gate_passed}")
    print(f"- best_lm_baseline={decision.best_lm_tokenizer}")
    print(f"- best_baseline_nll_per_token={decision.best_baseline_nll_per_token:.3f}")
    print(f"- morph_nll_per_token={decision.morph_nll_per_token:.3f}")
    print(f"- lm_nll_gain={decision.lm_nll_gain:.3f}")
    print(f"- best_boundary_tokenizer={decision.best_boundary_tokenizer}")
    print(f"- best_baseline_boundary_f1={decision.best_baseline_boundary_f1:.2%}")
    print(f"- morph_boundary_f1={decision.morph_boundary_f1:.2%}")
    print(f"- boundary_f1_gain={decision.boundary_f1_gain:.2%}")
    print(f"- recommendation={decision.recommendation}")

    print("")
    print("LM ranking")
    for report in lm_reports:
        print(
            f"- {report.tokenizer}: nll={report.nll_per_token:.3f}, "
            f"ppl={report.perplexity:.2f}, vocab={report.vocabulary}"
        )


if __name__ == "__main__":
    main()
