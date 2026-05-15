"""Run a tiny held-out token-LM probe for tokenizer comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.data import limit_texts_by_bytes, load_train_validation
from melm.tokenization import (
    build_default_tokenizers,
    compare_unigram_lms,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Text file or directory to evaluate.")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--max-train-bytes", type=int, help="Optional deterministic train byte cap for large corpora.")
    parser.add_argument("--max-validation-bytes", type=int, help="Optional deterministic validation byte cap.")
    args = parser.parse_args()

    texts, train, validation, source = load_train_validation(path=args.path, manifest_path=args.manifest)
    train = limit_texts_by_bytes(train, args.max_train_bytes)
    validation = limit_texts_by_bytes(validation, args.max_validation_bytes)
    texts = train + validation
    tokenizers = build_default_tokenizers(texts, train_texts=train)

    print(f"Tokenizer LM probe for {source}")
    print(f"- train_docs={len(train)}")
    print(f"- validation_docs={len(validation)}")
    print(f"- train_bytes={sum(len(text.encode('utf-8')) for text in train)}")
    print(f"- validation_bytes={sum(len(text.encode('utf-8')) for text in validation)}")
    for report in compare_unigram_lms(tokenizers, train, validation):
        print(
            f"- {report.tokenizer}: nll/token={report.nll_per_token:.3f}, "
            f"bits/byte={report.bits_per_byte:.3f}, "
            f"ppl={report.perplexity:.2f}, vocab={report.vocabulary}, "
            f"train_tokens={report.train_tokens}, validation_tokens={report.validation_tokens}"
        )

if __name__ == "__main__":
    main()
