"""Run dependency-free tokenizer metrics on local text files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.data import load_texts
from melm.tokenization import (
    build_default_tokenizers,
    compare_tokenizers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Text file or directory to evaluate.")
    parser.add_argument("--patch-size", type=int, default=4)
    args = parser.parse_args()

    texts = load_texts(args.path)
    tokenizers = build_default_tokenizers(texts, byte_patch_size=args.patch_size)

    print(f"Tokenizer report for {args.path}")
    print(f"- documents={len(texts)}")
    for report in compare_tokenizers(tokenizers, texts):
        print(
            f"- {report.tokenizer}: words={report.words}, tokens={report.tokens}, "
            f"unique={report.unique_tokens}, tokens/word={report.tokens_per_word:.3f}, "
            f"chars_ratio={report.compression_vs_chars:.3f}, fallback={report.fallback_rate:.2%}"
        )

if __name__ == "__main__":
    main()
