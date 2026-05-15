"""Run the dependency-free morphology boundary probe."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import morphology_fixture
from melm.tokenization import (
    build_default_tokenizers,
    evaluate_boundary_f1,
)


def main() -> None:
    examples = morphology_fixture()
    training_text = " ".join(example.word for example in examples)
    tokenizers = build_default_tokenizers([training_text], train_texts=[training_text])

    print("Morphology boundary probe")
    print(f"- examples={len(examples)}")
    for tokenizer in tokenizers:
        report = evaluate_boundary_f1(tokenizer, examples)
        print(
            f"- {report.tokenizer}: precision={report.precision:.2%}, "
            f"recall={report.recall:.2%}, f1={report.f1:.2%}, "
            f"exact={report.exact_match:.2%}"
        )


if __name__ == "__main__":
    main()
