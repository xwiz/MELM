"""Run a dependency-free Phase 1 smoke check."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import episodic_memory_fixture
from melm.evaluation import memory_gate
from melm.memory import EventMemory, evaluate_memory
from melm.tokenization import (
    BytePatchTokenizer,
    HeuristicMorphemeTokenizer,
    UnigramLikeTokenizer,
    WhitespaceTokenizer,
    compare_tokenizers,
)


def main() -> None:
    texts = [
        "Maya put the red cup on the kitchen table.",
        "Leo moved the blue book from the table to the shelf.",
        "Maya looked for the red cup before snack time.",
    ]

    tokenizers = [
        WhitespaceTokenizer(),
        HeuristicMorphemeTokenizer(),
        BytePatchTokenizer(patch_size=4),
        UnigramLikeTokenizer(frozenset({"maya", "put", "red", "cup", "table", "leo", "book", "shelf"})),
    ]

    print("Tokenizer smoke reports")
    for report in compare_tokenizers(tokenizers, texts):
        print(
            f"- {report.tokenizer}: tokens={report.tokens}, "
            f"tokens/word={report.tokens_per_word:.2f}, fallback={report.fallback_rate:.2%}"
        )

    events, cases = episodic_memory_fixture()
    memory = EventMemory(events)
    comparison = evaluate_memory(memory, cases, k=2)
    gate = memory_gate(comparison.event_memory_recall_at_k, comparison.rag_recall_at_k)

    print("\nMemory smoke report")
    print(f"- cases={comparison.cases}")
    print(f"- rag_recall@2={comparison.rag_recall_at_k:.2%}")
    print(f"- event_memory_recall@2={comparison.event_memory_recall_at_k:.2%}")
    print(f"- gate_gain={gate.metric:.2%}, passed={gate.passed}")


if __name__ == "__main__":
    main()
