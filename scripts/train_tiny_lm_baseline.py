"""Train a tiny causal LM baseline for end-to-end pipeline validation."""

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

from melm.data import limit_texts_by_bytes, load_train_validation
from melm.tokenization import build_default_tokenizers
from melm.training import TinyLMConfig, train_tiny_lm_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="MELM_whitepaper.md", help="Text file or directory to train on.")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--max-train-bytes", type=int, help="Optional deterministic train byte cap for large corpora.")
    parser.add_argument("--max-validation-bytes", type=int, help="Optional deterministic validation byte cap.")
    parser.add_argument("--tokenizer", default="simple_bpe", help="Tokenizer arm to use.")
    parser.add_argument("--steps", type=int, default=10, help="Training steps.")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-vocab-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", default="reports/tiny_lm_baseline.json")
    args = parser.parse_args()

    texts, train_texts, validation_texts, source = load_train_validation(
        path=args.path,
        manifest_path=args.manifest,
    )
    train_texts = limit_texts_by_bytes(train_texts, args.max_train_bytes)
    validation_texts = limit_texts_by_bytes(validation_texts, args.max_validation_bytes)
    texts = train_texts + validation_texts
    tokenizer = _select_tokenizer(args.tokenizer, texts, train_texts)
    config = TinyLMConfig(
        tokenizer_name=tokenizer.name,
        max_vocab_size=args.max_vocab_size,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        heads=args.heads,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    report = train_tiny_lm_baseline(train_texts, validation_texts, tokenizer, config)

    payload = {
        "source": source,
        "config": _to_jsonable(config),
        "report": _to_jsonable(report),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Tiny LM baseline")
    print(f"- source={source}")
    print(f"- tokenizer={report.tokenizer}")
    print(f"- train_docs={report.train_documents}")
    print(f"- validation_docs={report.validation_documents}")
    print(f"- train_tokens={report.train_tokens}")
    print(f"- validation_tokens={report.validation_tokens}")
    print(f"- validation_bytes={report.validation_bytes}")
    print(f"- train_sequences={report.train_sequences}")
    print(f"- validation_sequences={report.validation_sequences}")
    print(f"- vocabulary_size={report.vocabulary_size}")
    print(f"- parameters={report.parameters}")
    print(f"- steps={report.steps}")
    print(f"- device={report.device}")
    print(f"- final_train_loss={report.final_train_loss:.3f}")
    print(f"- validation_nll={report.validation_nll:.3f}")
    print(f"- validation_bits_per_byte={report.validation_bits_per_byte:.3f}")
    print(f"- validation_perplexity={report.validation_perplexity:.2f}")
    print(f"Wrote {out_path}")


def _select_tokenizer(name: str, texts: list[str], train_texts: list[str]):
    tokenizers = build_default_tokenizers(texts, train_texts=train_texts)
    for tokenizer in tokenizers:
        if tokenizer.name == name:
            return tokenizer
    available = ", ".join(tokenizer.name for tokenizer in tokenizers)
    raise ValueError(f"Unknown tokenizer {name!r}; available: {available}")


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
