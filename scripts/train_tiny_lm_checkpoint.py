"""Train one tiny LM arm and save reusable artifacts/checkpoint."""

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
from melm.tokenization import build_tokenizer_arms, save_tokenizer_artifact
from melm.training import TinyLMConfig, train_tiny_lm_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="MELM_whitepaper.md")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--tokenizer", default="capped_morpheme")
    parser.add_argument("--max-train-bytes", type=int)
    parser.add_argument("--max-validation-bytes", type=int)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="artifacts/tiny_lm/capped_morpheme_seed13")
    args = parser.parse_args()

    texts, train_texts, validation_texts, source = load_train_validation(
        path=args.path,
        manifest_path=args.manifest,
    )
    train_texts = limit_texts_by_bytes(train_texts, args.max_train_bytes)
    validation_texts = limit_texts_by_bytes(validation_texts, args.max_validation_bytes)
    texts = train_texts + validation_texts
    tokenizers = build_tokenizer_arms(
        [args.tokenizer],
        texts,
        train_texts,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
    )
    tokenizer = tokenizers[args.tokenizer]

    out_dir = Path(args.out_dir)
    tokenizer_dir = out_dir / "tokenizer"
    checkpoint_dir = out_dir / "checkpoint"
    tokenizer_metadata = save_tokenizer_artifact(tokenizer, tokenizer_dir)

    config = TinyLMConfig(
        tokenizer_name=tokenizer.name,
        max_vocab_size=args.max_vocab_size,
        pad_vocab_to_max_size=True,
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
    report = train_tiny_lm_baseline(
        train_texts,
        validation_texts,
        tokenizer,
        config,
        checkpoint_dir=checkpoint_dir,
    )
    manifest = {
        "source": source,
        "tokenizer_metadata": str(tokenizer_metadata),
        "checkpoint_dir": str(checkpoint_dir),
        "report": _to_jsonable(report),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Tiny LM checkpoint")
    print(f"- source={source}")
    print(f"- tokenizer={report.tokenizer}")
    print(f"- checkpoint_dir={checkpoint_dir}")
    print(f"- tokenizer_metadata={tokenizer_metadata}")
    print(f"- parameters={report.parameters}")
    print(f"- validation_bits_per_byte={report.validation_bits_per_byte:.3f}")
    print(f"- run_manifest={out_dir / 'run_manifest.json'}")


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
