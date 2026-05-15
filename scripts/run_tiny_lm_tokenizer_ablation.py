"""Run a tiny neural LM tokenizer ablation.

This is a smoke comparison only. Prefer bits-per-byte over token NLL when
comparing tokenizers with different segmentation lengths.
"""

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
from melm.tokenization import (
    build_tokenizer_arms,
)
from melm.training import TinyLMConfig, train_tiny_lm_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="MELM_whitepaper.md", help="Text file or directory to train on.")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--max-train-bytes", type=int, help="Optional deterministic train byte cap for large corpora.")
    parser.add_argument("--max-validation-bytes", type=int, help="Optional deterministic validation byte cap.")
    parser.add_argument(
        "--tokenizers",
        default="simple_bpe,unigram_like,heuristic_morpheme",
        help=(
            "Comma-separated tokenizer arms. Supports default arms plus "
            "hf_bpe,hf_unigram,capped_morpheme."
        ),
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        help="Vocabulary size for trained tokenizer arms. Defaults to --max-vocab-size.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-vocab-size", type=int, default=2048)
    parser.add_argument(
        "--pad-vocab-to-max-size",
        action="store_true",
        help="Pad each model vocabulary to --max-vocab-size for exact parameter matching.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", default="reports/tiny_lm_tokenizer_ablation.json")
    parser.add_argument("--out-md", default="reports/tiny_lm_tokenizer_ablation.md")
    args = parser.parse_args()

    requested = [name.strip() for name in args.tokenizers.split(",") if name.strip()]
    texts, train_texts, validation_texts, source = load_train_validation(
        path=args.path,
        manifest_path=args.manifest,
    )
    train_texts = limit_texts_by_bytes(train_texts, args.max_train_bytes)
    validation_texts = limit_texts_by_bytes(validation_texts, args.max_validation_bytes)
    texts = train_texts + validation_texts
    tokenizers = _build_requested_tokenizers(
        requested,
        texts,
        train_texts,
        tokenizer_vocab_size=args.tokenizer_vocab_size or args.max_vocab_size,
    )

    reports = []
    for tokenizer_name in requested:
        tokenizer = tokenizers[tokenizer_name]
        config = TinyLMConfig(
            tokenizer_name=tokenizer.name,
            max_vocab_size=args.max_vocab_size,
            pad_vocab_to_max_size=args.pad_vocab_to_max_size,
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
        reports.append(train_tiny_lm_baseline(train_texts, validation_texts, tokenizer, config))

    reports.sort(key=lambda report: report.validation_bits_per_byte)
    payload = {
        "source": source,
        "note": "Tiny neural smoke comparison; use bits_per_byte for cross-tokenizer ranking.",
        "config": {
            "steps": args.steps,
            "sequence_length": args.sequence_length,
            "embedding_dim": args.embedding_dim,
            "layers": args.layers,
            "heads": args.heads,
            "batch_size": args.batch_size,
            "max_vocab_size": args.max_vocab_size,
            "pad_vocab_to_max_size": args.pad_vocab_to_max_size,
            "tokenizer_vocab_size": args.tokenizer_vocab_size or args.max_vocab_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": args.device,
        },
        "reports": [_to_jsonable(report) for report in reports],
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Tiny LM tokenizer ablation")
    print(f"- source={source}")
    print(f"- reports={len(reports)}")
    for rank, report in enumerate(reports, start=1):
        print(
            f"- #{rank} {report.tokenizer}: bpb={report.validation_bits_per_byte:.3f}, "
            f"nll/token={report.validation_nll:.3f}, val_tokens={report.validation_tokens}"
        )
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


def _build_requested_tokenizers(
    requested: list[str],
    texts: list[str],
    train_texts: list[str],
    *,
    tokenizer_vocab_size: int,
):
    return build_tokenizer_arms(
        requested,
        texts,
        train_texts,
        tokenizer_vocab_size=tokenizer_vocab_size,
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tiny LM Tokenizer Ablation",
        "",
        f"Source: `{payload['source']}`",
        "",
        "This is a neural training smoke test, not a BabyLM-scale result.",
        "Rank by bits per byte because token NLL is not directly comparable across segmentations.",
        "",
        "| Rank | Tokenizer | Bits/Byte | NLL/Token | Validation Tokens | Parameters |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, report in enumerate(payload["reports"], start=1):
        lines.append(
            f"| {rank} | {report['tokenizer']} | {report['validation_bits_per_byte']:.3f} | "
            f"{report['validation_nll']:.3f} | {report['validation_tokens']} | "
            f"{report['parameters']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
