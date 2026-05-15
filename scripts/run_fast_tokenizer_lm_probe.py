"""Run fast tokenizer baselines with optional Hugging Face tokenizers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.data import limit_texts_by_bytes, load_train_validation
from melm.tokenization import (
    build_tokenizer_arms,
    compare_unigram_lms,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="MELM_whitepaper.md")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--max-train-bytes", type=int, help="Optional deterministic train byte cap.")
    parser.add_argument("--max-validation-bytes", type=int, help="Optional deterministic validation byte cap.")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--arms",
        default="hf_bpe,hf_unigram,heuristic_morpheme",
        help=(
            "Comma-separated arms: hf_bpe,hf_unigram,capped_morpheme,"
            "hybrid_morph_unigram,tiered_morph_unigram,heuristic_morpheme,"
            "whitespace,byte_patch."
        ),
    )
    parser.add_argument(
        "--rank-by",
        choices=("bits_per_byte", "nll_per_token"),
        default="bits_per_byte",
        help="Ranking metric. Use bits_per_byte for cross-tokenizer comparisons.",
    )
    parser.add_argument("--out-json", default="reports/fast_tokenizer_lm_probe.json")
    parser.add_argument("--out-md", default="reports/fast_tokenizer_lm_probe.md")
    args = parser.parse_args()

    requested = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    texts, train_texts, validation_texts, source = load_train_validation(
        path=args.path,
        manifest_path=args.manifest,
    )
    train_texts = limit_texts_by_bytes(train_texts, args.max_train_bytes)
    validation_texts = limit_texts_by_bytes(validation_texts, args.max_validation_bytes)
    texts = train_texts + validation_texts

    start = time.perf_counter()
    tokenizers = list(
        build_tokenizer_arms(
            requested,
            texts,
            train_texts,
            tokenizer_vocab_size=args.vocab_size,
        ).values()
    )

    reports = compare_unigram_lms(tokenizers, train_texts, validation_texts)
    reports = sorted(
        reports,
        key=lambda report: (
            report.bits_per_byte
            if args.rank_by == "bits_per_byte"
            else report.nll_per_token
        ),
    )
    elapsed = time.perf_counter() - start
    payload = {
        "source": source,
        "vocab_size": args.vocab_size,
        "arms": requested,
        "rank_by": args.rank_by,
        "train_documents": len(train_texts),
        "validation_documents": len(validation_texts),
        "train_bytes": sum(len(text.encode("utf-8")) for text in train_texts),
        "validation_bytes": sum(len(text.encode("utf-8")) for text in validation_texts),
        "elapsed_seconds": elapsed,
        "reports": [_to_jsonable(report) for report in reports],
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Fast tokenizer LM probe")
    print(f"- source={source}")
    print(f"- train_bytes={payload['train_bytes']}")
    print(f"- validation_bytes={payload['validation_bytes']}")
    print(f"- rank_by={args.rank_by}")
    print(f"- elapsed_seconds={elapsed:.2f}")
    for rank, report in enumerate(reports, start=1):
        print(
            f"- #{rank} {report.tokenizer}: nll/token={report.nll_per_token:.3f}, "
            f"bits/byte={report.bits_per_byte:.3f}, vocab={report.vocabulary}, "
            f"validation_tokens={report.validation_tokens}"
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


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Fast Tokenizer LM Probe",
        "",
        f"Source: `{payload['source']}`",
        f"Train bytes: `{payload['train_bytes']}`",
        f"Validation bytes: `{payload['validation_bytes']}`",
        f"Elapsed seconds: `{payload['elapsed_seconds']:.2f}`",
        f"Rank by: `{payload['rank_by']}`",
        "",
        "| Rank | Tokenizer | NLL/Token | Bits/Byte | Vocab | Train Tokens | Validation Tokens |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, report in enumerate(payload["reports"], start=1):
        lines.append(
            f"| {rank} | {report['tokenizer']} | {report['nll_per_token']:.3f} | "
            f"{report['bits_per_byte']:.3f} | {report['vocabulary']} | "
            f"{report['train_tokens']} | {report['validation_tokens']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
