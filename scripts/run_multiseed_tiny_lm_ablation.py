"""Run a multi-seed tiny LM tokenizer ablation."""

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
from melm.tokenization import build_tokenizer_arms
from melm.training import (
    TinyLMConfig,
    decide_tiny_lm_tokenizer_ablation,
    summaries_as_decision_reports,
    summarize_multiseed_reports,
    train_tiny_lm_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="MELM_whitepaper.md")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--max-train-bytes", type=int)
    parser.add_argument("--max-validation-bytes", type=int)
    parser.add_argument("--tokenizers", default="hf_bpe,hf_unigram,capped_morpheme")
    parser.add_argument("--seeds", default="3,13,23")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate", default="capped_morpheme")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-run cache files when --run-cache-dir is provided.",
    )
    parser.add_argument(
        "--run-cache-dir",
        help="Optional directory for one JSON report per tokenizer/seed run.",
    )
    parser.add_argument("--out-json", default="reports/multiseed_tiny_lm_ablation.json")
    parser.add_argument("--out-md", default="reports/multiseed_tiny_lm_ablation.md")
    args = parser.parse_args()

    requested = [name.strip() for name in args.tokenizers.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    texts, train_texts, validation_texts, source = load_train_validation(
        path=args.path,
        manifest_path=args.manifest,
    )
    train_texts = limit_texts_by_bytes(train_texts, args.max_train_bytes)
    validation_texts = limit_texts_by_bytes(validation_texts, args.max_validation_bytes)
    texts = train_texts + validation_texts
    tokenizers = build_tokenizer_arms(
        requested,
        texts,
        train_texts,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
    )

    run_reports: list[dict[str, Any]] = []
    cached_runs = 0
    cache_dir = Path(args.run_cache_dir) if args.run_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for tokenizer_name in requested:
            cache_path = _cache_path(cache_dir, tokenizer_name, seed)
            if args.resume and cache_path is not None and cache_path.exists():
                run_reports.append(json.loads(cache_path.read_text(encoding="utf-8")))
                cached_runs += 1
                continue
            tokenizer = tokenizers[tokenizer_name]
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
                seed=seed,
                device=args.device,
            )
            report = train_tiny_lm_baseline(
                train_texts,
                validation_texts,
                tokenizer,
                config,
            )
            item = _to_jsonable(report)
            item["seed"] = seed
            if cache_path is not None:
                cache_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
            run_reports.append(item)

    summaries = summarize_multiseed_reports(run_reports)
    decision = decide_tiny_lm_tokenizer_ablation(
        summaries_as_decision_reports(summaries),
        candidate_tokenizer=args.candidate,
    )
    payload = {
        "source": source,
        "config": {
            "tokenizers": requested,
            "seeds": seeds,
            "steps": args.steps,
            "sequence_length": args.sequence_length,
            "embedding_dim": args.embedding_dim,
            "layers": args.layers,
            "heads": args.heads,
            "batch_size": args.batch_size,
            "max_vocab_size": args.max_vocab_size,
            "tokenizer_vocab_size": args.tokenizer_vocab_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "pad_vocab_to_max_size": True,
            "resume": args.resume,
            "run_cache_dir": str(cache_dir) if cache_dir is not None else None,
            "cached_runs": cached_runs,
        },
        "runs": run_reports,
        "summaries": [_to_jsonable(summary) for summary in summaries],
        "decision": _to_jsonable(decision),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("Multi-seed tiny LM ablation")
    print(f"- source={source}")
    print(f"- seeds={seeds}")
    print(f"- runs={len(run_reports)}")
    if cache_dir is not None:
        print(f"- run_cache_dir={cache_dir}")
        print(f"- cached_runs={cached_runs}")
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"- #{rank} {summary.tokenizer}: mean_bpb={summary.mean_bits_per_byte:.3f}, "
            f"std={summary.std_bits_per_byte:.3f}, runs={summary.runs}"
        )
    print(f"- decision={decision.decision}")
    print(f"- relative_gain={decision.relative_bits_per_byte_gain:.2%}")
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


def _cache_path(cache_dir: Path | None, tokenizer_name: str, seed: int) -> Path | None:
    if cache_dir is None:
        return None
    safe_tokenizer = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in tokenizer_name
    )
    return cache_dir / f"{safe_tokenizer}_seed{seed}.json"


def _markdown(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Multi-Seed Tiny LM Ablation",
        "",
        f"Source: `{payload['source']}`",
        f"Seeds: `{', '.join(str(seed) for seed in payload['config']['seeds'])}`",
        f"Steps: `{payload['config']['steps']}`",
        "",
        "| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, summary in enumerate(payload["summaries"], start=1):
        lines.append(
            f"| {rank} | {summary['tokenizer']} | {summary['mean_bits_per_byte']:.3f} | "
            f"{summary['std_bits_per_byte']:.3f} | {summary['mean_nll_per_token']:.3f} | "
            f"{summary['mean_parameters']:.0f} | {summary['runs']} |"
        )
    lines.extend(
        [
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Best baseline: `{decision['best_baseline_tokenizer']}`",
            f"- Relative gain: `{decision['relative_bits_per_byte_gain']:.2%}`",
            f"- Recommendation: {decision['recommendation']}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
