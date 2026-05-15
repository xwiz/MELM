"""Run public LoCoMo memory retrieval comparisons for MELM SLM Appliance."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import (
    LOCOMO_URL,
    evaluate_public_context_budget,
    evaluate_public_memory_architectures,
    load_locomo_public_memory_benchmark,
)
from melm.evaluation import bootstrap_mean_ci, bootstrap_paired_difference_ci


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="local_data/locomo10.json")
    parser.add_argument("--download", action="store_true", help="Download LoCoMo if the dataset file is missing.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=1200)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--no-event-summaries",
        action="store_true",
        help="Disable LoCoMo event-summary memory for a stricter MELM ablation.",
    )
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--out-json", default="reports/melm_public_memory_locomo.json")
    parser.add_argument("--out-md", default="reports/melm_public_memory_locomo.md")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        if not args.download:
            raise SystemExit(
                f"{dataset_path} does not exist. Rerun with --download or place LoCoMo there."
            )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(LOCOMO_URL, dataset_path)

    started = time.perf_counter()
    benchmark = load_locomo_public_memory_benchmark(dataset_path)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    report = evaluate_public_memory_architectures(
        benchmark,
        k=args.k,
        max_questions=args.max_questions,
        include_event_summaries=not args.no_event_summaries,
    )
    strict_ablation = None
    if not args.no_event_summaries and not args.skip_ablation:
        strict_ablation = evaluate_public_memory_architectures(
            benchmark,
            k=args.k,
            max_questions=args.max_questions,
            include_event_summaries=False,
        )
    eval_seconds = time.perf_counter() - started

    payload = asdict(report)
    payload["dataset_url"] = LOCOMO_URL
    payload["load_seconds"] = load_seconds
    payload["eval_seconds"] = eval_seconds
    payload["include_event_summaries"] = not args.no_event_summaries
    payload["statistics"] = _statistics(report, samples=args.bootstrap_samples)
    context_budget = evaluate_public_context_budget(
        benchmark,
        report,
        token_budget=args.context_budget,
    )
    payload["context_budget"] = {
        name: asdict(context_report)
        for name, context_report in context_budget.items()
    }
    payload["context_budget_statistical_gate_passed"] = _context_budget_gate(
        context_budget,
        samples=args.bootstrap_samples,
    )
    payload["statistical_gate_passed"] = all(
        payload["statistics"][key]["low"] > 0.0
        for key in (
            "melm_vs_mem0_additive_arch_recall_diff_ci",
            "melm_vs_memgpt_tiered_arch_recall_diff_ci",
            "melm_vs_zep_temporal_graph_arch_recall_diff_ci",
        )
    )
    payload["strict_melm_ablation"] = _strict_ablation_summary(strict_ablation) if strict_ablation else None
    payload["comparison_scope"] = {
        "claim": "local architecture-family comparison on public LoCoMo evidence retrieval",
        "not_claimed": "official Mem0, Zep, Letta, or MemGPT vendor benchmark numbers",
        "why": (
            "The proxies implement documented memory architecture ideas over the same "
            "public records without LLM APIs, so the run is reproducible and isolates "
            "memory representation/retrieval choices."
        ),
    }
    _write_outputs(payload, Path(args.out_json), Path(args.out_md))

    print("MELM public memory benchmark")
    print(f"- benchmark={report.benchmark}")
    print(f"- documents={report.documents}")
    print(f"- questions={report.questions}")
    print(f"- k={report.k}")
    print(f"- include_event_summaries={payload['include_event_summaries']}")
    print(f"- gate_passed={report.gate_passed}")
    print(f"- statistical_gate_passed={payload['statistical_gate_passed']}")
    for name, architecture in report.architectures.items():
        print(
            f"- {name}: recall={architecture.mean_recall:.2%} "
            f"hit@k={architecture.hit_at_k:.2%} full@k={architecture.full_evidence_at_k:.2%}"
        )
    if strict_ablation:
        strict_melm = strict_ablation.architectures["melm_memory_os"]
        print(f"- melm_without_event_summaries: recall={strict_melm.mean_recall:.2%}")
    melm_context = context_budget["melm_memory_os"]
    print(
        f"- melm_context_budget_{args.context_budget}: evidence={melm_context.evidence_recall:.2%} "
        f"answer_support={melm_context.answer_support_rate:.2%}"
    )


def _statistics(report, *, samples: int) -> dict:
    melm = report.architectures["melm_memory_os"].predictions
    rows = {
        "melm_memory_os_recall_ci": asdict(
            bootstrap_mean_ci(
                [prediction.recall for prediction in melm],
                samples=samples,
                seed=401,
            )
        )
    }
    for baseline_name in ("vector_rag", "mem0_additive_arch", "memgpt_tiered_arch", "zep_temporal_graph_arch"):
        baseline = report.architectures[baseline_name].predictions
        rows[f"melm_vs_{baseline_name}_recall_diff_ci"] = asdict(
            bootstrap_paired_difference_ci(
                [prediction.recall for prediction in melm],
                [prediction.recall for prediction in baseline],
                samples=samples,
                seed=500 + len(rows),
            )
        )
    return rows


def _write_outputs(payload: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _context_budget_gate(context_budget: dict, *, samples: int) -> dict:
    melm = context_budget["melm_memory_os"].predictions
    rows = {}
    for baseline_name in ("vector_rag", "mem0_additive_arch", "memgpt_tiered_arch", "zep_temporal_graph_arch"):
        baseline = context_budget[baseline_name].predictions
        rows[f"melm_vs_{baseline_name}_evidence_diff_ci"] = asdict(
            bootstrap_paired_difference_ci(
                [prediction.evidence_recall for prediction in melm],
                [prediction.evidence_recall for prediction in baseline],
                samples=samples,
                seed=700 + len(rows),
            )
        )
        answer_pairs = [
            (candidate.answer_supported, baseline_prediction.answer_supported)
            for candidate, baseline_prediction in zip(melm, baseline)
            if candidate.answer_supported is not None
            and baseline_prediction.answer_supported is not None
        ]
        rows[f"melm_vs_{baseline_name}_answer_support_diff_ci"] = asdict(
            bootstrap_paired_difference_ci(
                [bool(candidate) for candidate, _ in answer_pairs],
                [bool(baseline_value) for _, baseline_value in answer_pairs],
                samples=samples,
                seed=800 + len(rows),
            )
        )
    rows["passed"] = all(
        rows[key]["low"] > 0.0
        for key in rows
        if key.endswith("_answer_support_diff_ci")
        and key
        in {
            "melm_vs_mem0_additive_arch_answer_support_diff_ci",
            "melm_vs_memgpt_tiered_arch_answer_support_diff_ci",
            "melm_vs_zep_temporal_graph_arch_answer_support_diff_ci",
        }
    )
    return rows


def _strict_ablation_summary(report) -> dict | None:
    if report is None:
        return None
    return {
        "include_event_summaries": False,
        "melm_memory_os": {
            "mean_recall": report.architectures["melm_memory_os"].mean_recall,
            "hit_at_k": report.architectures["melm_memory_os"].hit_at_k,
            "full_evidence_at_k": report.architectures["melm_memory_os"].full_evidence_at_k,
        },
        "advantage_vs_mem0_arch": report.advantage_vs_mem0_arch,
        "advantage_vs_memgpt_arch": report.advantage_vs_memgpt_arch,
        "advantage_vs_zep_arch": report.advantage_vs_zep_arch,
    }


def _markdown(payload: dict) -> str:
    architectures = payload["architectures"]
    stats = payload["statistics"]
    lines = [
        "# MELM Public Memory Benchmark: LoCoMo",
        "",
        f"Dataset: `{payload['source_path']}`",
        f"Dataset URL: {payload['dataset_url']}",
        f"k: `{payload['k']}`",
        f"Documents/questions: `{payload['documents']}` / `{payload['questions']}`",
        f"LoCoMo event summaries enabled for MELM: `{payload['include_event_summaries']}`",
        f"Gate passed: `{payload['gate_passed']}`",
        f"Statistical gate passed: `{payload['statistical_gate_passed']}`",
        f"Context budget: `{next(iter(payload['context_budget'].values()))['token_budget']}` tokens",
        f"Context-budget answer gate passed: `{payload['context_budget_statistical_gate_passed']['passed']}`",
        f"Load/eval seconds: `{payload['load_seconds']:.3f}` / `{payload['eval_seconds']:.3f}`",
        "",
        "## Architecture-Family Results",
        "",
        "| Architecture | Mean recall@k | Hit@k | Full evidence@k | Scope |",
        "|---|---:|---:|---:|---|",
    ]
    for name, report in architectures.items():
        lines.append(
            f"| `{name}` | {report['mean_recall']:.2%} | {report['hit_at_k']:.2%} | "
            f"{report['full_evidence_at_k']:.2%} | {report['description']} |"
        )
    if payload.get("strict_melm_ablation"):
        ablation = payload["strict_melm_ablation"]
        melm = ablation["melm_memory_os"]
        lines.extend(
            [
                "",
                "## MELM Event-Summary Ablation",
                "",
                "This ablation disables LoCoMo event-summary memory and leaves MELM with raw turns, observations, session summaries, entity boosts, and temporal routing.",
                "",
                f"- MELM recall without event summaries: `{melm['mean_recall']:.2%}`",
                f"- Difference vs MemGPT-style proxy: `{ablation['advantage_vs_memgpt_arch']:.2%}`",
                f"- Difference vs Zep-style proxy: `{ablation['advantage_vs_zep_arch']:.2%}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Bounded Context Support",
            "",
            "This metric packs retrieved memories into a fixed token budget and asks whether gold evidence sessions and gold answer tokens survive in the context. It is closer to the MemGPT question than retrieval@k alone.",
            "",
            "| Architecture | Evidence recall in budget | Answer support | Mean answer-token recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, context_report in payload["context_budget"].items():
        lines.append(
            f"| `{name}` | {context_report['evidence_recall']:.2%} | "
            f"{context_report['answer_support_rate']:.2%} | "
            f"{context_report['mean_answer_token_recall']:.2%} |"
        )
    budget_stats = payload["context_budget_statistical_gate_passed"]
    lines.extend(
        [
            "",
            "| Context-budget estimate | Mean | 95% CI |",
            "|---|---:|---:|",
            _ci_row("MELM - Mem0-style answer support", budget_stats["melm_vs_mem0_additive_arch_answer_support_diff_ci"]),
            _ci_row("MELM - MemGPT-style answer support", budget_stats["melm_vs_memgpt_tiered_arch_answer_support_diff_ci"]),
            _ci_row("MELM - Zep-style answer support", budget_stats["melm_vs_zep_temporal_graph_arch_answer_support_diff_ci"]),
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Paired Bootstrap Intervals",
            "",
            "| Estimate | Mean | 95% CI |",
            "|---|---:|---:|",
            _ci_row("MELM recall", stats["melm_memory_os_recall_ci"]),
            _ci_row("MELM - vector RAG recall", stats["melm_vs_vector_rag_recall_diff_ci"]),
            _ci_row("MELM - Mem0-style recall", stats["melm_vs_mem0_additive_arch_recall_diff_ci"]),
            _ci_row("MELM - MemGPT-style recall", stats["melm_vs_memgpt_tiered_arch_recall_diff_ci"]),
            _ci_row("MELM - Zep-style recall", stats["melm_vs_zep_temporal_graph_arch_recall_diff_ci"]),
            "",
            "## Category Recall",
            "",
            "| Architecture | Multi-hop | Single-hop | Temporal | Open-domain | Adversarial |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, report in architectures.items():
        categories = report["by_category"] or {}
        lines.append(
            f"| `{name}` | {_cat(categories, 'multi_hop')} | {_cat(categories, 'single_hop')} | "
            f"{_cat(categories, 'temporal')} | {_cat(categories, 'open_domain')} | {_cat(categories, 'adversarial')} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            f"- Claim: {payload['comparison_scope']['claim']}.",
            f"- Not claimed: {payload['comparison_scope']['not_claimed']}.",
            f"- Reason: {payload['comparison_scope']['why']}",
            "",
            "## References",
            "",
            "- LoCoMo public dataset: https://github.com/snap-research/locomo",
            "- Mem0 benchmark repository: https://github.com/mem0ai/memory-benchmarks",
            "- MemGPT paper: https://arxiv.org/abs/2310.08560",
            "- Zep/Graphiti temporal knowledge graph paper: https://arxiv.org/abs/2501.13956",
            "",
        ]
    )
    return "\n".join(lines)


def _ci_row(label: str, ci: dict) -> str:
    return f"| {label} | {ci['estimate']:.2%} | [{ci['low']:.2%}, {ci['high']:.2%}] |"


def _cat(categories: dict, key: str) -> str:
    report = categories.get(key)
    if not report:
        return ""
    return f"{report['mean_recall']:.2%}"


if __name__ == "__main__":
    main()
