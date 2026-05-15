"""Decision helper for fast BabyLM tokenizer probes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FastTokenizerDecision:
    candidate_tokenizer: str
    best_baseline_tokenizer: str
    candidate_bits_per_byte: float
    best_baseline_bits_per_byte: float
    bits_per_byte_gain: float
    relative_bits_per_byte_gain: float
    decision: str
    recommendation: str


def decide_fast_tokenizer_ablation(
    reports: list[dict],
    *,
    candidate_tokenizer: str = "capped_morpheme",
    baseline_prefixes: tuple[str, ...] = ("hf_",),
    minimum_relative_gain: float = 0.01,
) -> FastTokenizerDecision:
    """Decide whether a fast tokenizer probe earns neural ablation budget."""

    candidate = _find_report(reports, candidate_tokenizer)
    baselines = [
        report
        for report in reports
        if any(str(report["tokenizer"]).startswith(prefix) for prefix in baseline_prefixes)
    ]
    if not baselines:
        raise ValueError("Fast tokenizer decision requires at least one baseline report")

    best_baseline = min(baselines, key=lambda report: float(report["bits_per_byte"]))
    candidate_bpb = float(candidate["bits_per_byte"])
    baseline_bpb = float(best_baseline["bits_per_byte"])
    gain = baseline_bpb - candidate_bpb
    relative_gain = gain / baseline_bpb if baseline_bpb else 0.0

    if relative_gain >= minimum_relative_gain:
        decision = "promote_to_neural_ablation"
        recommendation = (
            "Run matched neural BabyLM ablations for capped morphology, HF BPE, and HF Unigram."
        )
    elif gain > 0.0:
        decision = "weak_candidate"
        recommendation = (
            "Keep morphology in one more low-cost neural smoke run before spending larger compute."
        )
    else:
        decision = "auxiliary_only"
        recommendation = (
            "Do not use morphology as a primary tokenizer unless downstream neural metrics recover."
        )

    return FastTokenizerDecision(
        candidate_tokenizer=str(candidate["tokenizer"]),
        best_baseline_tokenizer=str(best_baseline["tokenizer"]),
        candidate_bits_per_byte=candidate_bpb,
        best_baseline_bits_per_byte=baseline_bpb,
        bits_per_byte_gain=gain,
        relative_bits_per_byte_gain=relative_gain,
        decision=decision,
        recommendation=recommendation,
    )


def _find_report(reports: list[dict], tokenizer: str) -> dict:
    for report in reports:
        if report.get("tokenizer") == tokenizer:
            return report
    raise ValueError(f"Missing tokenizer report for {tokenizer!r}")
