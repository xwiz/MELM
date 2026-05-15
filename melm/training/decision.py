"""Decision helpers for tiny neural tokenizer ablations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyLMTokenizerDecision:
    candidate_tokenizer: str
    best_baseline_tokenizer: str
    candidate_bits_per_byte: float
    best_baseline_bits_per_byte: float
    bits_per_byte_gain: float
    relative_bits_per_byte_gain: float
    decision: str
    recommendation: str


def decide_tiny_lm_tokenizer_ablation(
    reports: list[dict],
    *,
    candidate_tokenizer: str = "capped_morpheme",
    baseline_prefixes: tuple[str, ...] = ("hf_",),
    minimum_relative_gain: float = 0.01,
) -> TinyLMTokenizerDecision:
    """Decide whether a tiny LM ablation justifies scaled neural runs."""

    candidate = _find_report(reports, candidate_tokenizer)
    baselines = [
        report
        for report in reports
        if any(str(report["tokenizer"]).startswith(prefix) for prefix in baseline_prefixes)
    ]
    if not baselines:
        raise ValueError("Tiny LM tokenizer decision requires at least one baseline report")

    best_baseline = min(baselines, key=lambda report: float(report["validation_bits_per_byte"]))
    candidate_bpb = float(candidate["validation_bits_per_byte"])
    baseline_bpb = float(best_baseline["validation_bits_per_byte"])
    gain = baseline_bpb - candidate_bpb
    relative_gain = gain / baseline_bpb if baseline_bpb else 0.0

    if relative_gain >= minimum_relative_gain:
        decision = "promote_to_scaled_neural_ablation"
        recommendation = (
            f"Run longer BabyLM neural ablations with matched compute for {candidate_tokenizer}, capped morphology, HF BPE, and HF Unigram."
        )
    elif gain > 0.0:
        decision = "weak_scaled_candidate"
        recommendation = (
            "Repeat with more seeds or more steps before scheduling larger runs."
        )
    else:
        decision = "do_not_scale_yet"
        recommendation = (
            "Keep morphology as auxiliary supervision unless longer neural ablations recover."
        )

    return TinyLMTokenizerDecision(
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
