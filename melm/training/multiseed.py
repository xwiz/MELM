"""Aggregate multi-seed tiny LM tokenizer ablations."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MultiSeedTokenizerSummary:
    tokenizer: str
    runs: int
    mean_bits_per_byte: float
    std_bits_per_byte: float
    mean_nll_per_token: float
    std_nll_per_token: float
    mean_parameters: float
    validation_tokens: int


def summarize_multiseed_reports(reports: list[dict]) -> list[MultiSeedTokenizerSummary]:
    """Aggregate flat tiny-LM reports by tokenizer."""

    buckets: dict[str, list[dict]] = {}
    for report in reports:
        buckets.setdefault(str(report["tokenizer"]), []).append(report)

    summaries = [
        MultiSeedTokenizerSummary(
            tokenizer=tokenizer,
            runs=len(items),
            mean_bits_per_byte=_mean(
                float(item["validation_bits_per_byte"]) for item in items
            ),
            std_bits_per_byte=_std(
                [float(item["validation_bits_per_byte"]) for item in items]
            ),
            mean_nll_per_token=_mean(float(item["validation_nll"]) for item in items),
            std_nll_per_token=_std([float(item["validation_nll"]) for item in items]),
            mean_parameters=_mean(float(item["parameters"]) for item in items),
            validation_tokens=int(items[0]["validation_tokens"]),
        )
        for tokenizer, items in buckets.items()
    ]
    return sorted(summaries, key=lambda item: item.mean_bits_per_byte)


def summaries_as_decision_reports(summaries: list[MultiSeedTokenizerSummary]) -> list[dict]:
    """Convert summaries into the report shape expected by decision helpers."""

    return [
        {
            "tokenizer": summary.tokenizer,
            "validation_bits_per_byte": summary.mean_bits_per_byte,
            "validation_nll": summary.mean_nll_per_token,
            "parameters": summary.mean_parameters,
        }
        for summary in summaries
    ]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
