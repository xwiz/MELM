"""Progression summaries across tiny LM ablation schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepTokenizerSummary:
    steps: int
    tokenizer: str
    mean_bits_per_byte: float
    std_bits_per_byte: float


@dataclass(frozen=True)
class TokenizerProgression:
    tokenizer: str
    points: list[StepTokenizerSummary]
    first_bits_per_byte: float
    last_bits_per_byte: float
    absolute_improvement: float
    relative_improvement: float


def summarize_progression(payloads: list[dict]) -> list[TokenizerProgression]:
    """Summarize tokenizer trends across multi-seed ablation payloads."""

    points_by_tokenizer: dict[str, list[StepTokenizerSummary]] = {}
    for payload in payloads:
        steps = int(payload["config"]["steps"])
        for summary in payload["summaries"]:
            point = StepTokenizerSummary(
                steps=steps,
                tokenizer=str(summary["tokenizer"]),
                mean_bits_per_byte=float(summary["mean_bits_per_byte"]),
                std_bits_per_byte=float(summary["std_bits_per_byte"]),
            )
            points_by_tokenizer.setdefault(point.tokenizer, []).append(point)

    progressions: list[TokenizerProgression] = []
    for tokenizer, points in points_by_tokenizer.items():
        ordered = sorted(points, key=lambda item: item.steps)
        first = ordered[0].mean_bits_per_byte
        last = ordered[-1].mean_bits_per_byte
        improvement = first - last
        relative = improvement / first if first else 0.0
        progressions.append(
            TokenizerProgression(
                tokenizer=tokenizer,
                points=ordered,
                first_bits_per_byte=first,
                last_bits_per_byte=last,
                absolute_improvement=improvement,
                relative_improvement=relative,
            )
        )

    return sorted(progressions, key=lambda item: item.last_bits_per_byte)
