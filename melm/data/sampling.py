"""Deterministic corpus sampling helpers."""

from __future__ import annotations


def limit_texts_by_bytes(texts: list[str], max_bytes: int | None) -> list[str]:
    """Cap a list of texts to an approximate UTF-8 byte budget.

    The budget is distributed across documents so multi-domain corpora keep some
    representation from each file. This is a deterministic probe helper, not a
    replacement for proper dataset sampling in full training.
    """

    if max_bytes is None or max_bytes <= 0:
        return texts
    if not texts:
        return []

    per_doc = max(1, max_bytes // len(texts))
    samples: list[str] = []
    used = 0
    for text in texts:
        remaining = max_bytes - used
        if remaining <= 0:
            break
        target = min(per_doc, remaining)
        sample = _prefix_by_bytes(text, target)
        if sample:
            samples.append(sample)
            used += len(sample.encode("utf-8"))

    if samples:
        return samples
    return [_prefix_by_bytes(texts[0], max_bytes)]


def _prefix_by_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
