"""Split loading helpers for reports and probes."""

from __future__ import annotations

from pathlib import Path

from .corpus import load_texts
from .manifest import CorpusManifest, load_manifest, texts_for_split


def load_train_validation(
    *,
    path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[list[str], list[str], list[str], str]:
    """Load all/train/validation texts from either a path or a manifest."""

    if manifest_path:
        manifest = load_manifest(manifest_path)
        train = texts_for_split(manifest, "train")
        validation = texts_for_split(manifest, "validation")
        test = texts_for_split(manifest, "test")
        all_texts = train + validation + test
        return all_texts, train, validation or test or train, f"manifest:{manifest_path}"

    if path is None:
        raise ValueError("Either path or manifest_path is required")

    from melm.tokenization import split_for_lm

    texts = load_texts(path)
    train, validation = split_for_lm(texts)
    return texts, train, validation, str(path)
