"""Minimal corpus loading helpers."""

from __future__ import annotations

from pathlib import Path


def load_texts(path: str | Path, *, suffixes: tuple[str, ...] = (".txt", ".md")) -> list[str]:
    """Load text files from a file or directory.

    This deliberately avoids dataset-specific assumptions. BabyLM and CHILDES
    adapters should wrap this with source manifests and license checks later.
    """

    source = Path(path)
    if source.is_file():
        return [source.read_text(encoding="utf-8")]
    if not source.exists():
        raise FileNotFoundError(source)

    texts: list[str] = []
    for file_path in sorted(source.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in suffixes:
            texts.append(file_path.read_text(encoding="utf-8"))
    return texts
