"""Optional Hugging Face tokenizers baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HFTokenizerWrapper:
    tokenizer: Any
    name: str

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens


def train_hf_bpe(
    texts: list[str],
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    lowercase: bool = True,
) -> HFTokenizerWrapper:
    """Train a fast BPE baseline using the optional `tokenizers` package."""

    tokenizers = _import_tokenizers()
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    _configure_normalization(tokenizer, tokenizers, lowercase)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return HFTokenizerWrapper(tokenizer=tokenizer, name="hf_bpe")


def train_hf_unigram(
    texts: list[str],
    *,
    vocab_size: int = 8192,
    lowercase: bool = True,
) -> HFTokenizerWrapper:
    """Train a fast Unigram baseline using the optional `tokenizers` package."""

    tokenizers = _import_tokenizers()
    tokenizer = tokenizers.Tokenizer(tokenizers.models.Unigram())
    _configure_normalization(tokenizer, tokenizers, lowercase)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    trainer = tokenizers.trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        unk_token="<unk>",
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return HFTokenizerWrapper(tokenizer=tokenizer, name="hf_unigram")


def _configure_normalization(tokenizer: Any, tokenizers: Any, lowercase: bool) -> None:
    normalizers = [tokenizers.normalizers.NFKC()]
    if lowercase:
        normalizers.append(tokenizers.normalizers.Lowercase())
    tokenizer.normalizer = tokenizers.normalizers.Sequence(normalizers)


def _import_tokenizers():
    try:
        import tokenizers
    except ImportError as exc:
        raise RuntimeError(
            "Fast tokenizer baselines require the optional `tokenizers` package."
        ) from exc
    return tokenizers
