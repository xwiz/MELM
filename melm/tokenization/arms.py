"""Named tokenizer-arm construction for experiments."""

from __future__ import annotations

from .factory import build_default_tokenizers
from .hf_tokenizers import train_hf_bpe, train_hf_unigram
from .hybrid import train_hybrid_morph_unigram, train_tiered_morph_unigram
from .simple_tokenizers import HeuristicMorphemeTokenizer, Tokenizer
from .vocab import cap_tokenizer_vocab


DEFAULT_TOKENIZER_ARMS = frozenset(
    {
        "whitespace",
        "simple_bpe",
        "heuristic_morpheme",
        "byte_patch",
        "unigram_like",
    }
)
FAST_TOKENIZER_ARMS = frozenset(
    {
        "hf_bpe",
        "hf_unigram",
        "capped_morpheme",
        "hybrid_morph_unigram",
        "tiered_morph_unigram",
    }
)


def build_tokenizer_arms(
    requested: list[str],
    texts: list[str],
    train_texts: list[str],
    *,
    tokenizer_vocab_size: int,
) -> dict[str, Tokenizer]:
    """Build the requested tokenizer arms in request order."""

    tokenizers: dict[str, Tokenizer] = {}
    if any(name in DEFAULT_TOKENIZER_ARMS for name in requested):
        tokenizers.update(
            {
                tokenizer.name: tokenizer
                for tokenizer in build_default_tokenizers(texts, train_texts=train_texts)
            }
        )
    if "hf_bpe" in requested:
        tokenizers["hf_bpe"] = train_hf_bpe(train_texts, vocab_size=tokenizer_vocab_size)
    if "hf_unigram" in requested:
        tokenizers["hf_unigram"] = train_hf_unigram(train_texts, vocab_size=tokenizer_vocab_size)
    if "capped_morpheme" in requested:
        tokenizers["capped_morpheme"] = cap_tokenizer_vocab(
            HeuristicMorphemeTokenizer(),
            train_texts,
            vocab_size=tokenizer_vocab_size,
            name="capped_morpheme",
        )
    if "hybrid_morph_unigram" in requested:
        tokenizers["hybrid_morph_unigram"] = train_hybrid_morph_unigram(
            train_texts,
            vocab_size=tokenizer_vocab_size,
        )
    if "tiered_morph_unigram" in requested:
        tokenizers["tiered_morph_unigram"] = train_tiered_morph_unigram(
            train_texts,
            vocab_size=tokenizer_vocab_size,
        )

    missing = [name for name in requested if name not in tokenizers]
    if missing:
        available = sorted(DEFAULT_TOKENIZER_ARMS | FAST_TOKENIZER_ARMS)
        raise ValueError(
            f"Unknown tokenizer arm(s) {missing}; available: {', '.join(available)}"
        )
    return {name: tokenizers[name] for name in requested}
