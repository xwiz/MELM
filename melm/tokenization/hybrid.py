"""Hybrid morphology plus unigram tokenizer arms."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from .hf_tokenizers import HFTokenizerWrapper, train_hf_unigram
from .simple_tokenizers import HeuristicMorphemeTokenizer, WORD_RE


@dataclass(frozen=True)
class HybridMorphUnigramTokenizer:
    """Tiered tokenizer: frequent words first, morphology second, chars last.

    This models the MELM intuition that high-frequency familiar forms should be
    retrieved as fast whole units, while rarer forms should decompose into
    reusable morphemes before falling back to character-level evidence.
    """

    whole_words: frozenset[str]
    morpheme_vocab: frozenset[str]
    morpheme_tokenizer: HeuristicMorphemeTokenizer = HeuristicMorphemeTokenizer()
    char_prefix: str = "<char:"
    number_token: str = "<num>"
    name: str = "hybrid_morph_unigram"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in WORD_RE.findall(text):
            lower = _normalize_word(raw)
            if raw.isdigit():
                tokens.append(self.number_token)
                continue
            if lower is None:
                tokens.append(raw)
                continue
            if lower in self.whole_words:
                tokens.append(lower)
                continue
            for piece in self.morpheme_tokenizer._split_word(lower):
                if piece in self.morpheme_vocab:
                    tokens.append(piece)
                else:
                    tokens.extend(self._char_fallback(piece))
        return tokens

    def _char_fallback(self, piece: str) -> list[str]:
        core = piece.replace("+", "")
        if not core:
            return [piece]
        return [f"{self.char_prefix}{char}>" for char in core]


@dataclass(frozen=True)
class TieredMorphUnigramTokenizer:
    """Unigram tokenizer with lexical fast-path and morphology fallback."""

    unigram: HFTokenizerWrapper
    whole_words: frozenset[str]
    morpheme_vocab: frozenset[str]
    morpheme_tokenizer: HeuristicMorphemeTokenizer = HeuristicMorphemeTokenizer()
    morph_length_slack: int = 1
    number_token: str = "<num>"
    name: str = "tiered_morph_unigram"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in WORD_RE.findall(text):
            lower = _normalize_word(raw)
            if raw.isdigit():
                tokens.append(self.number_token)
                continue
            if lower is None:
                tokens.append(raw)
                continue
            if lower in self.whole_words:
                tokens.append(lower)
                continue

            unigram_pieces = self.unigram.tokenize(lower)
            morpheme_pieces = self.morpheme_tokenizer._split_word(lower)
            if self._should_use_morphemes(morpheme_pieces, unigram_pieces):
                tokens.extend(morpheme_pieces)
            else:
                tokens.extend(unigram_pieces)
        return tokens

    def _should_use_morphemes(
        self,
        morpheme_pieces: list[str],
        unigram_pieces: list[str],
    ) -> bool:
        if len(morpheme_pieces) <= 1:
            return False
        if not all(piece in self.morpheme_vocab for piece in morpheme_pieces):
            return False
        return len(morpheme_pieces) <= len(unigram_pieces) + self.morph_length_slack


def train_hybrid_morph_unigram(
    texts: list[str],
    *,
    vocab_size: int = 8192,
    whole_word_fraction: float = 0.70,
    min_word_frequency: int = 2,
    name: str = "hybrid_morph_unigram",
) -> HybridMorphUnigramTokenizer:
    """Train a deterministic hybrid morphology/unigram tokenizer."""

    if vocab_size < 64:
        raise ValueError("vocab_size must be at least 64 for the hybrid tokenizer")
    if not 0.0 < whole_word_fraction < 1.0:
        raise ValueError("whole_word_fraction must be between 0 and 1")

    morpheme_tokenizer = HeuristicMorphemeTokenizer()
    word_counts: Counter[str] = Counter()
    morpheme_counts: Counter[str] = Counter()
    for text in texts:
        for raw in WORD_RE.findall(text):
            lower = _normalize_word(raw)
            if lower is None:
                continue
            word_counts[lower] += 1
            morpheme_counts.update(morpheme_tokenizer._split_word(lower))

    char_tokens = {f"<char:{char}>" for char in "abcdefghijklmnopqrstuvwxyz"}
    reserved_tokens = char_tokens | {"<num>"}
    forced_morphemes = {
        f"{prefix}+"
        for prefix in morpheme_tokenizer.prefixes
    } | {
        f"+{suffix}"
        for suffix in morpheme_tokenizer.suffixes
    }

    whole_word_budget = max(1, int(vocab_size * whole_word_fraction))
    whole_words = [
        token
        for token, count in sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_word_frequency
    ][:whole_word_budget]

    reserved = len(reserved_tokens | forced_morphemes)
    morpheme_budget = max(0, vocab_size - len(whole_words) - reserved)
    learned_morphemes = [
        token
        for token, _count in sorted(morpheme_counts.items(), key=lambda item: (-item[1], item[0]))
        if token not in whole_words and token not in forced_morphemes
    ][:morpheme_budget]
    morpheme_vocab = frozenset(reserved_tokens | forced_morphemes | set(learned_morphemes))

    return HybridMorphUnigramTokenizer(
        whole_words=frozenset(whole_words),
        morpheme_vocab=morpheme_vocab,
        morpheme_tokenizer=morpheme_tokenizer,
        name=name,
    )


def train_tiered_morph_unigram(
    texts: list[str],
    *,
    vocab_size: int = 8192,
    whole_word_fraction: float = 0.20,
    morpheme_fraction: float = 0.20,
    min_word_frequency: int = 2,
    morph_length_slack: int = 1,
    name: str = "tiered_morph_unigram",
) -> TieredMorphUnigramTokenizer:
    """Train a true Unigram-plus-morphology tiered tokenizer."""

    if vocab_size < 64:
        raise ValueError("vocab_size must be at least 64 for the tiered tokenizer")
    if not 0.0 <= whole_word_fraction < 1.0:
        raise ValueError("whole_word_fraction must be in [0, 1)")
    if not 0.0 < morpheme_fraction < 1.0:
        raise ValueError("morpheme_fraction must be in (0, 1)")

    unigram_budget = max(32, int(vocab_size * (1.0 - morpheme_fraction)))
    unigram = train_hf_unigram(texts, vocab_size=unigram_budget)
    morpheme_tokenizer = HeuristicMorphemeTokenizer()
    word_counts: Counter[str] = Counter()
    morpheme_counts: Counter[str] = Counter()
    for text in texts:
        for raw in WORD_RE.findall(text):
            lower = _normalize_word(raw)
            if lower is None:
                continue
            word_counts[lower] += 1
            morpheme_counts.update(morpheme_tokenizer._split_word(lower))

    whole_word_budget = int(vocab_size * whole_word_fraction)
    whole_words = [
        token
        for token, count in sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_word_frequency
    ][:whole_word_budget]
    forced_morphemes = {
        f"{prefix}+"
        for prefix in morpheme_tokenizer.prefixes
    } | {
        f"+{suffix}"
        for suffix in morpheme_tokenizer.suffixes
    }
    morpheme_budget = max(0, vocab_size - unigram_budget - len(whole_words))
    learned_morphemes = [
        token
        for token, _count in sorted(morpheme_counts.items(), key=lambda item: (-item[1], item[0]))
        if token not in whole_words and token not in forced_morphemes
    ][:morpheme_budget]
    morpheme_vocab = frozenset(forced_morphemes | set(learned_morphemes))

    return TieredMorphUnigramTokenizer(
        unigram=unigram,
        whole_words=frozenset(whole_words),
        morpheme_vocab=morpheme_vocab,
        morpheme_tokenizer=morpheme_tokenizer,
        morph_length_slack=morph_length_slack,
        name=name,
    )


def _normalize_word(raw: str) -> str | None:
    lower = raw.lower()
    compact = lower.replace("'", "")
    if compact.isalpha():
        return compact
    return None
