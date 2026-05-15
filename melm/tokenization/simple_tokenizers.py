"""Small deterministic tokenizer prototypes.

These are not intended to replace Hugging Face tokenizers. They provide a
dependency-free harness for early MELM validation metrics and smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    name: str

    def tokenize(self, text: str) -> list[str]:
        """Return tokens for a string."""


@dataclass(frozen=True)
class WhitespaceTokenizer:
    """Word-ish baseline using regex splitting."""

    name: str = "whitespace"

    def tokenize(self, text: str) -> list[str]:
        return WORD_RE.findall(text)


@dataclass(frozen=True)
class UnigramLikeTokenizer:
    """A tiny longest-prefix tokenizer over an explicit vocabulary.

    This is a cheap stand-in for a trained Unigram tokenizer. Unknown alphabetic
    spans fall back to characters so OOV behavior remains measurable.
    """

    vocab: frozenset[str]
    name: str = "unigram_like"

    def tokenize(self, text: str) -> list[str]:
        pieces: list[str] = []
        for raw in WORD_RE.findall(text):
            lower = raw.lower()
            if not lower.isalpha():
                pieces.append(raw)
                continue
            i = 0
            while i < len(lower):
                match = None
                for j in range(len(lower), i, -1):
                    candidate = lower[i:j]
                    if candidate in self.vocab:
                        match = candidate
                        break
                if match is None:
                    pieces.append(f"<unk:{lower[i]}>")
                    i += 1
                else:
                    pieces.append(match)
                    i += len(match)
        return pieces


@dataclass(frozen=True)
class HeuristicMorphemeTokenizer:
    """English-oriented morpheme heuristic for early ablations.

    The rules deliberately stay simple: common prefixes, common suffixes, and a
    residual root. This creates a fast proxy for measuring whether morphology is
    worth deeper tooling.
    """

    prefixes: tuple[str, ...] = (
        "anti",
        "counter",
        "dis",
        "inter",
        "mis",
        "non",
        "over",
        "pre",
        "re",
        "sub",
        "trans",
        "un",
        "under",
    )
    suffixes: tuple[str, ...] = (
        "ability",
        "able",
        "ation",
        "ed",
        "er",
        "est",
        "ful",
        "ing",
        "ion",
        "ish",
        "ity",
        "less",
        "ly",
        "ment",
        "ness",
        "ous",
        "s",
        "tion",
    )
    min_root_len: int = 3
    name: str = "heuristic_morpheme"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in WORD_RE.findall(text):
            lower = raw.lower()
            if not lower.isalpha():
                tokens.append(raw)
                continue
            tokens.extend(self._split_word(lower))
        return tokens

    def _split_word(self, word: str) -> list[str]:
        pieces: list[str] = []
        rest = word

        for prefix in sorted(self.prefixes, key=len, reverse=True):
            if rest.startswith(prefix) and len(rest) - len(prefix) >= self.min_root_len:
                pieces.append(f"{prefix}+")
                rest = rest[len(prefix) :]
                break

        suffix_stack: list[str] = []
        changed = True
        while changed:
            changed = False
            for suffix in sorted(self.suffixes, key=len, reverse=True):
                if rest.endswith(suffix) and len(rest) - len(suffix) >= self.min_root_len:
                    suffix_stack.append(f"+{suffix}")
                    rest = rest[: -len(suffix)]
                    changed = True
                    break

        if rest:
            pieces.append(rest)
        pieces.extend(reversed(suffix_stack))
        return pieces or [word]


@dataclass(frozen=True)
class BytePatchTokenizer:
    """Byte/patch baseline using fixed-size UTF-8 byte chunks."""

    patch_size: int = 4
    name: str = "byte_patch"

    def tokenize(self, text: str) -> list[str]:
        data = text.encode("utf-8")
        return [
            data[i : i + self.patch_size].hex()
            for i in range(0, len(data), self.patch_size)
        ]
