"""Llguidance backend for constrained decoding — M4.

Requires `llguidance` and `transformers`. Falls back gracefully (empty string)
when model is unavailable or generation fails, triggering the ConstrainedDecoder's
template fallback chain.
"""

from __future__ import annotations

import re
from typing import Any

from .assistant_authority import AnswerPlan, DecoderResult
from .assistant_decoder import DecoderBackend, DecodingGrammar


class HFCompatTokenizer:
    """Adapter making HuggingFace tokenizers compatible with llguidance TokenizerWrapper.

    HuggingFace tokenizers (fast or slow) don't expose a ``tokens`` list attribute.
    This adapter builds it from ``get_vocab()`` and exposes the required interface.
    """

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tok = hf_tokenizer
        self.eos_token_id: int | None = hf_tokenizer.eos_token_id
        self.bos_token_id: int | None = hf_tokenizer.bos_token_id

        vocab = hf_tokenizer.get_vocab()
        max_id = max(vocab.values()) if vocab else 0
        tokens: list[bytes] = [b""] * (max_id + 1)
        for token_str, token_id in vocab.items():
            if token_id < len(tokens):
                tokens[token_id] = token_str.encode("utf-8")
        self.tokens: list[bytes] = tokens

        special_ids: set[int] = set()
        if hf_tokenizer.all_special_ids:
            special_ids.update(hf_tokenizer.all_special_ids)
        self.special_token_ids: list[int] = list(special_ids)
        self.is_tokenizer_wrapper = True

    def __call__(self, s: str | bytes) -> list[int]:
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="replace")
        return self._tok.encode(s)


def build_llguidance_grammar(grammar: DecodingGrammar) -> str:
    """Build an llguidance grammar definition from DecodingGrammar constraints.

    Uses the ``regex`` format, which lets us encode mood-specific sentence
    structure and prohibited-token exclusion in a single pattern.
    """
    from llguidance import grammar_from

    pattern = build_regex_pattern(grammar)
    return grammar_from("regex", pattern)


def build_regex_pattern(grammar: DecodingGrammar) -> str:
    """Build a regex pattern constraining output structure per mood + constraints."""
    if grammar.mood == "refusal":
        base = r"[A-Z][^.!?]*[.!?]"
    elif grammar.mood == "factual":
        base = r"[A-Z][^.!?]*[.!?]( [A-Z][^.!?]*[.!?])*"
    elif grammar.mood == "narrative":
        base = r"[A-Z][^.!?]*[.!?]( [A-Z][^.!?]*[.!?]){1,5}"
    else:
        base = r"[A-Za-z ,.!?]+"

    if grammar.prohibited_tokens:
        excluded = "|".join(re.escape(t) for t in grammar.prohibited_tokens)
        base = f"(?![^.]*\\b(?:{excluded})\\b){base}"

    return base


def build_llm_prompt(grammar: DecodingGrammar) -> str:
    """Build prompt for the causal LM from grammar constraints."""
    parts: list[str] = []
    if grammar.template_hint:
        parts.append(f"Continue: {grammar.template_hint}")
    else:
        if grammar.required_constraints:
            parts.append(f"Topics: {', '.join(grammar.required_constraints)}.")
        if grammar.allowed_entities:
            parts.append(f"Entities: {', '.join(grammar.allowed_entities)}.")
        if not parts:
            parts.append("Respond concisely.")
    return " ".join(parts)


# Private aliases for internal use (kept for backward compat within this module)
_build_llguidance_grammar = build_llguidance_grammar
_build_regex_pattern = build_regex_pattern
_build_llm_prompt = build_llm_prompt


class LlguidanceBackend:
    """Constrained decoder backend using llguidance + HuggingFace CausalLM.

    Lazy-loads the model on first ``decode()`` call. Returns empty string
    on any failure (no model, load failure, generation error), which the
    ``ConstrainedDecoder`` treats as "skip this backend" and falls through
    to the template fallback.
    """

    name = "llguidance"

    def __init__(self, model_name: str = "microsoft/phi-1_5") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_tok = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._tokenizer = HFCompatTokenizer(hf_tok)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model.eval()
            return True
        except Exception:
            self._model = None
            self._tokenizer = None
            return False

    def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
        if not self._ensure_loaded():
            return ""
        try:
            import torch

            from llguidance import LLInterpreter, TokenizerWrapper

            llg_grammar = build_llguidance_grammar(grammar)
            prompt = build_llm_prompt(grammar)

            tokenizer_wrapper = TokenizerWrapper(self._tokenizer)
            interpreter = LLInterpreter(tokenizer_wrapper, llg_grammar)
            interpreter.process_prompt(prompt)

            input_ids = self._tokenizer(prompt)
            generated: list[int] = list(input_ids)
            vocab_size = len(self._tokenizer.tokens)

            for _ in range(grammar.max_tokens):
                mask = interpreter.compute_mask()
                if mask is None:
                    break

                with torch.no_grad():
                    outputs = self._model(
                        torch.tensor([generated])
                    )
                    logits = outputs.logits[0, -1, :]

                allowed = [bool(m) for m in mask[:vocab_size]]
                for idx in range(logits.shape[-1]):
                    if idx < len(allowed) and not allowed[idx]:
                        logits[idx] = float("-inf")

                next_token = int(logits.argmax().item())
                generated.append(next_token)
                interpreter.commit_token(next_token)

                if interpreter.has_pending_stop():
                    break

            hf_tok = self._tokenizer._tok
            result = hf_tok.decode(
                generated[len(input_ids):], skip_special_tokens=True
            )
            return result.strip()
        except Exception:
            return ""
