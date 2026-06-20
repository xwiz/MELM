"""Pluggable constrained decoder for bounded generation — M4 scaffold.

Architecture:
    AnswerPlan → ConstrainedDecoder.dispatch() → DecoderResult | None
                                                   ├─ if verified → use answer
                                                   └─ if fails verification → template fallback

Backend registry:
    - template: zero-dep, returns template_hint as-is
    - llguidance: CFG-constrained HuggingFace generation (optional, requires llguidance)
    - llamacpp: GGUF model via llama-cpp-python (optional, requires llama-cpp-python + .gguf)
    - bitnet: TQ2_0/TQ1_0 + LoRA (future)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .assistant_authority import AnswerPlan, DecoderResult
from .assistant_decoder_llama_cpp import LlamaCppBackend


@dataclass(frozen=True)
class DecodingGrammar:
    """Constraint grammar that guides decoder output.

    Fields mirror AnswerPlan constraints plus synthesis metadata:
      allowed_entities      — evidence entity labels the decoder may reference
      required_constraints  — must-include concepts (from AnswerPlan.requires)
      prohibited_tokens     — must-avoid tokens (from AnswerPlan.forbids)
      max_tokens            — maximum generation length
      template_hint         — template answer as generation seed
      mood                  — output tone (neutral, narrative, factual, refusal)
    """
    allowed_entities: tuple[str, ...] = ()
    required_constraints: tuple[str, ...] = ()
    prohibited_tokens: tuple[str, ...] = ()
    max_tokens: int = 256
    template_hint: str = ""
    mood: str = "neutral"


class DecoderBackend(Protocol):
    """Protocol each decoder backend must implement."""

    name: str

    def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
        """Return generated text or empty string on failure."""
        ...


class ConstrainedDecoder:
    """Registry + dispatcher for pluggable decoder backends.

    Backends are tried in registration order. The first non-empty result
    is returned. If all backends fail, returns None (caller falls back to template).
    """

    def __init__(self, preferred: str = "template", *, model_path: str = "") -> None:
        self._backends: dict[str, DecoderBackend] = {}
        self._preferred = preferred
        self._register_defaults()
        self.register(LlamaCppBackend(model_path=model_path))

    def _register_defaults(self) -> None:
        self.register(TemplateBackend())

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(self._backends)

    def register(self, backend: DecoderBackend) -> None:
        self._backends[backend.name] = backend

    def preferred(self, name: str | None = None) -> str | None:
        if name is not None and name in self._backends:
            self._preferred = name
        return self._preferred if self._preferred in self._backends else None

    def dispatch(
        self,
        plan: AnswerPlan,
        grammar: DecodingGrammar,
    ) -> DecoderResult | None:
        """Try preferred backend first, then all others in registration order."""
        ordered = [self._preferred] if self._preferred in self._backends else []
        ordered.extend(n for n in self._backends if n not in ordered)
        for name in ordered:
            backend = self._backends.get(name)
            if backend is None:
                continue
            try:
                text = backend.decode(plan, grammar)
                if text:
                    return DecoderResult(
                        answer=text,
                        decoder=name,
                        tokens_generated=len(text.split()),
                    )
            except Exception:
                continue
        return None


class TemplateBackend:
    """Zero-dependency fallback — returns template_hint as-is."""

    name = "template"

    def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
        return grammar.template_hint


def build_decoding_grammar(
    plan: AnswerPlan,
    template_hint: str = "",
    allowed_entities: tuple[str, ...] = (),
) -> DecodingGrammar:
    """Build a DecodingGrammar from an AnswerPlan and synthesis context."""
    return DecodingGrammar(
        allowed_entities=allowed_entities,
        required_constraints=plan.requires,
        prohibited_tokens=plan.forbids,
        template_hint=template_hint,
        mood=plan.mode,
    )
