"""GGUF decoder backend via llama-cpp-python (lazy-loaded).

M4 pluggable decoder architecture — registered in ``ConstrainedDecoder``
alongside ``template`` and ``llguidance``.  On first ``decode()`` call the
backend attempts to load a local GGUF model (e.g. Qwen2.5-0.5B-Instruct
Q4_K_M).  If the package or model is missing it returns an empty string,
which the dispatcher treats as "skip this backend" and falls through to
the next registered backend.

Platform notes:
- **Linux/macOS**: ``pip install llama-cpp-python`` uses pre-built wheels
  when available, or compiles from source via CMake.
- **Windows**: Verified on Windows 11 + Python 3.13.  If ``pip install``
  fails with long-path errors during source extraction, use a short temp
  directory::

      $env:TEMP = "C:\tmp"; $env:TMP = "C:\tmp"
      pip install llama-cpp-python --no-cache-dir

The backend respects the ``DecodingGrammar`` contract:
  * ``template_hint``  → prompt seed / system instruction
  * ``mood``           → temperature mapping (factual=0.3, narrative=0.7)
  * ``max_tokens``     → generation budget
  * ``required_constraints`` / ``prohibited_tokens``  → currently advisory
    (0.5B models have limited instruction-following for token-level bans;
    authority verification remains the safety gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .assistant_authority import AnswerPlan, DecoderResult


# Temperature map keyed by ``DecodingGrammar.mood``.
_TEMPERATURE_MAP: dict[str, float] = {
    "factual": 0.30,
    "neutral": 0.50,
    "narrative": 0.70,
    "refusal": 0.20,
}

# Default context length for Qwen2.5-0.5B on CPU with 8 GB RAM.
_DEFAULT_N_CTX = 1024

# Conservative thread count — avoids thermal penalty on laptops.
_DEFAULT_N_THREADS = 4


@dataclass
class LlamaCppBackend:
    """Lazy-loading GGUF backend using llama-cpp-python.

    Parameters
    ----------
    model_path:
        Absolute or relative path to the ``.gguf`` file.
    n_ctx:
        Context window in tokens.  Smaller = lower RAM.
    n_threads:
        CPU threads for matrix multiplication.
    verbose:
        Passed through to ``llama_cpp.Llama``.
    """

    name: str = "llamacpp"
    model_path: str = ""
    n_ctx: int = _DEFAULT_N_CTX
    n_threads: int = _DEFAULT_N_THREADS
    verbose: bool = False
    _llm: Any | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._llm is not None:
            return True
        if not self.model_path:
            return False
        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=self.verbose,
            )
            return True
        except Exception:
            self._llm = None
            return False

    def _build_messages(self, plan: AnswerPlan, grammar: DecodingGrammar) -> list[dict[str, str]]:
        """Assemble chat messages for ``create_chat_completion``.

        ``grammar.template_hint`` may contain a double-newline split:
        everything before ``\n\n`` is the system message; everything after
        is the user message.  If there is no split we treat the whole
        string as the system message and derive the user message from
        ``plan.question``.
        """
        hint = grammar.template_hint or (
            "You are a helpful, concise assistant running entirely on the user's device. "
            "Keep answers short and accurate."
        )
        if "\n\n" in hint:
            system, user = hint.split("\n\n", 1)
        else:
            system = hint
            user = f"Provide a {plan.mode} answer for route '{plan.route}'."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _temperature(self, grammar: DecodingGrammar) -> float:
        return _TEMPERATURE_MAP.get(grammar.mood, 0.50)

    # ------------------------------------------------------------------
    # DecoderBackend protocol
    # ------------------------------------------------------------------

    def decode(self, plan: AnswerPlan, grammar: DecodingGrammar) -> str:
        if not self._ensure_loaded():
            return ""
        try:
            messages = self._build_messages(plan, grammar)
            temperature = self._temperature(grammar)
            output = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=grammar.max_tokens,
                temperature=temperature,
            )
            text = str(output["choices"][0]["message"]["content"]).strip()
            return text
        except Exception:
            return ""

