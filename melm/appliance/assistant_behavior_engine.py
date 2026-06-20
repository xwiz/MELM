"""Creative-behaviors engine (merged-plan slice 3).

Mood-gated post-processor for synthesis output. Pure module: no store or
synthesis imports. Behaviors come from ``creative_behaviors.v1.json``; the
synthesizer calls :func:`build_behavior_context` from decision fields, then
:meth:`BehaviorEngine.evaluate` + :func:`apply_behaviors`.

Invariants:
- Disabled gate is checked by the caller; when on, behaviors only ever add a
  preamble/postamble, replace the answer, or cap its length.
- ``apply_behaviors(..., protect=True)`` (reasoning results / refusals) blocks
  ``replace_answer`` and ``max_words_override`` — faithfulness precedence.
- ``ConditionEvaluator`` is safe: no ``eval``; only the canonical variable set
  (kept in lockstep with the contract validator).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from melm.contracts.validation import load_creative_behaviors


# Behaviors whose template is prepended (vs the postamble default).
_PREAMBLE_BEHAVIORS = frozenset({"mood_narrative", "distress_callback"})


@dataclass
class BehaviorContext:
    """Condition variables, 1:1 with the contract's allowed identifiers."""
    current_mood_id: str = "neutral"
    prev_mood_id: str = "none"
    engagement: float = 0.5
    intent: str = ""
    occurrence: int = 0
    response_mode: str = "normal"
    prev_affect_has_pain: bool = False
    affect_has_fatigue: bool = False
    ambient_valence_delta: float = 0.0


@dataclass
class BehaviorResult:
    behavior_id: str
    preamble: str = ""
    postamble: str = ""
    replace_answer: str | None = None
    max_words_override: int | None = None


# --------------------------------------------------------------------------
# Safe condition evaluator
# --------------------------------------------------------------------------

_CONDITION_VARS = frozenset({
    "current_mood_id", "prev_mood_id", "engagement", "intent", "occurrence",
    "response_mode", "prev_affect_has_pain", "affect_has_fatigue",
    "ambient_valence_delta",
})
# Operators ordered by specificity so "NOT IN"/">=" win over "in"/">".
_OPERATORS = (" NOT IN ", ">=", "<=", "==", "!=", " in ", ">", "<")
_ABS_RE = re.compile(r"^abs\(\s*([A-Za-z_][A-Za-z_0-9]*)\s*\)$")


class ConditionEvaluator:
    """Evaluate a behavior condition string against a BehaviorContext."""

    def evaluate(self, condition: str, ctx: BehaviorContext) -> bool:
        # OR binds looser than AND.
        for or_clause in self._split(condition, " OR "):
            if all(self._term(t, ctx) for t in self._split(or_clause, " AND ")):
                return True
        return False

    @staticmethod
    def _split(text: str, sep: str) -> list[str]:
        return [p.strip() for p in text.split(sep) if p.strip()]

    def _term(self, term: str, ctx: BehaviorContext) -> bool:
        for op in _OPERATORS:
            if op in term:
                lhs, rhs = term.split(op, 1)
                return self._apply(op.strip(), lhs.strip(), rhs.strip(), ctx)
        # Bare boolean variable.
        return bool(self._resolve_lhs(term.strip(), ctx))

    def _apply(self, op: str, lhs: str, rhs: str, ctx: BehaviorContext) -> bool:
        lval = self._resolve_lhs(lhs, ctx)
        if op in ("in", "NOT IN"):
            members = self._parse_tuple(rhs)
            inside = str(lval) in members
            return (not inside) if op == "NOT IN" else inside
        rval = self._parse_scalar(rhs)
        if op in (">", "<", ">=", "<="):
            try:
                l, r = float(lval), float(rval)
            except (TypeError, ValueError):
                return False
            return {">": l > r, "<": l < r, ">=": l >= r, "<=": l <= r}[op]
        eq = self._eq(lval, rval)
        return eq if op == "==" else (not eq)

    def _resolve_lhs(self, token: str, ctx: BehaviorContext):
        m = _ABS_RE.match(token)
        if m:
            name = m.group(1)
            if name not in _CONDITION_VARS:
                raise ValueError(f"unknown condition variable: {name}")
            try:
                return abs(float(getattr(ctx, name)))
            except (TypeError, ValueError):
                return 0.0
        if token in _CONDITION_VARS:
            return getattr(ctx, token)
        raise ValueError(f"unknown condition variable: {token}")

    @staticmethod
    def _parse_scalar(s: str):
        s = s.strip()
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
            return s[1:-1]
        try:
            return float(s)
        except ValueError:
            return s

    @staticmethod
    def _parse_tuple(s: str) -> set[str]:
        s = s.strip().lstrip("(").rstrip(")")
        out: set[str] = set()
        for part in s.split(","):
            part = part.strip()
            if len(part) >= 2 and part[0] == "'" and part[-1] == "'":
                part = part[1:-1]
            if part:
                out.add(part)
        return out

    @staticmethod
    def _eq(lval, rval) -> bool:
        if isinstance(rval, float):
            try:
                return float(lval) == rval
            except (TypeError, ValueError):
                return False
        return str(lval) == str(rval)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class BehaviorEngine:
    """Loads behaviors, evaluates conditions, enforces per-behavior cooldowns."""

    def __init__(self, behaviors: list | None = None, evaluator: ConditionEvaluator | None = None):
        if behaviors is None:
            behaviors = load_creative_behaviors().get("behaviors", [])
        self._behaviors = list(behaviors)
        self._evaluator = evaluator or ConditionEvaluator()
        self._cooldowns: dict[str, int] = {}   # behavior_id -> turn last fired
        self._turn = 0

    def evaluate(self, ctx: BehaviorContext) -> list[BehaviorResult]:
        self._turn += 1
        results: list[BehaviorResult] = []
        for behavior in self._behaviors:
            bid = str(behavior.get("id", ""))
            cooldown = int(behavior.get("cooldown_turns", 0) or 0)
            last = self._cooldowns.get(bid)
            if last is not None and (self._turn - last) < cooldown:
                continue
            try:
                if self._evaluator.evaluate(str(behavior.get("condition", "")), ctx):
                    results.append(self._make_result(behavior, ctx))
                    self._cooldowns[bid] = self._turn
            except Exception:
                continue
        return results

    def _make_result(self, behavior: dict, ctx: BehaviorContext) -> BehaviorResult:
        bid = str(behavior.get("id", ""))
        mwo = behavior.get("max_words_override")
        max_words = int(mwo) if isinstance(mwo, int) else None
        if "templates" in behavior:
            return BehaviorResult(
                behavior_id=bid,
                replace_answer=self._render_replace(behavior, ctx),
                max_words_override=max_words,
            )
        template = str(behavior.get("template", ""))
        if bid in _PREAMBLE_BEHAVIORS:
            return BehaviorResult(behavior_id=bid, preamble=template, max_words_override=max_words)
        return BehaviorResult(behavior_id=bid, postamble=template, max_words_override=max_words)

    @staticmethod
    def _render_replace(behavior: dict, ctx: BehaviorContext) -> str:
        templates = behavior.get("templates", {})
        polarity = "negative" if ctx.ambient_valence_delta < 0 else "positive"
        arr = templates.get(polarity) or templates.get("positive") or []
        if not arr:
            return ""
        idx = (max(ctx.occurrence, 1) - 1) % len(arr)  # deterministic
        text = str(arr[idx])
        mood_word = behavior.get("mood_word_map", {}).get(ctx.prev_mood_id, ctx.prev_mood_id)
        return text.replace("{mood_word}", str(mood_word))


def build_behavior_context(decision) -> BehaviorContext:
    """Build a BehaviorContext from a fully-marshalled AssistantDecision."""
    cm = getattr(decision, "session_mood", None)
    pm = getattr(decision, "prev_mood", None)
    affect = getattr(decision, "utterance_affect", None)
    tags = tuple(getattr(affect, "dominant_tags", ()) or ()) if affect else ()
    fatigue = "fatigue" in tags or (
        affect is not None
        and getattr(affect, "confidence", 0.0) > 0
        and getattr(affect, "arousal", 1.0) < 0.2
    )
    return BehaviorContext(
        current_mood_id=getattr(cm, "mood_id", "neutral") if cm else "neutral",
        prev_mood_id=getattr(pm, "mood_id", "none") if pm else "none",
        engagement=float(getattr(cm, "engagement_level", 0.5)) if cm else 0.5,
        intent=str(getattr(decision, "intent", "")),
        occurrence=int(getattr(decision, "intent_occurrence", 0) or 0),
        response_mode=str(getattr(cm, "response_mode", "normal")) if cm else "normal",
        prev_affect_has_pain=bool(pm is not None and float(getattr(pm, "valence", 0.0)) < -0.3),
        affect_has_fatigue=bool(fatigue),
        ambient_valence_delta=float(getattr(decision, "ambient_valence_delta", 0.0) or 0.0),
    )


def apply_behaviors(answer: str, results: list[BehaviorResult], *, protect: bool = False) -> str:
    """Weave behavior modifications into *answer*.

    Precedence: ``replace_answer`` wins entirely; else ``max_words_override``
    caps length; then preamble + answer + postamble. When *protect* is True
    (reasoning result / refusal), ``replace_answer`` and ``max_words_override``
    are ignored — only preamble/postamble may apply.
    """
    pre: list[str] = []
    post: list[str] = []
    max_words: int | None = None
    for r in results:
        if not protect and r.replace_answer is not None:
            return r.replace_answer
        if not protect and r.max_words_override is not None:
            max_words = r.max_words_override if max_words is None else min(max_words, r.max_words_override)
        if r.preamble:
            pre.append(r.preamble)
        if r.postamble:
            post.append(r.postamble)
    base = answer
    if max_words is not None:
        base = " ".join(base.split()[:max_words])
    return " ".join(part for part in [*pre, base, *post] if part).strip()
