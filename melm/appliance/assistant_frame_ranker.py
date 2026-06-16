"""E3 candidate reranker for learned frame linking (M5).

Takes rule-generated FrameCandidate list from FrameLinker and reranks using
UOL semantic support: token role alignment, class specificity, and action
centrality. K0 (FrameLinker) remains gate owner — this module only
reorders, never filters candidates below threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import CONTRACT_ROOT, validate_frame_templates
from .assistant_frame_linker import FrameCandidate


# Weights for reranking features (when token_roles is None, use first weights).
_W_RULE_SCORE = 0.50
_W_ACTION_ALIGNMENT = 0.20
_W_CLASS_SPECIFICITY = 0.15
_W_ROLE_COVERAGE = 0.15

# Additional features used when token_roles are available.
_W_PREDICATE_ALIGNMENT = 0.25  # replaces action_alignment when token_roles present
_W_OBJECT_ALIGNMENT = 0.05
_W_RULE_SCORE_UOL = 0.40  # adjusted to keep total = 1.0 with UOL features

# Tokens treated as structural stopwords (not semantically load-bearing).
_STOP_TOKENS: frozenset[str] = frozenset({
    "a", "an", "the", "me", "my", "our", "your", "I", "you", "we", "us",
    "is", "are", "was", "were", "be", "been", "am",
    "what", "when", "where", "why", "how", "who",
    "will", "can", "could", "would", "should", "may", "might",
    "to", "for", "of", "in", "on", "at", "by", "with", "about", "without",
    "it", "its", "this", "that", "these", "those",
    "and", "or", "but", "if", "so", "then",
    "do", "does", "did", "done", "doing",
    "not", "no", "nor",
    "going",  # part of "be going to" future construction
    "events",  # generic plural, no semantic class signal
})


@dataclass(frozen=True)
class ScoredCandidate:
    """Reranked candidate with combined rule + reranker score."""

    frame_id: str
    intent: str
    rule_score: float
    rerank_score: float
    rerank_explanation: str
    score_components: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0

    @property
    def score(self) -> float:
        """Expose the rerank score as `score` for threshold checks.

        The router uses this property to decide whether a candidate passes
        its effective threshold. Returning rerank_score lets the learned
        reranker influence acceptance, not just reordering.
        """
        return self.rerank_score


def _load_frame_templates() -> dict[str, dict[str, Any]]:
    path = CONTRACT_ROOT / "frame_templates.v1.json"
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_frame_templates(payload)
    return payload["templates"]


def _content_tokens(tokens: tuple[str, ...]) -> list[str]:
    """Return non-stopword tokens in order."""
    return [t for t in tokens if t not in _STOP_TOKENS]


def _action_alignment_score(
    frame_id: str,
    templates: dict[str, dict[str, Any]],
    content: list[str],
) -> float:
    """Score how well frame's action tokens align with content-bearing tokens.

    Returns 1.0 if any content token matches the frame's action_tokens list,
    0.0 otherwise.
    """
    tmpl = templates.get(frame_id)
    if tmpl is None:
        return 0.0
    action_tokens = set(tmpl.get("activation", {}).get("action_tokens", []))
    if not action_tokens:
        return 0.0
    for token in content:
        if token in action_tokens:
            return 1.0
    return 0.0


def _class_specificity_score(
    frame_id: str,
    content: list[str],
    lexicon: dict[str, frozenset[str]],
    semantic_hierarchy: dict[str, int],
    templates: dict[str, dict[str, Any]],
) -> float:
    """Score based on depth of semantic classes relevant to this frame.

    Only considers classes that appear in the frame's activation sets,
    so the score differs per frame (serving as a discriminator).
    Normalized by max depth in the hierarchy.
    """
    max_depth = max(semantic_hierarchy.values()) if semantic_hierarchy else 1
    tmpl = templates.get(frame_id)
    if tmpl is None:
        return 0.0
    act = tmpl.get("activation", {})
    frame_classes: set[str] = set()
    for key in ("required_classes", "required_all_classes", "optional_classes", "exclude_classes"):
        frame_classes.update(act.get(key, []))
    frame_classes.update(act.get("action_tokens", []))

    depths: list[float] = []
    for token in content:
        token_classes = lexicon.get(token, frozenset())
        relevant = frame_classes & token_classes
        for cls in relevant:
            depth = semantic_hierarchy.get(cls, 0)
            if depth > 0:
                depths.append(depth / max_depth)
    if not depths:
        return 0.0
    return sum(depths) / len(depths)


def _role_coverage_score(
    frame_id: str,
    tokens: tuple[str, ...],
    lexicon: dict[str, frozenset[str]],
    templates: dict[str, dict[str, Any]],
) -> float:
    """Fraction of content-bearing tokens that contribute to this frame.

    A token contributes if its semantic classes overlap with the frame's
    activation classes (required, optional, required_all, or action_tokens).
    Exclude_classes are intentionally excluded — triggering an exclude should
    penalize, not reward.
    """
    tmpl = templates.get(frame_id)
    if tmpl is None:
        return 0.0
    act = tmpl.get("activation", {})
    frame_classes: set[str] = set()
    for key in ("required_classes", "required_all_classes", "optional_classes"):
        frame_classes.update(act.get(key, []))
    frame_classes.update(act.get("action_tokens", []))

    content = _content_tokens(tokens)
    if not content:
        return 0.0

    contributing = 0
    for token in content:
        token_classes = lexicon.get(token, frozenset())
        if frame_classes & token_classes:
            contributing += 1

    return contributing / len(content)


def _find_token_by_role(
    token_roles: tuple[dict[str, Any], ...],
    target_role: str,
) -> dict[str, Any] | None:
    """Return the first token role entry matching *target_role*, or None."""
    for entry in token_roles:
        if entry.get("role") == target_role:
            return entry
    return None


def _predicate_action_alignment_score(
    frame_id: str,
    templates: dict[str, dict[str, Any]],
    token_roles: tuple[dict[str, Any], ...],
) -> float:
    """Score frame action alignment using main_predicate UOL role.

    If the main_predicate token's lemma matches any frame action token,
    returns 1.0 (strong alignment — the utterance's central verb
    directly names the frame action).

    If the main_predicate's meaning (canonical:class) contains the
    action token as a substring (e.g. complement of "with_complement"),
    returns 0.8.

    Otherwise, tries partial match: secondary_predicate_candidate,
    content_nominal tokens that match action_tokens — returns 0.5.

    Returns 0.0 if no match is found.
    """
    tmpl = templates.get(frame_id)
    if tmpl is None:
        return 0.0
    action_tokens = set(tmpl.get("activation", {}).get("action_tokens", []))
    if not action_tokens:
        return 0.0

    pred_entry = _find_token_by_role(token_roles, "main_predicate")
    if pred_entry is not None:
        lemma = pred_entry.get("lemma", "")
        if lemma in action_tokens:
            return 1.0
        meaning = pred_entry.get("meaning", "")
        if any(at in meaning for at in action_tokens):
            return 0.8

    for entry in token_roles:
        role = entry.get("role", "")
        if role in ("secondary_predicate_candidate", "content_nominal"):
            lemma = entry.get("lemma", "")
            if lemma in action_tokens:
                return 0.5
    return 0.0


def _object_alignment_score(
    frame_id: str,
    templates: dict[str, dict[str, Any]],
    token_roles: tuple[dict[str, Any], ...],
    lexicon: dict[str, frozenset[str]],
) -> float:
    """Score based on semantic_object UOL role matching frame classes.

    Returns 1.0 if the semantic_object's token has a semantic class
    that overlaps with the frame's required or optional classes.
    Returns 0.5 if the object's class overlaps with action_tokens.
    Returns 0.0 otherwise.
    """
    tmpl = templates.get(frame_id)
    if tmpl is None:
        return 0.0
    act = tmpl.get("activation", {})
    frame_classes: set[str] = set()
    for key in ("required_classes", "required_all_classes", "optional_classes"):
        frame_classes.update(act.get(key, []))
    if not frame_classes:
        return 0.0

    obj_entry = _find_token_by_role(token_roles, "semantic_object")
    if obj_entry is None:
        return 0.0

    obj_token = obj_entry.get("token", "")
    obj_classes = lexicon.get(obj_token, frozenset())
    if frame_classes & obj_classes:
        return 1.0

    action_tokens = set(act.get("action_tokens", []))
    if action_tokens & obj_classes:
        return 0.5
    return 0.0


def _build_semantic_hierarchy() -> dict[str, int]:
    """Build semantic class depth map from semantic_classes.v1.json.

    Returns {class_id: depth} where entity=1, direct children=2, etc.
    Returns empty dict if the hierarchy cannot be loaded.
    """
    try:
        path = CONTRACT_ROOT / "semantic_classes.v1.json"
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = payload.get("classes", [])
    except Exception:
        classes = []
    if not classes:
        return {}

    parent_map: dict[str, str | None] = {}
    for entry in classes:
        parent_map[entry["class_id"]] = entry.get("parent_id")

    depth_map: dict[str, int] = {}
    def _depth(cid: str) -> int:
        if cid in depth_map:
            return depth_map[cid]
        parent = parent_map.get(cid)
        if parent is None or parent == cid:
            depth_map[cid] = 1
        else:
            depth_map[cid] = _depth(parent) + 1
        return depth_map[cid]

    for cid in parent_map:
        _depth(cid)
    return depth_map


class E3CandidateReranker:
    """Reranks FrameLinker candidates using UOL semantic support features.

    Usage:
        linker = FrameLinker()
        candidates = linker.score(tokens, lexicon, is_question_like, is_request_like)
        reranker = E3CandidateReranker()
        reranked = reranker.rerank(candidates, tokens, lexicon)
        best = reranked[0]  # highest rerank_score
    """

    def __init__(self) -> None:
        self._templates = _load_frame_templates()
        self._hierarchy = _build_semantic_hierarchy()

    def rerank(
        self,
        candidates: list[FrameCandidate],
        tokens: tuple[str, ...],
        lexicon: dict[str, frozenset[str]],
        is_question_like: bool = False,
        is_request_like: bool = False,
        token_roles: tuple[dict[str, Any], ...] | None = None,
    ) -> list[ScoredCandidate]:
        """Rerank candidates by combining rule score with UOL semantic features.

        When *token_roles* is provided, uses predicate-action alignment
        (main_predicate role) and object-class alignment (semantic_object role)
        to produce more discriminative rankings. Falls back to content-token
        matching when token_roles is None.

        Args:
            candidates: output from FrameLinker.score().
            tokens: input token tuple (same as passed to FrameLinker).
            lexicon: token → frozenset of semantic class IDs.
            is_question_like: passed through for future use.
            is_request_like: passed through for future use.
            token_roles: optional tuple of token role dicts from
                FunctionalParse.token_roles. Each entry has keys:
                index, token, lemma, role, meaning, weight.
                When provided, enables predicate + object alignment features.

        Returns:
            Reranked list of ScoredCandidate, highest rerank_score first.
        """
        if not candidates:
            return []

        content = _content_tokens(tokens)
        use_uol = token_roles is not None

        scored: list[ScoredCandidate] = []
        for c in candidates:
            if use_uol:
                rule_part = c.score * _W_RULE_SCORE_UOL
            else:
                rule_part = c.score * _W_RULE_SCORE

            if use_uol:
                pred_align = _predicate_action_alignment_score(
                    c.frame_id, self._templates, token_roles,  # type: ignore[arg-type]
                )
                pred_part = pred_align * _W_PREDICATE_ALIGNMENT

                obj_align = _object_alignment_score(
                    c.frame_id, self._templates, token_roles, lexicon,  # type: ignore[arg-type]
                )
                obj_part = obj_align * _W_OBJECT_ALIGNMENT
            else:
                action_align = _action_alignment_score(
                    c.frame_id, self._templates, content,
                )
                pred_part = action_align * _W_ACTION_ALIGNMENT
                obj_part = 0.0

            spec = _class_specificity_score(
                c.frame_id, content, lexicon, self._hierarchy, self._templates,
            )
            spec_part = spec * _W_CLASS_SPECIFICITY

            coverage = _role_coverage_score(
                c.frame_id, tokens, lexicon, self._templates,
            )
            coverage_part = coverage * _W_ROLE_COVERAGE

            rerank_score = rule_part + pred_part + obj_part + spec_part + coverage_part

            # Build explanation parts (only non-zero).
            parts: list[str] = []
            if use_uol:
                parts.append(f"rule={c.score:.2f}×{_W_RULE_SCORE_UOL:.2f}={rule_part:.3f}")
                if pred_align > 0:
                    parts.append(f"predicate={pred_align:.2f}×{_W_PREDICATE_ALIGNMENT:.2f}={pred_part:.3f}")
                if obj_align > 0:
                    parts.append(f"object={obj_align:.2f}×{_W_OBJECT_ALIGNMENT:.2f}={obj_part:.3f}")
            else:
                parts.append(f"rule={c.score:.2f}×{_W_RULE_SCORE:.2f}={rule_part:.3f}")
                if action_align > 0:
                    parts.append(f"action={action_align:.2f}×{_W_ACTION_ALIGNMENT:.2f}={pred_part:.3f}")
            if spec > 0:
                parts.append(f"depth={spec:.2f}×{_W_CLASS_SPECIFICITY:.2f}={spec_part:.3f}")
            if coverage > 0:
                parts.append(f"coverage={coverage:.2f}×{_W_ROLE_COVERAGE:.2f}={coverage_part:.3f}")

            scored.append(ScoredCandidate(
                frame_id=c.frame_id,
                intent=c.intent,
                rule_score=c.score,
                rerank_score=round(rerank_score, 4),
                rerank_explanation=" + ".join(parts),
                score_components=c.score_components,
                threshold=c.threshold,
            ))

        scored.sort(key=lambda s: (-s.rerank_score, -s.rule_score, s.frame_id))
        return scored
