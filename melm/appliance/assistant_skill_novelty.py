"""Novelty detection skill for the Local Assistant OS.

Detects lexical/cultural novelty as a side-effect of the UOL parse pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class NoveltyCandidate:
    surface_form: str
    utterance_context: str
    detection_reason: str
    decomposition: str = ""
    proposed_class_id: str = ""
    confidence: float = 0.5


_NOVELTY_PATTERNS = None


def _load_novelty_patterns():
    global _NOVELTY_PATTERNS
    if _NOVELTY_PATTERNS is None:
        try:
            from melm.contracts import load_novelty_patterns
            _NOVELTY_PATTERNS = load_novelty_patterns()
        except Exception:
            _NOVELTY_PATTERNS = {}
    return _NOVELTY_PATTERNS


def _is_palindrome(word: str, min_length: int = 3) -> bool:
    cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
    return len(cleaned) >= min_length and cleaned == cleaned[::-1]


def _has_morpheme_boundary(word: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat.startswith("-") and word.endswith(pat[1:]):
            return True
        if pat.endswith("-") and word.startswith(pat[:-1]):
            return True
    return False


def _check_cultural_symbol(word: str, symbols: dict) -> str | None:
    word_lower = word.lower()
    for category, config in symbols.items():
        for pattern in config.get("patterns", []):
            if pattern.lower() in word_lower:
                return category
    return None


def detect_novelty(
    parse_bundle: Any,
    lexicon: Any,
    store: Any,
) -> list[NoveltyCandidate]:
    """Detect novelty from the parse bundle. Returns list of NoveltyCandidate."""
    patterns = _load_novelty_patterns()
    candidates: list[NoveltyCandidate] = []
    
    unknown_tokens = []
    if parse_bundle is not None:
        unknown_tokens = getattr(parse_bundle, "semantic_unknown_tokens", [])
    
    utterance = ""
    if parse_bundle is not None:
        utterance = getattr(parse_bundle, "text", "") or ""
    
    for token in unknown_tokens:
        reasons = []
        decomposition_parts = []
        
        palindrome_config = patterns.get("palindrome", {})
        min_len = palindrome_config.get("min_length", 3)
        if _is_palindrome(token, min_len):
            reasons.append("palindrome")
            decomposition_parts.append(f"palindrome({len(token)} chars)")
        
        morpheme_patterns = patterns.get("morpheme_patterns", [])
        if _has_morpheme_boundary(token, morpheme_patterns):
            reasons.append("morpheme_cluster")
            for mp in morpheme_patterns:
                if mp.startswith("-") and token.endswith(mp[1:]):
                    stem = token[:-len(mp[1:])] if len(token) > len(mp[1:]) else ""
                    if stem:
                        decomposition_parts.append(f"stem={stem}+suffix={mp}")
                elif mp.endswith("-") and token.startswith(mp[:-1]):
                    rest = token[len(mp[:-1]):] if len(token) > len(mp[:-1]) else ""
                    if rest:
                        decomposition_parts.append(f"prefix={mp}+root={rest}")
        
        cultural_symbols = patterns.get("cultural_symbols", {})
        symbol_category = _check_cultural_symbol(token, cultural_symbols)
        if symbol_category:
            reasons.append("cultural_symbol")
            decomposition_parts.append(f"category={symbol_category}")
        
        if reasons:
            candidates.append(NoveltyCandidate(
                surface_form=token,
                utterance_context=utterance[:200],
                detection_reason=",".join(reasons),
                decomposition="; ".join(decomposition_parts),
                confidence=0.5 + 0.15 * len(reasons),
            ))
    
    return candidates


def record_novelty_candidates(
    store: Any,
    candidates: list[NoveltyCandidate],
) -> list[str]:
    """Record novelty candidates as entities in the store. Returns entity_ids."""
    if store is None:
        return []
    import uuid
    entity_ids = []
    for cand in candidates:
        try:
            entity_id = f"nc_{uuid.uuid4().hex[:12]}"
            store.add_entity(
                entity_id=entity_id,
                kind="novelty_candidate",
                label=f"novelty: {cand.surface_form[:40]}",
                semantic_class_id="novelty_candidate",
                canonical_lemma=cand.surface_form[:80],
            )
            store.set_entity_slot(entity_id, "surface_form", cand.surface_form)
            store.set_entity_slot(entity_id, "utterance_context", cand.utterance_context)
            store.set_entity_slot(entity_id, "detection_reason", cand.detection_reason)
            store.set_entity_slot(entity_id, "decomposition", cand.decomposition)
            store.set_entity_slot(entity_id, "review_status", "flagged")
            store.set_entity_slot(entity_id, "confidence", str(cand.confidence))
            entity_ids.append(entity_id)
        except Exception:
            pass
    return entity_ids


def build_novelty_digest(store: Any, session_id: str) -> str:
    """Build a summary of novel items from a session."""
    if store is None:
        return ""
    try:
        entities = store.find_entities(kind="novelty_candidate")
        flagged = []
        for ent in entities:
            eid = ent.entity_id
            status_slot = store.get_entity_slot(eid, "review_status")
            if status_slot and status_slot.value_json and "flagged" in status_slot.value_json:
                form_slot = store.get_entity_slot(eid, "surface_form")
                reason_slot = store.get_entity_slot(eid, "detection_reason")
                form = form_slot.value_json if form_slot and form_slot.value_json else "unknown"
                flagged.append(f"'{form}' ({reason_slot.value_json if reason_slot and reason_slot.value_json else 'unknown'})")
        if flagged:
            return f"Novel items found: {', '.join(flagged)}."
        return ""
    except Exception:
        return ""
