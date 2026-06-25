"""Extract trigger response candidates from external text sources.

Reads folk tales, Gutenberg texts, and conversation transcripts to find
phrases that match trigger detection patterns. Produces
``learned_trigger_response`` entity entries in the store.

Usage::

    pipeline = TriggerLearningPipeline(store)
    pipeline.run()
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from typing import Any


_CONTRADICTION_MARKERS = re.compile(
    r"\b(?:but\s+that\s+is\s+not|that\s+is\s+not\s+true|"
    r"i\s+disagree|you\s+are\s+wrong|no[.,!]?\s+that|"
    r"actually[.,!]\s+|in\s+fact[.,!]\s+)",
    re.IGNORECASE,
)
_NEGATION_ASSERTION_MARKERS = re.compile(
    r"\b(?:is\s+not|are\s+not|was\s+not|were\s+not|"
    r"has\s+no|have\s+no|cannot|can\s+not|could\s+not)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_TRIGGER_MARKERS: dict[str, re.Pattern] = {
    "assistant_negative_assertion": _NEGATION_ASSERTION_MARKERS,
    "assistant_contradiction": _CONTRADICTION_MARKERS,
}


def _compute_confidence(text: str) -> float:
    """Score 0.4-0.9 based on how clean the sentence is."""
    words = text.split()
    if len(words) < 3:
        return 0.4
    if len(words) > 30:
        return 0.5
    has_punct = bool(re.search(r"[.!?]$", text.strip()))
    has_cap = text[0].isupper() if text else False
    score = 0.6
    if has_punct and has_cap:
        score = 0.8
    elif has_cap:
        score = 0.7
    return score


def extract_trigger_responses_from_text(
    text: str,
    source: str,
    trigger_id: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Extract response candidates matching a trigger from free text.

    Returns a list of dicts with keys ``response_text``, ``source``,
    ``context_uol``, ``confidence``, ``learned_at``.
    """
    pattern = _TRIGGER_MARKERS.get(trigger_id)
    if pattern is None:
        return []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    sentences = _SENTENCE_SPLIT.split(text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        if pattern.search(sentence):
            key = sentence.lower().strip()
            if key not in seen:
                seen.add(key)
                confidence = _compute_confidence(sentence)
                results.append({
                    "response_text": sentence,
                    "source": source,
                    "context_uol": "",
                    "confidence": confidence,
                    "learned_at": datetime.now(timezone.utc).isoformat(),
                })
    return results


class TriggerLearningPipeline:
    """Batch extraction pipeline that populates the entity store.

    Call ``run()`` after conversation processing to process folk tales,
    Gutenberg texts, and transcripts for trigger response candidates.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self._rng = random.Random(42)

    def run(self, trigger_ids: list[str] | None = None) -> dict[str, int]:
        if self.store is None:
            return {}
        if trigger_ids is None:
            trigger_ids = list(_TRIGGER_MARKERS.keys())
        counts: dict[str, int] = {}
        for tid in trigger_ids:
            entries = self._run_for_trigger(tid)
            if entries:
                counts[tid] = len(entries)
        return counts

    def _run_for_trigger(self, trigger_id: str) -> int:
        sources: list[tuple[str, str]] = self._find_source_texts()
        total = 0
        for text, source_label in sources:
            candidates = extract_trigger_responses_from_text(
                text, source_label, trigger_id, self._rng,
            )
            for c in candidates:
                self._store_candidate(trigger_id, c)
                total += 1
        return total

    def _find_source_texts(self) -> list[tuple[str, str]]:
        texts: list[tuple[str, str]] = []
        folk_tales = self._load_entities_by_kind("folk_tale")
        for tale in folk_tales:
            content = self._get_entity_text(tale.entity_id)
            if content:
                texts.append((content, "folk_tale"))
        gutenberg = self._load_entities_by_kind("gutenberg_text")
        for book in gutenberg:
            content = self._get_entity_text(book.entity_id)
            if content:
                texts.append((content, "gutenberg"))
        transcripts = self._load_entities_by_kind("transcript")
        for t in transcripts:
            content = self._get_entity_text(t.entity_id)
            if content:
                texts.append((content, "transcript"))
        return texts

    def _load_entities_by_kind(self, kind: str) -> list[Any]:
        try:
            return list(self.store.find_entities(kind=kind))
        except Exception:
            return []

    def _get_entity_text(self, entity_id: str) -> str:
        try:
            slot = self.store.get_entity_slot(entity_id, "text")
            if slot is not None and slot.value_json:
                return str(json.loads(slot.value_json))
            slot = self.store.get_entity_slot(entity_id, "content")
            if slot is not None and slot.value_json:
                return str(json.loads(slot.value_json))
        except Exception:
            pass
        return ""

    def _store_candidate(self, trigger_id: str, candidate: dict[str, Any]) -> None:
        try:
            eid = self.store._new_entity_id()
            self.store.connection.execute(
                "INSERT INTO entities(entity_id, kind, created_at) VALUES (?, ?, ?)",
                (eid, "learned_trigger_response", candidate.get("learned_at", datetime.now(timezone.utc).isoformat())),
            )
            slots = {
                "trigger_id": trigger_id,
                "response_text": candidate["response_text"],
                "variables": json.dumps({}),
                "source": candidate["source"],
                "context_uol": candidate.get("context_uol", ""),
                "confidence": candidate.get("confidence", 0.5),
                "use_count": 0,
                "learned_at": candidate.get("learned_at", datetime.now(timezone.utc).isoformat()),
            }
            for slot_name, value in slots.items():
                self.store.set_entity_slot(eid, slot_name, value)
            self.store.connection.commit()
        except Exception:
            pass
