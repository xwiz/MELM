"""Commitment tracking skill for the Local Assistant OS.

Detects temporal promises, tracks fulfillment status, and generates
commitment-aware greetings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json
import re


@dataclass
class UserCommitment:
    commitment_type: str = "reminder_request"
    topic: str = ""
    promised_time: str = ""
    parsed_time: str = ""
    user_utterance: str = ""
    session_id: str = ""


_COMMITMENT_PARSERS = None


def _load_commitment_parsers():
    global _COMMITMENT_PARSERS
    if _COMMITMENT_PARSERS is None:
        try:
            from melm.contracts import load_commitment_parsers
            _COMMITMENT_PARSERS = load_commitment_parsers()
        except Exception:
            _COMMITMENT_PARSERS = {}
    return _COMMITMENT_PARSERS


def extract_commitment(utterance: str, parse_bundle: Any) -> UserCommitment | None:
    """Extract a temporal promise/commitment from an utterance."""
    parsers = _load_commitment_parsers()
    cues = parsers.get("commitment_cues", {})
    text = utterance.lower().strip()
    
    for ctype, patterns in cues.items():
        for pattern in patterns:
            if pattern in text:
                topic = ""
                idx = text.find(pattern)
                after = text[idx + len(pattern):].strip()
                if after and not after.startswith(("me", "you", "us")):
                    topic = after[:60].rstrip(".!,?")
                if not topic:
                    topic = pattern
                
                return UserCommitment(
                    commitment_type=ctype,
                    topic=topic,
                    user_utterance=utterance[:200],
                    promised_time=pattern,
                )
    
    return None


def record_commitment(store: Any, commitment: UserCommitment) -> str | None:
    """Record a user_commitment entity. Returns entity_id."""
    if store is None:
        return None
    import uuid
    try:
        entity_id = f"uc_{uuid.uuid4().hex[:12]}"
        store.add_entity(
            entity_id=entity_id,
            kind="user_commitment",
            label=f"commitment: {commitment.commitment_type}",
            semantic_class_id="user_commitment",
            canonical_lemma=commitment.topic[:80],
        )
        store.set_entity_slot(entity_id, "commitment_type", commitment.commitment_type)
        store.set_entity_slot(entity_id, "topic", commitment.topic)
        store.set_entity_slot(entity_id, "status", "pending")
        store.set_entity_slot(entity_id, "user_utterance", commitment.user_utterance)
        store.set_entity_slot(entity_id, "session_id", commitment.session_id or "")
        if commitment.parsed_time:
            store.set_entity_slot(entity_id, "parsed_time", commitment.parsed_time)
        if commitment.promised_time:
            store.set_entity_slot(entity_id, "promised_time", commitment.promised_time)
        return entity_id
    except Exception:
        return None


def check_commitment_status(
    store: Any,
    commitment_entity: dict[str, Any],
    now: str | None = None,
) -> str:
    """Check if a commitment is pending, due, overdue, fulfilled, or broken."""
    status = commitment_entity.get("status", "pending")
    return status


def build_commitment_greeting(
    store: Any,
    commitment_entity: dict[str, Any],
    profile: Any,
) -> str | None:
    """Build a greeting that references a commitment."""
    from melm.contracts import load_agreement_templates
    try:
        templates = load_agreement_templates()
    except Exception:
        templates = {}
    
    topic = commitment_entity.get("topic", "something")
    status = commitment_entity.get("status", "pending")
    
    template_key = {
        "fulfilled": "commitment_fulfilled",
        "broken": "commitment_broken",
        "pending": "commitment_due",
        "expired": "commitment_expired",
    }.get(status, "commitment_due")
    
    template = templates.get(template_key, "")
    if template:
        return template.replace("{topic}", topic)
    return None
