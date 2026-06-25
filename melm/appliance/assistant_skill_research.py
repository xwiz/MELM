"""Research skill — open-domain topic extraction and learned-fact management.

Radial consumer of:
- contracts/open_domain_templates.v1.json (templates)
- entity store learned_fact schema (storage)
- functional_grammar FunctionalParse (topic extraction)

Does NOT contain inline knowledge. All strings come from the contract registry.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from melm.contracts import load_open_domain_templates

from .assistant_lexicon import (
    _compute_class_candidates,
    _controlled_lexemes,
    _extract_genus_lemma,
    _normalize_term,
    _timestamp,
    lexicon_ingest,
)
from .functional_grammar import FunctionalParse


# ---------------------------------------------------------------------------
# ResearchProvider protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchResult:
    """Structured result from a research provider."""

    topic: str
    summary: str
    source: str
    found: bool


class ResearchProvider(ABC):
    """Protocol for external research adapters.

    A provider queries an external source (web API, offline dictionary, etc.)
    and returns a structured ResearchResult. The kernel may then store the
    result as a learned_fact and re-answer locally.
    """

    @abstractmethod
    def research(self, topic: str) -> ResearchResult:
        """Fetch a brief summary for *topic*. Must not raise on failure."""
        ...


class StubResearchProvider(ResearchProvider):
    """In-memory provider for tests. Returns canned results by topic."""

    def __init__(self, canned: dict[str, str] | None = None) -> None:
        self.canned = canned or {}

    def research(self, topic: str) -> ResearchResult:
        summary = self.canned.get(topic.lower(), "")
        return ResearchResult(
            topic=topic,
            summary=summary,
            source="stub",
            found=bool(summary),
        )


class WikipediaResearchProvider(ResearchProvider):
    """Wikipedia REST API provider — zero-dep, no API key required."""

    BASE_URL: str = "https://en.wikipedia.org/api/rest_v1/page/summary"
    TIMEOUT_SECONDS: int = 8

    def research(self, topic: str) -> ResearchResult:
        safe = quote(topic.replace(" ", "_"), safe="_")
        url = f"{self.BASE_URL}/{safe}"
        try:
            with urlopen(url, timeout=self.TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (OSError, HTTPError, json.JSONDecodeError):
            return ResearchResult(topic=topic, summary="", source="", found=False)
        extract = payload.get("extract", "")
        if not extract:
            return ResearchResult(topic=topic, summary="", source="", found=False)
        return ResearchResult(
            topic=topic,
            summary=extract,
            source=f"wikipedia:{payload.get('content_urls', {}).get('desktop', {}).get('page', url)}",
            found=True,
        )


_DEFINITION_PATTERN = re.compile(
    r"^(?:a|an|the)\s+(?P<word>[a-z0-9' -]+)\s+is\s+(?:a|an|the)\s+(?P<rest>.+)$",
    re.IGNORECASE,
)


def research_to_lexicon(store: Any, topic: str, summary: str) -> dict | None:
    """Parse a Wikipedia summary as a definition and build a lexicon candidate.

    If the summary matches ``_DEFINITION_PATTERN`` (e.g. "A vase is an open
    container used to hold cut flowers"), extracts the genus and resolves a
    semantic class. Returns a candidate dict for ``lexicon_ingest()``, or
    ``None`` if the summary doesn't match a noun-definition pattern.
    """
    cleaned = summary.strip()
    m = _DEFINITION_PATTERN.match(cleaned)
    if not m:
        return None
    word = _normalize_term(m.group("word"))
    if word != _normalize_term(topic):
        return None
    rest = m.group("rest").strip().rstrip(".!?")
    genus_lemma = _extract_genus_lemma(rest)
    if genus_lemma:
        candidates = _compute_class_candidates(store, genus_lemma, "noun")
        if not candidates or candidates[0].get("confidence", 0) < 0.1:
            # Genus unresolvable — omit genus_lemma to bypass
            # the unresolved-genus guard in lexicon_ingest.
            genus_lemma = ""
            candidates = [{"class_id": "abstract", "method": "genus_walk", "confidence": 0.01}]
    else:
        candidates = [{"class_id": "abstract", "method": "genus_walk", "confidence": 0.01}]
    reserved, policy = _controlled_lexemes()
    candidate: dict[str, Any] = {
        "schema_id": "melm.sense_candidate.v1",
        "lemma": word,
        "language": "en",
        "pos": "noun",
        "source": {
            "provenance": "auto_research",
            "source_ref": f"wikipedia:{topic}",
            "license": "cc-by-sa",
        },
        "definition": rest,
        "semantic_class_candidates": candidates,
        "forms": [],
        "relations": [],
        "safety": {
            "reserved_conflict": word in reserved,
            "policy_term_overlap": word in policy,
        },
        "suggested_status": "active",
        "confidence_prior": 0.70,
    }
    if genus_lemma:
        candidate["genus_lemma"] = genus_lemma
    return candidate


def learn_topic(
    store: Any,
    topic: str,
    provider: ResearchProvider,
) -> ResearchResult:
    """Orchestrate auto-research: fetch, store, return.

    Idempotent — if a matching learned_fact already exists, returns it
    without calling the provider again.

    After storing a learned_fact, also attempts to parse the Wikipedia
    summary as a definition and ingest into the lexicon via
    ``research_to_lexicon()``, making the term recognizable.
    """
    if store is None:
        return provider.research(topic)
    existing = find_learned_fact(store, topic)
    if existing is not None:
        return ResearchResult(
            topic=existing["topic"],
            summary=existing["summary"],
            source=existing["source"],
            found=True,
        )
    result = provider.research(topic)
    if result.found and result.summary:
        record_learned_fact(
            store,
            topic=result.topic,
            summary=result.summary,
            source=result.source,
        )
        candidate = research_to_lexicon(store, topic, result.summary)
        if candidate is not None:
            try:
                lexicon_ingest(store, candidate, expected_provenance="auto_research")
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Topic extraction from T1 parse
# ---------------------------------------------------------------------------

def _get_attr(parse: Any, key: str) -> Any:
    """Unified accessor for dataclass or dict."""
    if parse is None:
        return None
    if isinstance(parse, dict):
        return parse.get(key)
    return getattr(parse, key, None)


def extract_topic(parse: FunctionalParse | dict[str, Any] | None) -> str | None:
    """Pick the most contentful noun from the functional parse.

    Accepts either a FunctionalParse dataclass or the dict produced by
    AssistantDecision.functional_parse.

    Priority: object > complement_action > target > subject > semantic_unknown_tokens
    """
    if parse is None:
        return None
    for key in ("object", "complement_action", "target", "subject"):
        val = _get_attr(parse, key)
        if val and len(str(val)) > 2:
            return str(val)
    unknowns = _get_attr(parse, "semantic_unknown_tokens")
    if unknowns:
        for u in unknowns:
            ustr = str(u)
            if len(ustr) > 2:
                return ustr
    return None


def extract_action(parse: FunctionalParse | dict[str, Any] | None) -> str:
    """Return the action verb from the parse, or 'learn about' as fallback."""
    action = _get_attr(parse, "action")
    return str(action) if action else "learn about"


# ---------------------------------------------------------------------------
# Learned-fact entity store helpers
# ---------------------------------------------------------------------------

def find_learned_fact(store: Any, topic: str) -> dict[str, Any] | None:
    """Search entity store for a learned_fact whose topic slot matches *topic*.

    Returns the first match (best-effort) as a dict with entity_id and slots.
    """
    if store is None:
        return None
    # find_entities(kind="object", semantic_class_id="learned_fact")
    entities = store.find_entities(kind="object", semantic_class_id="learned_fact")
    topic_lower = topic.lower()
    for ent in entities:
        slots = store.get_entity_slots(ent.entity_id)
        slot_map: dict[str, str] = {}
        for s in slots:
            raw = s.value_json
            try:
                val = json.loads(raw) if raw else ""
            except json.JSONDecodeError:
                val = raw
            slot_map[s.slot_name] = str(val) if val is not None else ""
        if topic_lower in (slot_map.get("topic") or "").lower():
            return {
                "entity_id": ent.entity_id,
                "topic": slot_map.get("topic", ""),
                "summary": slot_map.get("summary", ""),
                "source": slot_map.get("source", ""),
                "learned_at": slot_map.get("learned_at", ""),
            }
    return None


def record_learned_fact(
    store: Any,
    topic: str,
    summary: str,
    source: str = "",
) -> str:
    """Create a learned_fact entity and populate its slots.

    Returns the entity_id.
    """
    now = datetime.now(timezone.utc).isoformat()
    entity_id = f"learned_fact:{topic.lower().replace(' ', '_')}:{now}"
    store.add_entity(
        entity_id=entity_id,
        kind="object",
        label=topic,
        semantic_class_id="learned_fact",
    )
    store.set_entity_slot(entity_id, "topic", topic)
    store.set_entity_slot(entity_id, "summary", summary)
    store.set_entity_slot(entity_id, "learned_at", now)
    if source:
        store.set_entity_slot(entity_id, "source", source)
    store.connection.commit()
    return entity_id


# ---------------------------------------------------------------------------
# Open-domain answer formatting (contract-driven)
# ---------------------------------------------------------------------------

def format_open_domain_answer(
    topic: str,
    action: str = "learn about",
    learned_fact: dict[str, Any] | None = None,
    speech_act: str = "",
) -> str:
    """Render an open_domain/unknown fallback using the contract template registry.

    Selects template variant by speech_act type:
    - wh_question → inquisitive tone
    - yes_no_question → search-oriented
    - statement/claim → reflective
    - request/command → action-oriented
    - default → generic handoff

    Within each speech_act group, three variants exist:
    - "handoff" (no learned_fact, needs external fetch)
    - "learned" (learned_fact evidence present)
    - "unknown" (fallback when no learned_fact and no handoff possible)
    """
    templates = load_open_domain_templates()
    speech_act_group = templates.get("speech_act_templates", {})
    group_templates = speech_act_group.get(speech_act) or speech_act_group.get("default", {})
    if learned_fact:
        tmpl = group_templates.get(
            "learned",
            "You asked about {topic}. This is what I found: {summary}",
        )
        return tmpl.format(
            topic=topic,
            action=action,
            summary=learned_fact.get("summary", ""),
        )
    tmpl = group_templates.get(
        "handoff",
        "You asked about {topic}. I do not have that information locally, so I will need to look it up.",
    )
    return tmpl.format(topic=topic, action=action)
