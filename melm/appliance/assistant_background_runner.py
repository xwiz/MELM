"""Passive background task checker for the Local Assistant OS.

NOT a threaded runner. Designed to be called from kernel.handle()
at the start of each turn to check for due tasks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from .assistant_os_store import StoredInventoryJob


class BackgroundTaskRunner:
    """Lightweight task checker that runs inline during handle().
    
    This is a passive checker: call tick() at the start of each turn
    to execute due tasks synchronously. No threading, no asyncio.
    """
    
    def __init__(
        self,
        store: Any,
        interval_seconds: int = 60,
        max_jobs_per_tick: int = 1,
    ):
        self._store = store
        self._interval = interval_seconds
        self._max_jobs = max_jobs_per_tick
        self._last_tick: float = 0.0
        self._running = False
    
    def start(self) -> None:
        """Mark runner as active (no-op — tick() is called manually)."""
        self._running = True
    
    def stop(self) -> None:
        """Mark runner as inactive."""
        self._running = False
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def tick(self) -> list[dict[str, Any]]:
        """Check for and execute due tasks. Returns list of completed job results."""
        if not self._running or self._store is None:
            return []
        now = datetime.now(timezone.utc)
        if self._last_tick > 0:
            elapsed = now.timestamp() - self._last_tick
            if elapsed < self._interval:
                return []
        self._last_tick = now.timestamp()
        
        results = []
        for _ in range(self._max_jobs):
            job = self._store.start_next_job(kinds=['deferred_research']) if hasattr(self._store, 'start_next_job') else None
            if job is None:
                break
            try:
                result = self._execute_job(job)
                self._store.complete_job(job.job_id, result=result)
                results.append(result)
            except Exception as exc:
                self._store.fail_job(job.job_id, error=str(exc))
        return results
    
    def _execute_job(self, job: StoredInventoryJob) -> dict[str, Any]:
        kind = job.kind
        payload = job.payload
        if kind == 'deferred_research':
            return self._run_deferred_research(payload)
        raise ValueError(f"Unknown job kind: {kind}")
    
    def _run_deferred_research(self, payload: dict) -> dict[str, Any]:
        """Execute a deferred research task with offline enrichment.
        
        Performs local knowledge enrichment: scans learned_fact entities,
        lexical_senses, and lightweight contracts for the topic. Records
        findings as a learned_fact entity and updates the deferred_task
        entity if applicable.
        """
        topic = payload.get('topic', '')
        if not topic:
            return {'kind': 'deferred_research', 'topic': '', 'status': 'failed', 'error': 'empty_topic'}
        
        enrichment = _enrich_topic_offline(self._store, topic)
        summary = enrichment.get('summary', '')
        confidence = enrichment.get('confidence', 0.0)
        source = enrichment.get('source', 'local:offline_enrichment')
        
        if not summary:
            summary = f"Noted topic '{topic}' but no local information available."
        
        # Record as learned_fact if we found something meaningful
        if summary and confidence >= 0.3 and self._store is not None:
            try:
                from .assistant_skill_research import record_learned_fact, find_learned_fact
                existing = find_learned_fact(self._store, topic)
                if existing is None:
                    record_learned_fact(self._store, topic, summary, source=source)
            except Exception:
                pass
        
        # Update deferred_task entity if session_id is available
        session_id = payload.get('session_id', '')
        if session_id and self._store is not None:
            try:
                from .assistant_skill_deferred_tasks import complete_deferred_task
                entity_id = f"dt_{topic.lower().replace(' ', '_')}_{session_id[:8]}"
                complete_deferred_task(self._store, entity_id, result_summary=summary)
            except Exception:
                pass
        
        status = 'completed' if summary else 'completed_no_results'
        return {
            'kind': 'deferred_research',
            'topic': topic,
            'status': status,
            'summary': summary,
            'confidence': confidence,
            'source': source,
        }


# ---------------------------------------------------------------------------
# Offline enrichment helpers
# ---------------------------------------------------------------------------

_ENRICHMENT_CACHE: dict[str, dict[str, Any]] = {}


def _enrich_topic_offline(store: Any, topic: str) -> dict[str, Any]:
    """Enrich a topic using only local data (no network, no cloud).
    
    Returns {summary, confidence, source}.
    """
    # Check cache
    cached = _ENRICHMENT_CACHE.get(topic)
    if cached is not None:
        return cached
    
    fragments: list[str] = []
    best_confidence = 0.0
    sources: list[str] = []
    topic_lower = topic.lower()
    
    # 1. Check learned_fact entities
    if store is not None:
        try:
            from .assistant_skill_research import find_learned_fact
            fact = find_learned_fact(store, topic)
            if fact and fact.get('summary'):
                fragments.append(fact['summary'])
                best_confidence = max(best_confidence, 0.95)
                sources.append('entity:learned_fact')
        except Exception:
            pass
    
    # 2. Check lexical_senses for semantic relatives
    if store is not None:
        try:
            from .assistant_lexicon import lookup_lexical_senses
            senses = lookup_lexical_senses(store, topic_lower)
            if senses:
                classes = {s.get('semantic_class_id', '') for s in senses if isinstance(s, dict)}
                domain_hints = [c.replace('_', ' ') for c in classes if c]
                if domain_hints:
                    fragments.append(f"Related semantic domains: {', '.join(domain_hints[:3])}.")
                    best_confidence = max(best_confidence, 0.7)
                    sources.append('lexical_senses')
                if not fragments:
                    fragments.append(f"I found {topic} in the local lexicon with domain: {domain_hints[0]}.")
                    best_confidence = max(best_confidence, 0.5)
                    sources.append('lexical_senses')
        except Exception:
            pass
    
    # 3. Check geo_atlas contract
    if best_confidence < 0.9:
        try:
            from melm.contracts import load_geo_atlas
            atlas = load_geo_atlas()
            entries = atlas.get('locations', {})
            if topic_lower in {k.lower() for k in entries}:
                fragments.append(f"I know from the local atlas that {topic} is a named location.")
                best_confidence = max(best_confidence, 0.8)
                sources.append('contract:geo_atlas.v1')
        except Exception:
            pass
    
    # 4. Check food_tags contract
    if best_confidence < 0.8:
        try:
            from melm.contracts import load_food_tags
            tags = load_food_tags()
            if topic_lower in {k.lower() for k in tags}:
                fragments.append(f"I know that {topic} is a food item.")
                best_confidence = max(best_confidence, 0.7)
                sources.append('contract:food_tags.v1')
        except Exception:
            pass
    
    # 5. Check weather_concepts contract
    if best_confidence < 0.7:
        try:
            from melm.contracts import load_weather_concepts
            concepts = load_weather_concepts()
            if topic_lower in {c.lower() for c in concepts}:
                fragments.append(f"I know that {topic} is a weather concept.")
                best_confidence = max(best_confidence, 0.7)
                sources.append('contract:weather_concepts.v1')
        except Exception:
            pass
    
    result = {
        'summary': ' '.join(fragments) if fragments else '',
        'confidence': best_confidence,
        'source': '+'.join(sources) if sources else 'local:offline_enrichment',
    }
    _ENRICHMENT_CACHE[topic] = result
    return result


def clear_enrichment_cache() -> None:
    """Clear the offline enrichment cache (for tests)."""
    _ENRICHMENT_CACHE.clear()
