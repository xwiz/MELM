"""Passive background task checker for the Local Assistant OS.

NOT a threaded runner. Designed to be called from kernel.handle()
at the start of each turn to check for due tasks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable


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
            job = self._store.start_next_job() if hasattr(self._store, 'start_next_job') else None
            if job is None:
                break
            try:
                result = self._execute_job(job)
                self._store.complete_job(job['job_id'], json.dumps(result))
                results.append(result)
            except Exception as exc:
                self._store.fail_job(job['job_id'], str(exc))
        return results
    
    def _execute_job(self, job: dict[str, Any]) -> dict[str, Any]:
        kind = job.get('kind', '')
        payload = json.loads(job.get('payload_json', '{}')) if job.get('payload_json') else {}
        if kind == 'deferred_research':
            return self._run_deferred_research(payload)
        raise ValueError(f"Unknown job kind: {kind}")
    
    def _run_deferred_research(self, payload: dict) -> dict[str, Any]:
        """Execute a deferred research task (stub — real research is offline)."""
        topic = payload.get('topic', '')
        return {
            'kind': 'deferred_research',
            'topic': topic,
            'status': 'completed',
            'summary': f'Research for {topic} completed.',
        }
