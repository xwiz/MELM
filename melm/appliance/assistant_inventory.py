"""Metadata-only local inventory builders for Assistant OS v0.1."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import gzip
from io import StringIO
import json
from pathlib import Path
import re
from time import sleep
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .local_assistant_router import LocalAssistantProfile


DEFAULT_STORY_METADATA = Path("benchmarks/public_domain_story_metadata.json")
DEFAULT_LOCAL_MEDIA_MANIFEST = Path("benchmarks/local_media_manifest.json")
DEFAULT_GUTENBERG_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
DEFAULT_INTERNET_ARCHIVE_SCRAPE_ENDPOINT = "https://archive.org/services/search/v1/scrape"
DEFAULT_INTERNET_ARCHIVE_QUERY = "collection:gutenberg AND mediatype:texts"
USER_AGENT = "MELM Local Assistant OS metadata importer/0.1"
MIN_STORY_METADATA_QUALITY = 0.5
MIN_INTERNET_ARCHIVE_PAGE_SIZE = 100
SUPPORTED_MEDIA_EXTENSIONS = {
    ".aac": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    ".mp3": "audio",
    ".ogg": "audio",
    ".wav": "audio",
    ".webm": "audio",
    ".mp4": "video",
    ".mkv": "video",
}


@dataclass(frozen=True)
class StoryMetadata:
    item_id: str
    title: str
    source: str
    source_url: str
    license: str
    age_min: int
    age_max: int
    topics: tuple[str, ...]
    cultures: tuple[str, ...]
    summary: str
    narrative_frame: str

    @property
    def template(self) -> str:
        """Backward-compatible alias for older fixture/database rows."""

        return self.narrative_frame


@dataclass(frozen=True)
class StoryInventoryBuildResult:
    story_models: dict[str, str]
    selected_items: tuple[StoryMetadata, ...]
    source_count: int
    source_path: str


@dataclass(frozen=True)
class StoryMetadataImportResult:
    source: str
    source_url: str
    items: tuple[StoryMetadata, ...]
    source_count: int
    selected_count: int
    rejected_count: int
    max_source_bytes: int
    network_used: bool
    observability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_url": self.source_url,
            "source_count": self.source_count,
            "selected_count": self.selected_count,
            "rejected_count": self.rejected_count,
            "max_source_bytes": self.max_source_bytes,
            "network_used": self.network_used,
            "observability": self.observability,
            "items": [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "source": item.source,
                    "source_url": item.source_url,
                    "license": item.license,
                    "age_min": item.age_min,
                    "age_max": item.age_max,
                    "topics": list(item.topics),
                    "cultures": list(item.cultures),
                    "summary": item.summary,
                }
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class LocalMediaItem:
    item_id: str
    title: str
    kind: str
    path: str
    tags: tuple[str, ...]
    source: str
    license: str
    path_exists: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalMediaImportResult:
    source: str
    source_path: str
    items: tuple[LocalMediaItem, ...]
    source_count: int
    selected_count: int
    rejected_count: int
    network_used: bool = False
    observability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_path": self.source_path,
            "source_count": self.source_count,
            "selected_count": self.selected_count,
            "rejected_count": self.rejected_count,
            "network_used": self.network_used,
            "observability": self.observability,
            "items": [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "kind": item.kind,
                    "path": item.path,
                    "tags": list(item.tags),
                    "source": item.source,
                    "license": item.license,
                    "path_exists": item.path_exists,
                }
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class InventoryRefreshRecommendation:
    kind: str
    reason: str
    priority: float
    job_id: str
    resource_budget: dict[str, Any]
    priority_signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "priority": self.priority,
            "job_id": self.job_id,
            "resource_budget": self.resource_budget,
            "priority_signals": self.priority_signals,
        }


@dataclass(frozen=True)
class InventoryRefreshScheduleReport:
    recommendations: tuple[InventoryRefreshRecommendation, ...]
    story_inventory_count: int
    weather_today_cached: bool
    queued_jobs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendations": [item.to_dict() for item in self.recommendations],
            "story_inventory_count": self.story_inventory_count,
            "weather_today_cached": self.weather_today_cached,
            "queued_jobs": self.queued_jobs,
        }


@dataclass(frozen=True)
class _FetchBytesResult:
    data: bytes
    attempts: int


class PublicDomainStoryMetadataAdapter:
    """Selects local story frames from source metadata, not raw scraping."""

    def __init__(self, metadata_path: str | Path = DEFAULT_STORY_METADATA) -> None:
        self.metadata_path = Path(metadata_path)

    def load(self) -> tuple[StoryMetadata, ...]:
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "melm.public_domain_story_metadata.v1":
            raise ValueError(f"unsupported story metadata schema: {payload.get('schema')!r}")
        return tuple(_metadata_from_record(record) for record in payload.get("items", []))

    def build_story_inventory(
        self,
        profile: LocalAssistantProfile,
        *,
        limit: int = 3,
    ) -> StoryInventoryBuildResult:
        items = self.load()
        ranked = sorted(
            items,
            key=lambda item: _rank_story(item, profile),
            reverse=True,
        )
        selected = tuple(item for item in ranked if _rank_story(item, profile) > 0.0)[:limit]
        if not selected:
            selected = ranked[:limit]
        return StoryInventoryBuildResult(
            story_models={item.item_id: item.narrative_frame for item in selected},
            selected_items=selected,
            source_count=len(items),
            source_path=str(self.metadata_path),
        )


class LocalMediaInventoryAdapter:
    """Builds local media inventory from a JSON manifest or device directory."""

    source = "local_media_manifest"

    def __init__(self, manifest_path: str | Path = DEFAULT_LOCAL_MEDIA_MANIFEST) -> None:
        self.manifest_path = Path(manifest_path)

    def import_manifest(
        self,
        profile: LocalAssistantProfile,
        *,
        limit: int = 24,
        require_files: bool = False,
    ) -> LocalMediaImportResult:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "melm.local_media_manifest.v1":
            raise ValueError(f"unsupported local media manifest schema: {payload.get('schema')!r}")
        rows = tuple(dict(record) for record in payload.get("items", []))
        parsed = tuple(
            item
            for record in rows
            if (item := _local_media_item_from_manifest(record, self.manifest_path)) is not None
        )
        if require_files:
            filtered = tuple(item for item in parsed if item.path_exists)
        else:
            filtered = parsed
        selected = _ranked_media_selection(filtered, profile, limit=limit)
        return LocalMediaImportResult(
            source=self.source,
            source_path=str(self.manifest_path),
            items=selected,
            source_count=len(rows),
            selected_count=len(selected),
            rejected_count=len(rows) - len(filtered),
            network_used=False,
            observability={
                "require_files": require_files,
                "missing_file_count": sum(1 for item in parsed if not item.path_exists),
                "candidate_count": len(parsed),
                "selected_tags": tuple(dict.fromkeys(tag for item in selected for tag in item.tags)),
            },
        )

    def import_directory(
        self,
        media_dir: str | Path,
        profile: LocalAssistantProfile,
        *,
        limit: int = 100,
        recursive: bool = True,
    ) -> LocalMediaImportResult:
        root = Path(media_dir)
        paths = root.rglob("*") if recursive else root.glob("*")
        candidates = tuple(
            _local_media_item_from_path(path, root)
            for path in sorted(paths)
            if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
        )
        selected = _ranked_media_selection(candidates, profile, limit=limit)
        return LocalMediaImportResult(
            source="local_media_directory",
            source_path=str(root),
            items=selected,
            source_count=len(candidates),
            selected_count=len(selected),
            rejected_count=0,
            network_used=False,
            observability={
                "recursive": recursive,
                "supported_extensions": tuple(sorted(SUPPORTED_MEDIA_EXTENSIONS)),
                "selected_tags": tuple(dict.fromkeys(tag for item in selected for tag in item.tags)),
            },
        )


class ProjectGutenbergCatalogImporter:
    """Metadata-only adapter for Project Gutenberg catalog CSV records."""

    source = "project_gutenberg_catalog_csv"

    def __init__(self, source_url: str = DEFAULT_GUTENBERG_CATALOG_URL) -> None:
        self.source_url = source_url

    def import_metadata(
        self,
        profile: LocalAssistantProfile,
        *,
        limit: int = 12,
        max_source_bytes: int = 6_500_000,
        timeout: float = 20.0,
        max_attempts: int = 2,
        backoff_seconds: float = 0.5,
    ) -> StoryMetadataImportResult:
        fetch = _fetch_url_bytes_observed(
            self.source_url,
            max_source_bytes=max_source_bytes,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        if self.source_url.endswith(".gz"):
            text = gzip.decompress(fetch.data).decode("utf-8", errors="replace")
        else:
            text = fetch.data.decode("utf-8", errors="replace")
        return self.import_csv_text(
            text,
            profile,
            limit=limit,
            max_source_bytes=max_source_bytes,
            source_url=self.source_url,
            network_used=True,
            fetch_observability={
                "fetch_attempts": fetch.attempts,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
            },
        )

    def import_csv_path(
        self,
        path: str | Path,
        profile: LocalAssistantProfile,
        *,
        limit: int = 12,
        max_source_bytes: int = 6_500_000,
    ) -> StoryMetadataImportResult:
        source = Path(path)
        raw = source.read_bytes()
        if len(raw) > max_source_bytes:
            raise ValueError(f"Gutenberg sample exceeds max_source_bytes={max_source_bytes}")
        if source.suffix == ".gz":
            text = gzip.decompress(raw).decode("utf-8", errors="replace")
        else:
            text = raw.decode("utf-8", errors="replace")
        return self.import_csv_text(
            text,
            profile,
            limit=limit,
            max_source_bytes=max_source_bytes,
            source_url=str(source),
            network_used=False,
        )

    def import_csv_text(
        self,
        text: str,
        profile: LocalAssistantProfile,
        *,
        limit: int = 12,
        max_source_bytes: int = 6_500_000,
        source_url: str = "memory:gutenberg_csv",
        network_used: bool = False,
        fetch_observability: dict[str, Any] | None = None,
    ) -> StoryMetadataImportResult:
        rows = tuple(csv.DictReader(StringIO(text)))
        candidates = tuple(
            item
            for row in rows
            if (item := _gutenberg_story_from_row(row)) is not None
        )
        selection = _ranked_selection_observed(candidates, profile, limit=limit)
        return StoryMetadataImportResult(
            source=self.source,
            source_url=source_url,
            items=selection["items"],
            source_count=len(rows),
            selected_count=len(selection["items"]),
            rejected_count=max(0, len(rows) - len(candidates)),
            max_source_bytes=max_source_bytes,
            network_used=network_used,
            observability={
                **selection["observability"],
                **(fetch_observability or {}),
            },
        )


class InternetArchiveSearchMetadataImporter:
    """Metadata-only adapter for Internet Archive search/scrape results."""

    source = "internet_archive_search_metadata"

    def __init__(
        self,
        endpoint: str = DEFAULT_INTERNET_ARCHIVE_SCRAPE_ENDPOINT,
        *,
        query: str = DEFAULT_INTERNET_ARCHIVE_QUERY,
    ) -> None:
        self.endpoint = endpoint
        self.query = query

    def import_metadata(
        self,
        profile: LocalAssistantProfile,
        *,
        limit: int = 12,
        max_source_bytes: int = 250_000,
        timeout: float = 20.0,
        max_attempts: int = 2,
        backoff_seconds: float = 0.5,
        page_size: int = 100,
        max_pages: int = 1,
        cursor: str | None = None,
        rate_limit_delay_seconds: float = 0.0,
    ) -> StoryMetadataImportResult:
        bounded_page_size = max(MIN_INTERNET_ARCHIVE_PAGE_SIZE, page_size)
        bounded_max_pages = max(1, max_pages)
        page_rows: list[dict[str, Any]] = []
        page_urls: list[str] = []
        page_item_counts: list[int] = []
        page_fetch_attempts: list[int] = []
        cursors_seen: list[str] = []
        total_source_bytes = 0
        fetch_attempts_total = 0
        sleep_count = 0
        sleep_total = 0.0
        current_cursor = cursor or ""
        next_cursor = ""
        byte_budget_exhausted = False
        for page_index in range(bounded_max_pages):
            if page_index > 0 and rate_limit_delay_seconds > 0:
                sleep(rate_limit_delay_seconds)
                sleep_count += 1
                sleep_total += rate_limit_delay_seconds
            source_url = self._source_url(page_size=bounded_page_size, cursor=current_cursor or None)
            remaining_bytes = max_source_bytes - total_source_bytes
            if remaining_bytes <= 0:
                byte_budget_exhausted = True
                break
            fetch = _fetch_url_bytes_observed(
                source_url,
                max_source_bytes=remaining_bytes,
                timeout=timeout,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
            total_source_bytes += len(fetch.data)
            fetch_attempts_total += fetch.attempts
            page_fetch_attempts.append(fetch.attempts)
            page_urls.append(source_url)
            cursors_seen.append(current_cursor)
            payload = json.loads(
                fetch.data.decode(
                    "utf-8",
                    errors="replace",
                )
            )
            rows = tuple(dict(item) for item in payload.get("items", []))
            page_rows.extend(rows)
            page_item_counts.append(len(rows))
            next_cursor = str(payload.get("cursor") or payload.get("next_cursor") or "")
            if not next_cursor or next_cursor in cursors_seen:
                break
            current_cursor = next_cursor
        source_url = page_urls[0] if page_urls else self._source_url(page_size=bounded_page_size, cursor=cursor)
        return self.import_search_payload(
            {"items": page_rows},
            profile,
            limit=limit,
            max_source_bytes=max_source_bytes,
            source_url=source_url,
            network_used=True,
            pagination={
                "page_size": bounded_page_size,
                "max_pages": bounded_max_pages,
                "page_count": len(page_urls),
                "page_item_counts": page_item_counts,
                "page_urls": page_urls,
                "cursors_seen": cursors_seen,
                "cursor": cursor or "",
                "next_cursor": next_cursor,
                "rate_limit_delay_seconds": rate_limit_delay_seconds,
                "rate_limit_sleep_count": sleep_count,
                "rate_limit_delay_total_seconds": round(sleep_total, 3),
                "fetch_attempts": fetch_attempts_total,
                "fetch_attempts_total": fetch_attempts_total,
                "page_fetch_attempts": page_fetch_attempts,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
                "total_source_bytes": total_source_bytes,
                "byte_budget_exhausted": byte_budget_exhausted,
            },
        )

    def _source_url(self, *, page_size: int, cursor: str | None = None) -> str:
        params = urlencode(
            {
                "q": self.query,
                "fields": "identifier,title,creator,subject,language,licenseurl,collection",
                "count": str(page_size),
                **({"cursor": cursor} if cursor else {}),
            }
        )
        return f"{self.endpoint}?{params}"

    def import_json_path(
        self,
        path: str | Path,
        profile: LocalAssistantProfile,
        *,
        limit: int = 12,
        max_source_bytes: int = 250_000,
    ) -> StoryMetadataImportResult:
        source = Path(path)
        raw = source.read_bytes()
        if len(raw) > max_source_bytes:
            raise ValueError(f"Internet Archive sample exceeds max_source_bytes={max_source_bytes}")
        return self.import_search_payload(
            json.loads(raw.decode("utf-8")),
            profile,
            limit=limit,
            max_source_bytes=max_source_bytes,
            source_url=str(source),
            network_used=False,
        )

    def import_search_payload(
        self,
        payload: dict[str, Any],
        profile: LocalAssistantProfile,
        *,
        limit: int = 12,
        max_source_bytes: int = 250_000,
        source_url: str = "memory:internet_archive_search",
        network_used: bool = False,
        pagination: dict[str, Any] | None = None,
    ) -> StoryMetadataImportResult:
        rows = tuple(dict(item) for item in payload.get("items", []))
        candidates = tuple(
            item
            for row in rows
            if (item := _internet_archive_story_from_row(row)) is not None
        )
        selection = _ranked_selection_observed(candidates, profile, limit=limit)
        return StoryMetadataImportResult(
            source=self.source,
            source_url=source_url,
            items=selection["items"],
            source_count=len(rows),
            selected_count=len(selection["items"]),
            rejected_count=max(0, len(rows) - len(candidates)),
            max_source_bytes=max_source_bytes,
            network_used=network_used,
            observability={
                **selection["observability"],
                **(pagination or {}),
            },
        )


def story_items_to_inventory_rows(
    items: Iterable[StoryMetadata],
    *,
    profile: LocalAssistantProfile,
) -> tuple[dict, ...]:
    return tuple(
        {
            "kind": "story_model",
            "item_id": item.item_id,
            "payload": {
                "title": item.title,
                "narrative_frame": item.narrative_frame,
                "summary": item.summary,
                "source_url": item.source_url,
                "age_min": item.age_min,
                "age_max": item.age_max,
                "topics": item.topics,
                "cultures": item.cultures,
                "quality_score": round(_rank_story(item, profile) + _metadata_quality_score(item), 3),
                "local_fit_score": round(_rank_story(item, profile), 3),
                "metadata_quality": round(_metadata_quality_score(item), 3),
            },
            "source": item.source,
            "license": item.license,
            "tags": tuple(
                dict.fromkeys(
                    (
                        "story",
                        f"age:{profile.age}",
                        *item.topics,
                        *item.cultures,
                    )
                )
            ),
        }
        for item in items
    )


def media_items_to_inventory_rows(items: Iterable[LocalMediaItem]) -> tuple[dict, ...]:
    return tuple(
        {
            "kind": "media",
            "item_id": item.item_id,
            "payload": {
                "title": item.title,
                "kind": item.kind,
                "path": item.path,
                "path_exists": item.path_exists,
                "tags": item.tags,
                "metadata": item.metadata,
            },
            "source": item.source,
            "license": item.license,
            "tags": tuple(dict.fromkeys(("media", item.kind, *item.tags))),
        }
        for item in items
    )


def schedule_inventory_refreshes(
    store: Any,
    profile: LocalAssistantProfile,
    *,
    min_story_models: int = 6,
    story_limit: int = 6,
    source: str = "both",
    use_offline_samples: bool = False,
    gutenberg_csv: str | Path | None = None,
    internet_archive_json: str | Path | None = None,
    internet_archive_query: str | None = None,
    gutenberg_max_source_bytes: int = 6_500_000,
    internet_archive_max_source_bytes: int = 250_000,
    internet_archive_page_size: int = 100,
    internet_archive_max_pages: int = 1,
    internet_archive_cursor: str | None = None,
    internet_archive_rate_limit_delay_seconds: float = 0.0,
) -> InventoryRefreshScheduleReport:
    """Queue Pi-budgeted inventory refresh jobs when local caches look thin."""

    story_count = len(store.load_inventory("story_model"))
    weather_today_cached = "today" in profile.weekly_weather
    priority_context = _priority_context(store)
    recommendations: list[InventoryRefreshRecommendation] = []
    if story_count < min_story_models:
        reason = f"story inventory has {story_count} items; target is {min_story_models}"
        story_priority_signals = _story_priority_signals(
            store,
            priority_context=priority_context,
            story_count=story_count,
            min_story_models=min_story_models,
        )
        story_priority = _bounded_priority(
            0.72
            + story_priority_signals["inventory_gap_ratio"] * 0.16
            + min(0.1, story_priority_signals["recent_story_cloud_handoffs"] * 0.025)
            + max(0.0, story_priority_signals["cloud_dependence_delta"]) * 0.08
            + story_priority_signals["avg_uncertainty"] * 0.05
            + max(0.0, -story_priority_signals["local_resolution_delta"]) * 0.04
            + story_priority_signals["story_inventory_gap_persistence"] * 0.03
            - story_priority_signals["recent_failed_jobs"] * 0.05
        )
        payload = {
            "source": source,
            "limit": story_limit,
            "reason": reason,
            "min_story_models": min_story_models,
            "gutenberg_max_source_bytes": gutenberg_max_source_bytes,
            "internet_archive_max_source_bytes": internet_archive_max_source_bytes,
            "internet_archive_page_size": max(MIN_INTERNET_ARCHIVE_PAGE_SIZE, internet_archive_page_size),
            "internet_archive_max_pages": max(1, internet_archive_max_pages),
            "internet_archive_cursor": internet_archive_cursor or "",
            "internet_archive_query": internet_archive_query or DEFAULT_INTERNET_ARCHIVE_QUERY,
            "internet_archive_rate_limit_delay_seconds": max(
                0.0,
                internet_archive_rate_limit_delay_seconds,
            ),
            "priority_signals": story_priority_signals,
        }
        if use_offline_samples:
            payload["gutenberg_csv"] = str(gutenberg_csv or "benchmarks/sample_gutenberg_catalog.csv")
            payload["internet_archive_json"] = str(
                internet_archive_json or "benchmarks/sample_internet_archive_search.json"
            )
        recommendations.append(
            _queue_refresh_job(
                store,
                kind="import_story_metadata",
                reason=reason,
                priority=story_priority,
                evidence_key="story_model_thin",
                proposed_action="refresh public-domain story metadata inventory",
                source_candidates=(
                    "project_gutenberg_catalog_metadata",
                    "internet_archive_item_search_and_metadata",
                ),
                payload=payload,
                resource_budget={
                    "max_items": story_limit,
                    "max_source_bytes": gutenberg_max_source_bytes + internet_archive_max_source_bytes,
                    "network": "metadata_only" if not use_offline_samples else "offline_fixture",
                    "cpu_class": "raspberry_pi",
                    "dedupe": "sqlite_primary_key",
                    "quality_floor": MIN_STORY_METADATA_QUALITY,
                    "internet_archive_page_size": max(MIN_INTERNET_ARCHIVE_PAGE_SIZE, internet_archive_page_size),
                    "internet_archive_max_pages": max(1, internet_archive_max_pages),
                    "internet_archive_query": internet_archive_query or DEFAULT_INTERNET_ARCHIVE_QUERY,
                    "internet_archive_rate_limit_delay_seconds": max(
                        0.0,
                        internet_archive_rate_limit_delay_seconds,
                    ),
                },
                priority_signals=story_priority_signals,
            )
        )
    if not weather_today_cached:
        reason = "today weather cache is missing or stale"
        weather_priority_signals = _weather_priority_signals(
            store,
            priority_context=priority_context,
            weather_today_cached=weather_today_cached,
        )
        weather_priority = _bounded_priority(
            0.66
            + (0.08 if not weather_today_cached else 0.0)
            + min(0.1, weather_priority_signals["recent_weather_misses"] * 0.04)
            + max(0.0, -weather_priority_signals["cache_freshness_delta"]) * 0.08
            + weather_priority_signals["avg_cloud_dependence"] * 0.03
            + weather_priority_signals["weather_cache_gap_persistence"] * 0.04
            - weather_priority_signals["recent_failed_jobs"] * 0.05
        )
        recommendations.append(
            _queue_refresh_job(
                store,
                kind="refresh_weather_cache",
                reason=reason,
                priority=weather_priority,
                evidence_key="weather_today_stale_or_missing",
                proposed_action="refresh local weather cache before answering weather-dependent requests",
                source_candidates=("weather_tool",),
                payload={"reason": reason, "target_day": "today"},
                resource_budget={
                    "max_items": 7,
                    "max_source_bytes": 20000,
                    "network": "tool_fetch",
                    "cpu_class": "raspberry_pi",
                    "cache_policy": "refresh_stale_or_missing_today",
                },
                priority_signals=weather_priority_signals,
            )
        )
    return InventoryRefreshScheduleReport(
        recommendations=tuple(recommendations),
        story_inventory_count=story_count,
        weather_today_cached=weather_today_cached,
        queued_jobs=len(store.load_jobs(status="queued")),
    )


def _queue_refresh_job(
    store: Any,
    *,
    kind: str,
    reason: str,
    priority: float,
    evidence_key: str,
    proposed_action: str,
    source_candidates: tuple[str, ...],
    payload: dict[str, Any],
    resource_budget: dict[str, Any],
    priority_signals: dict[str, Any],
) -> InventoryRefreshRecommendation:
    evidence_event_ids = (f"inventory_scheduler_{evidence_key}",)
    base_job_id = f"{kind}:{'_'.join(evidence_event_ids)}"
    store.save_opportunity(
        kind=kind,
        priority=priority,
        reason=reason,
        evidence_event_ids=evidence_event_ids,
        expected_cloud_reduction=1 if kind == "import_story_metadata" else 0,
        proposed_action=proposed_action,
        source_candidates=source_candidates,
    )
    job_id, refresh_cycle = _next_refresh_job_id(store, base_job_id)
    store.enqueue_job(
        kind=kind,
        payload={
            **payload,
            "opportunity_id": base_job_id,
            "refresh_cycle": refresh_cycle,
            "refresh_cycle_base_job_id": base_job_id,
            "evidence_event_ids": evidence_event_ids,
            "proposed_action": proposed_action,
            "source_candidates": source_candidates,
            "priority_signals": priority_signals,
        },
        priority=priority,
        resource_budget={
            **resource_budget,
            "refresh_cycle": refresh_cycle,
            "refresh_cycle_base_job_id": base_job_id,
        },
        job_id=job_id,
    )
    return InventoryRefreshRecommendation(
        kind=kind,
        reason=reason,
        priority=priority,
        job_id=job_id,
        resource_budget=resource_budget,
        priority_signals=priority_signals,
    )


def _next_refresh_job_id(store: Any, base_job_id: str) -> tuple[str, int]:
    rows = store.connection.execute(
        """
        SELECT job_id, status
        FROM jobs
        WHERE job_id=? OR job_id GLOB ?
        ORDER BY created_at
        """,
        (base_job_id, f"{base_job_id}:cycle_*"),
    ).fetchall()
    for row in rows:
        status = str(row["status"])
        if status in {"queued", "running"}:
            job_id = str(row["job_id"])
            return job_id, _refresh_cycle_from_job_id(base_job_id, job_id)
    if not rows:
        return base_job_id, 1
    next_cycle = max(_refresh_cycle_from_job_id(base_job_id, str(row["job_id"])) for row in rows) + 1
    return f"{base_job_id}:cycle_{next_cycle}", next_cycle


def _refresh_cycle_from_job_id(base_job_id: str, job_id: str) -> int:
    if job_id == base_job_id:
        return 1
    match = re.fullmatch(rf"{re.escape(base_job_id)}:cycle_(\d+)", job_id)
    return int(match.group(1)) if match else 1


def _priority_context(store: Any) -> dict[str, Any]:
    homeostasis = _recent_homeostasis(store)
    return {
        "homeostasis": homeostasis,
        "events": _recent_event_counts(store),
        "failed_jobs": _failed_job_counts(store),
        "self_observation": _self_observation_priority_context(store),
    }


def _story_priority_signals(
    store: Any,
    *,
    priority_context: dict[str, Any],
    story_count: int,
    min_story_models: int,
) -> dict[str, Any]:
    homeostasis = priority_context["homeostasis"]
    events = priority_context["events"]
    failed_jobs = priority_context["failed_jobs"]
    self_observation = priority_context["self_observation"]
    inventory_gap = max(0, min_story_models - story_count)
    return {
        "inventory_gap": inventory_gap,
        "inventory_gap_ratio": round(inventory_gap / max(1, min_story_models), 3),
        "recent_story_cloud_handoffs": int(events.get("story_cloud_handoffs", 0)),
        "avg_cloud_dependence": homeostasis["avg_cloud_dependence"],
        "cloud_dependence_delta": homeostasis["cloud_dependence_delta"],
        "avg_uncertainty": homeostasis["avg_uncertainty"],
        "avg_inventory_coverage": homeostasis["avg_inventory_coverage"],
        "self_observation_points": self_observation["points"],
        "local_resolution_delta": self_observation["local_resolution_delta"],
        "story_inventory_gap_persistence": self_observation["story_inventory_gap_persistence"],
        "recent_failed_jobs": int(failed_jobs.get("import_story_metadata", 0)),
        "expected_local_resolution_gain": round(inventory_gap / max(1, min_story_models), 3),
    }


def _weather_priority_signals(
    store: Any,
    *,
    priority_context: dict[str, Any],
    weather_today_cached: bool,
) -> dict[str, Any]:
    homeostasis = priority_context["homeostasis"]
    events = priority_context["events"]
    failed_jobs = priority_context["failed_jobs"]
    self_observation = priority_context["self_observation"]
    return {
        "weather_today_cached": weather_today_cached,
        "recent_weather_misses": int(events.get("weather_cache_misses", 0)),
        "avg_cloud_dependence": homeostasis["avg_cloud_dependence"],
        "avg_cache_freshness": homeostasis["avg_cache_freshness"],
        "cache_freshness_delta": homeostasis["cache_freshness_delta"],
        "self_observation_points": self_observation["points"],
        "local_resolution_delta": self_observation["local_resolution_delta"],
        "weather_cache_gap_persistence": self_observation["weather_cache_gap_persistence"],
        "recent_failed_jobs": int(failed_jobs.get("refresh_weather_cache", 0)),
        "expected_local_resolution_gain": 1.0 if not weather_today_cached else 0.0,
    }


def _recent_homeostasis(store: Any, *, limit: int = 12) -> dict[str, float]:
    rows = store.connection.execute(
        """
        SELECT cloud_dependence, uncertainty, inventory_coverage, cache_freshness
        FROM homeostatic_snapshots
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return {
            "avg_cloud_dependence": 0.0,
            "cloud_dependence_delta": 0.0,
            "avg_uncertainty": 0.0,
            "avg_inventory_coverage": 0.0,
            "avg_cache_freshness": 0.0,
            "cache_freshness_delta": 0.0,
        }
    newest = rows[0]
    oldest = rows[-1]
    return {
        "avg_cloud_dependence": _avg([float(row["cloud_dependence"]) for row in rows]),
        "cloud_dependence_delta": round(
            float(newest["cloud_dependence"]) - float(oldest["cloud_dependence"]),
            3,
        ),
        "avg_uncertainty": _avg([float(row["uncertainty"]) for row in rows]),
        "avg_inventory_coverage": _avg([float(row["inventory_coverage"]) for row in rows]),
        "avg_cache_freshness": _avg([float(row["cache_freshness"]) for row in rows]),
        "cache_freshness_delta": round(
            float(newest["cache_freshness"]) - float(oldest["cache_freshness"]),
            3,
        ),
    }


def _recent_event_counts(store: Any, *, limit: int = 24) -> dict[str, int]:
    rows = store.connection.execute(
        """
        SELECT intent, route, reason
        FROM events
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return {
        "story_cloud_handoffs": sum(
            1 for row in rows if row["intent"] == "story" and row["route"] == "cloud_handoff"
        ),
        "weather_cache_misses": sum(
            1 for row in rows if row["intent"] == "weather" and row["reason"] == "weather_cache_miss"
        ),
    }


def _failed_job_counts(store: Any) -> dict[str, int]:
    rows = store.connection.execute(
        """
        SELECT kind, COUNT(*) AS count
        FROM jobs
        WHERE status='failed'
        GROUP BY kind
        """
    ).fetchall()
    return {str(row["kind"]): int(row["count"]) for row in rows}


def _self_observation_priority_context(store: Any) -> dict[str, float | int]:
    load_self_state = getattr(store, "load_self_state", None)
    if not callable(load_self_state):
        return {
            "points": 0,
            "local_resolution_delta": 0.0,
            "weather_cache_gap_persistence": 0.0,
            "story_inventory_gap_persistence": 0.0,
        }
    state = load_self_state()
    history = state.get("runtime_health_history", [])
    if not isinstance(history, list):
        return {
            "points": 0,
            "local_resolution_delta": 0.0,
            "weather_cache_gap_persistence": 0.0,
            "story_inventory_gap_persistence": 0.0,
        }
    points = [dict(item) for item in history if isinstance(item, dict)]
    if not points:
        return {
            "points": 0,
            "local_resolution_delta": 0.0,
            "weather_cache_gap_persistence": 0.0,
            "story_inventory_gap_persistence": 0.0,
        }
    first = points[0]
    latest = points[-1]
    count = len(points)
    return {
        "points": count,
        "local_resolution_delta": round(
            float(latest.get("local_resolution_rate", 0.0) or 0.0)
            - float(first.get("local_resolution_rate", 0.0) or 0.0),
            3,
        ),
        "weather_cache_gap_persistence": round(
            sum(1 for item in points if not bool(item.get("weather_cache_ready"))) / max(1, count),
            3,
        ),
        "story_inventory_gap_persistence": round(
            sum(1 for item in points if not bool(item.get("story_inventory_ready"))) / max(1, count),
            3,
        ),
    }


def _bounded_priority(value: float) -> float:
    return round(min(0.99, max(0.1, value)), 3)


def _avg(values: Iterable[float]) -> float:
    value_tuple = tuple(values)
    if not value_tuple:
        return 0.0
    return round(sum(value_tuple) / len(value_tuple), 3)


def _local_media_item_from_manifest(
    record: dict[str, Any],
    manifest_path: Path,
) -> LocalMediaItem | None:
    title = _clean_text(record.get("title", ""))
    path_text = _clean_text(record.get("path", ""))
    kind = _clean_text(record.get("kind", "")).lower()
    if not title and path_text:
        title = _title_from_media_path(Path(path_text))
    if not title:
        return None
    if not path_text:
        return None
    path = Path(path_text)
    resolved = path if path.is_absolute() else manifest_path.parent / path
    suffix_kind = SUPPORTED_MEDIA_EXTENSIONS.get(path.suffix.lower(), "")
    media_kind = kind or suffix_kind
    if media_kind not in {"audio", "video"}:
        return None
    item_id = _media_item_id(str(record.get("item_id") or title))
    tags = tuple(
        dict.fromkeys(
            _media_tags(
                tuple(str(item) for item in record.get("tags", [])),
                title=title,
                kind=media_kind,
            )
        )
    )
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"item_id", "title", "kind", "path", "tags", "source", "license"}
    }
    metadata["resolved_path"] = str(resolved)
    return LocalMediaItem(
        item_id=item_id,
        title=title,
        kind=media_kind,
        path=path_text,
        tags=tags,
        source=str(record.get("source") or "local_media_manifest"),
        license=str(record.get("license") or "local_device"),
        path_exists=resolved.exists(),
        metadata=metadata,
    )


def _local_media_item_from_path(path: Path, root: Path) -> LocalMediaItem:
    kind = SUPPORTED_MEDIA_EXTENSIONS[path.suffix.lower()]
    title = _title_from_media_path(path)
    try:
        display_path = str(path.relative_to(root))
    except ValueError:
        display_path = str(path)
    return LocalMediaItem(
        item_id=_media_item_id(title),
        title=title,
        kind=kind,
        path=display_path,
        tags=_media_tags((), title=title, kind=kind),
        source="local_media_directory",
        license="local_device",
        path_exists=True,
        metadata={"bytes": path.stat().st_size, "resolved_path": str(path)},
    )


def _ranked_media_selection(
    items: tuple[LocalMediaItem, ...],
    profile: LocalAssistantProfile,
    *,
    limit: int,
) -> tuple[LocalMediaItem, ...]:
    deduped: dict[str, LocalMediaItem] = {}
    for item in items:
        current = deduped.get(item.item_id)
        if current is None or _rank_media(item, profile) > _rank_media(current, profile):
            deduped[item.item_id] = item
    ranked = sorted(
        deduped.values(),
        key=lambda item: (_rank_media(item, profile), item.path_exists, item.title.lower()),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _rank_media(item: LocalMediaItem, profile: LocalAssistantProfile) -> float:
    text = " ".join([item.title, *item.tags]).lower()
    preference_text = " ".join([*profile.preferences.values(), *profile.facts.values()]).lower()
    score = 1.0
    score += sum(0.5 for tag in item.tags if tag.lower() in preference_text)
    if item.title.lower() in preference_text:
        score += 1.0
    if item.path_exists:
        score += 0.25
    if "preferred" in item.metadata and bool(item.metadata["preferred"]):
        score += 0.5
    if "song" in preference_text and item.kind == "audio":
        score += 0.25
    if any(term in preference_text for term in ("calm", "sleep", "bedtime")) and "calm" in text:
        score += 0.25
    return score


def _media_tags(tags: Iterable[str], *, title: str, kind: str) -> tuple[str, ...]:
    inferred: list[str] = [kind]
    text = title.lower()
    for tag in tags:
        cleaned = _clean_text(tag).lower()
        if cleaned:
            inferred.append(cleaned)
    for needle in ("calm", "focus", "piano", "rain", "sleep", "lofi", "story", "song"):
        if needle in text:
            inferred.append(needle)
    return tuple(dict.fromkeys(inferred))


def _title_from_media_path(path: Path) -> str:
    words = re.sub(r"[_\-]+", " ", path.stem).split()
    return " ".join(word.capitalize() for word in words) or path.stem


def _media_item_id(value: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9]+", " ", value).strip().lower().split()
    return " ".join(words) or "media item"


def _metadata_from_record(record: dict) -> StoryMetadata:
    return StoryMetadata(
        item_id=str(record["item_id"]),
        title=str(record["title"]),
        source=str(record["source"]),
        source_url=str(record["source_url"]),
        license=str(record.get("license", "public_domain")),
        age_min=int(record.get("age_min", 0)),
        age_max=int(record.get("age_max", 99)),
        topics=tuple(str(item) for item in record.get("topics", [])),
        cultures=tuple(str(item) for item in record.get("cultures", [])),
        summary=str(record.get("summary", "")),
        narrative_frame=_story_frame_from_record(record),
    )


def _gutenberg_story_from_row(row: dict[str, str]) -> StoryMetadata | None:
    text_id = _clean_text(row.get("Text#", ""))
    title = _clean_text(row.get("Title", ""))
    language = _clean_text(row.get("Language", "")).lower()
    ebook_type = _clean_text(row.get("Type", ""))
    authors = _clean_text(row.get("Authors", ""))
    subjects = _split_terms(row.get("Subjects", ""))
    bookshelves = _split_terms(row.get("Bookshelves", ""))
    haystack = " ".join([title, *subjects, *bookshelves]).lower()
    if not text_id or not title or title == "--":
        return None
    if ebook_type and ebook_type != "Text":
        return None
    if language and language != "en":
        return None
    if not _looks_story_like(haystack):
        return None
    topics = _topics_from_terms(subjects + bookshelves + (title,))
    cultures = _cultures_from_terms(subjects + bookshelves)
    age_min, age_max = _age_range_from_topics(topics, haystack)
    author_text = f" by {authors}" if authors else ""
    summary = f"Project Gutenberg catalog entry for {title}{author_text}; subjects: {_short_terms(subjects)}."
    return StoryMetadata(
        item_id=f"pg_{text_id}",
        title=title,
        source="project_gutenberg_catalog_metadata",
        source_url=f"https://www.gutenberg.org/ebooks/{text_id}",
        license="public_domain_catalog_metadata",
        age_min=age_min,
        age_max=age_max,
        topics=topics,
        cultures=cultures,
        summary=summary,
        narrative_frame=_narrative_frame_from_metadata_title(title),
    )


def _internet_archive_story_from_row(row: dict[str, Any]) -> StoryMetadata | None:
    identifier = _clean_text(_first_value(row.get("identifier", "")))
    title = _clean_text(_first_value(row.get("title", "")))
    creator = _clean_text(_first_value(row.get("creator", "")))
    language = _clean_text(_first_value(row.get("language", ""))).lower()
    subjects = tuple(_clean_text(item) for item in _as_tuple(row.get("subject", ())) if _clean_text(item))
    collections = tuple(_clean_text(item).lower() for item in _as_tuple(row.get("collection", ())) if _clean_text(item))
    haystack = " ".join([title, *subjects, *collections]).lower()
    if not identifier or not title or title == "--":
        return None
    if language and language not in {"en", "eng", "english"}:
        return None
    if "gutenberg" not in collections and not _public_license(row.get("licenseurl", "")):
        return None
    if not _looks_story_like(haystack):
        return None
    topics = _topics_from_terms(subjects + (title,))
    cultures = _cultures_from_terms(subjects)
    age_min, age_max = _age_range_from_topics(topics, haystack)
    creator_text = f" by {creator}" if creator else ""
    summary = f"Internet Archive metadata entry for {title}{creator_text}; subjects: {_short_terms(subjects)}."
    return StoryMetadata(
        item_id=f"ia_{_slug(identifier)}",
        title=title,
        source="internet_archive_item_search_and_metadata",
        source_url=f"https://archive.org/details/{identifier}",
        license="public_domain_or_gutenberg_collection_metadata",
        age_min=age_min,
        age_max=age_max,
        topics=topics,
        cultures=cultures,
        summary=summary,
        narrative_frame=_narrative_frame_from_metadata_title(title),
    )


def _rank_story(item: StoryMetadata, profile: LocalAssistantProfile) -> float:
    score = 0.0
    if item.age_min <= profile.age <= item.age_max:
        score += 3.0
    elif item.age_min - 1 <= profile.age <= item.age_max + 1:
        score += 1.0
    culture_terms = {profile.culture.lower(), profile.location.lower()}
    score += len(culture_terms & {item.lower() for item in item.cultures}) * 1.5
    preference_text = " ".join([*profile.preferences.values(), *profile.facts.values()]).lower()
    score += sum(1.0 for topic in item.topics if topic.lower() in preference_text)
    if "bedtime" in preference_text and "bedtime" in item.topics:
        score += 0.8
    return score


def _metadata_quality_score(item: StoryMetadata) -> float:
    score = 0.0
    if item.title:
        score += 0.2
    if item.summary:
        score += 0.25
    if item.source_url:
        score += 0.15
    if item.topics:
        score += min(0.2, 0.05 * len(item.topics))
    if item.cultures:
        score += min(0.1, 0.05 * len(item.cultures))
    if item.license:
        score += 0.1
    return min(score, 1.0)


def _ranked_selection(
    items: tuple[StoryMetadata, ...],
    profile: LocalAssistantProfile,
    *,
    limit: int,
) -> tuple[StoryMetadata, ...]:
    return _ranked_selection_observed(items, profile, limit=limit)["items"]


def _ranked_selection_observed(
    items: tuple[StoryMetadata, ...],
    profile: LocalAssistantProfile,
    *,
    limit: int,
) -> dict[str, Any]:
    quality_items = tuple(
        item for item in items if _metadata_quality_score(item) >= MIN_STORY_METADATA_QUALITY
    )
    deduped = _dedupe_story_items(quality_items, profile)
    ranked = sorted(
        deduped,
        key=lambda item: (
            _rank_story(item, profile),
            _metadata_quality_score(item),
            item.title,
        ),
        reverse=True,
    )
    selected = tuple(item for item in ranked if _rank_story(item, profile) > 0.0)[:limit]
    if not selected:
        selected = ranked[:limit]
    return {
        "items": selected,
        "observability": _selection_observability(
            source_candidates=items,
            quality_candidates=quality_items,
            deduped_candidates=deduped,
            selected=selected,
            profile=profile,
        ),
    }


def _selection_observability(
    *,
    source_candidates: tuple[StoryMetadata, ...],
    quality_candidates: tuple[StoryMetadata, ...],
    deduped_candidates: tuple[StoryMetadata, ...],
    selected: tuple[StoryMetadata, ...],
    profile: LocalAssistantProfile,
) -> dict[str, Any]:
    metadata_quality = [_metadata_quality_score(item) for item in selected]
    local_fit = [_rank_story(item, profile) for item in selected]
    quality_score = [
        _metadata_quality_score(item) + _rank_story(item, profile)
        for item in selected
    ]
    return {
        "candidate_count": len(source_candidates),
        "quality_floor": MIN_STORY_METADATA_QUALITY,
        "quality_rejected_count": len(source_candidates) - len(quality_candidates),
        "duplicate_rejected_count": len(quality_candidates) - len(deduped_candidates),
        "deduped_candidate_count": len(deduped_candidates),
        "selected_avg_metadata_quality": _avg(metadata_quality),
        "selected_avg_local_fit_score": _avg(local_fit),
        "selected_avg_quality_score": _avg(quality_score),
    }


def _fetch_url_bytes(
    url: str,
    *,
    max_source_bytes: int,
    timeout: float,
    max_attempts: int = 2,
    backoff_seconds: float = 0.5,
) -> bytes:
    return _fetch_url_bytes_observed(
        url,
        max_source_bytes=max_source_bytes,
        timeout=timeout,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    ).data


def _fetch_url_bytes_observed(
    url: str,
    *,
    max_source_bytes: int,
    timeout: float,
    max_attempts: int = 2,
    backoff_seconds: float = 0.5,
) -> _FetchBytesResult:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    attempts = max(1, max_attempts)
    last_error: Exception | None = None
    data = b""
    used_attempts = 0
    for attempt in range(attempts):
        used_attempts = attempt + 1
        try:
            with urlopen(request, timeout=timeout) as response:
                data = response.read(max_source_bytes + 1)
            break
        except Exception as exc:  # pragma: no cover - covered through importer tests
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            if backoff_seconds > 0:
                sleep(backoff_seconds * (2**attempt))
    else:  # pragma: no cover - defensive; loop either breaks or raises
        raise RuntimeError(f"failed to fetch {url}") from last_error
    if len(data) > max_source_bytes:
        raise ValueError(f"source exceeded max_source_bytes={max_source_bytes}: {url}")
    return _FetchBytesResult(data=data, attempts=used_attempts)


def _dedupe_story_items(
    items: tuple[StoryMetadata, ...],
    profile: LocalAssistantProfile,
) -> tuple[StoryMetadata, ...]:
    best: dict[str, StoryMetadata] = {}
    for item in items:
        key = _canonical_story_title(item.title)
        current = best.get(key)
        if current is None or _story_selection_score(item, profile) > _story_selection_score(current, profile):
            best[key] = item
    return tuple(best.values())


def _story_selection_score(item: StoryMetadata, profile: LocalAssistantProfile) -> tuple[float, float, str]:
    return (_rank_story(item, profile), _metadata_quality_score(item), item.source)


def _canonical_story_title(title: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
    stopwords = {"a", "an", "the"}
    return " ".join(word for word in words if word not in stopwords)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _split_terms(value: str | None) -> tuple[str, ...]:
    return tuple(_clean_text(item) for item in str(value or "").split(";") if _clean_text(item))


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _first_value(value: Any) -> str:
    values = _as_tuple(value)
    return str(values[0]) if values else ""


def _looks_story_like(text: str) -> bool:
    substring_terms = (
        "adventure",
        "bedtime",
        "children",
        "child",
        "fairy",
        "fable",
        "folk",
        "folklore",
        "juvenile",
        "legend",
        "myth",
        "stories",
        "tale",
        "tales",
    )
    return any(term in text for term in substring_terms) or bool(re.search(r"\bstory\b", text))


def _topics_from_terms(terms: Iterable[str]) -> tuple[str, ...]:
    text = " ".join(terms).lower()
    topics: list[str] = []
    mapping = (
        ("bedtime", ("bedtime",)),
        ("adventure", ("adventure", "voyage", "journey")),
        ("folktale", ("folk", "folklore", "tale", "fairy", "fable", "legend", "myth")),
        ("school", ("school", "student")),
        ("kindness", ("kind", "moral", "virtue")),
        ("questions", ("question", "curiosity")),
        ("animal", ("animal", "tortoise", "rabbit", "lion", "fox")),
    )
    for topic, needles in mapping:
        if any(needle in text for needle in needles):
            topics.append(topic)
    if not topics:
        topics.append("story")
    return tuple(dict.fromkeys(topics))


def _cultures_from_terms(terms: Iterable[str]) -> tuple[str, ...]:
    text = " ".join(terms).lower()
    cultures: list[str] = []
    mapping = (
        ("Yoruba", ("yoruba",)),
        ("West Africa", ("west africa", "africa")),
        ("Lagos", ("lagos",)),
        ("global", ("children", "juvenile", "fairy", "folk")),
    )
    for culture, needles in mapping:
        if any(needle in text for needle in needles):
            cultures.append(culture)
    if not cultures:
        cultures.append("global")
    return tuple(dict.fromkeys(cultures))


def _age_range_from_topics(topics: tuple[str, ...], haystack: str) -> tuple[int, int]:
    if "children" in haystack or "juvenile" in haystack or "bedtime" in topics:
        return (5, 10)
    if "fairy" in haystack or "fable" in haystack:
        return (6, 11)
    return (7, 12)


def _short_terms(terms: Iterable[str], *, limit: int = 3) -> str:
    selected = [term for term in terms if term][:limit]
    return "; ".join(selected) if selected else "catalog metadata only"


def _story_frame_from_record(record: dict[str, Any]) -> str:
    return str(record.get("narrative_frame") or record.get("template") or "")


def _narrative_frame_from_metadata_title(title: str) -> str:
    safe_title = _clean_text(title)
    return (
        f"Using the public-domain catalog entry for {safe_title}, "
        "{name} made a small adventure in {location}: listen first, ask one "
        "good question, help someone nearby, and come home safely."
    )


def _public_license(value: Any) -> bool:
    text = " ".join(str(item).lower() for item in _as_tuple(value))
    return "publicdomain" in text or "public-domain" in text or "creativecommons" in text


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return slug or "item"
