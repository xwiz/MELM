"""SQLite persistence for the Local Assistant OS v0.1 kernel.

The store is intentionally small and dependency-free so the MVP can run on a
Raspberry Pi-class device. JSON columns are used for flexible payloads while
the core OS ledgers remain queryable tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from .local_assistant_router import LocalAssistantProfile


SCHEMA_VERSION = "melm.local_assistant_os.sqlite.v1"
SEED_SCHEMA = "melm.local_assistant_os.seed.v1"
MEMORY_DIGEST_QUALITY_FLOOR = 0.72
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


@dataclass(frozen=True)
class StoredAssistantEvent:
    event_id: str
    session_id: str
    previous_event_id: str
    next_event_id: str
    utterance: str
    intent: str
    route: str
    reason: str
    cloud_needed: bool
    evidence_keys: tuple[str, ...]
    capture_surface: str = ""
    capture_source: str = ""


@dataclass(frozen=True)
class StoredInventoryJob:
    job_id: str
    kind: str
    status: str
    priority: float
    attempts: int
    max_attempts: int
    resource_budget: dict[str, Any]
    payload: dict[str, Any]


class AssistantOSStore:
    """Small SQLite ledger for assistant memory, policy, state, and inventory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self._active_session_id = ""
        self._configure()
        self.initialize()

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                consent INTEGER NOT NULL,
                local_only INTEGER NOT NULL,
                cloud_eligible INTEGER NOT NULL DEFAULT 0,
                scope TEXT NOT NULL DEFAULT 'private_local',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS self_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT 'legacy_session',
                previous_event_id TEXT NOT NULL DEFAULT '',
                next_event_id TEXT NOT NULL DEFAULT '',
                utterance TEXT NOT NULL,
                intent TEXT NOT NULL,
                route TEXT NOT NULL,
                reason TEXT NOT NULL,
                answer TEXT NOT NULL,
                cloud_needed INTEGER NOT NULL,
                external_fetch_needed INTEGER NOT NULL,
                device_action INTEGER NOT NULL,
                local_memory_used INTEGER NOT NULL,
                capture_surface TEXT NOT NULL DEFAULT '',
                capture_source TEXT NOT NULL DEFAULT '',
                evidence_keys_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS membrane_decisions (
                decision_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                route TEXT NOT NULL,
                allowed INTEGER NOT NULL,
                boundary_crossed TEXT NOT NULL,
                personal_facts_included_json TEXT NOT NULL,
                personal_facts_excluded_json TEXT NOT NULL,
                confirmation_required INTEGER NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );

            CREATE TABLE IF NOT EXISTS homeostatic_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                privacy_risk REAL NOT NULL,
                cloud_dependence REAL NOT NULL,
                local_capability REAL NOT NULL,
                uncertainty REAL NOT NULL,
                cache_freshness REAL NOT NULL,
                action_risk REAL NOT NULL,
                user_trust REAL NOT NULL,
                inventory_coverage REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );

            CREATE TABLE IF NOT EXISTS synthesis_traces (
                event_id TEXT PRIMARY KEY,
                route TEXT NOT NULL,
                applied INTEGER NOT NULL,
                refused INTEGER NOT NULL,
                quality_score REAL NOT NULL,
                citation_count INTEGER NOT NULL,
                evidence_count INTEGER NOT NULL,
                warnings_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                boundary_crossed TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );

            CREATE TABLE IF NOT EXISTS response_integrity (
                event_id TEXT PRIMARY KEY,
                understanding_score REAL NOT NULL,
                response_integrity_score REAL NOT NULL,
                overall_score REAL NOT NULL,
                band TEXT NOT NULL,
                research_recommended INTEGER NOT NULL,
                components_json TEXT NOT NULL,
                flags_json TEXT NOT NULL,
                research_topics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );

            CREATE TABLE IF NOT EXISTS session_improvement_consent (
                session_id TEXT PRIMARY KEY,
                opted_in INTEGER NOT NULL,
                consent_scope TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS improvement_candidates (
                candidate_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL,
                priority REAL NOT NULL,
                candidate_kinds_json TEXT NOT NULL,
                research_topics_json TEXT NOT NULL,
                redaction_state TEXT NOT NULL,
                cloud_export_allowed INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );

            CREATE TABLE IF NOT EXISTS opportunities (
                opportunity_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                priority REAL NOT NULL,
                reason TEXT NOT NULL,
                evidence_event_ids_json TEXT NOT NULL,
                expected_cloud_reduction INTEGER NOT NULL,
                proposed_action TEXT NOT NULL,
                source_candidates_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS inventories (
                kind TEXT NOT NULL,
                item_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                source TEXT NOT NULL,
                license TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(kind, item_id)
            );

            CREATE TABLE IF NOT EXISTS pending_actions (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                target TEXT NOT NULL,
                utterance TEXT NOT NULL,
                evidence_keys_json TEXT NOT NULL,
                confirmation_state TEXT NOT NULL,
                executed INTEGER NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                priority REAL NOT NULL,
                attempts INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                resource_budget_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lexemes (
                lexeme_id TEXT PRIMARY KEY,
                lemma TEXT NOT NULL,
                normalized_lemma TEXT NOT NULL,
                language TEXT NOT NULL,
                pos TEXT NOT NULL,
                reserved INTEGER NOT NULL DEFAULT 0,
                frequency_rank INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(language, normalized_lemma, pos)
            );

            CREATE TABLE IF NOT EXISTS word_forms (
                form_id TEXT PRIMARY KEY,
                lexeme_id TEXT NOT NULL,
                surface TEXT NOT NULL,
                normalized_surface TEXT NOT NULL,
                morph_features TEXT NOT NULL,
                provenance TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(lexeme_id, normalized_surface, morph_features),
                FOREIGN KEY(lexeme_id) REFERENCES lexemes(lexeme_id)
            );

            CREATE TABLE IF NOT EXISTS lexical_senses (
                sense_id TEXT PRIMARY KEY,
                lexeme_id TEXT NOT NULL,
                semantic_class_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                argument_template_id TEXT NOT NULL,
                definition TEXT NOT NULL,
                genus_lemma TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                superseded_by TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                UNIQUE(lexeme_id, semantic_class_id),
                FOREIGN KEY(lexeme_id) REFERENCES lexemes(lexeme_id)
            );

            CREATE TABLE IF NOT EXISTS lexical_provenance (
                provenance_id TEXT PRIMARY KEY,
                sense_id TEXT NOT NULL,
                provenance TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                license TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                definition TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(sense_id, provenance, source_ref),
                FOREIGN KEY(sense_id) REFERENCES lexical_senses(sense_id)
            );

            CREATE TABLE IF NOT EXISTS lexical_relation_candidates (
                relation_id TEXT PRIMARY KEY,
                sense_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_lemma TEXT NOT NULL,
                target_sense_ref TEXT NOT NULL,
                status TEXT NOT NULL,
                provenance TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(sense_id, relation, target_lemma, target_sense_ref),
                FOREIGN KEY(sense_id) REFERENCES lexical_senses(sense_id)
            );

            CREATE TABLE IF NOT EXISTS lexicon_ingestions (
                ingestion_id TEXT PRIMARY KEY,
                candidate_hash TEXT NOT NULL UNIQUE,
                batch_id TEXT NOT NULL,
                schema_id TEXT NOT NULL,
                provenance TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                status TEXT NOT NULL,
                lexeme_id TEXT NOT NULL,
                sense_id TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_intent_route
                ON events(intent, route);
            CREATE INDEX IF NOT EXISTS idx_events_created_at
                ON events(created_at);
            CREATE INDEX IF NOT EXISTS idx_events_session
                ON events(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_membrane_event
                ON membrane_decisions(event_id);
            CREATE INDEX IF NOT EXISTS idx_homeostasis_event
                ON homeostatic_snapshots(event_id);
            CREATE INDEX IF NOT EXISTS idx_synthesis_quality
                ON synthesis_traces(quality_score, applied, refused);
            CREATE INDEX IF NOT EXISTS idx_response_integrity_score
                ON response_integrity(overall_score, research_recommended);
            CREATE INDEX IF NOT EXISTS idx_improvement_candidates_status
                ON improvement_candidates(status, priority);
            CREATE INDEX IF NOT EXISTS idx_improvement_candidates_session
                ON improvement_candidates(session_id, status);
            CREATE INDEX IF NOT EXISTS idx_opportunities_status_kind
                ON opportunities(status, kind);
            CREATE INDEX IF NOT EXISTS idx_inventories_kind
                ON inventories(kind);
            CREATE INDEX IF NOT EXISTS idx_pending_actions_state
                ON pending_actions(confirmation_state, executed);
            CREATE INDEX IF NOT EXISTS idx_jobs_status_priority
                ON jobs(status, priority);
            CREATE INDEX IF NOT EXISTS idx_lexemes_lookup
                ON lexemes(language, normalized_lemma, pos);
            CREATE INDEX IF NOT EXISTS idx_word_forms_lookup
                ON word_forms(normalized_surface);
            CREATE INDEX IF NOT EXISTS idx_lexical_senses_status
                ON lexical_senses(status, semantic_class_id);
            CREATE INDEX IF NOT EXISTS idx_lexical_provenance_sense
                ON lexical_provenance(sense_id);
            CREATE INDEX IF NOT EXISTS idx_lexical_relations_sense
                ON lexical_relation_candidates(sense_id, status);
            """
        )
        self._ensure_event_link_columns()
        self._ensure_event_provenance_columns()
        self._ensure_user_fact_policy_columns()
        self.connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            WHERE metadata.value <> excluded.value
            """,
            ("schema_version", SCHEMA_VERSION),
        )
        self.connection.commit()

    def save_profile(self, profile: LocalAssistantProfile) -> None:
        self.upsert_user_fact(
            "profile.user_name",
            profile.user_name,
            source="profile",
            confidence=1.0,
            preserve_policy=True,
        )
        self.upsert_user_fact(
            "profile.age",
            str(profile.age),
            source="profile",
            confidence=1.0,
            preserve_policy=True,
        )
        self.upsert_user_fact(
            "profile.location",
            profile.location,
            source="profile",
            confidence=1.0,
            preserve_policy=True,
        )
        self.upsert_user_fact(
            "profile.culture",
            profile.culture,
            source="profile",
            confidence=0.8,
            preserve_policy=True,
        )
        for key, value in profile.facts.items():
            evidence_key = f"facts.{key}"
            self.upsert_user_fact(
                evidence_key,
                value,
                source="profile.facts",
                confidence=0.86,
                scope=_default_user_fact_scope(evidence_key),
                preserve_policy=True,
            )
        for key, value in profile.preferences.items():
            self.upsert_user_fact(
                f"preferences.{key}",
                value,
                source="profile.preferences",
                confidence=0.86,
                preserve_policy=True,
            )
        for index, goal in enumerate(profile.health_goals):
            self.upsert_user_fact(
                f"health_goals.{index}",
                goal,
                source="profile.health_goals",
                confidence=0.76,
                preserve_policy=True,
            )
        for key, value in profile.story_models.items():
            self.upsert_profile_inventory(
                "story_model",
                key,
                {"narrative_frame": value},
                source="local_seed_or_builder",
                license="public_domain_or_local_story_frame",
                tags=("story", profile.culture, f"age:{profile.age}"),
            )
        for key, value in profile.weekly_weather.items():
            self.upsert_profile_inventory(
                "weather",
                key,
                {"forecast": value, "location": profile.location},
                source="weather_cache",
                license="local_cache",
                tags=("weather", profile.location),
            )
        for key, value in profile.contacts.items():
            self.upsert_profile_inventory(
                "contact",
                key,
                {"number": value},
                source="user_profile",
                license="private_local",
                tags=("contact", "local_only"),
            )
        for item in profile.media_library:
            self.upsert_profile_inventory(
                "media",
                item,
                {"title": item},
                source="local_media_index",
                license="local_device",
                tags=("media",),
            )
        for item in profile.food_inventory:
            self.upsert_profile_inventory(
                "food",
                item,
                {"name": item},
                source="local_food_inventory",
                license="local_seed",
                tags=("food",),
            )
        self.connection.commit()

    def upsert_profile_inventory(
        self,
        kind: str,
        item_id: str,
        payload: dict[str, Any],
        *,
        source: str,
        license: str,
        tags: Iterable[str] = (),
    ) -> None:
        existing = self.connection.execute(
            """
            SELECT payload_json, source, license, tags_json
            FROM inventories
            WHERE kind=? AND item_id=?
            """,
            (kind, item_id),
        ).fetchone()
        if existing is None:
            self.upsert_inventory(kind, item_id, payload, source=source, license=license, tags=tags)
            return
        merged_payload = {**dict(_loads(existing["payload_json"], default={})), **payload}
        self.upsert_inventory(
            kind,
            item_id,
            merged_payload,
            source=str(existing["source"]),
            license=str(existing["license"]),
            tags=tuple(str(item) for item in _loads(existing["tags_json"], default=list(tags))),
        )

    def load_profile(self, base: LocalAssistantProfile | None = None) -> LocalAssistantProfile:
        profile = base or LocalAssistantProfile()
        facts = dict(profile.facts)
        preferences = dict(profile.preferences)
        health_goals = list(profile.health_goals)
        user_name = profile.user_name
        age = profile.age
        location = profile.location
        culture = profile.culture
        fact_rows = self.connection.execute("SELECT key, value, consent FROM user_facts").fetchall()
        revoked_keys = {str(row["key"]) for row in fact_rows if not bool(row["consent"])}
        for row in fact_rows:
            if not bool(row["consent"]):
                continue
            key = str(row["key"])
            value = str(row["value"])
            if key == "profile.user_name":
                user_name = value
            elif key == "profile.age":
                try:
                    age = int(value)
                except ValueError:
                    age = profile.age
            elif key == "profile.location":
                location = value
            elif key == "profile.culture":
                culture = value
            elif key.startswith("facts."):
                facts[key.split(".", 1)[1]] = value
            elif key.startswith("preferences."):
                preferences[key.split(".", 1)[1]] = value
            elif key.startswith("health_goals."):
                health_goals.append(value)
        _remove_revoked_profile_values(
            revoked_keys,
            facts=facts,
            preferences=preferences,
            health_goals=health_goals,
        )
        story_models = {
            item_id: _story_frame_payload_text(payload)
            for item_id, payload in self.load_inventory("story_model").items()
            if _story_frame_payload_text(payload)
        } or dict(profile.story_models)
        weather_inventory = self.load_inventory("weather")
        weekly_weather = {
            item_id: str(payload.get("forecast", ""))
            for item_id, payload in weather_inventory.items()
            if payload.get("forecast") and _is_inventory_payload_fresh(payload)
        }
        if not weather_inventory:
            weekly_weather = dict(profile.weekly_weather)
        contacts = {
            item_id: str(payload.get("number", ""))
            for item_id, payload in self.load_inventory("contact").items()
            if payload.get("number")
        } or dict(profile.contacts)
        media = tuple(self.load_inventory("media").keys()) or profile.media_library
        food = tuple(self.load_inventory("food").keys()) or profile.food_inventory
        return LocalAssistantProfile(
            user_name=user_name,
            age=age,
            location=location,
            culture=culture,
            facts=facts,
            preferences=preferences,
            health_goals=tuple(dict.fromkeys(health_goals)),
            contacts=contacts,
            weekly_weather=weekly_weather,
            story_models=story_models,
            media_library=media,
            food_inventory=food,
        )

    def upsert_user_fact(
        self,
        key: str,
        value: str,
        *,
        source: str,
        confidence: float,
        consent: bool = True,
        local_only: bool = True,
        cloud_eligible: bool = False,
        scope: str = "private_local",
        preserve_policy: bool = False,
    ) -> None:
        if preserve_policy:
            existing = self.connection.execute(
                """
                SELECT consent, local_only, cloud_eligible, scope
                FROM user_facts
                WHERE key=?
                """,
                (key,),
            ).fetchone()
            if existing is not None:
                consent = bool(existing["consent"])
                local_only = bool(existing["local_only"])
                cloud_eligible = bool(existing["cloud_eligible"])
                scope = str(existing["scope"])
        safe_scope = _memory_scope(scope, local_only=local_only, cloud_eligible=cloud_eligible)
        self.connection.execute(
            """
            INSERT INTO user_facts(
                key, value, source, confidence, consent, local_only,
                cloud_eligible, scope, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                source=excluded.source,
                confidence=excluded.confidence,
                consent=excluded.consent,
                local_only=excluded.local_only,
                cloud_eligible=excluded.cloud_eligible,
                scope=excluded.scope,
                updated_at=excluded.updated_at
            """,
            (
                key,
                value,
                source,
                confidence,
                int(consent),
                int(local_only),
                int(cloud_eligible and not local_only),
                safe_scope,
                _now(),
            ),
        )

    def set_user_fact_consent(self, key: str, *, consent: bool) -> None:
        default_scope = _default_user_fact_scope(key)
        self.connection.execute(
            """
            INSERT INTO user_facts(
                key, value, source, confidence, consent, local_only,
                cloud_eligible, scope, updated_at
            )
            VALUES (?, '', 'consent_revocation', 1.0, ?, 1, 0, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                consent=excluded.consent,
                cloud_eligible=0,
                scope=CASE
                    WHEN user_facts.scope IS NOT NULL AND user_facts.scope != ''
                    THEN user_facts.scope
                    ELSE excluded.scope
                END,
                updated_at=excluded.updated_at
            """,
            (key, int(consent), default_scope, _now()),
        )
        self.connection.commit()

    def load_user_fact_privacy_index(self) -> dict[str, dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT key, consent, local_only, cloud_eligible, scope, source, confidence
            FROM user_facts
            ORDER BY key
            """
        ).fetchall()
        return {
            str(row["key"]): {
                "consent": bool(row["consent"]),
                "local_only": bool(row["local_only"]),
                "cloud_eligible": bool(row["cloud_eligible"]),
                "scope": str(row["scope"]),
                "source": str(row["source"]),
                "confidence": float(row["confidence"]),
            }
            for row in rows
        }

    def save_self_model(self, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            self.connection.execute(
                """
                INSERT INTO self_state(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, _json(value), _now()),
            )
        self.connection.commit()

    def load_self_state(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT key, value_json FROM self_state")
        return {str(row["key"]): _loads(row["value_json"]) for row in rows}

    def record_turn(
        self,
        *,
        event_id: str,
        utterance: str,
        intent: str,
        route: str,
        reason: str,
        answer: str,
        cloud_needed: bool,
        external_fetch_needed: bool,
        device_action: bool,
        local_memory_used: bool,
        evidence_keys: Iterable[str],
        membrane: dict[str, Any],
        homeostasis: dict[str, Any],
        capture_surface: str = "",
        capture_source: str = "",
    ) -> None:
        now = _now()
        session_id = self._current_session_id()
        previous_event_id = self.latest_event_id()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO events(
                event_id, session_id, previous_event_id, next_event_id,
                utterance, intent, route, reason, answer, cloud_needed,
                external_fetch_needed, device_action, local_memory_used,
                capture_surface, capture_source,
                evidence_keys_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                previous_event_id,
                "",
                utterance,
                intent,
                route,
                reason,
                answer,
                int(cloud_needed),
                int(external_fetch_needed),
                int(device_action),
                int(local_memory_used),
                str(capture_surface),
                str(capture_source),
                _json(tuple(evidence_keys)),
                now,
            ),
        )
        if previous_event_id:
            self.connection.execute(
                """
                UPDATE events
                SET next_event_id=?
                WHERE event_id=? AND next_event_id=''
                """,
                (event_id, previous_event_id),
            )
        self.connection.execute(
            """
            INSERT OR REPLACE INTO membrane_decisions(
                decision_id, event_id, route, allowed, boundary_crossed,
                personal_facts_included_json, personal_facts_excluded_json,
                confirmation_required, reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"membrane_{event_id}",
                event_id,
                route,
                int(bool(membrane["allowed"])),
                str(membrane["boundary_crossed"]),
                _json(membrane["personal_facts_included"]),
                _json(membrane["personal_facts_excluded"]),
                int(bool(membrane["confirmation_required"])),
                str(membrane["reason"]),
                now,
            ),
        )
        self.connection.execute(
            """
            INSERT OR REPLACE INTO homeostatic_snapshots(
                snapshot_id, event_id, privacy_risk, cloud_dependence,
                local_capability, uncertainty, cache_freshness, action_risk,
                user_trust, inventory_coverage, reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"homeostasis_{event_id}",
                event_id,
                float(homeostasis["privacy_risk"]),
                float(homeostasis["cloud_dependence"]),
                float(homeostasis["local_capability"]),
                float(homeostasis["uncertainty"]),
                float(homeostasis["cache_freshness"]),
                float(homeostasis["action_risk"]),
                float(homeostasis["user_trust"]),
                float(homeostasis["inventory_coverage"]),
                str(homeostasis["reason"]),
                now,
            ),
        )
        self.connection.commit()

    def record_synthesis_trace(
        self,
        *,
        event_id: str,
        route: str,
        applied: bool,
        refused: bool,
        quality: dict[str, Any],
        reason: str,
        boundary_crossed: str,
    ) -> None:
        now = _now()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO synthesis_traces(
                event_id, route, applied, refused, quality_score, citation_count,
                evidence_count, warnings_json, reason, boundary_crossed, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                route,
                int(applied),
                int(refused),
                float(quality.get("score", 0.0)),
                int(quality.get("citation_count", 0)),
                int(quality.get("evidence_count", 0)),
                _json(tuple(str(item) for item in quality.get("warnings", ()))),
                reason,
                boundary_crossed,
                now,
            ),
        )
        self.connection.commit()

    def record_response_integrity(
        self,
        *,
        event_id: str,
        assessment: dict[str, Any],
    ) -> None:
        now = _now()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO response_integrity(
                event_id, understanding_score, response_integrity_score,
                overall_score, band, research_recommended, components_json,
                flags_json, research_topics_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                float(assessment.get("understanding_score", 0.0)),
                float(assessment.get("response_integrity_score", 0.0)),
                float(assessment.get("overall_score", 0.0)),
                str(assessment.get("band", "low")),
                int(bool(assessment.get("research_recommended", False))),
                _json(dict(assessment.get("components", {}))),
                _json(tuple(str(item) for item in assessment.get("flags", ()))),
                _json(tuple(str(item) for item in assessment.get("research_topics", ()))),
                now,
            ),
        )
        self.connection.commit()

    def set_session_improvement_consent(
        self,
        session_id: str,
        *,
        opted_in: bool,
        consent_scope: str = "local_quarantined_research",
    ) -> None:
        normalized = _validated_session_id(session_id)
        now = _now()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO session_improvement_consent(
                session_id, opted_in, consent_scope, updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (normalized, int(opted_in), str(consent_scope), now),
        )
        if not opted_in:
            self.connection.execute(
                """
                UPDATE improvement_candidates
                SET status='consent_revoked', updated_at=?
                WHERE session_id=? AND status IN ('queued', 'research_ready')
                """,
                (now, normalized),
            )
        self.connection.commit()

    def session_improvement_consent(self, session_id: str) -> dict[str, Any]:
        normalized = _validated_session_id(session_id)
        row = self.connection.execute(
            """
            SELECT session_id, opted_in, consent_scope, updated_at
            FROM session_improvement_consent
            WHERE session_id=?
            """,
            (normalized,),
        ).fetchone()
        if row is None:
            return {
                "session_id": normalized,
                "opted_in": False,
                "consent_scope": "",
                "updated_at": "",
            }
        return {
            "session_id": str(row["session_id"]),
            "opted_in": bool(row["opted_in"]),
            "consent_scope": str(row["consent_scope"]),
            "updated_at": str(row["updated_at"]),
        }

    def queue_improvement_candidate(
        self,
        *,
        event_id: str,
        assessment: dict[str, Any],
    ) -> bool:
        if not bool(assessment.get("research_recommended", False)):
            return False
        event = self.connection.execute(
            "SELECT session_id FROM events WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if event is None:
            return False
        session_id = str(event["session_id"])
        consent = self.session_improvement_consent(session_id)
        if not consent["opted_in"]:
            return False
        overall_score = float(assessment.get("overall_score", 0.0) or 0.0)
        priority = round(min(1.0, max(0.0, 1.0 - overall_score)), 3)
        now = _now()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO improvement_candidates(
                candidate_id, event_id, session_id, status, priority,
                candidate_kinds_json, research_topics_json, redaction_state,
                cloud_export_allowed, created_at, updated_at
            )
            VALUES (?, ?, ?, 'queued', ?, ?, ?, 'not_redacted', 0, ?, ?)
            """,
            (
                f"improvement_{event_id}",
                event_id,
                session_id,
                priority,
                _json(tuple(str(item) for item in assessment.get("candidate_kinds", ()))),
                _json(tuple(str(item) for item in assessment.get("research_topics", ()))),
                now,
                now,
            ),
        )
        self.connection.commit()
        return True

    def improvement_queue(
        self,
        *,
        session_id: str = "",
        status: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        bounded_limit = max(1, min(500, int(limit)))
        where: list[str] = []
        params: list[Any] = []
        if session_id:
            where.append("c.session_id=?")
            params.append(_validated_session_id(session_id))
        if status:
            where.append("c.status=?")
            params.append(str(status))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self.connection.execute(
            f"""
            SELECT c.candidate_id, c.event_id, c.session_id, c.status, c.priority,
                   c.candidate_kinds_json, c.research_topics_json,
                   c.redaction_state, c.cloud_export_allowed,
                   e.utterance, e.intent, e.route, e.reason,
                   r.understanding_score, r.response_integrity_score,
                   r.overall_score, r.band, r.flags_json
            FROM improvement_candidates AS c
            JOIN events AS e ON e.event_id=c.event_id
            JOIN response_integrity AS r ON r.event_id=c.event_id
            {clause}
            ORDER BY c.priority DESC, c.created_at
            LIMIT ?
            """,
            (*params, bounded_limit),
        ).fetchall()
        candidates = [
            {
                "candidate_id": str(row["candidate_id"]),
                "event_id": str(row["event_id"]),
                "session_id": str(row["session_id"]),
                "status": str(row["status"]),
                "priority": float(row["priority"]),
                "candidate_kinds": list(_loads(row["candidate_kinds_json"], default=[])),
                "research_topics": list(_loads(row["research_topics_json"], default=[])),
                "redaction_state": str(row["redaction_state"]),
                "cloud_export_allowed": bool(row["cloud_export_allowed"]),
                "utterance": str(row["utterance"]),
                "intent": str(row["intent"]),
                "route": str(row["route"]),
                "reason": str(row["reason"]),
                "integrity": {
                    "understanding_score": float(row["understanding_score"]),
                    "response_integrity_score": float(row["response_integrity_score"]),
                    "overall_score": float(row["overall_score"]),
                    "band": str(row["band"]),
                    "flags": list(_loads(row["flags_json"], default=[])),
                },
            }
            for row in rows
        ]
        status_rows = self.connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM improvement_candidates
            GROUP BY status
            ORDER BY status
            """
        ).fetchall()
        consent_rows = self.connection.execute(
            """
            SELECT opted_in, COUNT(*) AS count
            FROM session_improvement_consent
            GROUP BY opted_in
            """
        ).fetchall()
        return {
            "schema": "melm.improvement_queue.v1",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "status_counts": {
                str(row["status"]): int(row["count"])
                for row in status_rows
            },
            "consent": {
                "opted_in_sessions": sum(
                    int(row["count"]) for row in consent_rows if bool(row["opted_in"])
                ),
                "opted_out_sessions": sum(
                    int(row["count"]) for row in consent_rows if not bool(row["opted_in"])
                ),
            },
            "policy": {
                "live_router_mutation": False,
                "cloud_export_allowed": False,
                "next_stage": "redact_research_evaluate_before_promotion",
            },
        }

    def session_id_for_event(self, event_id: str) -> str:
        row = self.connection.execute(
            "SELECT session_id FROM events WHERE event_id=?",
            (event_id,),
        ).fetchone()
        return str(row["session_id"]) if row is not None else ""

    def next_event_id(self) -> str:
        self.connection.execute("BEGIN IMMEDIATE")
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='event_counter'"
        ).fetchone()
        next_index = int(row["value"]) + 1 if row is not None else 1
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('event_counter', ?)",
            (str(next_index),),
        )
        self.connection.commit()
        return f"os_e{next_index}"

    def latest_event_id(self) -> str:
        row = self.connection.execute(
            """
            SELECT event_id
            FROM events
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()
        return str(row["event_id"]) if row is not None else ""

    def load_events(self) -> list[StoredAssistantEvent]:
        rows = self.connection.execute(
            """
            SELECT event_id, session_id, previous_event_id, next_event_id,
                   utterance, intent, route, reason, cloud_needed, evidence_keys_json,
                   capture_surface, capture_source
            FROM events
            ORDER BY rowid
            """
        )
        return [
            StoredAssistantEvent(
                event_id=str(row["event_id"]),
                session_id=str(row["session_id"]),
                previous_event_id=str(row["previous_event_id"]),
                next_event_id=str(row["next_event_id"]),
                utterance=str(row["utterance"]),
                intent=str(row["intent"]),
                route=str(row["route"]),
                reason=str(row["reason"]),
                cloud_needed=bool(row["cloud_needed"]),
                evidence_keys=tuple(str(item) for item in _loads(row["evidence_keys_json"], default=[])),
                capture_surface=str(row["capture_surface"]),
                capture_source=str(row["capture_source"]),
            )
            for row in rows
        ]

    def query_event_memory(
        self,
        *,
        query: str = "",
        intent: str = "",
        route: str = "",
        session_id: str = "",
        limit: int = 12,
    ) -> dict[str, Any]:
        """Replay/query bounded autobiographical event memory from SQLite."""

        bounded_limit = max(1, min(100, limit))
        selected_session_id = self._resolve_session_selector(session_id)
        where = []
        params: list[Any] = []
        if selected_session_id:
            where.append("session_id=?")
            params.append(selected_session_id)
        if intent:
            where.append("intent=?")
            params.append(intent)
        if route:
            where.append("route=?")
            params.append(route)
        if query:
            like = f"%{query}%"
            where.append(
                """
                (
                    utterance LIKE ?
                    OR answer LIKE ?
                    OR reason LIKE ?
                    OR intent LIKE ?
                    OR route LIKE ?
                    OR evidence_keys_json LIKE ?
                )
                """
            )
            params.extend([like, like, like, like, like, like])
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self.connection.execute(
            f"""
            SELECT rowid, event_id, session_id, previous_event_id, next_event_id,
                   utterance, intent, route, reason, answer, cloud_needed,
                   external_fetch_needed, device_action, local_memory_used,
                   evidence_keys_json, created_at
            FROM events
            {where_sql}
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (*params, bounded_limit),
        ).fetchall()
        chronological = list(reversed(rows))
        event_ids = {
            str(row["event_id"])
            for row in self.connection.execute("SELECT event_id FROM events").fetchall()
        }
        events = [_event_memory_record(row, event_ids=event_ids) for row in chronological]
        return {
            "local_only": True,
            "query": query,
            "intent": intent,
            "route": route,
            "session_id": selected_session_id,
            "limit": bounded_limit,
            "matches": len(events),
            "first_event_id": events[0]["event_id"] if events else "",
            "last_event_id": events[-1]["event_id"] if events else "",
            "chain": _event_memory_chain_summary(events),
            "events": events,
        }

    def query_recent_session_memory(
        self,
        *,
        session_limit: int = 3,
        events_per_session: int = 4,
    ) -> dict[str, Any]:
        """Return bounded events from the most recent autobiographical sessions."""

        bounded_session_limit = max(1, min(10, session_limit))
        bounded_events_per_session = max(1, min(12, events_per_session))
        session_rows = self.connection.execute(
            """
            SELECT session_id, MAX(rowid) AS latest_rowid
            FROM events
            GROUP BY session_id
            ORDER BY latest_rowid DESC
            LIMIT ?
            """,
            (bounded_session_limit,),
        ).fetchall()
        session_ids = [str(row["session_id"]) for row in reversed(session_rows)]
        rows = []
        for session_id in session_ids:
            session_events = self.connection.execute(
                """
                SELECT rowid, event_id, session_id, previous_event_id, next_event_id,
                       utterance, intent, route, reason, answer, cloud_needed,
                       external_fetch_needed, device_action, local_memory_used,
                       evidence_keys_json, created_at
                FROM events
                WHERE session_id=?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (session_id, bounded_events_per_session),
            ).fetchall()
            rows.extend(reversed(session_events))
        event_ids = {
            str(row["event_id"])
            for row in self.connection.execute("SELECT event_id FROM events").fetchall()
        }
        events = [_event_memory_record(row, event_ids=event_ids) for row in rows]
        return {
            "local_only": True,
            "session_limit": bounded_session_limit,
            "events_per_session": bounded_events_per_session,
            "session_count": len(session_ids),
            "session_ids": session_ids,
            "matches": len(events),
            "first_event_id": events[0]["event_id"] if events else "",
            "last_event_id": events[-1]["event_id"] if events else "",
            "chain": _event_memory_chain_summary(events),
            "sessions": [
                summary
                for summary in self.memory_session_summaries(limit=bounded_session_limit)
                if summary["session_id"] in set(session_ids)
            ],
            "events": events,
        }

    def memory_session_summaries(self, *, limit: int = 5) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT rowid, event_id, session_id, previous_event_id, next_event_id,
                   intent, route, created_at
            FROM events
            ORDER BY rowid
            """
        ).fetchall()
        event_ids = {str(row["event_id"]) for row in rows}
        sessions: dict[str, dict[str, Any]] = {}
        for row in rows:
            session_id = str(row["session_id"])
            summary = sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "event_count": 0,
                    "first_event_id": str(row["event_id"]),
                    "last_event_id": str(row["event_id"]),
                    "first_created_at": str(row["created_at"]),
                    "last_created_at": str(row["created_at"]),
                    "intent_counts": {},
                    "route_counts": {},
                    "dangling_links": 0,
                },
            )
            summary["event_count"] += 1
            summary["last_event_id"] = str(row["event_id"])
            summary["last_created_at"] = str(row["created_at"])
            intent_counts = summary["intent_counts"]
            route_counts = summary["route_counts"]
            intent_counts[str(row["intent"])] = int(intent_counts.get(str(row["intent"]), 0)) + 1
            route_counts[str(row["route"])] = int(route_counts.get(str(row["route"]), 0)) + 1
            previous_event_id = str(row["previous_event_id"])
            next_event_id = str(row["next_event_id"])
            if previous_event_id and previous_event_id not in event_ids:
                summary["dangling_links"] += 1
            if next_event_id and next_event_id not in event_ids:
                summary["dangling_links"] += 1
        return tuple(sessions.values())[-max(1, min(20, limit)) :]

    def build_memory_digest(
        self,
        *,
        digest_id: str = "long_horizon_latest",
        session_limit: int = 20,
        events_per_session: int = 3,
    ) -> dict[str, Any]:
        """Compact many sessions into one cited local-only memory inventory row."""

        bounded_session_limit = max(1, min(30, session_limit))
        bounded_events_per_session = max(1, min(8, events_per_session))
        session_rows = self.connection.execute(
            """
            SELECT session_id, MIN(rowid) AS first_rowid, MAX(rowid) AS latest_rowid
            FROM events
            GROUP BY session_id
            ORDER BY latest_rowid DESC
            LIMIT ?
            """,
            (bounded_session_limit,),
        ).fetchall()
        session_ids = [str(row["session_id"]) for row in reversed(session_rows)]
        rows: list[sqlite3.Row] = []
        for session_id in session_ids:
            session_events = self.connection.execute(
                """
                SELECT rowid, event_id, session_id, previous_event_id, next_event_id,
                       utterance, intent, route, reason, answer, cloud_needed,
                       external_fetch_needed, device_action, local_memory_used,
                       evidence_keys_json, created_at
                FROM events
                WHERE session_id=?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (session_id, bounded_events_per_session),
            ).fetchall()
            rows.extend(reversed(session_events))
        event_ids = {
            str(row["event_id"])
            for row in self.connection.execute("SELECT event_id FROM events").fetchall()
        }
        events = [_event_memory_record(row, event_ids=event_ids) for row in rows]
        payload = _memory_digest_payload(
            digest_id=digest_id,
            session_ids=session_ids,
            events=events,
            session_limit=bounded_session_limit,
            events_per_session=bounded_events_per_session,
        )
        self.upsert_inventory(
            "memory_digest",
            digest_id,
            payload,
            source="assistant_event_ledger_compactor",
            license="private_local",
            tags=("memory_digest", "autobiographical_memory", "local_only"),
        )
        self.connection.commit()
        return payload

    def load_memory_digest(self, digest_id: str = "long_horizon_latest") -> dict[str, Any]:
        return self.load_inventory("memory_digest").get(digest_id, {})

    def save_opportunity(
        self,
        *,
        kind: str,
        priority: float,
        reason: str,
        evidence_event_ids: Iterable[str],
        expected_cloud_reduction: int,
        proposed_action: str,
        source_candidates: Iterable[str],
        status: str = "open",
    ) -> None:
        evidence_tuple = tuple(evidence_event_ids)
        opportunity_id = _opportunity_id(kind, evidence_tuple)
        now = _now()
        self.connection.execute(
            """
            INSERT INTO opportunities(
                opportunity_id, kind, priority, reason, evidence_event_ids_json,
                expected_cloud_reduction, proposed_action, source_candidates_json,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(opportunity_id) DO UPDATE SET
                priority=excluded.priority,
                reason=excluded.reason,
                expected_cloud_reduction=excluded.expected_cloud_reduction,
                proposed_action=excluded.proposed_action,
                source_candidates_json=excluded.source_candidates_json,
                status=CASE
                    WHEN opportunities.status='executed' THEN opportunities.status
                    ELSE excluded.status
                END,
                updated_at=excluded.updated_at
            """,
            (
                opportunity_id,
                kind,
                priority,
                reason,
                _json(evidence_tuple),
                expected_cloud_reduction,
                proposed_action,
                _json(tuple(source_candidates)),
                status,
                now,
                now,
            ),
        )
        self.enqueue_job(
            kind=kind,
            payload={
                "opportunity_id": opportunity_id,
                "reason": reason,
                "evidence_event_ids": evidence_tuple,
                "proposed_action": proposed_action,
                "source_candidates": tuple(source_candidates),
            },
            priority=priority,
            resource_budget=_default_resource_budget(kind),
        )
        self.connection.commit()

    def mark_opportunity_executed(self, kind: str) -> None:
        self.connection.execute(
            "UPDATE opportunities SET status='executed', updated_at=? WHERE kind=?",
            (_now(), kind),
        )
        self.complete_jobs(kind=kind, result={"status": "executed_from_kernel"})
        self.connection.commit()

    def mark_opportunity_executed_by_id(self, opportunity_id: str) -> None:
        self.connection.execute(
            "UPDATE opportunities SET status='executed', updated_at=? WHERE opportunity_id=?",
            (_now(), opportunity_id),
        )
        self.connection.commit()

    def load_executed_jobs(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT kind FROM opportunities WHERE status='executed' ORDER BY updated_at"
        )
        return [str(row["kind"]) for row in rows]

    def enqueue_job(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        priority: float,
        resource_budget: dict[str, Any],
        max_attempts: int = 2,
        job_id: str | None = None,
    ) -> str:
        job_id = str(job_id or payload.get("opportunity_id") or f"{kind}:{self.count('jobs') + 1}")
        now = _now()
        self.connection.execute(
            """
            INSERT INTO jobs(
                job_id, kind, status, priority, attempts, max_attempts, payload_json,
                resource_budget_json, result_json, error, created_at, updated_at
            )
            VALUES (?, ?, 'queued', ?, 0, ?, ?, ?, '{}', '', ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                priority=CASE
                    WHEN jobs.status IN ('completed', 'running') THEN jobs.priority
                    ELSE excluded.priority
                END,
                payload_json=CASE
                    WHEN jobs.status IN ('completed', 'running') THEN jobs.payload_json
                    ELSE excluded.payload_json
                END,
                resource_budget_json=CASE
                    WHEN jobs.status IN ('completed', 'running') THEN jobs.resource_budget_json
                    ELSE excluded.resource_budget_json
                END,
                status=CASE
                    WHEN jobs.status IN ('completed', 'running') THEN jobs.status
                    ELSE excluded.status
                END,
                updated_at=CASE
                    WHEN jobs.status IN ('completed', 'running') THEN jobs.updated_at
                    ELSE excluded.updated_at
                END
            """,
            (
                job_id,
                kind,
                priority,
                max_attempts,
                _json(payload),
                _json(resource_budget),
                now,
                now,
            ),
        )
        self.connection.commit()
        return job_id

    def start_next_job(self, *, kinds: Iterable[str] = ()) -> StoredInventoryJob | None:
        kind_list = tuple(kinds)
        params: tuple[Any, ...] = ()
        where = "status='queued'"
        if kind_list:
            placeholders = ",".join("?" for _ in kind_list)
            where += f" AND kind IN ({placeholders})"
            params = kind_list
        row = self.connection.execute(
            f"""
            SELECT * FROM jobs
            WHERE {where}
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is None:
            return None
        self.connection.execute(
            """
            UPDATE jobs
            SET status='running', attempts=attempts + 1, updated_at=?
            WHERE job_id=?
            """,
            (_now(), row["job_id"]),
        )
        self.connection.commit()
        return _stored_job(row, attempts=int(row["attempts"]) + 1, status="running")

    def complete_job(self, job_id: str, *, result: dict[str, Any]) -> None:
        self.connection.execute(
            """
            UPDATE jobs
            SET status='completed', result_json=?, error='', updated_at=?
            WHERE job_id=?
            """,
            (_json(result), _now(), job_id),
        )
        self.connection.commit()

    def fail_job(self, job_id: str, *, error: str) -> None:
        row = self.connection.execute(
            "SELECT attempts, max_attempts FROM jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            return
        status = "failed" if int(row["attempts"]) >= int(row["max_attempts"]) else "queued"
        self.connection.execute(
            "UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
            (status, error, _now(), job_id),
        )
        self.connection.commit()

    def complete_jobs(self, *, kind: str, result: dict[str, Any]) -> None:
        self.connection.execute(
            """
            UPDATE jobs
            SET status='completed', result_json=?, error='', updated_at=?
            WHERE kind=? AND status IN ('queued', 'running')
            """,
            (_json(result), _now(), kind),
        )

    def load_jobs(self, *, status: str | None = None) -> list[StoredInventoryJob]:
        if status is None:
            rows = self.connection.execute("SELECT * FROM jobs ORDER BY priority DESC, created_at")
        else:
            rows = self.connection.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY priority DESC, created_at",
                (status,),
            )
        return [_stored_job(row) for row in rows]

    def upsert_inventory(
        self,
        kind: str,
        item_id: str,
        payload: dict[str, Any],
        *,
        source: str,
        license: str,
        tags: Iterable[str] = (),
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO inventories(kind, item_id, payload_json, source, license, tags_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kind, item_id) DO UPDATE SET
                payload_json=excluded.payload_json,
                source=excluded.source,
                license=excluded.license,
                tags_json=excluded.tags_json,
                updated_at=excluded.updated_at
            """,
            (kind, item_id, _json(payload), source, license, _json(tuple(tags)), _now()),
        )

    def load_inventory(self, kind: str) -> dict[str, dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT item_id, payload_json FROM inventories WHERE kind=? ORDER BY item_id",
            (kind,),
        )
        return {str(row["item_id"]): dict(_loads(row["payload_json"], default={})) for row in rows}

    def record_pending_action(
        self,
        *,
        action_type: str,
        target: str,
        utterance: str,
        evidence_keys: Iterable[str],
    ) -> str:
        action_id = f"action_{self.count('pending_actions') + 1}"
        now = _now()
        self.connection.execute(
            """
            INSERT INTO pending_actions(
                action_id, action_type, target, utterance, evidence_keys_json,
                confirmation_state, executed, result, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'pending', 0, '', ?, ?)
            """,
            (action_id, action_type, target, utterance, _json(tuple(evidence_keys)), now, now),
        )
        self.connection.commit()
        return action_id

    def mark_latest_pending_action_executed(self, result: Any, *, executed: bool = True) -> None:
        row = self.connection.execute(
            """
            SELECT action_id FROM pending_actions
            WHERE executed=0
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return
        self.mark_pending_action_executed(str(row["action_id"]), result=result, executed=executed)

    def mark_pending_action_executed(self, action_id: str, *, result: Any, executed: bool = True) -> None:
        self.connection.execute(
            """
            UPDATE pending_actions
            SET confirmation_state='confirmed', executed=?, result=?, updated_at=?
            WHERE action_id=?
            """,
            (int(executed), _action_result_json(result), _now(), action_id),
        )
        self.connection.commit()

    def mark_latest_pending_action_cancelled(self, result: str) -> None:
        row = self.connection.execute(
            """
            SELECT action_id FROM pending_actions
            WHERE executed=0 AND confirmation_state='pending'
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return
        self.connection.execute(
            """
            UPDATE pending_actions
            SET confirmation_state='cancelled', executed=0, result=?, updated_at=?
            WHERE action_id=?
            """,
            (result, _now(), row["action_id"]),
        )
        self.connection.commit()

    def latest_pending_action(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT action_id, action_type, target, utterance, evidence_keys_json
            FROM pending_actions
            WHERE executed=0 AND confirmation_state='pending'
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return {
            "action_id": str(row["action_id"]),
            "action_type": str(row["action_type"]),
            "target": str(row["target"]),
            "utterance": str(row["utterance"]),
            "evidence_keys": tuple(str(item) for item in _loads(row["evidence_keys_json"], default=[])),
        }

    def count(self, table: str) -> int:
        if table not in {
            "events",
            "membrane_decisions",
            "homeostatic_snapshots",
            "synthesis_traces",
            "response_integrity",
            "session_improvement_consent",
            "improvement_candidates",
            "opportunities",
            "inventories",
            "pending_actions",
            "jobs",
            "user_facts",
            "self_state",
            "lexemes",
            "word_forms",
            "lexical_senses",
            "lexical_provenance",
            "lexical_relation_candidates",
            "lexicon_ingestions",
        }:
            raise ValueError(f"unsupported table: {table}")
        row = self.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"])

    def table_counts(self) -> dict[str, int]:
        return {
            table: self.count(table)
            for table in (
                "events",
                "membrane_decisions",
                "homeostatic_snapshots",
                "synthesis_traces",
                "response_integrity",
                "session_improvement_consent",
                "improvement_candidates",
                "opportunities",
                "inventories",
                "pending_actions",
                "jobs",
                "user_facts",
                "self_state",
                "lexemes",
                "word_forms",
                "lexical_senses",
                "lexical_provenance",
                "lexical_relation_candidates",
                "lexicon_ingestions",
            )
        }

    def _configure(self) -> None:
        self.connection.execute("PRAGMA foreign_keys=ON")
        if str(self.path) != ":memory:":
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")

    def _ensure_event_link_columns(self) -> None:
        rows = self.connection.execute("PRAGMA table_info(events)").fetchall()
        columns = {str(row["name"]) for row in rows}
        migrations = (
            ("session_id", "ALTER TABLE events ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy_session'"),
            ("previous_event_id", "ALTER TABLE events ADD COLUMN previous_event_id TEXT NOT NULL DEFAULT ''"),
            ("next_event_id", "ALTER TABLE events ADD COLUMN next_event_id TEXT NOT NULL DEFAULT ''"),
        )
        for column, statement in migrations:
            if column not in columns:
                self.connection.execute(statement)
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, created_at)"
        )

    def _ensure_event_provenance_columns(self) -> None:
        rows = self.connection.execute("PRAGMA table_info(events)").fetchall()
        columns = {str(row["name"]) for row in rows}
        migrations = (
            ("capture_surface", "ALTER TABLE events ADD COLUMN capture_surface TEXT NOT NULL DEFAULT ''"),
            ("capture_source", "ALTER TABLE events ADD COLUMN capture_source TEXT NOT NULL DEFAULT ''"),
        )
        for column, statement in migrations:
            if column not in columns:
                self.connection.execute(statement)

    def _ensure_user_fact_policy_columns(self) -> None:
        rows = self.connection.execute("PRAGMA table_info(user_facts)").fetchall()
        columns = {str(row["name"]) for row in rows}
        migrations = (
            ("cloud_eligible", "ALTER TABLE user_facts ADD COLUMN cloud_eligible INTEGER NOT NULL DEFAULT 0"),
            ("scope", "ALTER TABLE user_facts ADD COLUMN scope TEXT NOT NULL DEFAULT 'private_local'"),
        )
        for column, statement in migrations:
            if column not in columns:
                self.connection.execute(statement)
        self.connection.execute(
            """
            UPDATE user_facts
            SET cloud_eligible=0
            WHERE local_only=1 OR consent=0
            """
        )

    def _resolve_session_selector(self, session_id: str) -> str:
        requested = session_id.strip()
        if requested and requested.lower() != "latest":
            return requested
        if requested.lower() == "latest":
            row = self.connection.execute(
                """
                SELECT session_id
                FROM events
                ORDER BY rowid DESC
                LIMIT 1
                """
            ).fetchone()
            return str(row["session_id"]) if row is not None else ""
        return ""

    def start_new_session(self) -> None:
        """Make the next recorded turn allocate a fresh local session id."""

        self._active_session_id = ""

    def use_session(self, session_id: str) -> None:
        """Bind subsequent turns to an explicit local capture session."""

        self._active_session_id = _validated_session_id(session_id)

    def _current_session_id(self) -> str:
        if self._active_session_id:
            return self._active_session_id
        self.connection.execute("BEGIN IMMEDIATE")
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='session_counter'"
        ).fetchone()
        next_index = int(row["value"]) + 1 if row is not None else 1
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('session_counter', ?)",
            (str(next_index),),
        )
        self.connection.commit()
        self._active_session_id = f"session_{next_index}"
        return self._active_session_id


def initialize_assistant_os_database(
    db_path: str | Path,
    *,
    seed_path: str | Path | None = None,
) -> AssistantOSStore:
    store = AssistantOSStore(db_path)
    if seed_path is not None:
        seed_assistant_os_store(store, seed_path)
    return store


def _validated_session_id(session_id: str) -> str:
    normalized = str(session_id or "").strip()
    if not SESSION_ID_RE.fullmatch(normalized):
        raise ValueError("session_id must be 1-96 safe identifier characters")
    return normalized


def seed_assistant_os_store(store: AssistantOSStore, seed_path: str | Path) -> None:
    seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    if seed.get("schema") != SEED_SCHEMA:
        raise ValueError(f"unsupported assistant OS seed schema: {seed.get('schema')!r}")
    for fact in seed.get("user_facts", []):
        store.upsert_user_fact(
            str(fact["key"]),
            str(fact["value"]),
            source=str(fact.get("source", "seed")),
            confidence=float(fact.get("confidence", 0.8)),
            consent=bool(fact.get("consent", True)),
            local_only=bool(fact.get("local_only", True)),
            cloud_eligible=bool(fact.get("cloud_eligible", False)),
            scope=str(fact.get("scope", "private_local")),
        )
    for item in seed.get("inventories", []):
        store.upsert_inventory(
            str(item["kind"]),
            str(item["item_id"]),
            dict(item.get("payload", {})),
            source=str(item.get("source", "seed")),
            license=str(item.get("license", "local_seed")),
            tags=tuple(str(tag) for tag in item.get("tags", [])),
        )
    store.connection.commit()


def seed_assistant_os_lexicon(store: AssistantOSStore) -> None:
    """Seed all legacy vocabulary and functional-grammar candidates into the store.

    Seeds functional-grammar verb/nominal candidates plus all 200+ LEGACY
    router vocabulary entries. All semantic classes must be in the ontology
    contract (semantic_classes.v1.json); callers must extend the contract
    before seeding new classes.
    """
    from .assistant_lexicon import (
        configure_lexicon_router_families,
        lexicon_ingest,
    )
    from .assistant_lexicon_legacy import (
        build_legacy_lexicon_candidates,
        build_legacy_router_candidates,
    )

    candidates = (
        build_legacy_lexicon_candidates()
        + build_legacy_router_candidates(seed_all=True)
    )
    for candidate in candidates:
        lexicon_ingest(store, candidate, expected_provenance="seed_authored")
    configure_lexicon_router_families(store, ("media", "story", "weather"))


def _opportunity_id(kind: str, evidence_event_ids: tuple[str, ...]) -> str:
    suffix = "_".join(evidence_event_ids) if evidence_event_ids else "no_evidence"
    return f"{kind}:{suffix}"


def _default_resource_budget(kind: str) -> dict[str, Any]:
    if kind == "build_story_inventory":
        return {
            "max_items": 12,
            "max_source_bytes": 250000,
            "network": "metadata_only",
            "cpu_class": "raspberry_pi",
        }
    if kind == "refresh_weather_cache":
        return {
            "max_items": 7,
            "max_source_bytes": 20000,
            "network": "tool_fetch",
            "cpu_class": "raspberry_pi",
        }
    return {
        "max_items": 1,
        "max_source_bytes": 10000,
        "network": "none",
        "cpu_class": "raspberry_pi",
    }


def _stored_job(
    row: sqlite3.Row,
    *,
    attempts: int | None = None,
    status: str | None = None,
) -> StoredInventoryJob:
    return StoredInventoryJob(
        job_id=str(row["job_id"]),
        kind=str(row["kind"]),
        status=status or str(row["status"]),
        priority=float(row["priority"]),
        attempts=int(row["attempts"]) if attempts is None else attempts,
        max_attempts=int(row["max_attempts"]),
        resource_budget=dict(_loads(row["resource_budget_json"], default={})),
        payload=dict(_loads(row["payload_json"], default={})),
    )


def _event_memory_record(row: sqlite3.Row, *, event_ids: set[str]) -> dict[str, Any]:
    previous_event_id = str(row["previous_event_id"])
    next_event_id = str(row["next_event_id"])
    return {
        "event_id": str(row["event_id"]),
        "session_id": str(row["session_id"]),
        "previous_event_id": previous_event_id,
        "next_event_id": next_event_id,
        "previous_link_valid": not previous_event_id or previous_event_id in event_ids,
        "next_link_valid": not next_event_id or next_event_id in event_ids,
        "utterance": str(row["utterance"]),
        "intent": str(row["intent"]),
        "route": str(row["route"]),
        "reason": str(row["reason"]),
        "answer": str(row["answer"]),
        "cloud_needed": bool(row["cloud_needed"]),
        "external_fetch_needed": bool(row["external_fetch_needed"]),
        "device_action": bool(row["device_action"]),
        "local_memory_used": bool(row["local_memory_used"]),
        "evidence_keys": tuple(str(item) for item in _loads(row["evidence_keys_json"], default=[])),
        "created_at": str(row["created_at"]),
    }


def _event_memory_chain_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "returned_events": 0,
            "linked_previous_in_ledger": 0,
            "linked_next_in_ledger": 0,
            "dangling_previous": 0,
            "dangling_next": 0,
        }
    return {
        "returned_events": len(events),
        "linked_previous_in_ledger": sum(
            1 for event in events if event["previous_event_id"] and event["previous_link_valid"]
        ),
        "linked_next_in_ledger": sum(
            1 for event in events if event["next_event_id"] and event["next_link_valid"]
        ),
        "dangling_previous": sum(
            1 for event in events if event["previous_event_id"] and not event["previous_link_valid"]
        ),
        "dangling_next": sum(
            1 for event in events if event["next_event_id"] and not event["next_link_valid"]
        ),
    }


def _memory_digest_payload(
    *,
    digest_id: str,
    session_ids: list[str],
    events: list[dict[str, Any]],
    session_limit: int,
    events_per_session: int,
) -> dict[str, Any]:
    intent_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for event in events:
        intent = str(event["intent"])
        route = str(event["route"])
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        route_counts[route] = route_counts.get(route, 0) + 1
    highlights = [_memory_digest_highlight(event) for event in events[:24]]
    threads = _memory_digest_threads(events)
    session_summaries = _memory_digest_session_summaries(events, session_ids=session_ids)
    capability_transitions = _memory_digest_capability_transitions(events)
    active_limits = _memory_digest_active_limits(events)
    open_loops = _memory_digest_open_loops(events)
    quality = _memory_digest_quality(
        local_only=True,
        session_count=len(session_ids),
        event_count=len(events),
        session_limit=session_limit,
        events_per_session=events_per_session,
        intent_counts=intent_counts,
        route_counts=route_counts,
        threads=threads,
        session_summaries=session_summaries,
        capability_transitions=capability_transitions,
        active_limits=active_limits,
        open_loops=open_loops,
    )
    summary = _memory_digest_summary(
        session_count=len(session_ids),
        event_count=len(events),
        intent_counts=intent_counts,
        route_counts=route_counts,
        highlights=highlights[:8],
        threads=threads,
        capability_transitions=capability_transitions,
        active_limits=active_limits,
        open_loops=open_loops,
    )
    return {
        "digest_id": digest_id,
        "local_only": True,
        "session_limit": session_limit,
        "events_per_session": events_per_session,
        "session_count": len(session_ids),
        "session_ids": session_ids,
        "event_count": len(events),
        "first_event_id": events[0]["event_id"] if events else "",
        "last_event_id": events[-1]["event_id"] if events else "",
        "intent_counts": dict(sorted(intent_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "highlights": highlights,
        "threads": threads,
        "session_summaries": session_summaries,
        "capability_transitions": capability_transitions,
        "active_limits": active_limits,
        "open_loops": open_loops,
        "quality": quality,
        "summary": summary,
    }


def _memory_digest_summary(
    *,
    session_count: int,
    event_count: int,
    intent_counts: dict[str, int],
    route_counts: dict[str, int],
    highlights: list[str],
    threads: list[dict[str, Any]],
    capability_transitions: list[str],
    active_limits: list[str],
    open_loops: list[str],
) -> str:
    if event_count == 0:
        return "No prior local conversation events are available yet."
    top_intents = _top_counts_text(intent_counts, limit=4)
    top_routes = _top_counts_text(route_counts, limit=4)
    thread_text = _memory_digest_thread_text(threads)
    transition_text = "; ".join(capability_transitions[:5])
    limit_text = "; ".join((active_limits + open_loops)[:5])
    highlight_text = "; ".join(highlights[:3])
    parts = [
        f"I remember {event_count} local events across {session_count} sessions",
        f"main threads: {thread_text}" if thread_text else f"main intents: {top_intents}",
        f"capability changes: {transition_text}" if transition_text else "",
        f"limits/open loops: {limit_text}" if limit_text else "",
        f"route mix: {top_routes}" if top_routes else "",
        f"sampled details: {highlight_text}" if highlight_text else "",
    ]
    return ". ".join(part for part in parts if part)[:1200]


def _memory_digest_highlight(event: dict[str, Any]) -> str:
    utterance = str(event["utterance"]).replace("\n", " ").strip()
    if len(utterance) > 90:
        utterance = f"{utterance[:87]}..."
    return (
        f"{event['session_id']}:{event['intent']} via {event['route']} "
        f"- \"{utterance}\""
    )


def _memory_digest_quality(
    *,
    local_only: bool,
    session_count: int,
    event_count: int,
    session_limit: int,
    events_per_session: int,
    intent_counts: dict[str, int],
    route_counts: dict[str, int],
    threads: list[dict[str, Any]],
    session_summaries: list[dict[str, Any]],
    capability_transitions: list[str],
    active_limits: list[str],
    open_loops: list[str],
) -> dict[str, Any]:
    if event_count <= 0:
        return {
            "score": 0.0,
            "floor": MEMORY_DIGEST_QUALITY_FLOOR,
            "passed": False,
            "components": {
                "local_only": 1.0 if local_only else 0.0,
                "long_horizon": 0.0,
                "event_density": 0.0,
                "thread_coverage": 0.0,
                "session_summary_coverage": 0.0,
                "key_moment_coverage": 0.0,
                "resolution_awareness": 0.0,
                "intent_route_diversity": 0.0,
            },
            "warnings": ["no_events"],
        }
    target_sessions = max(1, min(3, session_limit))
    target_events = max(1, target_sessions * max(1, min(2, events_per_session)))
    meaningful_threads = [item for item in threads if item.get("thread") != "other"]
    key_moment_count = sum(len(item.get("key_moments", [])) for item in threads)
    component_scores = {
        "local_only": 1.0 if local_only else 0.0,
        "long_horizon": _clamp01(session_count / target_sessions),
        "event_density": _clamp01(event_count / target_events),
        "thread_coverage": _clamp01(len(meaningful_threads) / max(1, min(5, event_count))),
        "session_summary_coverage": _clamp01(len(session_summaries) / max(1, session_count)),
        "key_moment_coverage": _clamp01(key_moment_count / max(1, min(event_count, len(threads) * 2))),
        "resolution_awareness": 1.0 if capability_transitions or active_limits or open_loops else 0.0,
        "intent_route_diversity": _clamp01((len(intent_counts) + len(route_counts)) / 6),
    }
    score = (
        component_scores["local_only"] * 0.1
        + component_scores["long_horizon"] * 0.15
        + component_scores["event_density"] * 0.15
        + component_scores["thread_coverage"] * 0.15
        + component_scores["session_summary_coverage"] * 0.1
        + component_scores["key_moment_coverage"] * 0.15
        + component_scores["resolution_awareness"] * 0.15
        + component_scores["intent_route_diversity"] * 0.05
    )
    warnings: list[str] = []
    if not local_only:
        warnings.append("not_local_only")
    if session_count < 2 or event_count < 3:
        warnings.append("insufficient_long_horizon")
        score = min(score, MEMORY_DIGEST_QUALITY_FLOOR - 0.08)
    if not meaningful_threads:
        warnings.append("no_meaningful_threads")
    if not session_summaries:
        warnings.append("no_session_summaries")
    if key_moment_count <= 0:
        warnings.append("no_key_moments")
    if not (capability_transitions or active_limits or open_loops):
        warnings.append("no_transition_limit_or_open_loop")
    rounded_components = {key: round(value, 3) for key, value in component_scores.items()}
    rounded_score = round(score, 3)
    return {
        "score": rounded_score,
        "floor": MEMORY_DIGEST_QUALITY_FLOOR,
        "passed": rounded_score >= MEMORY_DIGEST_QUALITY_FLOOR,
        "components": rounded_components,
        "warnings": warnings,
    }


def _memory_digest_threads(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for event in events:
        for thread_id, label in _memory_digest_event_threads(event):
            bucket = buckets.setdefault(
                thread_id,
                {
                    "thread": thread_id,
                    "label": label,
                    "event_count": 0,
                    "sessions": set(),
                    "route_counts": {},
                    "key_moments": [],
                    "latest_session": "",
                },
            )
            bucket["event_count"] += 1
            bucket["sessions"].add(str(event["session_id"]))
            bucket["latest_session"] = str(event["session_id"])
            routes = bucket["route_counts"]
            route = str(event["route"])
            routes[route] = int(routes.get(route, 0)) + 1
            moment = _memory_digest_moment(event)
            if moment and moment not in bucket["key_moments"] and len(bucket["key_moments"]) < 4:
                bucket["key_moments"].append(moment)
    threads = []
    for bucket in buckets.values():
        threads.append(
            {
                "thread": bucket["thread"],
                "label": bucket["label"],
                "event_count": int(bucket["event_count"]),
                "session_count": len(bucket["sessions"]),
                "latest_session": bucket["latest_session"],
                "route_counts": dict(sorted(bucket["route_counts"].items())),
                "key_moments": list(bucket["key_moments"]),
            }
        )
    return sorted(threads, key=lambda item: (-int(item["event_count"]), str(item["label"])))


def _memory_digest_event_threads(event: dict[str, Any]) -> list[tuple[str, str]]:
    intent = str(event["intent"])
    route = str(event["route"])
    text = _memory_digest_event_text(event)
    threads: list[tuple[str, str]] = []
    if "household" in text or "family" in text:
        threads.append(("household_memory", "household memory"))
    if "routine" in text or "schedule" in text:
        threads.append(("routine_memory", "routine memory"))
    if intent == "personal_memory" and not any(
        thread_id in {"household_memory", "routine_memory"} for thread_id, _ in threads
    ):
        threads.append(("profile_memory", "profile memory"))
    if intent == "story":
        threads.append(("story_inventory", "story inventory"))
    if intent == "weather" or "weather" in text:
        threads.append(("weather_cache", "weather cache"))
    if intent == "common_sense_safety":
        threads.append(("safety_guidance", "safety guidance"))
    if intent == "meal_suggestion":
        threads.append(("meal_advice", "meal advice"))
    if intent == "health_advice":
        threads.append(("health_advice", "health advice"))
    if intent == "media_playback":
        threads.append(("media_playback", "media playback"))
    if intent == "social_contact":
        threads.append(("trusted_contact", "trusted contact"))
    if intent == "autobiographical_memory":
        threads.append(("conversation_recall", "conversation recall"))
    if (
        route == "reject"
        or "cloud_unavailable" in text
        or "private_facts_to_cloud" in text
        or "previous conversation to the cloud" in text
    ):
        threads.append(("boundary_control", "privacy/offline boundary"))
    if not threads:
        threads.append(("other", "other conversation"))
    return _dedupe_pairs(threads)


def _memory_digest_session_summaries(
    events: list[dict[str, Any]],
    *,
    session_ids: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {session_id: [] for session_id in session_ids}
    for event in events:
        grouped.setdefault(str(event["session_id"]), []).append(event)
    summaries = []
    for session_id in session_ids:
        session_events = grouped.get(session_id, [])
        if not session_events:
            continue
        intent_counts: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        moments: list[str] = []
        for event in session_events:
            intent = str(event["intent"])
            route = str(event["route"])
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            moment = _memory_digest_moment(event)
            if moment and moment not in moments and len(moments) < 3:
                moments.append(moment)
        summaries.append(
            {
                "session_id": session_id,
                "event_count": len(session_events),
                "main_intents": _top_count_keys(intent_counts, limit=3),
                "route_counts": dict(sorted(route_counts.items())),
                "key_moments": moments,
            }
        )
    return summaries


def _memory_digest_capability_transitions(events: list[dict[str, Any]]) -> list[str]:
    reasons = {str(event["reason"]) for event in events}
    transitions: list[str] = []
    if "missing_story_model" in reasons and "local_story_inventory" in reasons:
        transitions.append("story requests moved from cloud handoff to local story inventory")
    if "weather_cache_miss" in reasons and "weather_cache_hit" in reasons:
        transitions.append("weather moved from cache miss to cached local forecast")
    if "empty_media_library" in reasons and (
        "local_media_action" in reasons or _has_confirmed_action(events, "media_playback")
    ):
        transitions.append("media moved from missing index to confirmed local action")
    if "missing_contact" in reasons and (
        "trusted_contact_action" in reasons or _has_confirmed_action(events, "social_contact")
    ):
        transitions.append("trusted contact moved from setup gap to confirmed local action")
    if _has_text_reason(events, "household", "personal_memory_empty") and "consented_household_memory_stored" in reasons:
        transitions.append("household memory moved from unknown to user-consented local fact")
    if _has_text_reason(events, "routine", "personal_memory_empty") and "consented_routine_memory_stored" in reasons:
        transitions.append("routine memory moved from unknown to user-consented local fact")
    if "autobiographical_memory_summary" in reasons or "autobiographical_memory_digest" in reasons:
        transitions.append("conversation recall stayed local using the event ledger")
    return transitions


def _memory_digest_active_limits(events: list[dict[str, Any]]) -> list[str]:
    reasons = {str(event["reason"]) for event in events}
    limits: list[str] = []
    if "blocked_private_facts_to_cloud" in reasons:
        limits.append("private conversation and user memory stayed local-only")
    if "cloud_unavailable" in reasons:
        limits.append("latest/open-web questions still need network or cloud and are not invented offline")
    if "cancelled_pending_action" in reasons or "no_pending_action_to_confirm" in reasons:
        limits.append("cancelled or stale actions were not replayed")
    return limits


def _memory_digest_open_loops(events: list[dict[str, Any]]) -> list[str]:
    reasons = {str(event["reason"]) for event in events}
    loops: list[str] = []
    if "missing_story_model" in reasons and "local_story_inventory" not in reasons:
        loops.append("story inventory still needs local story models")
    if "weather_cache_miss" in reasons and "weather_cache_hit" not in reasons:
        loops.append("weather cache still needs a refresh")
    if "empty_media_library" in reasons and "local_media_action" not in reasons:
        loops.append("media library still needs indexing")
    if "missing_contact" in reasons and "consented_trusted_contact_stored" not in reasons:
        loops.append("trusted contact still needs user setup")
    if _has_text_reason(events, "household", "personal_memory_empty") and "consented_household_memory_stored" not in reasons:
        loops.append("household memory still needs consented setup")
    if _has_text_reason(events, "routine", "personal_memory_empty") and "consented_routine_memory_stored" not in reasons:
        loops.append("routine memory still needs a user-provided fact")
    if any(
        event["reason"] == "consent_revoked_user_fact" and "routine" in _memory_digest_event_text(event)
        for event in events
    ):
        loops.append("routine memory was revoked and needs fresh consent before reuse")
    return loops


def _memory_digest_moment(event: dict[str, Any]) -> str:
    reason = str(event["reason"])
    intent = str(event["intent"])
    utterance = _short_digest_utterance(event)
    if reason == "profile_update":
        return f"learned profile fact from \"{utterance}\""
    if reason == "consented_household_memory_stored":
        return f"stored household memory from \"{utterance}\""
    if reason == "consented_routine_memory_stored":
        return f"stored routine memory from \"{utterance}\""
    if reason == "consented_trusted_contact_stored":
        return f"stored trusted contact from \"{utterance}\""
    if reason == "personal_memory_empty":
        return f"noticed a missing memory for \"{utterance}\""
    if reason == "personal_memory_recall":
        return f"answered from saved memory for \"{utterance}\""
    if reason == "personal_memory_summary":
        return "summarized saved personal facts locally"
    if reason == "weather_cache_miss":
        return "needed a weather refresh before answering"
    if reason == "weather_cache_hit":
        return "answered weather from cached local forecast"
    if reason in {"school_clothing_weather_policy", "local_common_sense_policy"}:
        return f"answered safety locally for \"{utterance}\""
    if reason == "missing_story_model":
        return "could not tell a local story yet"
    if reason == "local_story_inventory":
        return "told a story from local story inventory"
    if reason == "memory_plus_weather_cache":
        return "combined food memory with cached weather"
    if reason == "bounded_general_health_guidance":
        return "gave bounded health guidance from local goals and policy"
    if reason == "empty_media_library":
        return "noticed media needed indexing"
    if reason == "local_media_action":
        return "prepared a local media action behind confirmation"
    if reason == "confirmed_device_action" and intent == "media_playback":
        return "confirmed the local media action"
    if reason == "cancelled_pending_action":
        return "cancelled a pending media action"
    if reason == "no_pending_action_to_confirm":
        return "blocked a stale confirmation replay"
    if reason == "missing_contact":
        return "noticed a trusted-contact setup gap"
    if reason == "trusted_contact_action":
        return "prepared a trusted-contact call behind confirmation"
    if reason == "confirmed_device_action" and intent == "social_contact":
        return "confirmed the trusted-contact action"
    if reason == "blocked_private_facts_to_cloud":
        return "refused to send private memory to cloud"
    if reason == "cloud_unavailable":
        return "did not invent latest news while offline"
    if reason in {"autobiographical_memory_summary", "autobiographical_memory_digest"}:
        return "recalled earlier conversation from local event memory"
    if reason == "consent_revoked_user_fact":
        return f"forgot a user fact after \"{utterance}\""
    return f"{str(event['route'])}/{reason} for \"{utterance}\""


def _memory_digest_thread_text(threads: list[dict[str, Any]]) -> str:
    if not threads:
        return ""
    main = threads[:5]
    return "; ".join(_memory_digest_thread_phrase(item) for item in main)


def _memory_digest_thread_phrase(item: dict[str, Any]) -> str:
    event_count = int(item["event_count"])
    session_count = int(item["session_count"])
    event_word = "event" if event_count == 1 else "events"
    session_word = "session" if session_count == 1 else "sessions"
    return f"{item['label']} ({event_count} {event_word}, {session_count} {session_word})"


def _memory_digest_event_text(event: dict[str, Any]) -> str:
    return (
        f"{event.get('utterance', '')} {event.get('intent', '')} "
        f"{event.get('route', '')} {event.get('reason', '')}"
    ).lower()


def _short_digest_utterance(event: dict[str, Any]) -> str:
    utterance = str(event["utterance"]).replace("\n", " ").strip()
    if len(utterance) > 72:
        utterance = f"{utterance[:69]}..."
    return utterance


def _has_confirmed_action(events: list[dict[str, Any]], intent: str) -> bool:
    return any(
        str(event["intent"]) == intent and str(event["reason"]) == "confirmed_device_action"
        for event in events
    )


def _has_text_reason(events: list[dict[str, Any]], marker: str, reason: str) -> bool:
    return any(
        str(event["reason"]) == reason and marker in _memory_digest_event_text(event)
        for event in events
    )


def _top_count_keys(counts: dict[str, int], *, limit: int) -> list[str]:
    return [key for key, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe_pairs(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for key, label in items:
        if key in seen:
            continue
        seen.add(key)
        deduped.append((key, label))
    return deduped


def _top_counts_text(counts: dict[str, int], *, limit: int) -> str:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{key}={value}" for key, value in ordered[:limit])


def _remove_revoked_profile_values(
    revoked_keys: set[str],
    *,
    facts: dict[str, str],
    preferences: dict[str, str],
    health_goals: list[str],
) -> None:
    for key in revoked_keys:
        if key.startswith("facts."):
            facts.pop(key.split(".", 1)[1], None)
        elif key.startswith("preferences."):
            preferences.pop(key.split(".", 1)[1], None)
        elif key.startswith("health_goals."):
            health_goals.clear()


def _is_inventory_payload_fresh(payload: dict[str, Any]) -> bool:
    stale = payload.get("stale")
    if isinstance(stale, bool) and stale:
        return False
    if isinstance(stale, str) and stale.strip().lower() in {"1", "true", "yes"}:
        return False
    fresh = payload.get("fresh")
    if isinstance(fresh, bool) and not fresh:
        return False
    if isinstance(fresh, str) and fresh.strip().lower() in {"0", "false", "no"}:
        return False
    expires_at = payload.get("expires_at") or payload.get("valid_until")
    if not expires_at:
        return True
    try:
        expires = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > datetime.now(timezone.utc)


def _story_frame_payload_text(payload: dict[str, Any]) -> str:
    return str(payload.get("narrative_frame") or payload.get("template") or "")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _memory_scope(scope: str, *, local_only: bool, cloud_eligible: bool) -> str:
    normalized = "_".join(scope.strip().lower().replace("-", "_").split())
    if local_only or not cloud_eligible:
        allowed_local = {
            "private_local",
            "child_local",
            "household_local",
            "routine_local",
            "profile_local",
            "trusted_contact_local",
        }
        return normalized if normalized in allowed_local else "private_local"
    allowed = {
        "user_approved_cloud",
        "shareable_profile",
        "public_profile",
        "tool_eligible",
    }
    return normalized if normalized in allowed else "user_approved_cloud"


def _default_user_fact_scope(key: str) -> str:
    if not key.startswith("facts."):
        return "private_local"
    fact_key = key.split(".", 1)[1]
    if any(marker in fact_key for marker in ("child", "son", "daughter")):
        return "child_local"
    if any(marker in fact_key for marker in ("household", "family", "shared_device")):
        return "household_local"
    if any(marker in fact_key for marker in ("routine", "schedule")):
        return "routine_local"
    return "private_local"


def _action_result_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _json(value)


def _loads(value: str, *, default: Any = None) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
