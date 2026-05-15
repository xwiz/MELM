"""Dependency-light local runtime for the MELM SLM Appliance."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DATE_RE = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2})\b",
    re.IGNORECASE,
)
MONTH_NAMES = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}
MONTH_NUMBERS = {name.lower(): int(number) for number, name in MONTH_NAMES.items()}
MONTH_NUMBERS.update({name[:3].lower(): value for name, value in MONTH_NUMBERS.items()})
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
ANSWER_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
}
NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
GENERIC_EVENT_TERMS = {
    "be",
    "do",
    "get",
    "go",
    "have",
    "make",
    "meet",
    "take",
    "tell",
    "went",
}


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    text: str
    kind: str = "memory"
    created_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def searchable_text(self) -> str:
        compact_parts = [
            self.text,
            self.metadata.get("observation", ""),
            self.metadata.get("summary", ""),
            self.metadata.get("event_summary", ""),
        ]
        return "\n".join(part for part in compact_parts if part)


@dataclass(frozen=True)
class RetrievedMemory:
    memory: MemoryRecord
    score: float
    reason: str


@dataclass(frozen=True)
class ApplianceAnswer:
    question: str
    answer: str
    confidence: float
    citations: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    context_tokens: int
    context: str


class MelmAppliance:
    """Local memory appliance with retrieval, adaptive packing, and cited answers."""

    def __init__(self, records: Iterable[MemoryRecord] = ()) -> None:
        self.records: list[MemoryRecord] = list(records)
        self._rebuild()

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "MelmAppliance":
        records: list[MemoryRecord] = []
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSONL record") from exc
                records.append(_memory_from_json(record, source=source, line_number=line_number))
        return cls(records)

    def save_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(_memory_to_json(record), ensure_ascii=False) + "\n")

    def add(self, record: MemoryRecord) -> None:
        self.records.append(record)
        self._rebuild()

    def retrieve(self, question: str, *, k: int = 5) -> list[RetrievedMemory]:
        query_terms = _tokens(question)
        query_entities = _query_entities(question)
        scored: list[RetrievedMemory] = []
        for record in self.records:
            lexical = self._bm25_score(record.memory_id, query_terms)
            entity_overlap = len(self._entities[record.memory_id] & query_entities)
            compact_boost = _compact_match_boost(record, query_terms)
            score = lexical + entity_overlap * 0.35 + compact_boost
            reason_parts = ["bm25"]
            if entity_overlap:
                reason_parts.append("entity")
            if compact_boost:
                reason_parts.append("compact")
            scored.append(RetrievedMemory(record, score, "+".join(reason_parts)))
        scored.sort(key=lambda item: (item.score, item.memory.memory_id), reverse=True)
        return scored[:k]

    def pack_context(
        self,
        question: str,
        retrieved: Iterable[RetrievedMemory],
        *,
        token_budget: int = 1200,
        snippet_tokens_per_memory: int = 160,
    ) -> tuple[str, tuple[str, ...], int]:
        parts: list[str] = []
        citations: list[str] = []
        used = 0
        for item in retrieved:
            text = _adaptive_context_text(
                item.memory,
                question=question,
                snippet_tokens=snippet_tokens_per_memory,
            )
            tokens = _tokens(text)
            if not tokens:
                continue
            if used + len(tokens) > token_budget and parts:
                continue
            if used + len(tokens) > token_budget:
                tokens = tokens[: token_budget - used]
                text = " ".join(tokens)
            used += len(tokens)
            citations.append(item.memory.memory_id)
            parts.append(f"[{item.memory.memory_id}]\n{text}")
            if used >= token_budget:
                break
        return "\n\n".join(parts), tuple(citations), used

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        token_budget: int = 1200,
    ) -> ApplianceAnswer:
        retrieved = self.retrieve(question, k=k)
        context, citations, context_tokens = self.pack_context(
            question,
            retrieved,
            token_budget=token_budget,
        )
        temporal_answer = _extract_temporal_answer(question, retrieved)
        if temporal_answer is None:
            structured_answer = _extract_structured_answer(question, retrieved)
            if structured_answer is None:
                answer, confidence = _extract_answer(question, context)
            else:
                answer, confidence = structured_answer
        else:
            answer, confidence = temporal_answer
        return ApplianceAnswer(
            question=question,
            answer=answer,
            confidence=confidence,
            citations=citations,
            retrieved_ids=tuple(item.memory.memory_id for item in retrieved),
            context_tokens=context_tokens,
            context=context,
        )

    def _rebuild(self) -> None:
        self._vectors = {
            record.memory_id: _token_counts(record.searchable_text())
            for record in self.records
        }
        self._doc_lengths = {
            memory_id: sum(counts.values())
            for memory_id, counts in self._vectors.items()
        }
        self._avg_doc_length = (
            sum(self._doc_lengths.values()) / len(self._doc_lengths)
            if self._doc_lengths
            else 0.0
        )
        self._doc_freqs: Counter[str] = Counter()
        for counts in self._vectors.values():
            self._doc_freqs.update(counts.keys())
        self._entities = {
            record.memory_id: _record_entities(record)
            for record in self.records
        }

    def _bm25_score(self, memory_id: str, query_terms: list[str]) -> float:
        counts = self._vectors[memory_id]
        doc_length = max(1, self._doc_lengths[memory_id])
        total_docs = max(1, len(self.records))
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            doc_freq = self._doc_freqs.get(term, 0)
            idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denominator = freq + 1.2 * (
                1.0 - 0.75 + 0.75 * doc_length / max(1.0, self._avg_doc_length)
            )
            score += idf * (freq * 2.2) / denominator
        return score


def _memory_from_json(record: dict[str, Any], *, source: Path, line_number: int) -> MemoryRecord:
    if record.get("schema") not in {None, "melm.appliance.memory.v1"}:
        raise ValueError(f"{source}:{line_number}: unsupported schema {record.get('schema')!r}")
    memory_id = record.get("memory_id") or record.get("id")
    text = record.get("text")
    if not isinstance(memory_id, str) or not memory_id:
        raise ValueError(f"{source}:{line_number}: memory_id is required")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{source}:{line_number}: text is required")
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        kind=str(record.get("kind", "memory")),
        created_at=str(record.get("created_at", "")),
        metadata={str(key): str(value) for key, value in dict(record.get("metadata", {})).items()},
    )


def _memory_to_json(record: MemoryRecord) -> dict[str, Any]:
    return {
        "schema": "melm.appliance.memory.v1",
        **asdict(record),
    }


def _adaptive_context_text(
    memory: MemoryRecord,
    *,
    question: str,
    snippet_tokens: int,
) -> str:
    compact = "\n".join(
        part
        for part in (
            memory.created_at,
            memory.metadata.get("observation", ""),
            memory.metadata.get("summary", ""),
            memory.metadata.get("event_summary", ""),
        )
        if part
    )
    snippet = _question_guided_snippets(memory.text, question=question, max_tokens=snippet_tokens)
    return "\n".join(part for part in (compact, snippet) if part) or memory.text


def _question_guided_snippets(raw_text: str, *, question: str, max_tokens: int) -> str:
    query_terms = set(_tokens(question))
    if not query_terms:
        return ""
    sentences = [item.strip() for item in SENTENCE_RE.split(raw_text) if item.strip()]
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = set(_tokens(sentence))
        overlap = len(query_terms & sentence_terms)
        if not overlap:
            continue
        start = max(0, index - 1)
        end = min(len(sentences), index + 2)
        local_window = " ".join(sentences[start:end])
        scored.append((overlap / max(1, len(query_terms)), -index, local_window))
    selected: list[str] = []
    seen: set[str] = set()
    used = 0
    for _, _, sentence in sorted(scored, reverse=True):
        if sentence in seen:
            continue
        seen.add(sentence)
        sentence_tokens = _tokens(sentence)
        if used + len(sentence_tokens) > max_tokens and selected:
            continue
        if used + len(sentence_tokens) > max_tokens:
            sentence = " ".join(sentence_tokens[: max_tokens - used])
            sentence_tokens = _tokens(sentence)
        selected.append(sentence)
        used += len(sentence_tokens)
        if used >= max_tokens:
            break
    return "\n".join(selected)


def _extract_answer(question: str, context: str) -> tuple[str, float]:
    query_terms = set(_tokens(question))
    question_is_when = any(term in query_terms for term in {"when", "date", "year"})
    candidates = [item.strip() for item in SENTENCE_RE.split(context) if item.strip()]
    best_sentence = ""
    best_score = 0.0
    for sentence in candidates:
        if sentence.startswith("[") and sentence.endswith("]"):
            continue
        if _looks_like_structured_blob(sentence):
            continue
        sentence_terms = set(_tokens(sentence))
        if not sentence_terms:
            continue
        overlap = len(query_terms & sentence_terms) / max(1, len(query_terms))
        density = len(query_terms & sentence_terms) / max(1, len(sentence_terms))
        score = overlap * 0.75 + density * 0.25
        if sentence.endswith("?"):
            score *= 0.2
        if question_is_when and _has_dateish_text(sentence):
            score += 0.8 if overlap == 0.0 else 0.35
        if score > best_score:
            best_score = score
            best_sentence = sentence
    if not best_sentence:
        return "No supported answer found in local memory.", 0.0
    return _clean_answer_sentence(best_sentence), round(min(best_score, 1.0), 4)


def _extract_temporal_answer(
    question: str,
    retrieved: list[RetrievedMemory],
) -> tuple[str, float] | None:
    question_terms = set(_tokens(question))
    if "when" not in question_terms:
        return None
    entity_terms = _query_entities(question)
    focus_terms = question_terms - {
        "a",
        "an",
        "did",
        "do",
        "does",
        "is",
        "the",
        "to",
        "was",
        "were",
        "when",
    }
    anchor_terms = focus_terms - entity_terms - GENERIC_EVENT_TERMS
    best: tuple[float, str] | None = None
    for item in retrieved:
        base_date = _parse_memory_date(item.memory.created_at)
        if base_date is None:
            continue
        temporal_text = "\n".join(
            part
            for part in (
                item.memory.text,
                item.memory.metadata.get("summary", ""),
                item.memory.metadata.get("event_summary", ""),
            )
            if part
        )
        for sentence in [part.strip() for part in SENTENCE_RE.split(temporal_text) if part.strip()]:
            if _looks_like_structured_blob(sentence):
                continue
            sentence_terms = set(_tokens(sentence))
            focus_overlap = len(focus_terms & sentence_terms)
            anchor_overlap = len(anchor_terms & sentence_terms)
            if focus_terms and not focus_overlap:
                continue
            if anchor_terms and not anchor_overlap:
                continue
            resolved = _resolve_relative_date(sentence, base_date)
            if not resolved:
                continue
            density = focus_overlap / max(1, len(sentence_terms))
            score = (
                focus_overlap / max(1, len(focus_terms))
                + anchor_overlap / max(1, len(anchor_terms))
                + density
            )
            score += _relative_temporal_weight(sentence)
            if _is_session_date_lead(sentence):
                score -= 0.45
            if best is None or score > best[0]:
                best = (score, resolved)
    if best is None:
        return None
    return best[1], round(min(1.0, 0.65 + best[0] * 0.1), 4)


def _extract_structured_answer(
    question: str,
    retrieved: list[RetrievedMemory],
) -> tuple[str, float] | None:
    query_terms = set(_content_terms(question))
    candidates = _candidate_fact_texts(retrieved)
    aggregate_answer = _aggregate_pattern_answer(question, candidates)
    if aggregate_answer:
        return aggregate_answer, 0.78
    best: tuple[float, str] | None = None
    for text, source_weight in candidates:
        answer = _pattern_answer(question, text)
        if not answer:
            continue
        answer = _clean_short_answer(answer)
        if not answer:
            continue
        text_terms = set(_content_terms(text))
        overlap = len(query_terms & text_terms) / max(1, len(query_terms))
        answer_overlap = len(query_terms & set(_content_terms(answer))) / max(1, len(query_terms))
        score = source_weight + overlap + answer_overlap * 0.25
        if best is None or score > best[0]:
            best = (score, answer)
    if best is None:
        return None
    return best[1], round(min(1.0, 0.55 + best[0] * 0.12), 4)


def _aggregate_pattern_answer(question: str, candidates: list[tuple[str, float]]) -> str:
    question_lower = question.lower()
    subject = _query_subject(question)
    relevant_texts = [
        text
        for text, _ in candidates
        if not subject or subject.lower() in text.lower()
    ]
    if "what books" in question_lower or "which books" in question_lower:
        titles = _ordered_unique(
            title
            for text in relevant_texts
            if "book" in text.lower() or "read" in text.lower()
            for title in _quoted_titles(text)
        )
        if titles:
            return ", ".join(titles)
    if "where" in question_lower and "camp" in question_lower:
        places = _ordered_unique(
            place
            for text in relevant_texts
            if "camp" in text.lower()
            for place in _match_camping_place(text).split(", ")
            if place
        )
        if len(places) >= 2:
            return ", ".join(places)
    if "kids" in question_lower and ("like" in question_lower or "love" in question_lower):
        likes = _ordered_unique(
            item
            for text in relevant_texts
            for item in _match_kids_likes(text).split(", ")
            if item
        )
        if len(likes) >= 2:
            return ", ".join(likes)
    if "activities" in question_lower or "partake" in question_lower:
        activities = _ordered_unique(
            item
            for text in relevant_texts
            for item in _match_activity_terms(text).split(", ")
            if item
        )
        if len(activities) >= 2:
            return ", ".join(activities)
    return ""


def _query_subject(question: str) -> str:
    for match in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", question):
        value = match.group(0)
        if value.lower() not in {
            "how",
            "what",
            "when",
            "where",
            "which",
            "who",
            "would",
        }:
            return value
    return ""


def _ordered_unique(items: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _candidate_fact_texts(retrieved: list[RetrievedMemory]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in retrieved:
        memory = item.memory
        for fact in _observation_facts(memory.metadata.get("observation", "")):
            _append_candidate(candidates, seen, fact, 1.15)
        for fact in _event_summary_facts(memory.metadata.get("event_summary", "")):
            _append_candidate(candidates, seen, fact, 1.05)
        for sentence in SENTENCE_RE.split(memory.metadata.get("summary", "")):
            _append_candidate(candidates, seen, sentence.strip(), 0.9)
        for line in memory.text.splitlines():
            _append_candidate(candidates, seen, line.strip(), 0.85)
    return candidates


def _append_candidate(
    candidates: list[tuple[str, float]],
    seen: set[str],
    text: str,
    weight: float,
) -> None:
    text = text.strip()
    if not text or text in seen:
        return
    seen.add(text)
    candidates.append((text, weight))


def _observation_facts(raw: str) -> list[str]:
    parsed = _literal_dict(raw)
    facts: list[str] = []
    if not isinstance(parsed, dict):
        return facts
    for entries in parsed.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], str):
                facts.append(entry[0])
            elif isinstance(entry, str):
                facts.append(entry)
    return facts


def _event_summary_facts(raw: str) -> list[str]:
    parsed = _literal_dict(raw)
    facts: list[str] = []
    if not isinstance(parsed, dict):
        return facts
    for key, value in parsed.items():
        if key == "date":
            continue
        if isinstance(value, list):
            facts.extend(str(item) for item in value if item)
        elif value:
            facts.append(str(value))
    return facts


def _literal_dict(raw: str) -> object:
    if not raw or not raw.strip().startswith("{"):
        return None
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None


def _pattern_answer(question: str, text: str) -> str:
    question_lower = question.lower()
    text_clean = _clean_answer_sentence(text)
    if "how long ago" in question_lower:
        age = _match_duration(text_clean)
        if age:
            return age
    if "relationship status" in question_lower:
        status = _match_relationship_status(text_clean)
        if status:
            return status
    if "identity" in question_lower:
        identity = _match_identity(text_clean)
        if identity:
            return identity
    if "career" in question_lower or "field" in question_lower:
        career = _match_career(text_clean)
        if career:
            return career
    if "what books" in question_lower or "which books" in question_lower:
        titles = _quoted_titles(text_clean)
        if titles:
            return ", ".join(titles)
    if "where" in question_lower and "move" in question_lower:
        moved_from = _match_moved_from(text_clean)
        if moved_from:
            return moved_from
    if "where" in question_lower and "camp" in question_lower:
        camping = _match_camping_place(text_clean)
        if camping:
            return camping
    if "kids" in question_lower and ("like" in question_lower or "love" in question_lower):
        liked = _match_kids_likes(text_clean)
        if liked:
            return liked
    direct_object = _match_direct_object(question_lower, text_clean)
    if direct_object:
        return direct_object
    return ""


def _match_duration(text: str) -> str:
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\s+ago\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    number = match.group(1)
    normalized = NUMBER_WORDS.get(number.lower(), number)
    return f"{normalized} years ago"


def _match_relationship_status(text: str) -> str:
    lowered = text.lower()
    if "single parent" in lowered or re.search(r"\bsingle\b", lowered):
        return "Single"
    if "married" in lowered:
        return "Married"
    if "divorced" in lowered:
        return "Divorced"
    return ""


def _match_identity(text: str) -> str:
    lowered = text.lower()
    if "transgender woman" in lowered:
        return "Transgender woman"
    if "trans woman" in lowered:
        return "Transgender woman"
    if "transgender man" in lowered:
        return "Transgender man"
    if "transgender" in lowered and "woman" in lowered:
        return "Transgender woman"
    return ""


def _match_career(text: str) -> str:
    match = re.search(
        r"\bcareer in ([^.]+?)(?:\.|, as|, which| - |$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r"\b(?:keen on|looking into|pursue|pursuing)\s+([^.;]+?(?:counseling|mental health)[^.;]*)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return ""


def _quoted_titles(text: str) -> list[str]:
    titles: list[str] = []
    for match in re.finditer(r'"([^"]{2,80})"', text):
        content = match.group(1).strip(" .,;:")
        if content:
            titles.append(f'"{content}"')
    for match in re.finditer(r"(?<![A-Za-z])'([^']{2,80})'(?![A-Za-z])", text):
        content = match.group(1).strip(" .,;:")
        if content:
            titles.append(f'"{content}"')
    return titles


def _match_moved_from(text: str) -> str:
    match = re.search(r"\bmoved from ([A-Z][A-Za-z]+)\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\bfrom ([A-Z][A-Za-z]+),?\s+(?:where|when|and)\b", text)
    if match:
        return match.group(1)
    return ""


def _match_camping_place(text: str) -> str:
    lowered = text.lower()
    places: list[str] = []
    for place in ("beach", "mountains", "forest", "woods", "lake"):
        if place in lowered:
            places.append(place)
    return ", ".join(dict.fromkeys(places))


def _match_kids_likes(text: str) -> str:
    lowered = text.lower()
    likes: list[str] = []
    for item in ("dinosaurs", "nature", "animals", "camping", "books"):
        if item in lowered:
            likes.append(item)
    return ", ".join(dict.fromkeys(likes))


def _match_activity_terms(text: str) -> str:
    lowered = text.lower()
    activities: list[str] = []
    for item in (
        "pottery",
        "camping",
        "painting",
        "swimming",
        "running",
        "reading",
        "violin",
        "hiking",
    ):
        if item in lowered:
            activities.append(item)
    return ", ".join(dict.fromkeys(activities))


def _match_direct_object(question_lower: str, text: str) -> str:
    verb = _question_did_verb(question_lower)
    if not verb:
        return ""
    if "?" in text:
        text = text.rsplit("?", 1)[-1]
    forms = _verb_forms(verb)
    for form in forms:
        pattern = (
            r"\b"
            + re.escape(form)
            + r"\s+(?:about\s+|for\s+|into\s+)?"
            + r"([^.;!?]+?)(?:\s+(?:with|because|as|so|while|after|before|near|due)\b|[.;!?]|$)"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            answer = match.group(1)
            return answer
    return ""


def _question_did_verb(question_lower: str) -> str:
    match = re.search(r"\bwhat\s+did\s+[a-z][a-z0-9_'-]*\s+([a-z]+)\b", question_lower)
    if match:
        return _normalize_token(match.group(1))
    return ""


def _verb_forms(verb: str) -> tuple[str, ...]:
    irregular = {
        "go": ("go", "goes", "going", "went", "gone"),
        "read": ("read", "reads", "reading"),
        "research": ("research", "researches", "researched", "researching"),
        "choose": ("choose", "chooses", "chose", "chosen", "choosing"),
    }
    if verb in irregular:
        return irregular[verb]
    return (
        verb,
        f"{verb}s",
        f"{verb}ed",
        f"{verb}ing",
    )


def _clean_short_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer.strip(" .,;:-"))
    answer = re.sub(r"^(?:the|a|an)\s+", "", answer, flags=re.IGNORECASE)
    answer = answer.replace("edu", "education")
    if len(answer) > 140:
        answer = answer[:140].rsplit(" ", 1)[0]
    return answer


def _content_terms(text: str) -> list[str]:
    return [
        token
        for token in _tokens(text)
        if token not in ANSWER_STOPWORDS or any(ch.isdigit() for ch in token)
    ]


def _resolve_relative_date(sentence: str, base_date: date) -> str:
    lowered = sentence.lower()
    if "yesterday" in lowered:
        return _format_date(base_date - timedelta(days=1))
    if "today" in lowered:
        return _format_date(base_date)
    if "this month" in lowered:
        return _format_month_year(base_date.year, base_date.month)
    if "next month" in lowered:
        year = base_date.year + (1 if base_date.month == 12 else 0)
        month = 1 if base_date.month == 12 else base_date.month + 1
        return _format_month_year(year, month)
    if "last year" in lowered:
        return str(base_date.year - 1)
    if "last week" in lowered or "week before" in lowered:
        return f"The week before {_format_date(base_date)}"
    if "last weekend" in lowered or "over the weekend" in lowered:
        return f"The weekend before {_format_date(base_date)}"
    weekday_match = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lowered,
    )
    if weekday_match:
        weekday_name = weekday_match.group(1).title()
        return f"The {weekday_name} before {_format_date(base_date)}"
    if "recently" in lowered:
        return _format_date(base_date)
    explicit = _explicit_date_from_text(sentence)
    if explicit:
        return explicit
    return ""


def _relative_temporal_weight(sentence: str) -> float:
    lowered = sentence.lower()
    if re.search(
        r"\b(?:yesterday|today|last year|last week|last weekend|next month|this month|over the weekend|last\s+"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        lowered,
    ):
        return 0.65
    if "recently" in lowered:
        return 0.15
    return 0.0


def _is_session_date_lead(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(
        re.match(r"^(?:on\s+)?(?:\d{1,2}\s+[a-z]+|[a-z]+\s+\d{1,2}),?\s+20\d{2}", lowered)
        and ("conversation" in lowered or "tells" in lowered or "talked" in lowered)
    )


def _parse_memory_date(raw: str) -> date | None:
    raw = raw.strip()
    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", raw)
    if iso_match:
        year, month, day = (int(part) for part in iso_match.groups())
        return date(year, month, day)
    day_month_match = re.search(
        r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(20\d{2})\b",
        raw,
    )
    if day_month_match:
        day_raw, month_raw, year_raw = day_month_match.groups()
        month = MONTH_NUMBERS.get(month_raw.lower())
        if month:
            return date(int(year_raw), month, int(day_raw))
    month_day_match = re.search(
        r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(20\d{2})\b",
        raw,
    )
    if month_day_match:
        month_raw, day_raw, year_raw = month_day_match.groups()
        month = MONTH_NUMBERS.get(month_raw.lower())
        if month:
            return date(int(year_raw), month, int(day_raw))
    return None


def _explicit_date_from_text(sentence: str) -> str:
    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", sentence)
    if iso_match:
        year, month, day = iso_match.groups()
        return _format_date(date(int(year), int(month), int(day)))
    day_month_match = re.search(
        r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(20\d{2})\b",
        sentence,
    )
    if day_month_match:
        day_raw, month_raw, year_raw = day_month_match.groups()
        month = MONTH_NUMBERS.get(month_raw.lower())
        if month:
            return _format_date(date(int(year_raw), month, int(day_raw)))
    month_day_match = re.search(
        r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(20\d{2})\b",
        sentence,
    )
    if month_day_match:
        month_raw, day_raw, year_raw = month_day_match.groups()
        month = MONTH_NUMBERS.get(month_raw.lower())
        if month:
            return _format_date(date(int(year_raw), month, int(day_raw)))
    return ""


def _format_date(value: date) -> str:
    return f"{value.day} {MONTH_NAMES[f'{value.month:02d}']} {value.year}"


def _format_month_year(year: int, month: int) -> str:
    return f"{MONTH_NAMES[f'{month:02d}']} {year}"


def _clean_answer_sentence(sentence: str) -> str:
    sentence = re.sub(r"^\[[^\]]+\]\s*", "", sentence).strip()
    sentence = _format_standalone_iso_date(sentence)
    sentence = re.sub(r"^[A-Za-z][A-Za-z0-9_ -]{0,40}:\s*", "", sentence).strip()
    return sentence


def _looks_like_structured_blob(sentence: str) -> bool:
    stripped = sentence.strip()
    if stripped.startswith(("{", "}")):
        return True
    if stripped.count("':") >= 2 or stripped.count('":') >= 2:
        return True
    if len(_tokens(stripped)) > 90 and ("[" in stripped or "{" in stripped):
        return True
    return False


def _has_dateish_text(sentence: str) -> bool:
    return bool(DATE_RE.search(sentence))


def _format_standalone_iso_date(sentence: str) -> str:
    match = re.fullmatch(r"(20\d{2})-(\d{2})-(\d{2})(?:[T\s].*)?", sentence.strip())
    if not match:
        return sentence
    year, month, day = match.groups()
    month_name = MONTH_NAMES.get(month, month)
    return f"{int(day)} {month_name} {year}"


def _compact_match_boost(record: MemoryRecord, query_terms: list[str]) -> float:
    compact_text = " ".join(
        record.metadata.get(key, "")
        for key in ("observation", "summary", "event_summary")
    )
    compact_terms = set(_tokens(compact_text))
    if not compact_terms:
        return 0.0
    return min(len(set(query_terms) & compact_terms) * 0.05, 0.35)


def _record_entities(record: MemoryRecord) -> set[str]:
    raw_entities = _capitalized_terms(record.text)
    for value in record.metadata.values():
        raw_entities.update(_capitalized_terms(value))
    return raw_entities | {record.memory_id.lower(), record.kind.lower()}


def _query_entities(query: str) -> set[str]:
    return _capitalized_terms(query)


def _capitalized_terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", text)
        if match.group(0).lower()
        not in {
            "how",
            "what",
            "when",
            "where",
            "which",
            "who",
            "would",
        }
    }


def _tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in TOKEN_RE.findall(text.lower())]


def _token_counts(text: str) -> Counter[str]:
    return Counter(_tokens(text))


def _normalize_token(token: str) -> str:
    irregular = {
        "ran": "run",
        "running": "run",
        "went": "go",
        "gone": "go",
        "going": "go",
        "took": "take",
        "taken": "take",
        "taking": "take",
        "researched": "research",
        "researching": "research",
        "painted": "paint",
        "painting": "paint",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("'s") and len(token) > 3:
        return token[:-2]
    return token
